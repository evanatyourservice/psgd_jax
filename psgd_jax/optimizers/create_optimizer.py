from typing import Optional, Union, Tuple, Callable

import flax
import jax.lax
import jax.numpy as jnp
import optax
from optax._src import combine, transform

from psgd_jax.optimizers.shampoo_caspr import distributed_shampoo, GraftingType
from psgd_jax.optimizers.first_order_optimizers import (
    scale_by_belief,
    scale_by_adam3,
    scale_by_adam,
    scale_by_sign_sgd,
)
from psgd_jax.optimizers.psgd import scale_by_psgd
from psgd_jax.optimizers.optimizer_utils import norm_grads as norm_grads_transformation


# only lets through kernel weights for weight decay
_kernels = flax.traverse_util.ModelParamTraversal(lambda p, _: "kernel" in p)


def _kernel_mask(params):
    all_false = jax.tree.map(lambda _: False, params)
    out = _kernels.update(lambda _: True, all_false)
    return out


def create_optimizer(
    optimizer: str,
    learning_rate: float,
    min_learning_rate: float,
    norm_grads: Optional[str],
    beta1: float,
    beta2: float,
    epsilon: float,
    nesterov: bool,
    weight_decay: float,
    lr_schedule: str,
    schedule_free: bool,
    warmup_steps: int,
    total_train_steps: int,
    gradient_clip: Optional[float],
    pmap_axis_name: Optional[str],
    graft: bool,
    shampoo_precond_every_n: int,
    shampoo_precond_block_size: int,
    psgd_precond_type: str,
    psgd_update_prob: float,
    psgd_rank: int,
    psgd_heavyball: bool,
    psgd_feed_into_adam: bool,
    psgd_precond_lr: float,
    psgd_precond_init_scale: Optional[float],
    cooldown_steps: int = 10000,
    rsqrt_timescale: int = 10000,
    exp_decay_rate: float = 0.1,
    mu_dtype: Union[jnp.dtype, str] = jnp.float32,
) -> Tuple[optax.GradientTransformationExtraArgs, Callable[[int], float]]:
    if norm_grads is not None and gradient_clip is not None:
        print(
            "WARNING: both norm_grads and gradient_clip are set. "
            "Only norm_grads will be applied."
        )
        gradient_clip = None
    warmup_steps = max(0, warmup_steps)
    # schedule-free momentum adjusting
    sf_b1 = beta1
    if schedule_free:
        if lr_schedule != "flat_w_warmup":
            print("WARNING: changing lr schedule to flat_w_warmup for schedule-free.")
            lr_schedule = "flat_w_warmup"
        beta1 = 0.0

    # learning rate schedule
    learning_rate_in = make_schedule(
        lr_schedule,
        learning_rate,
        min_learning_rate,
        warmup_steps,
        total_train_steps,
        cooldown_steps,
        rsqrt_timescale,
        exp_decay_rate,
    )

    # norm grads
    if norm_grads is not None:
        chain = [
            norm_grads_transformation(layerwise=norm_grads in ["layer", "layerwise"])
        ]
    else:
        chain = []

    # gradient clipping
    if gradient_clip is not None:
        if gradient_clip > 0.0:
            chain += [optax.clip_by_global_norm(gradient_clip)]

    # optimizer
    if optimizer in ["adam", "adamw", "lamb"]:
        chain += [
            scale_by_adam(
                b1=beta1, b2=beta2, eps=epsilon, mu_dtype=mu_dtype, nesterov=nesterov
            )
        ]
    elif optimizer == "adagrad":
        chain += [transform.scale_by_rss(initial_accumulator_value=0.0, eps=1e-10)]
    elif optimizer == "adagrad_momentum":
        chain += [
            transform.scale_by_rss(initial_accumulator_value=0.0, eps=1e-10),
            optax.trace(beta1),
        ]
    elif optimizer == "sgd":
        chain += [
            transform.trace(decay=beta1, nesterov=nesterov, accumulator_dtype=mu_dtype)
        ]
    elif optimizer == "sign_sgd":
        # skip norm and clip for sign_sgd
        chain = [scale_by_sign_sgd()]
    elif optimizer == "rmsprop":
        chain += [transform.scale_by_rms(decay=beta2, eps=epsilon)]
    elif optimizer == "radam":
        chain += [
            transform.scale_by_radam(b1=beta1, b2=beta2, eps=epsilon, nesterov=nesterov)
        ]
    elif optimizer == "sm3":
        chain += [transform.scale_by_sm3(b1=beta1, b2=beta2, eps=epsilon)]
    elif optimizer == "adabelief":
        chain += [
            scale_by_belief(
                b1=sf_b1,  # momentum needed
                b2=beta2,
                eps=epsilon,
                rectify=not schedule_free,
                mu_dtype=mu_dtype,
                schedulefree=schedule_free,
            )
        ]
    elif optimizer == "novograd":
        chain += [
            transform.scale_by_novograd(
                b1=beta1, b2=beta2, eps=epsilon, mu_dtype=mu_dtype
            )
        ]
    elif optimizer == "adam3":
        chain += [
            scale_by_adam3(
                b1=sf_b1,  # momentum needed
                b2=beta2,
                eps=epsilon,
                mu_dtype=mu_dtype,
                schedulefree=schedule_free,
            )
        ]
    elif optimizer == "lion":
        # skip norm and clip for lion
        chain = [optax.scale_by_lion(b1=beta1, b2=beta2, mu_dtype=mu_dtype)]
    elif optimizer in ["shampoo", "caspr"]:
        # skip norm and clip for shampoo
        chain = [
            distributed_shampoo(
                learning_rate=learning_rate_in,
                caspr_variant=optimizer == "caspr",
                block_size=shampoo_precond_block_size,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                preconditioning_compute_steps=shampoo_precond_every_n,
                nesterov=nesterov,
                batch_axis_name=pmap_axis_name,
                clip_by_scaled_gradient_norm=gradient_clip,
                graft_type=(
                    GraftingType.RMSPROP_NORMALIZED if graft else GraftingType.SGD
                ),
            )
        ]
    elif optimizer == "psgd":
        # skip norm and clip for psgd
        chain = [
            scale_by_psgd(
                preconditioner_type=psgd_precond_type,
                preconditioner_update_probability=psgd_update_prob,
                b1=beta1,
                heavyball=psgd_heavyball,
                nesterov=nesterov,
                gradient_clip=gradient_clip,
                uvd_rank_of_approximation=psgd_rank,
                precond_lr=psgd_precond_lr,
                precond_init_scale=psgd_precond_init_scale,
                feed_into_adam=psgd_feed_into_adam,
                graft_adam_lr=graft,
                adam_b2=beta2,
                adam_norm_grads=norm_grads,
                mu_dtype=mu_dtype,
            )
        ]
    else:
        raise NotImplementedError(f"{optimizer} optimizer not implemented.")

    if optimizer in ["shampoo", "caspr"]:  # standalone
        optimizer = chain[0]
    else:
        if weight_decay > 0:
            chain += [transform.add_decayed_weights(weight_decay, mask=_kernel_mask)]

        if optimizer == "lamb":
            # trust ratio after weight decay
            chain += [optax.scale_by_trust_ratio()]

        chain += [transform.scale_by_learning_rate(learning_rate_in)]

        optimizer = combine.chain(*chain)

    if schedule_free:
        # wraps any optimizer in schedule-free
        optimizer = optax.contrib.schedule_free(optimizer, learning_rate_in, sf_b1)

    return optimizer, learning_rate_in


def make_schedule(
    lr_schedule: str,
    learning_rate: float,
    min_learning_rate: float,
    warmup_steps: int,
    total_train_steps: int,
    cooldown_steps: int = 10000,
    rsqrt_timescale: int = 10000,
    exp_decay_rate: float = 0.1,
) -> Callable[[int], float]:
    if lr_schedule == "cosine":
        if warmup_steps == 0:
            schedule = optax.cosine_decay_schedule(learning_rate, total_train_steps)
        else:
            schedule = optax.warmup_cosine_decay_schedule(
                0.0,
                learning_rate,
                warmup_steps,
                decay_steps=total_train_steps,
                end_value=min_learning_rate,
            )
    elif lr_schedule == "linear":
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
        )
        decay_fn = optax.linear_schedule(
            init_value=learning_rate,
            end_value=min_learning_rate,
            transition_steps=total_train_steps - warmup_steps,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
        )
    elif lr_schedule == "flat_w_warmup":
        warmup_fn = optax.linear_schedule(
            init_value=min_learning_rate,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        decay_fn = optax.constant_schedule(learning_rate)
        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn], boundaries=[warmup_steps]
        )
    elif lr_schedule == "exponential":
        exp_ts = 100_000
        if warmup_steps == 0:
            schedule = optax.exponential_decay(
                learning_rate, exp_ts, exp_decay_rate, end_value=min_learning_rate
            )
        else:
            schedule = optax.warmup_exponential_decay_schedule(
                0.0,
                learning_rate,
                warmup_steps,
                exp_ts,
                exp_decay_rate,
                end_value=min_learning_rate,
            )
    elif lr_schedule == "rsqrt":
        if cooldown_steps == 0:
            print("WARNING: cooldown_steps is 0 but rsqrt schedule is used")
        schedule = _rsqrt_lr_schedule(
            learning_rate,
            total_train_steps,
            warmup_steps,
            cooldown_steps,
            rsqrt_timescale,
        )
    elif lr_schedule in ["trapezoid", "trapezoidal"]:
        # from https://arxiv.org/abs/2405.18392
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps
        )
        constant_fn = optax.constant_schedule(learning_rate)
        decay_fn = _one_minus_sqrt_schedule(learning_rate, cooldown_steps)
        schedule = optax.join_schedules(
            schedules=[warmup_fn, constant_fn, decay_fn],
            boundaries=[warmup_steps, total_train_steps - cooldown_steps],
        )
    elif lr_schedule is None:
        if warmup_steps > 0:
            print("WARNING: warmup_steps is ignored when lr_schedule is None.")
        schedule = lambda _: learning_rate
    else:
        raise NotImplementedError(f"lr_schedule {lr_schedule} not implemented.")
    return schedule


def _rsqrt_lr_schedule(
    learning_rate, total_steps, warmup_steps=0, cooldown_steps=0, timescale=10000
):
    def step_fn(step):
        lr = learning_rate
        shift = timescale - warmup_steps
        lr = jnp.where(
            warmup_steps < step, lr / jnp.sqrt((step + shift) / timescale), lr
        )
        if warmup_steps > 0:
            lr = lr * jnp.minimum(1.0, step / warmup_steps)
        if cooldown_steps > 0:
            lr = lr * jnp.minimum(1.0, (total_steps - step) / cooldown_steps)

        return jnp.clip(lr, min=0.0)

    return step_fn


def _one_minus_sqrt_schedule(learning_rate: float, transition_steps: int):
    def step_fn(step):
        factor = 1 - jnp.sqrt(step / transition_steps)
        lr = learning_rate * factor
        return jnp.clip(lr, min=0.0)

    return step_fn
