from typing import Any, Optional, Union, Callable, NamedTuple

import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from optax import tree_utils as otu
from optax._src import base, transform, clipping
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

from psgd_jax.utils import add_eps, apply_momentum


class PSGDXMatState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    a: jax.Array
    b: jax.Array


def scale_by_xmat(
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        update_global_norm_clip: optional float, clip updates by global norm.
        step_normalizer_order: str, '1st' or '2nd'.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)

    def init_fn(params):
        key = seed if seed is not None else jax.random.PRNGKey(36)

        # momentum
        mu = None
        if b1 > 0:
            print("PSGD: Using momentum.")
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioner
        n_params = sum([x.size for x in jax.tree.leaves(params)])
        a = jnp.ones((n_params,), jnp.float32)
        b = jnp.zeros((n_params,), jnp.float32)

        # initial state
        return PSGDXMatState(count=jnp.zeros([], jnp.int32), key=key, mu=mu, a=a, b=b)

    def update_fn(
        updates: base.Updates,
        state: PSGDXMatState,
        params: base.Params = None,
        Hvp: Optional[base.Updates] = None,
        vector: Optional[base.Updates] = None,
        update_preconditioner: Optional[bool] = None,
    ):
        del params
        # use hessian preconditioning if hessian provided
        # otherwise use gg^T whitening type preconditioning
        hessian_based_preconditioning = Hvp is not None
        if hessian_based_preconditioning and (
            vector is None or update_preconditioner is None
        ):
            raise ValueError(
                "If using Hessian-based preconditioning, must also pass in random vector and "
                "update_preconditioner to PSGD's update function. See README for more info."
            )

        count_inc = safe_int32_increment(state.count)
        key = state.key

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        def _update_precond(key: PRNGKey, state: PSGDXMatState, Hvs, vs):
            v = jnp.concatenate([jnp.reshape(x, (-1,)) for x in jax.tree.leaves(vs)], 0)
            h = jnp.concatenate(
                [jnp.reshape(x, (-1,)) for x in jax.tree.leaves(Hvs)], 0
            )

            # init a
            if precond_init_scale is not None:
                init_scale = precond_init_scale
            else:
                if hessian_based_preconditioning:
                    init_scale = (jnp.sum(v * v) / jnp.sum(h * h)) ** 0.25
                else:
                    init_scale = (len(h) / jnp.sum(jnp.square(h))) ** 0.25
            a = jax.lax.cond(
                state.count == 0, lambda: state.a * init_scale, lambda: state.a
            )

            # update preconditioner
            a, b = _update_precond_Xmat_math_(
                a, state.b, v, h, precond_lr_in, step_normalizer_order, precision
            )

            return key, a, b

        def _dont_update_precond(key, state, Hvs, vs):
            return key, state.a, state.b

        if not hessian_based_preconditioning:
            # update cond and vector not passed in, create here
            key, subkey = jax.random.split(key)
            update_preconditioner = jnp.logical_or(
                jax.random.uniform(subkey) < preconditioner_update_probability,
                state.count < 2,
            )
            key, subkey = jax.random.split(key)
            vector = otu.tree_random_like(subkey, updates, jax.random.normal)
            # use grads as Hvp
            Hvp = updates

        key, a, b = jax.lax.cond(
            update_preconditioner,
            _update_precond,
            _dont_update_precond,
            key,
            state,
            Hvp,
            vector,
        )

        # momentum
        mu = None
        if state.mu is not None:
            updates, mu = apply_momentum(updates, state.mu, count_inc, b1, nesterov)

        # preconditioning
        flat_updates = jnp.concatenate(
            [jnp.reshape(x, (-1,)) for x in jax.tree.leaves(updates)], 0
        )
        flat_updates = _precond_grad_Xmat_math(a, b, flat_updates)
        with jax.ensure_compile_time_eval():
            params_struct = jax.tree.structure(updates)
            param_sizes = [x.size for x in jax.tree.leaves(updates)]
            param_cumsizes = [x.item() for x in jnp.cumsum(jnp.array(param_sizes))]
            param_shapes = [x.shape for x in jax.tree.leaves(updates)]
        flat_updates = [
            jnp.reshape(flat_updates[idx - size : idx], s)
            for idx, size, s in zip(param_cumsizes, param_sizes, param_shapes)
        ]
        updates = jax.tree.unflatten(params_struct, flat_updates)

        # clipping
        if update_global_norm_clip:
            updates, _ = clipping.clip_by_global_norm(update_global_norm_clip).update(
                updates, base.EmptyState
            )

        mu = otu.tree_cast(mu, mu_dtype)
        state = PSGDXMatState(count=count_inc, key=key, mu=mu, a=a, b=b)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def xmat(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements XMat PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate for the optimizer.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay.
        mask: optional mask for weight decay.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        update_global_norm_clip: optional float, clip updates by global norm.
        step_normalizer_order: str, '1st' or '2nd'.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_xmat(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            update_global_norm_clip=update_global_norm_clip,
            step_normalizer_order=step_normalizer_order,
            seed=seed,
            mu_dtype=mu_dtype,
            precision=precision,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def _update_precond_Xmat_math_(a, b, v, h, precond_lr, step_normalizer, precision):
    """
    Update preconditioner Q = diag(a) + adiag(b) with (vector, Hessian-vector product) = (v, h).
    """
    with jax.default_matmul_precision(precision):
        Qh = a * h + b * jnp.flip(h, 0)
        aflip, bflip = jnp.flip(a, 0), jnp.flip(b, 0)
        invQtv = (aflip * v - bflip * jnp.flip(v, 0)) / (a * aflip - b * bflip)

        u, v = Qh * Qh, invQtv * invQtv
        nablaA = u - v
        nablaB = Qh * jnp.flip(Qh, 0) - invQtv * jnp.flip(invQtv, 0)
        q, r = jnp.divmod(len(nablaB), 2)
        nablaB = jnp.where(r == 1, nablaB.at[q].set(0), nablaB)

        if step_normalizer == "2nd":
            mu = precond_lr / add_eps(jnp.max(u + v))
        else:
            mu = precond_lr / add_eps(
                jnp.maximum(jnp.max(jnp.abs(nablaA)), jnp.max(jnp.abs(nablaB)))
            )

        a -= mu * (nablaA * a + nablaB * bflip)
        b -= mu * (nablaA * b + nablaB * aflip)

        return a, b


def _precond_grad_Xmat_math(a, b, g):
    """
    Preconditioning gradient g with Q = diag(a) + adiag(b).

    All variables here are either matrices or column vectors.
    """
    ab = a * b
    return (a * a + jnp.flip(b * b, 0)) * g + (ab + jnp.flip(ab, 0)) * jnp.flip(g, 0)
