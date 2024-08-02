import os
import time
from functools import partial
from pprint import pprint
from typing import Optional
import wandb

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from psgd_jax.optimizers.create_optimizer import create_optimizer
from psgd_jax.image_classification.network_utils import normal_init
from psgd_jax.image_classification.models.ViT import LearnablePositionalEncoding
from psgd_jax.optimizers.psgd import psgd_hvp_helper


# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

wandb.require("core")
jax.config.update("jax_default_matmul_precision", jax.lax.Precision.HIGH)


class RNN(nn.Module):
    dim_hidden: int
    batch_size: int

    @nn.compact
    def __call__(self, xs):
        W1x = self.param(
            "W1x", nn.initializers.truncated_normal(0.1), (2, self.dim_hidden)
        )
        W1h = self.param(
            "W1h", nn.initializers.orthogonal(), (self.dim_hidden, self.dim_hidden)
        )
        b1 = self.param("b1", nn.initializers.zeros_init(), (self.dim_hidden,))
        W2 = self.param(
            "W2", nn.initializers.truncated_normal(0.1), (self.dim_hidden, 1)
        )
        b2 = self.param("b2", nn.initializers.zeros_init(), [1])

        def _loop_body(h, x):
            h = jnp.tanh(x @ W1x + h @ W1h + b1)
            return h, None

        h = jnp.zeros((self.batch_size, self.dim_hidden))
        xs = jnp.swapaxes(xs, 0, 1)
        h, _ = jax.lax.scan(_loop_body, h, xs, unroll=2)

        return h @ W2 + b2


class TransformerBlock(nn.Module):
    n_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, q):
        enc_dim = q.shape[-1]

        q2 = nn.LayerNorm(use_bias=False)(q)
        q2 = nn.SelfAttention(
            num_heads=self.n_heads,
            kernel_init=normal_init,
            use_bias=False,
            normalize_qk=True,
            deterministic=True,
        )(q2)
        b = self.param("att_bias", nn.initializers.zeros, (enc_dim,))
        q2 = q2 + jnp.reshape(b, (1, 1, enc_dim))
        q = q + q2

        q2 = nn.LayerNorm(use_bias=False)(q)
        q2 = nn.Dense(features=self.ff_dim, kernel_init=normal_init)(q2)
        q2 = nn.silu(q2)
        q2 = nn.Dense(features=enc_dim, kernel_init=normal_init)(q2)
        q = q + q2

        return q


class Transformer(nn.Module):
    n_layers: int
    enc_dim: int
    n_heads: int
    ff_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        # embed
        x = nn.Dense(features=self.enc_dim, kernel_init=normal_init, use_bias=False)(x)
        x *= jnp.sqrt(self.enc_dim)

        # positional encoding
        x = LearnablePositionalEncoding()(x)

        # transformer blocks
        for _ in range(self.n_layers):
            x = TransformerBlock(n_heads=self.n_heads, ff_dim=self.ff_dim)(x)
        x = nn.LayerNorm(use_bias=False)(x)

        # mean pool
        x = jnp.mean(x, axis=-2)

        return nn.Dense(features=1, kernel_init=normal_init, use_bias=False)(x)


def run_xor_experiment(
    log_to_wandb: bool = True,
    wandb_entity: str = "",
    seed: int = 5,
    criteria_threshold: float = 0.1,
    total_steps: int = 100000,
    batch_size: int = 128,
    seq_len: int = 32,
    model_type: str = "rnn",
    dim_hidden: int = 32,
    n_layers: int = 1,
    n_heads: int = 2,
    ff_dim: int = 32,
    l2_reg: float = 0.0,
    group_n_train_steps: int = 100,
    learning_rate: float = 0.01,
    min_learning_rate: float = 0.0,
    lr_schedule: str = None,
    warmup_steps: int = 0,
    cooldown_steps: int = 0,
    schedule_free: bool = False,
    optimizer: str = "psgd",
    norm_grads: str = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    gradient_clip: float = 1.0,
    graft: bool = False,
    shampoo_precondition_every_n: int = 10,
    shampoo_precond_block_size: int = 128,
    psgd_precond_type: str = "uvd",
    psgd_use_hessian: bool = False,
    psgd_update_probability: float = 1.0,
    psgd_rank: int = 10,
    psgd_heavyball: bool = False,
    psgd_feed_into_adam: bool = False,
    psgd_precond_lr: float = 0.01,
    psgd_precond_init_scale: Optional[float] = None,
    mu_dtype: str = "float32",
):
    """Train a model on the XOR task.

    Args:
        log_to_wandb: bool, whether to log to wandb.
        wandb_entity: str, wandb entity to log to.
        seed: int, seed for random number generation.
        criteria_threshold: float, loss threshold for successful training.
        total_steps: int, total number of training steps.
        batch_size: int, batch size for training.
        seq_len: int, length of the sequence to solve.
        model_type: str, model type to use, either "rnn" or "transformer".
        dim_hidden: int, hidden dimension for the model.
        n_layers: int, number of layers for the transformer model.
        n_heads: int, number of heads for the transformer model.
        ff_dim: int, feed-forward dimension for the transformer model.
        l2_reg: float, L2 regularization, psgd style with random strength per param.
        group_n_train_steps: int, number of training steps to group within jit.
        learning_rate: float, learning rate for the optimizer.
        min_learning_rate: float, minimum learning rate for the optimizer.
        lr_schedule: str, learning rate schedule for the optimizer.
        warmup_steps: int, number of warmup steps for the learning rate schedule.
        cooldown_steps: int, number of cooldown steps for the learning rate schedule.
        schedule_free: bool, whether to use a schedule-free optimizer.
        optimizer: str, optimizer to use.
        norm_grads: str, 'global', 'layer', or None, whether to normalize gradients.
        beta1: float, beta1 for the optimizer.
        beta2: float, beta2 for the optimizer.
        epsilon: float, epsilon for the optimizer.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay for the optimizer.
        gradient_clip: float, gradient clip value.
        graft: bool, whether to graft to adam in psgd, shampoo, caspr. Default for
            psgd should be False, for shampoo and caspr should be True.
        shampoo_precondition_every_n: int, precondition every n steps.
        shampoo_precond_block_size: int, precondition block size.
        psgd_precond_type: str, preconditioner type for psgd.
        psgd_use_hessian: bool, whether to use hessian for psgd, otherwise
            use gradient whitening.
        psgd_update_probability: float, precond update probability for psgd.
        psgd_rank: int, rank for UVd psgd.
        psgd_heavyball: bool, whether to use heavyball momentum for psgd.
        psgd_feed_into_adam: bool, whether to feed precond grads into adam.
        psgd_precond_lr: float, preconditioner learning rate for psgd.
        psgd_precond_init_scale: float, initial scale for the preconditioner.
        mu_dtype: str, momentum dtype for the optimizer.
    """
    if log_to_wandb:
        if not wandb_entity:
            print(
                "WARNING: No wandb entity provided, running without logging to wandb."
            )
            log_to_wandb = False
        else:
            locals_ = locals()
            if not os.environ["WANDB_API_KEY"]:
                raise ValueError(
                    "No WANDB_API_KEY found in environment variables, see readme "
                    "for instructions on setting wandb API key."
                )
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
            if schedule_free:
                name = f"schedule-free_{optimizer}_seqlen={seq_len}"
            else:
                name = f"{optimizer}_seqlen_{seq_len}"
            wandb.init(
                entity=wandb_entity, project="opt_xor", name=name, config=locals_
            )

    print(f"Number of devices: {jax.device_count()}")
    if jax.local_device_count() > 1:
        print("Performing separate experiment on each device.")

    def generate_train_data(key):
        @jax.vmap
        def single_sample(key):
            x = jnp.zeros([seq_len, 2])
            y = jnp.zeros([1])

            key, subkey = jax.random.split(key)
            x = x.at[:, 0].set(
                jax.random.rademacher(subkey, [seq_len], dtype=jnp.float32)
            )

            key, subkey = jax.random.split(key)
            i1 = (jax.random.uniform(subkey) * 0.1 * seq_len).astype(jnp.int32)
            key, subkey = jax.random.split(key)
            i2 = (jax.random.uniform(subkey) * 0.4 * seq_len + 0.1 * seq_len).astype(
                jnp.int32
            )
            x = x.at[i1, 1].set(1.0)
            x = x.at[i2, 1].set(1.0)

            one = jnp.ones([1])
            xor1 = jax.lax.dynamic_slice(x, [i1, 0], [1, 1])[0][0]
            xor2 = jax.lax.dynamic_slice(x, [i2, 0], [1, 1])[0][0]
            y = jnp.where(jnp.equal(xor1, xor2), -one, one)

            return x, y

        keys = jax.random.split(key, batch_size)
        return single_sample(keys)

    opt, _ = create_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        norm_grads=norm_grads,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        nesterov=nesterov,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        schedule_free=schedule_free,
        warmup_steps=warmup_steps,
        total_train_steps=total_steps,
        gradient_clip=gradient_clip,
        graft=graft,
        pmap_axis_name=None,
        shampoo_precond_every_n=shampoo_precondition_every_n,
        shampoo_precond_block_size=shampoo_precond_block_size,
        psgd_precond_type=psgd_precond_type,
        psgd_update_prob=psgd_update_probability,
        psgd_rank=psgd_rank,
        psgd_heavyball=psgd_heavyball,
        psgd_feed_into_adam=psgd_feed_into_adam,
        psgd_precond_lr=psgd_precond_lr,
        psgd_precond_init_scale=psgd_precond_init_scale,
        cooldown_steps=cooldown_steps,
        mu_dtype=mu_dtype,
    )

    # model
    if model_type == "rnn":
        model = RNN(dim_hidden=dim_hidden, batch_size=batch_size)
    elif model_type == "transformer":
        model = Transformer(
            n_layers=n_layers, enc_dim=dim_hidden, n_heads=n_heads, ff_dim=ff_dim
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    def init_fn(key):
        key, subkey = jax.random.split(key)
        params = model.init(subkey, jnp.ones((1, seq_len, 2)))["params"]
        print(f"Number of parameters: {sum(p.size for p in jax.tree.leaves(params))}")
        opt_state = opt.init(params)
        return params, opt_state, key

    # initialize
    # unique keys for separate experiments
    keys = jnp.stack(
        [jax.random.PRNGKey(seed + i) for i in range(jax.local_device_count())]
    )
    params, opt_state, key = jax.pmap(init_fn)(keys)

    def train_loss(params, key, xy_pair):  # logistic loss
        """Compute the loss for training."""
        key1, key2 = jax.random.split(key)
        pred = model.apply({"params": params}, xy_pair[0], rngs={"params": key1})
        loss = -jnp.mean(jnp.log(jax.nn.sigmoid(xy_pair[1] * pred)))

        if l2_reg > 0:
            # randomized l2 regularization psgd style
            rand = jax.random.uniform(key2)
            loss += rand * l2_reg * optax.global_norm(params) ** 2

        return loss

    def train_step(params, opt_state, key):
        """Train for one step."""
        key, subkey = jax.random.split(key)
        batch = generate_train_data(subkey)

        if optimizer == "psgd" and psgd_use_hessian:
            # use helper to calc hvp and pass into PSGD
            loss, grads, hvp, vector, update_precond = psgd_hvp_helper(
                key,
                train_loss,
                params,
                loss_fn_extra_args=(subkey, batch),
                has_aux=False,
                preconditioner_update_probability=psgd_update_probability,
            )
            updates, opt_state = opt.update(
                grads,
                opt_state,
                params,
                Hvp=hvp,
                vector=vector,
                update_preconditioner=update_precond,
            )
        else:
            key, subkey = jax.random.split(key)
            loss, grads = jax.value_and_grad(train_loss)(params, subkey, batch)

            # unit norm grads
            if norm_grads is not None:
                global_norm = optax.global_norm(grads)
                grads = jax.tree.map(lambda g: g / global_norm, grads)

            updates, opt_state = opt.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        return params, opt_state, key, loss

    def scan_n_steps(n, params, opt_state, key):
        """Train for n steps."""

        def _loop_body(carry, _):
            params, opt_state, key = carry
            params, opt_state, key, loss = train_step(params, opt_state, key)
            return (params, opt_state, key), loss

        carry, losses = jax.lax.scan(
            _loop_body, (params, opt_state, key), length=n, unroll=2
        )
        params, opt_state, key = carry
        return params, opt_state, key, jnp.mean(losses)

    scan_n_steps = jax.pmap(
        scan_n_steps, static_broadcasted_argnums=(0,), donate_argnums=(1, 2, 3)
    )

    losses = []
    for num_iter in range(total_steps // group_n_train_steps):
        params, opt_state, key, loss = scan_n_steps(
            group_n_train_steps, params, opt_state, key
        )

        losses.append(loss)
        print(
            ("Iteration: {}; loss: " + "{:.4f} " * jax.device_count()).format(
                (num_iter + 1) * group_n_train_steps, *losses[-1]
            )
        )

        if log_to_wandb:
            wandb.log(
                {
                    f"device_{i:02}_loss": loss.item()
                    for i, loss in enumerate(losses[-1])
                },
                step=(num_iter + 1) * group_n_train_steps,
            )

        concat_losses = jnp.stack(losses)
        if jnp.all(jnp.any(jnp.less(concat_losses, criteria_threshold), axis=0)):
            print("All experiments successful, stopping early.")
            break

    number_successful = jnp.sum(
        jnp.any(jnp.less(concat_losses, criteria_threshold), axis=0)
    )
    print(
        f"Number of successful experiments: {number_successful} out of {jax.device_count()}"
    )
    step_solved = (
        jnp.argmax(jnp.less(concat_losses, criteria_threshold), axis=0) + 1
    ) * group_n_train_steps
    step_solved = jnp.where(step_solved == group_n_train_steps, -1, step_solved)
    print(f"Step solved: {step_solved} (-1 means not solved)")
    average_steps = jnp.mean(step_solved[step_solved != -1])
    print(f"Average number of steps to solve: {average_steps}")

    if log_to_wandb:
        wandb.finish()

    return number_successful.item(), step_solved, average_steps.item()


if __name__ == "__main__":
    fn = partial(
        run_xor_experiment,
        log_to_wandb=True,
        wandb_entity="",
        seed=5,
        criteria_threshold=0.1,
        total_steps=100_000,
        batch_size=128,
        seq_len=40,
        model_type="rnn",
        dim_hidden=32,
        n_layers=1,
        n_heads=2,
        ff_dim=32,
        l2_reg=1e-6,
        group_n_train_steps=100,
        learning_rate=0.01,
        min_learning_rate=0.0,
        lr_schedule="linear",
        warmup_steps=0,
        cooldown_steps=0,
        schedule_free=False,
        optimizer="psgd",
        norm_grads=None,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        nesterov=True,
        weight_decay=0.0,
        gradient_clip=1.0,
        graft=False,
        shampoo_precondition_every_n=10,
        shampoo_precond_block_size=128,
        psgd_precond_type="uvd",
        psgd_use_hessian=True,
        psgd_update_probability=1.0,
        psgd_rank=10,
        psgd_heavyball=False,
        psgd_feed_into_adam=False,
        psgd_precond_lr=0.01,
        psgd_precond_init_scale=None,
        mu_dtype="float32",
    )

    results = []

    print("TESTING PSGD UVD")
    t_start = time.time()
    number_successful, step_solved, average_steps = jax.block_until_ready(fn())
    time_taken = time.time() - t_start
    print(f"Time taken: {time_taken:.2f}s")
    results.append(
        {
            "optimizer": "psgd uvd",
            "number_successful": number_successful,
            "step_solved": step_solved,
            "average_steps": average_steps,
            "time_taken": time_taken,
        }
    )

    print("TESTING CASPR")
    t_start = time.time()
    number_successful, step_solved, average_steps = jax.block_until_ready(
        fn(learning_rate=0.0003, optimizer="caspr", beta1=0.9, graft=True)
    )
    time_taken = time.time() - t_start
    print(f"Time taken: {time_taken:.2f}s")
    results.append(
        {
            "optimizer": "caspr",
            "number_successful": number_successful,
            "step_solved": step_solved,
            "average_steps": average_steps,
            "time_taken": time_taken,
        }
    )

    print("TESTING ADAM")
    t_start = time.time()
    number_successful, step_solved, average_steps = jax.block_until_ready(
        fn(learning_rate=0.0001, optimizer="adam", beta1=0.9)
    )
    time_taken = time.time() - t_start
    print(f"Time taken: {time_taken:.2f}s")
    results.append(
        {
            "optimizer": "adam",
            "number_successful": number_successful,
            "step_solved": step_solved,
            "average_steps": average_steps,
            "time_taken": time_taken,
        }
    )

    print("TESTING ADABELIEF")
    t_start = time.time()
    number_successful, step_solved, average_steps = jax.block_until_ready(
        fn(learning_rate=0.0001, optimizer="adabelief", beta1=0.9)
    )
    time_taken = time.time() - t_start
    print(f"Time taken: {time_taken:.2f}s")
    results.append(
        {
            "optimizer": "adabelief",
            "number_successful": number_successful,
            "step_solved": step_solved,
            "average_steps": average_steps,
            "time_taken": time_taken,
        }
    )

    print("TESTING SCHEDULE-FREE")
    t_start = time.time()
    number_successful, step_solved, average_steps = jax.block_until_ready(
        fn(learning_rate=0.0025, optimizer="adam", beta1=0.95, schedule_free=True)
    )
    time_taken = time.time() - t_start
    print(f"Time taken: {time_taken:.2f}s")
    results.append(
        {
            "optimizer": "adam schedule-free",
            "number_successful": number_successful,
            "step_solved": step_solved,
            "average_steps": average_steps,
            "time_taken": time_taken,
        }
    )

    pprint(results, width=120, sort_dicts=False)
