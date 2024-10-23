import os
from functools import partial
from typing import Union, Optional
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint

import jax
from jax import numpy as jnp, jit, sharding
from jax.random import uniform
from jax.experimental import mesh_utils
import optax

from psgd_jax import hessian_helper
from psgd_jax.xmat import xmat
from psgd_jax.low_rank_approximation import low_rank_approximation
from psgd_jax.affine import affine
from psgd_jax.kron import kron


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"


def _plot_rosenbrock(test_iter, plot_title, losses, save_dir=None):
    """plot rosenbrock test from batch of results.

    Adapted from https://github.com/jettify/pytorch-optimizer"""

    def rosenbrock(p):
        x, y = p
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    # plot 4 subplots
    fig = plt.figure(figsize=(16, 10))

    # plot losses in top left
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(losses)
    ax.set_title(f"Losses (final loss = {losses[-1]})")
    ax.set_yscale("log")
    ax.set_ylim([min(losses) * 0.5, max(losses) * 2])

    # plot three examples
    for i, sample in enumerate(test_iter):
        iter_x, iter_y = sample[0, :], sample[1, :]
        ax = fig.add_subplot(2, 2, i + 2)
        ax.contour(X, Y, Z, 90, cmap="jet")
        ax.plot(iter_x, iter_y, color="r", marker="x", markersize=4)
        ax.set_title(f"{plot_title}, {len(iter_x) - 1} steps")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
        ax.plot(*minimum, "gD")
        if i == 2:
            break

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{plot_title}.png"))
    plt.show()


@jit
def _loss_fn_rosenbrock(xs):
    # rosenbrock function
    l = lambda x, y: (1 - x) ** 2 + 1 * (y - x**2) ** 2
    flat_xs = jax.tree.leaves(xs)
    return sum([l(x[0], x[1]) for x in flat_xs]) / len(flat_xs)


@jit
def _make_params(key):
    # params in [-2, 2] and [-1, 3]
    n_sets = 16
    keys = jax.random.split(key, n_sets * 2)
    keys = jnp.reshape(keys, (n_sets, 2, 2))
    params = {
        f"{i:02}": jnp.array(
            [
                uniform(k[0], [], jnp.float32, -2, -1),
                uniform(k[1], [], jnp.float32, -1, 3),
            ]
        )
        for i, k in enumerate(keys)
    }
    params["00"] = jnp.array([-2, 2], dtype=jnp.float32)
    return params


def _run_test(
    optimizer: Union[
        optax.GradientTransformation, optax.GradientTransformationExtraArgs
    ],
    opt_state: optax.OptState,
    params: dict,
    steps: int,
    psgd_use_hessian: Optional[bool] = False,
    psgd_update_probability: float = 1.0,
):
    def loop_body(i, state):
        params, opt_state, key, losses, recorded_params = state

        key, subkey = jax.random.split(key)
        if psgd_use_hessian:
            # use helper to compute hvp and pass into PSGD
            loss_out, grads, hvp, vector, update_precond = hessian_helper(
                subkey,
                i,
                _loss_fn_rosenbrock,
                params,
                loss_fn_extra_args=(),
                has_aux=False,
                preconditioner_update_probability=psgd_update_probability,
            )
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                Hvp=hvp,
                vector=vector,
                update_preconditioner=update_precond,
            )
        else:
            loss_out, updates = jax.value_and_grad(_loss_fn_rosenbrock)(params)
            updates, opt_state = optimizer.update(updates, opt_state, params)

        params = optax.apply_updates(params, updates)
        losses = losses.at[i].set(loss_out)
        recorded_params = recorded_params.at[:, :, i + 1].set(
            jnp.stack(jax.tree.leaves(params))
        )
        return params, opt_state, key, losses, recorded_params

    losses = jnp.zeros([steps])
    recorded_params = jnp.zeros([len(jax.tree.leaves(params)), 2, steps + 1])
    recorded_params = recorded_params.at[:, :, 0].set(
        jnp.stack(jax.tree.leaves(params))
    )
    init_state = (params, opt_state, jax.random.PRNGKey(0), losses, recorded_params)
    params, opt_state, _, losses, recorded_params = jax.lax.fori_loop(
        0, steps, loop_body, init_state
    )

    return params, opt_state, losses, recorded_params


def main():
    print("Testing PSGD variants on Rosenbrock function")

    for use_hessian in [False, True]:
        for precond_type in ["kron", "xmat", "low_rank_approximation", "affine"]:
            if use_hessian and precond_type == "kron":
                # kron just uses whitening (gg^T)
                continue
            steps = 500
            psgd_update_probability = 1.0
            learning_rate = optax.linear_schedule(0.1, 0.0, steps)
            kwargs = {
                "learning_rate": learning_rate,
                "preconditioner_update_probability": psgd_update_probability,
                "b1": 0.9,
                "precond_lr": 0.1,
                "update_global_norm_clip": np.sqrt(32.0),
            }
            if precond_type == "xmat":
                optimizer = partial(xmat, **kwargs)
            elif precond_type == "low_rank_approximation":
                optimizer = partial(low_rank_approximation, **kwargs)
            elif precond_type == "affine":
                optimizer = partial(affine, **kwargs)
            elif precond_type == "kron":
                del kwargs["precond_lr"]
                del kwargs["update_global_norm_clip"]
                optimizer = partial(kron, memory_save_mode="one_diag", **kwargs)
            else:
                optimizer = None

            plot_title = f"{precond_type} PSGD {'Hvp' if use_hessian else 'gg^T'}"
            print(plot_title)

            seed = np.random.randint(0, 2**30)

            params = _make_params(jax.random.PRNGKey(seed))

            optimizer = optimizer()
            opt_state = optimizer.init(params)
            pprint(opt_state)

            P = sharding.PartitionSpec
            devices = mesh_utils.create_device_mesh((2,))
            mesh = sharding.Mesh(devices, ("m",))

            def create_spec(x):
                if x.size > 1:
                    shards = ("m" if x.shape[0] % 2 == 0 else None,) + (None,) * (
                        x.ndim - 1
                    )
                    return sharding.NamedSharding(mesh, P(*shards))
                else:
                    return sharding.NamedSharding(mesh, P())

            params_sharding = jax.tree.map(create_spec, params)
            opt_state_sharding = jax.tree.map(create_spec, opt_state)

            params = jax.device_put(params, params_sharding)
            opt_state = jax.device_put(opt_state, opt_state_sharding)

            initial_loss = _loss_fn_rosenbrock(params)
            print(f"Initial loss = {initial_loss}")

            run_test_fn = jit(
                _run_test,
                static_argnums=(0, 3, 4, 5),
                out_shardings=(
                    params_sharding,
                    opt_state_sharding,
                    sharding.NamedSharding(mesh, P()),
                    sharding.NamedSharding(mesh, P()),
                ),
            )

            params, opt_state, losses, recorded_params = run_test_fn(
                optimizer,
                opt_state,
                params,
                steps,
                use_hessian,
                psgd_update_probability,
            )

            final_loss = _loss_fn_rosenbrock(params)
            print(f"Final loss = {final_loss}")

            print("Output sharding:")
            print(jax.tree.map(lambda x: x.sharding, params))
            print(jax.tree.map(lambda x: x.sharding, opt_state))

            _plot_rosenbrock(recorded_params, plot_title, losses)


if __name__ == "__main__":
    main()
