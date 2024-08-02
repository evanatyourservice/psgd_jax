import os
from functools import partial
from typing import Union, Optional
import numpy as np
from matplotlib import pyplot as plt

import jax
from jax import numpy as jnp, jit
from jax.random import uniform
import optax

from psgd_jax.optimizers.psgd import psgd_hvp_helper
from psgd_jax.optimizers.create_optimizer import create_optimizer


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
    ax.set_title("Losses")
    ax.set_yscale("log")
    ax.set_ylim([min(losses) * 0.5, max(losses) * 2])

    # plot three examples
    for i, sample in enumerate(test_iter):
        iter_x, iter_y = sample[0, :], sample[1, :]
        ax = fig.add_subplot(2, 2, i + 2)
        ax.contour(X, Y, Z, 90, cmap="jet")
        ax.plot(iter_x, iter_y, color="r", marker="x", markersize=3)
        ax.set_title(f"{plot_title}, {len(iter_x)} steps")
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


def _make_params(key):
    # params in [-2, 2] and [-1, 3]
    n_sets = 16
    keys = jax.random.split(key, n_sets * 2)
    keys = jnp.reshape(keys, (n_sets, 2, 2))
    return {
        f"{i:02}": jnp.array(
            [
                uniform(k[0], [], jnp.float32, -2, -1),
                uniform(k[1], [], jnp.float32, -1, 3),
            ]
        )
        for i, k in enumerate(keys)
    }


@partial(jit, static_argnums=(0, 3, 4, 5))
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
            loss_out, grads, hvp, vector, update_precond = psgd_hvp_helper(
                subkey,
                _loss_fn_rosenbrock,
                params,
                loss_fn_extra_args=(),
                has_aux=False,
                pmap_axis_name=None,
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
        recorded_params = recorded_params.at[:, :, i].set(
            jnp.stack(jax.tree.leaves(params))
        )
        return params, opt_state, key, losses, recorded_params

    losses = jnp.zeros([steps])
    recorded_params = jnp.zeros([len(jax.tree.leaves(params)), 2, steps])
    init_state = (params, opt_state, jax.random.PRNGKey(0), losses, recorded_params)
    params, _, _, losses, recorded_params = jax.lax.fori_loop(
        0, steps, loop_body, init_state
    )

    return params, losses, recorded_params


def plot_rosenbrock(
    optimizer: Union[
        optax.GradientTransformation, optax.GradientTransformationExtraArgs
    ],
    steps: int = 500,
    plot_title: str = "Rosenbrock",
    save_dir: Optional[str] = None,
    psgd_use_hessian: Optional[bool] = False,
    psgd_update_probability: float = 1.0,
    seed: Optional[int] = None,
    plot: bool = True,
):
    """Test and plot an optimizer on the rosenbrock function.

    Args:
        optimizer: optax optimizer
        steps: number of steps to run
        plot_title: title for the plot
        save_dir: directory to save the plot
        psgd_use_hessian: If using PSGD, set this to True to calculate
            and pass in the Hvp, otherwise PSGD will default to gg^T
            (gradient whitening) preconditioning.
        psgd_update_probability: probability of updating the preconditioner
            if using PSGD and psgd_use_hessian is True.
        seed: random seed for reproducibility
        plot: whether to plot the results
    """
    print(plot_title)

    if seed is None:
        seed = np.random.randint(0, 2**30)
    params = _make_params(jax.random.PRNGKey(seed))

    opt_state = optimizer.init(params)

    initial_loss = _loss_fn_rosenbrock(params)
    print(f"Initial loss = {initial_loss}")

    params, losses, recorded_params = _run_test(
        optimizer,
        opt_state,
        params,
        steps,
        psgd_use_hessian=psgd_use_hessian,
        psgd_update_probability=psgd_update_probability,
    )

    final_loss = _loss_fn_rosenbrock(params)
    print(f"Final loss = {final_loss}")

    if plot:
        _plot_rosenbrock(recorded_params, plot_title, losses, save_dir)

    return final_loss.item()


if __name__ == "__main__":
    steps = 500
    fn = partial(
        create_optimizer,
        optimizer="psgd",
        learning_rate=0.2,
        min_learning_rate=0.0,
        norm_grads=None,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        nesterov=True,
        weight_decay=0.0,
        lr_schedule="linear",
        schedule_free=False,
        warmup_steps=0,
        total_train_steps=steps,
        gradient_clip=None,
        pmap_axis_name=None,
        graft=False,
        shampoo_precond_every_n=2,
        shampoo_precond_block_size=128,
        psgd_precond_type="affine",
        psgd_update_prob=1.0,
        psgd_rank=10,
        psgd_heavyball=False,
        psgd_feed_into_adam=True,
        psgd_precond_lr=0.3,
        psgd_precond_init_scale=None,
    )
    plot = partial(plot_rosenbrock, seed=np.random.randint(0, 2**30))

    opt = fn()[0]
    plot(
        optimizer=opt,
        steps=steps,
        plot_title="PSGD",
        psgd_use_hessian=True,
        psgd_update_probability=1.0,
    )

    opt = fn(optimizer="shampoo", learning_rate=0.02, graft=True)[0]
    plot(optimizer=opt, steps=steps, plot_title="Shampoo")

    opt = fn(optimizer="adam", learning_rate=0.3)[0]
    plot(optimizer=opt, steps=steps, plot_title="Adam")

    opt = fn(optimizer="adam", learning_rate=0.5, beta1=0.95, schedule_free=True)[0]
    plot(optimizer=opt, steps=steps, plot_title="Schedule-free Adam")
