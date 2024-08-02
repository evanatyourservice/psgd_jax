import warnings
import numpy as np
from numpy.random import uniform
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import optax

from psgd_jax.optimizers.create_optimizer import create_optimizer
from psgd_jax.optimizers.psgd import psgd_hvp_helper


# warnings.filterwarnings("ignore", category=DeprecationWarning)


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def plot_rosenbrock(grad_iter, optimizer_name, lr, losses):
    """plot 4 rosenbrock functions from batch.

    grad_iter is shape [b, 2, n_steps]"""
    x = np.linspace(-2, 2, 250)
    y = np.linspace(-1, 3, 250)
    minimum = (1.0, 1.0)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    # plot 4 subplots
    fig = plt.figure(figsize=(16, 10))
    for i, sample in enumerate(grad_iter):
        if i == 0:
            # plot losses in top left
            ax = fig.add_subplot(2, 2, i + 1)
            ax.plot(losses)
            ax.set_title("Losses")
            ax.set_yscale("log")
            ax.set_ylim([min(losses) * 0.5, max(losses) * 2])

        iter_x, iter_y = sample[0, :], sample[1, :]
        ax = fig.add_subplot(2, 2, i + 2)
        ax.contour(X, Y, Z, 90, cmap="jet")
        ax.plot(iter_x, iter_y, color="r", marker="x", markersize=3)
        ax.set_title("{}, {} steps, lr={:.6}".format(optimizer_name, len(iter_x), lr))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-1, 3])
        ax.plot(*minimum, "gD")

        # plt.plot(iter_x[-1], iter_y[-1], "rD")
        # plt.plot(*minimum, "gD")
        if i == 2:
            break

    plt.show()


@jax.jit
def loss_fn_rosenbrock(xs):
    # rosenbrock function
    l = lambda x, y: (1 - x) ** 2 + 1 * (y - x**2) ** 2
    flat_xs = jax.tree.leaves(xs)
    return sum([l(x[0], x[1]) for x in flat_xs]) / len(flat_xs)


def make_params():
    # params in [-2, 2] and [-1, 3]
    return {
        "a": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "d": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "b": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "c": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "e": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "f": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "g": np.array([uniform(-2, 0), uniform(-1, 3)]),
        "h": np.array([uniform(-2, 0), uniform(-1, 3)]),
    }


print("Testing PSGD")
for use_hessian in [True, False]:
    for precond_type in ["xmat", "uvd", "affine"]:
        print(f"Preconditioner = {precond_type}, Hessian = {use_hessian}")

        steps = 500
        lr = 0.5 if use_hessian else 0.1
        psgd_update_probability = 1.0
        opt, _ = create_optimizer(
            optimizer="psgd",
            learning_rate=lr,
            min_learning_rate=0.0,
            norm_grads=None,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            nesterov=True,
            weight_decay=0.0,
            lr_schedule="linear",
            schedule_free=False,
            warmup_steps=50,
            total_train_steps=steps,
            gradient_clip=10.0,
            pmap_axis_name=None,
            graft=False,
            shampoo_precond_every_n=1,
            shampoo_precond_block_size=128,
            psgd_precond_type=precond_type,
            psgd_update_prob=psgd_update_probability,
            psgd_rank=10,
            psgd_heavyball=False,
            psgd_feed_into_adam=True,
            psgd_precond_lr=0.1,
            psgd_precond_init_scale=None,
        )

        params = make_params()

        opt_state = opt.init(params)

        initial_loss = loss_fn_rosenbrock(params)
        print(f"Initial loss = {initial_loss}")

        def loop_body(i, state):
            params, opt_state, key, losses, recorded_params = state

            key, subkey = jax.random.split(key)
            if use_hessian:
                # use helper to compute hvp and pass into PSGD
                loss_out, grads, hvp, vector, update_precond = psgd_hvp_helper(
                    subkey,
                    loss_fn_rosenbrock,
                    params,
                    loss_fn_extra_args=(),
                    has_aux=False,
                    pmap_axis_name=None,
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
                loss_out, updates = jax.value_and_grad(loss_fn_rosenbrock)(params)
                updates, opt_state = opt.update(updates, opt_state, params)

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

        final_loss = loss_fn_rosenbrock(params)
        if not final_loss.item() < 0.01:
            print("TEST FAILED")
        else:
            print("test passed")
        print(f"Final loss = {final_loss}")

        # graph losses
        graph = True
        if graph:
            hess_name = "Hvp" if use_hessian else "gg^T"
            title = f"{precond_type} PSGD {hess_name}"
            plot_rosenbrock(recorded_params, title, lr, losses)
