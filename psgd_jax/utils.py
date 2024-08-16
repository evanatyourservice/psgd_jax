from typing import Callable, Tuple

import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from optax import tree_utils as otu
from optax._src import base
from optax._src.numerics import safe_int32_increment


def hessian_helper(
    key: PRNGKey,
    train_step: int,
    loss_fn: Callable,
    params: base.Params,
    loss_fn_extra_args: Tuple = (),
    has_aux: bool = False,
    preconditioner_update_probability: float = 1.0,
):
    """Helper function for computing hessian vector product for PSGD.

    This helps handle the calculation of a hessian vector product if wanting to use exact
    hvp instead of the default gradient whitening style preconditioner. It returns the
    loss fn output, gradients, hvp, random vector, and a bool of whether we're updating the
    preconditioner this step. The hvp, vector, and update cond are then passed into PSGD's
    update fn. This fn is not needed if wanting to use the default gradient whitening style
    preconditioner.

    Args:
        key: PRNGKey, random key.
        train_step: int, current train step needed to init preconditioner on first step.
        loss_fn: callable, loss function.
        params: flax.Params, model parameters.
        loss_fn_extra_args: tuple, extra arguments for loss function to be used as
            `loss_fn(params, *loss_fn_extra_args)`.
        has_aux: bool, whether loss function has aux output.
        preconditioner_update_probability: float, probability of updating the preconditioner.

    Returns:
        loss_out: jnp.ndarray, output of loss function.
        grads: flax.Params, gradients.
        hvp: flax.Params, hessian vector product.
        vector: flax.Params, random vector.
        update_preconditioner: bool, whether we're updating preconditioner this step.
    """
    obj_fn = lambda params: loss_fn(params, *loss_fn_extra_args)
    key1, key2 = jax.random.split(key)

    def grad_fn(params):
        loss_out, grad = jax.value_and_grad(obj_fn, has_aux=has_aux)(params)
        return grad, loss_out

    def hvp_fn(params):
        vector = otu.tree_random_like(key1, params, jax.random.normal)
        grad, hvp, loss_out = jax.jvp(grad_fn, (params,), (vector,), has_aux=True)
        return grad, loss_out, hvp, vector

    # TODO (evanatyourservice): finite difference hvp option

    def g_fn(params):
        grad, loss_out = grad_fn(params)
        dummy_hvp = jax.tree.map(jnp.zeros_like, params)
        dummy_vector = jax.tree.map(jnp.zeros_like, params)
        return grad, loss_out, dummy_hvp, dummy_vector

    update_precond = jnp.logical_or(
        jax.random.uniform(key2) < preconditioner_update_probability, train_step < 2
    )

    grad, loss_out, hvp, vector = jax.lax.cond(update_precond, hvp_fn, g_fn, params)
    return loss_out, grad, hvp, vector, update_precond


def apply_momentum(
    updates: base.Updates, momentum: base.Updates, step, b1, nesterov
) -> Tuple[base.Updates, base.Updates]:
    # ema
    mu = otu.tree_update_moment(updates, momentum, b1, 1)
    if nesterov:
        # nesterov momentum for ema with bias correction
        # https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
        updates = jax.tree.map(
            lambda m, g: b1 * m + (1 - b1) * g,
            otu.tree_bias_correction(mu, b1, safe_int32_increment(step)),
            otu.tree_bias_correction(updates, b1, step),
        )
    else:
        # bias correction only
        updates = otu.tree_bias_correction(mu, b1, step)

    return updates, mu


def add_eps(x):
    return jnp.clip(x, 1e-25, None)
