"""
Some first-order optimizers that are either not in optax or modified from optax.
"""

from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from optax._src import base, numerics, utils, transform
from optax import tree_utils as otu


class ScaleByBeliefState(NamedTuple):
    count: chex.Array
    mu: base.Updates
    nu: base.Updates


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    rectify: bool = False,
    mu_dtype: Optional[chex.ArrayDType] = None,
    schedulefree: bool = False,
) -> base.GradientTransformation:
    """Same as optax version but with rectify option as in radam.

    Args:
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the second moment estimates.
        eps: Small constant to avoid division by zero.
        rectify: Whether to apply RAdam rectification.
        mu_dtype: The dtype for momentum buffer.
        schedulefree: Set to true if optimizer is wrapped in schedule_free.
            This will use grads for scaling instead of mu_hat. Adabelief
            needs b1>0 for its preconditioner, so do not set b1 to zero if
            wrapping in schedule_free, simply set this flag to True.
    """
    assert b1 > 0.0, "b1 must be greater than 0 for adabelief"
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        s = jax.tree.map(jnp.zeros_like, params)
        return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

    def update_fn(updates, state, params=None):
        del params
        count_inc = numerics.safe_int32_increment(state.count)

        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        prediction_error = jax.tree.map(lambda g, m: g - m, updates, mu)
        nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: m / (jnp.sqrt(v) + eps),
            updates if schedulefree else mu_hat,
            nu_hat,
        )
        if rectify:
            sma_inf = 2.0 / (1.0 - b2) - 1.0
            sma_t = sma_inf - 2.0 * count_inc * jnp.power(b2, count_inc) / (
                1.0 - jnp.power(b2, count_inc)
            )
            r_t = jnp.sqrt(
                (sma_t - 4.0)
                / (sma_inf - 4.0)
                * (sma_t - 2.0)
                / (sma_inf - 2.0)
                * sma_inf
                / sma_t
            )
            updates = jax.tree.map(
                lambda u, m: jnp.where(sma_t >= 5, r_t * u, m), updates, mu_hat
            )

        mu = otu.tree_cast(mu, mu_dtype)
        return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_adam3(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    mu_dtype: Optional[chex.ArrayDType] = None,
    schedulefree: bool = False,
) -> base.GradientTransformation:
    """adam3 optimizer from https://github.com/wyzjack/AdaM3

    Args:
        b1: Exponential decay rate for the first moment estimates.
        b2: Exponential decay rate for the second moment estimates.
        eps: Small constant to avoid division by zero.
        mu_dtype: The dtype for momentum buffer.
        schedulefree: Set to true if optimizer is wrapped in schedule_free.
            This will use grads for scaling instead of mu_hat. Adam3
            needs b1>0 for its preconditioner, so do not set b1 to zero if
            wrapping in schedule_free, simply set this flag to True.
    """
    assert b1 > 0.0, "b1 must be greater than 0 for adam3"
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree.map(lambda p: jnp.zeros_like(p, dtype=mu_dtype), params)
        nu = jax.tree.map(jnp.zeros_like, params)
        return transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        count_inc = numerics.safe_int32_increment(state.count)

        mu = otu.tree_update_moment(updates, state.mu, b1, 1)
        # adam3 passes momentum into preconditioner
        nu = otu.tree_update_moment_per_elem_norm(mu, state.nu, b2, 2)
        # adam3 adds epsilon to nu directly
        nu = jax.tree.map(lambda v: v + eps, nu)
        mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: m / jnp.sqrt(v), updates if schedulefree else mu_hat, nu_hat
        )

        mu = otu.tree_cast(mu, mu_dtype)
        return updates, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    nesterov: bool = False,
) -> base.GradientTransformation:
    """Same as optax version but doesn't create momentum buffer if b1 == 0."""
    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        if b1 > 0:
            mu = otu.tree_zeros_like(params, dtype=mu_dtype)
        else:
            mu = None
        nu = otu.tree_zeros_like(params)
        return transform.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        count_inc = numerics.safe_int32_increment(state.count)
        if b1 > 0:
            mu = otu.tree_update_moment(updates, state.mu, b1, 1)
            if nesterov:
                mu_hat = jax.tree.map(
                    lambda m, g: b1 * m + (1 - b1) * g,
                    otu.tree_bias_correction(
                        mu, b1, numerics.safe_int32_increment(count_inc)
                    ),
                    otu.tree_bias_correction(updates, b1, count_inc),
                )
            else:
                mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
            mu = otu.tree_cast(mu, mu_dtype)
        else:
            mu = None
            mu_hat = updates
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
        updates = jax.tree.map(
            lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat
        )
        return updates, transform.ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_sign_sgd() -> base.GradientTransformation:
    def init_fn(params):
        del params
        return base.EmptyState()

    def update_fn(updates, state, params=None):
        del params
        updates = jax.tree_map(jnp.sign, updates)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
