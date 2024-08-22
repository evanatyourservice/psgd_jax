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


class PSGDLRAState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    U: jax.Array
    V: jax.Array
    d: jax.Array


def scale_by_lra(
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    uvd_rank_of_approximation: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements UVd PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        uvd_rank_of_approximation: int, rank of approximation for uvd preconditioner.
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

        # preconditioners
        n_params = sum([x.size for x in jax.tree.leaves(params)])
        key, subkey = jax.random.split(key)
        U = jax.random.normal(
            subkey, (n_params, uvd_rank_of_approximation), dtype=jnp.float32
        )
        U /= (n_params * (uvd_rank_of_approximation + 10)) ** 0.5

        key, subkey = jax.random.split(key)
        V = jax.random.normal(
            subkey, (n_params, uvd_rank_of_approximation), dtype=jnp.float32
        )
        V /= (n_params * (uvd_rank_of_approximation + 10)) ** 0.5

        d = jnp.ones((n_params, 1), jnp.float32)

        # initial state
        return PSGDLRAState(
            count=jnp.zeros([], jnp.int32), key=key, mu=mu, U=U, V=V, d=d
        )

    def update_fn(
        updates: base.Updates,
        state: PSGDLRAState,
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

        def _update_precond(key: PRNGKey, state: PSGDLRAState, Hvs, vs):
            v = jnp.concatenate(
                [jnp.reshape(x, (-1, 1)) for x in jax.tree.leaves(vs)], 0
            )
            h = jnp.concatenate(
                [jnp.reshape(x, (-1, 1)) for x in jax.tree.leaves(Hvs)], 0
            )

            # init d
            if precond_init_scale is not None:
                init_scale = precond_init_scale
            else:
                if hessian_based_preconditioning:
                    init_scale = (jnp.sum(v * v) / jnp.sum(h * h)) ** 0.25
                else:
                    init_scale = (len(h) / jnp.sum(jnp.square(h))) ** 0.25
            d = jax.lax.cond(
                state.count == 0, lambda: state.d * init_scale, lambda: state.d
            )

            # update preconditioner
            key, subkey = jax.random.split(key)
            U, V, d = _update_precond_UVd_math(
                subkey,
                state.U,
                state.V,
                d,
                v,
                h,
                precond_lr_in,
                step_normalizer_order,
                precision,
            )

            return key, U, V, d

        def _dont_update_precond(key, state, Hvs, vs):
            return key, state.U, state.V, state.d

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

        key, U, V, d = jax.lax.cond(
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
            [jnp.reshape(x, (-1, 1)) for x in jax.tree.leaves(updates)], 0
        )
        flat_updates = _precond_grad_UVd_math(U, V, d, flat_updates)
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
        if update_global_norm_clip is not None:
            updates, _ = clipping.clip_by_global_norm(update_global_norm_clip).update(
                updates, base.EmptyState
            )

        mu = otu.tree_cast(mu, mu_dtype)
        state = PSGDLRAState(count=count_inc, key=key, mu=mu, U=U, V=V, d=d)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def low_rank_approximation(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    uvd_rank_of_approximation: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements UVd PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate for the optimizer.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay.
        mask: optional mask for weight decay.
        uvd_rank_of_approximation: int, rank of approximation for uvd preconditioner.
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
        scale_by_lra(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            uvd_rank_of_approximation=uvd_rank_of_approximation,
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


def _IpUVtmatvec(U, V, x):
    """Returns (I + U*V')*x. All variables are either matrices or column vectors."""
    return x + jnp.matmul(U, jnp.matmul(V.T, x))


def _update_precond_UVd_math(
    key, U, V, d, v, h, precond_lr, step_normalizer, precision
):
    """
    Update preconditioner Q = (I + U*V')*diag(d) with (vector, Hessian-vector product) = (v, h).
    State variables U, V and d are updated inplace.

    U, V, d, v, and h are either matrices or column vectors.
    """
    with jax.default_matmul_precision(precision):
        # balance the numerical dynamic ranges of U and V; optional
        def _balance(U, V):
            normU = jnp.linalg.norm(U)
            normV = jnp.linalg.norm(V)
            rho = jnp.sqrt(normU / normV)
            U = U / rho
            V = V * rho
            return U, V

        key, subkey = jax.random.split(key)
        U, V = jax.lax.cond(
            jax.random.uniform(subkey) < 0.01, _balance, lambda u, v: (u, v), U, V
        )

        Qh = _IpUVtmatvec(U, V, d * h)
        Ph = d * _IpUVtmatvec(V, U, Qh)

        VtU = V.T @ U
        I = jnp.eye(VtU.shape[0], dtype=VtU.dtype)
        IpVtU = I + VtU
        invQtv = v / d

        # cast to float32 for accuracy, no slowdown as 'a' is only (rank, rank)
        orig_dtype = U.dtype
        IpVtU = IpVtU.astype(jnp.float32)
        U_solve = jnp.linalg.solve(IpVtU.T, (U.T @ invQtv).astype(jnp.float32))
        invQtv = invQtv - V @ U_solve.astype(orig_dtype)
        V_solve = jnp.linalg.solve(IpVtU, (V.T @ invQtv).astype(jnp.float32))
        invPv = invQtv - U @ V_solve.astype(orig_dtype)
        IpVtU = IpVtU.astype(orig_dtype)
        invPv = invPv / d

        nablaD = Ph * h - v * invPv
        if step_normalizer == "2nd":
            mu = precond_lr * jnp.min(
                jax.lax.rsqrt(add_eps(Ph * Ph + v * v))
                * jax.lax.rsqrt(add_eps(h * h + invPv * invPv))
            )  # two seperate rsqrt's to avoid underflow
        else:
            mu = precond_lr / add_eps(jnp.max(jnp.abs(nablaD)))
        d -= mu * d * nablaD

        # update either U or V, not both at the same time
        a, b = Qh, invQtv

        def _update_U(U, V):
            atV = a.T @ V
            btV = b.T @ V
            atVVt = atV @ V.T
            btVVt = btV @ V.T
            if step_normalizer == "2nd":
                mu = precond_lr / add_eps(
                    jnp.linalg.norm(a) * jnp.linalg.norm(atVVt)
                    + jnp.linalg.norm(b) * jnp.linalg.norm(btVVt)
                )
            else:  # '1st'
                norm = jnp.sqrt(
                    jnp.abs(
                        (a.T @ a) * (atVVt @ atVVt.T)
                        + (b.T @ b) * (btVVt @ btVVt.T)
                        - 2 * (a.T @ b) * (atVVt @ btVVt.T)
                    )
                )
                mu = precond_lr / add_eps(norm)

            U -= mu * (a @ (atV @ IpVtU) - b @ (btV @ IpVtU))

            return U, V

        def _update_V(U, V):
            atU = a.T @ U
            btU = b.T @ U
            UUta = U @ atU.T
            UUtb = U @ btU.T
            if step_normalizer == "2nd":
                mu = precond_lr / add_eps(
                    jnp.linalg.norm(a) * jnp.linalg.norm(UUta)
                    + jnp.linalg.norm(b) * jnp.linalg.norm(UUtb)
                )
            else:  # '1st'
                norm = jnp.sqrt(
                    jnp.abs(
                        (UUta.T @ UUta) * (a.T @ a)
                        + (UUtb.T @ UUtb) * (b.T @ b)
                        - 2 * (UUta.T @ UUtb) * (a.T @ b)
                    )
                )
                mu = precond_lr / add_eps(norm)

            V -= mu * ((a + V @ atU.T) @ atU - (b + V @ btU.T) @ btU)

            return U, V

        U, V = jax.lax.cond(jax.random.uniform(key) < 0.5, _update_U, _update_V, U, V)

        return U, V, d


def _precond_grad_UVd_math(U, V, d, g):
    """
    Preconditioning gradient g with Q = (I + U*V')*diag(d).

    All variables here are either matrices or column vectors.
    """
    g = _IpUVtmatvec(U, V, d * g)
    g = d * _IpUVtmatvec(V, U, g)
    return g
