from collections import defaultdict
from typing import Any, Optional, Union, Callable, NamedTuple, List, Tuple

import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.linear_algebra import global_norm
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


class PSGDAffineState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    Qs: base.Updates


# decays to 0.01 by around 2000 steps
update_prob_schedule_1_to_0_01 = lambda n: jnp.maximum(1.0 * jnp.exp(-0.002 * n), 0.01)
# opposite of update_prob_schedule from 0.01 to 0.1
precond_lr_schedule_0_01_to_0_1 = lambda n: (-0.9 * jnp.exp(-0.002 * n) + 1.0) / 10


# PSGD reshapes params into matrices. Here we get sharding rules for them
# which are similar to the rules for params. We can pass this into PSGD for
# internal sharding constraints, although it's not absolutely necessary.
def get_reshaped_params_shapes(params):
    """Get shapes of reshaped parameters for PSGD.

    PSGD reshapes parameters into matrices to apply affine preconditioning.
    This function returns the shapes of these matrices. Useful for making
    sharding rules for these reshaped parameters to pass into PSGD's
    `reshaped_params_sharding` argument.
    """
    # TODO account for scanned layers
    affine_reshapers = jax.tree.map(
        _shape_as_matrix, params
    )  # returns tuples of (reshape_fn, unreshape_fn, shape)
    p_struct = jax.tree.structure(params)
    affine_reshapers = p_struct.flatten_up_to(affine_reshapers)
    matrix_shapes = [jax.ShapeDtypeStruct(r[2], jnp.float32) for r in affine_reshapers]
    return p_struct.unflatten(matrix_shapes)


def scale_by_affine(
    preconditioner_update_probability: Union[float, Callable[[int], float]] = 0.5,
    b1: float = 0.9,
    nesterov: bool = False,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 16,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "bfloat16",
    reshaped_params_sharding: Any = None,
    best_effort_scan: bool = True,
) -> base.GradientTransformationExtraArgs:
    """
    Implements Affine PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        max_size_triangular: int, max size for affine preconditioner to be
            triangular.
        max_skew_triangular: int, max skew for affine preconditioner to be
            triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: float, initial scale for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        reshaped_params_sharding: optional Any, sharding spec for reshaped parameters.
        best_effort_scan: bool, try to automatically stack same-shaped matrices
            and use lax.map to update them to save on compile time and memory.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    def init_fn(params):
        key = jax.random.PRNGKey(36)

        # momentum
        mu = None
        if b1 > 0:
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioners
        affine_reshapers = [_shape_as_matrix(x) for x in jax.tree.leaves(params)]
        Qs = [
            _initQ(s[2], max_size_triangular, max_skew_triangular, precond_dtype)
            for s in affine_reshapers
        ]
        Qs = jax.tree.structure(params).unflatten(Qs)
        Qs = jax.tree.map(lambda q: jnp.sqrt(precond_init_scale) * q, Qs)

        # initial state
        return PSGDAffineState(count=jnp.zeros([], jnp.int32), key=key, mu=mu, Qs=Qs)

    def update_fn(
        updates: base.Updates, state: PSGDAffineState, params: base.Params = None
    ):
        del params
        count_inc = safe_int32_increment(state.count)
        key = state.key

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        # momentum
        mu = None
        if state.mu is not None:
            updates, mu = _apply_momentum(updates, state.mu, count_inc, b1, nesterov)

        # get reshapers
        affine_reshapers = jax.tree.map(_shape_as_matrix, updates)

        # flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        Qs = grads_structure.flatten_up_to(state.Qs)
        affine_reshapers = grads_structure.flatten_up_to(affine_reshapers)
        flat_sharding = grads_structure.flatten_up_to(reshaped_params_sharding)

        # reshape updates using affine reshapers
        gs = [r[0](x) for x, r in zip(updates, affine_reshapers)]
        if flat_sharding is not None:
            gs = jax.lax.with_sharding_constraint(gs, flat_sharding)

        # stack same-shaped matrices to scan over
        if best_effort_scan:
            gs, Qs, revert_indices = _stack_matrices(gs, Qs)

        map_batch_size = 1  # interpolates between scan and vmap, 1 being scan unroll=1

        # update preconditioner
        key, subkey = jax.random.split(key)
        do_update = jax.random.uniform(subkey, dtype=jnp.float32) < update_prob_in

        def update_preconditioner():
            ukey = key
            if best_effort_scan:
                key_list = []
                for g in gs:
                    n_keys = jax.random.split(ukey, g.shape[0] + 1)
                    ukey, keys = n_keys[0], n_keys[1:]
                    key_list.append(keys)

                new_Qs = [
                    jax.lax.map(
                        lambda xs: _update_precond_affine_dropv_math(
                            *xs, precond_lr_in, precision
                        ),
                        (k, Ql, Qr, g),
                        batch_size=map_batch_size,
                    )
                    for (k, (Ql, Qr), g) in zip(key_list, Qs, gs)
                ]
            else:
                keys = jax.random.split(ukey, len(Qs))
                new_Qs = [
                    _update_precond_affine_dropv_math(
                        k, Ql, Qr, g, precond_lr_in, precision
                    )
                    for (k, (Ql, Qr), g) in zip(keys, Qs, gs)
                ]

            new_Qs = otu.tree_cast(new_Qs, precond_dtype)
            return new_Qs

        Qs = _efficient_cond(do_update, update_preconditioner, Qs)

        # precondition grads
        if best_effort_scan:
            precond_grad_fn = lambda l, r, g: jax.lax.map(
                lambda xs: _precond_grad_affine_math(*xs),
                (l, r, g),
                batch_size=map_batch_size,
            )
        else:
            precond_grad_fn = _precond_grad_affine_math

        gs = [precond_grad_fn(Ql, Qr, g) for ((Ql, Qr), g) in zip(Qs, gs)]

        # unstack matrices
        if best_effort_scan:
            gs, Qs = _unstack_matrices(gs, Qs, revert_indices)
            if flat_sharding is not None:
                gs = jax.lax.with_sharding_constraint(gs, flat_sharding)

        # global clipping, sqrt(n_params) / 2 seems to work well empirically
        n_params = sum(p.size for p in jax.tree.leaves(gs))
        max_norm = jnp.sqrt(n_params) / 2
        gs = _global_clip(gs, max_norm)
        # element-wise clipping
        gs = jax.tree.map(lambda x: jnp.clip(x, -1.0, 1.0), gs)

        # reshape updates back to original shapes and unflatten pytrees
        updates = [r[1](u) for u, r in zip(gs, affine_reshapers)]
        updates = grads_structure.unflatten(updates)
        Qs = grads_structure.unflatten(Qs)

        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = PSGDAffineState(count=count_inc, key=key, mu=mu, Qs=Qs)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def affine(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    preconditioner_update_probability: Union[float, Callable[[int], float]] = 0.5,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 16,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "bfloat16",
    reshaped_params_sharding: Any = None,
    best_effort_scan: bool = True,
) -> base.GradientTransformationExtraArgs:
    """
    Implements Affine PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        nesterov: bool, whether to use Nesterov momentum.
        weight_decay: float, weight decay.
        mask: optional Any or callable, mask to apply to parameters.
        max_size_triangular: int, max size for affine preconditioner to be
            triangular.
        max_skew_triangular: int, max skew for affine preconditioner to be
            triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: float, initial scale for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.
        reshaped_params_sharding: optional Any, sharding spec for parameters.
        best_effort_scan: bool, try to automatically stack same-shaped matrices
            and use lax.map to update them to save on compile time and memory.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_affine(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precision=precision,
            reshaped_params_sharding=reshaped_params_sharding,
            best_effort_scan=best_effort_scan,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def _apply_momentum(
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


def _add_eps(x):
    return jnp.clip(x, 1e-25, None)


def _global_clip(updates, max_norm):
    g_norm = global_norm(updates)
    g_norm = jnp.maximum(max_norm, g_norm)
    updates = jax.tree.map(
        lambda u: (u / g_norm.astype(u.dtype)) * max_norm.astype(u.dtype), updates
    )
    return updates


def _norm_lower_bound(A: jax.Array):
    """
    Returns a cheap lower bound for the spectral norm of A.
    Numerical results on random matrices with a wide range of distributions and sizes suggest,
        norm(A) <= sqrt(2) * norm_lower_bound(A)
    Looks to be a very tight lower bound.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs

        aa = jnp.real(A * A.conj())

        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x.conj() @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A.conj().T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x.conj()
            return max_abs * jnp.linalg.norm(A.conj().T @ (x / jnp.linalg.norm(x)))

        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    def pass_calc(A):
        return max_abs

    return jax.lax.cond(max_abs > 0, calc, pass_calc, A)


def _shape_as_matrix(arr: jax.Array) -> tuple:
    """Reshapes tensor x to a matrix with conditions to improve efficiency.

    From original pytorch version.

    Args:
        arr: jax.Array, tensor to be reshaped.

    Returns:
        tuple where first element is function that convert x to matrix, second
            element is function that converts matrix back to x, and third element
            is the shape of x as a matrix.
    """

    def prod(arr):
        # prod = lambda arr: 1 if len(arr)==0 else arr[0]*prod(arr[1:])
        result = 1
        for a in arr:
            result *= a
        return result

    def permutations(p0):
        # generate all the permutations of the original one p0
        if len(p0) == 1:
            yield p0
        else:
            for i in range(len(p0)):
                for q in permutations(p0[:i] + p0[i + 1 :]):
                    yield p0[i], *q

    # here begins the processing
    if arr.ndim == 2:  # t already is a matrix, do nothing
        return lambda u: u, lambda v: v, arr.shape
    elif arr.ndim < 2:  # scalar or vector, simple reshape to matrix
        mtx_shape = (1, arr.size)
        return (
            lambda u, shape=mtx_shape: u.reshape(shape),
            lambda v, shape=arr.shape: v.reshape(shape),
            mtx_shape,
        )
    else:  # higher order tensor, a little complicated
        p0, s0 = tuple(range(arr.ndim)), arr.shape  # original permutation and shape
        min_precond_size, opt_p, opt_s, opt_i = float("inf"), None, None, None
        for p in permutations(p0):
            s = tuple(s0[j] for j in p)
            for i in range(1, len(p)):
                if (new_size := prod(s[:i]) ** 2 + prod(s[i:]) ** 2) < min_precond_size:
                    min_precond_size = new_size
                    opt_p, opt_s, opt_i = p, s, i

        if opt_p == p0:  # no permutation is needed, just reshaping
            mtx_shape = (prod(s0[:opt_i]), prod(s0[opt_i:]))
            return (
                lambda u, shape=mtx_shape: u.reshape(shape),
                lambda v, shape=s0: v.reshape(shape),
                mtx_shape,
            )
        else:  # need both permutation and reshaping
            mtx_shape = (prod(opt_s[:opt_i]), prod(opt_s[opt_i:]))
            q = tuple(
                pair[1] for pair in sorted([(k, i) for (i, k) in enumerate(opt_p)])
            )
            return (
                lambda u, permute=opt_p, shape=mtx_shape: u.transpose(permute).reshape(
                    shape
                ),
                lambda v, permute=q, shape=opt_s: v.reshape(shape).transpose(permute),
                mtx_shape,
            )


def _initQ(shape, max_size, max_skew, dtype=None):
    """
    It initializes Q = kron(Q2, Q1) for param p to scale * I,
    where Q1 and Q2 can reduce to diagonal matrices to save memory if
    max_size or max_skew are set to small numbers.
    """
    assert len(shape) == 2, "preconditioned param shape must be 2D"
    s1, s2 = shape
    if s1 < 2 or s1 > max_size or s1 > max_skew * s2:
        Q1 = jnp.ones(s1, dtype=dtype)
    else:
        Q1 = jnp.eye(s1, dtype=dtype)

    if s2 < 2 or s2 > max_size or s2 > max_skew * s1:
        Q2 = jnp.ones(s2, dtype=dtype)
    else:
        Q2 = jnp.eye(s2, dtype=dtype)

    return [Q1, Q2]


def _solve_triangular(a, b, upper, left=True):
    """jax.lax.linalg.triangular_solve rewritten to match PyTorch convention."""
    dtype_in = jnp.promote_types(a.dtype, b.dtype)
    a, b = a.astype(dtype_in), b.astype(dtype_in)
    return jax.lax.linalg.triangular_solve(a, b, left_side=left, lower=not upper)


def _update_precond_affine_math_(key, Ql, Qr, dX, dG, precond_lr, precision):
    step_normalizer = "2nd"

    with jax.default_matmul_precision(precision):
        if Ql.ndim == 2:
            if Qr.ndim == 2:  # Ql.dim()=2 and Qr.dim()=2:
                A = jnp.linalg.multi_dot([Ql, dG, Qr.conj().T])
                Bh = _solve_triangular(
                    Ql.conj().T,
                    _solve_triangular(Qr, dX, upper=True, left=False),
                    upper=False,
                )

                AhA, BhB = A.conj().T @ A, Bh @ Bh.conj().T
                AAh, BBh = A @ A.conj().T, Bh.conj().T @ Bh
                grad1 = jnp.triu(AAh - BhB)
                grad2 = jnp.triu(AhA - BBh)

                if step_normalizer == "2nd":
                    step1 = precond_lr / _add_eps(_norm_lower_bound(AAh + BhB))
                    step2 = precond_lr / _add_eps(_norm_lower_bound(AhA + BBh))
                else:
                    step1 = precond_lr / _add_eps(_norm_lower_bound(grad1))
                    step2 = precond_lr / _add_eps(_norm_lower_bound(grad2))

                Ql -= step1 * grad1 @ Ql
                Qr -= step2 * grad2 @ Qr
            else:  # Ql.dim()=2 and Qr.dim()=1:
                A = Ql @ (dG * Qr.conj())
                Bh = _solve_triangular(Ql.conj().T, dX / Qr, upper=False)

                AAh, BhB = A @ A.conj().T, Bh @ Bh.conj().T
                AAc, BBc = jnp.sum(A * A.conj(), axis=0), jnp.sum(
                    Bh * Bh.conj(), axis=0
                )
                grad1 = jnp.triu(AAh - BhB)
                grad2 = AAc - BBc

                if step_normalizer == "2nd":
                    step1 = precond_lr / _add_eps(_norm_lower_bound(AAh + BhB))
                    step2 = precond_lr / _add_eps(jnp.max(jnp.real(AAc + BBc)))
                else:
                    step1 = precond_lr / _add_eps(_norm_lower_bound(grad1))
                    step2 = precond_lr / _add_eps(jnp.max(jnp.abs(grad2)))

                Ql -= step1 * grad1 @ Ql
                Qr -= step2 * grad2 * Qr
        else:
            if Qr.ndim == 2:  # Ql.dim()=1 and Qr.dim()=2:
                A = (Ql[:, None] * dG) @ Qr.conj().T
                Bh = _solve_triangular(Qr, dX, upper=True, left=False) / (
                    Ql.conj()[:, None]
                )

                AAc, BBc = jnp.sum(A * A.conj(), axis=1), jnp.sum(
                    Bh * Bh.conj(), axis=1
                )
                AhA, BBh = A.conj().T @ A, Bh.conj().T @ Bh
                grad1 = AAc - BBc
                grad2 = jnp.triu(AhA - BBh)

                if step_normalizer == "2nd":
                    step1 = precond_lr / _add_eps(jnp.max(jnp.real(AAc + BBc)))
                    step2 = precond_lr / _add_eps(_norm_lower_bound(AhA + BBh))
                else:
                    step1 = precond_lr / _add_eps(jnp.max(jnp.abs(grad1)))
                    step2 = precond_lr / _add_eps(_norm_lower_bound(grad2))

                Ql -= step1 * grad1 * Ql
                Qr -= step2 * grad2 @ Qr
            else:  # Ql.dim()=1 and Qr.dim()=1:
                A = Ql[:, None] * dG * Qr.conj()
                Bh = dX / Qr / Ql.conj()[:, None]

                AAc1, BBc1 = jnp.sum(A * A.conj(), axis=1), jnp.sum(
                    Bh * Bh.conj(), axis=1
                )
                AAc2, BBc2 = jnp.sum(A * A.conj(), axis=0), jnp.sum(
                    Bh * Bh.conj(), axis=0
                )
                grad1 = AAc1 - BBc1
                grad2 = AAc2 - BBc2

                if step_normalizer == "2nd":
                    step1 = precond_lr / _add_eps(jnp.max(jnp.real(AAc1 + BBc1)))
                    step2 = precond_lr / _add_eps(jnp.max(jnp.real(AAc2 + BBc2)))
                else:
                    step1 = precond_lr / _add_eps(jnp.max(jnp.abs(grad1)))
                    step2 = precond_lr / _add_eps(jnp.max(jnp.abs(grad2)))

                Ql -= step1 * grad1 * Ql
                Qr -= step2 * grad2 * Qr

        def _balance():
            max_l = jnp.max(jnp.abs(Ql))
            max_r = jnp.max(jnp.abs(Qr))

            rho = jnp.sqrt(max_l / max_r)
            new_Ql = Ql / rho
            new_Qr = Qr * rho
            return new_Ql, new_Qr

        key, subkey = jax.random.split(key)
        Ql, Qr = _efficient_cond(jax.random.uniform(subkey) < 0.01, _balance, (Ql, Qr))

        return [Ql, Qr]


def _update_precond_affine_dropv_math(key, Ql, Qr, dG, precond_lr, precision):
    step_normalizer = "2nd"

    with jax.default_matmul_precision(precision):
        if Ql.ndim == 1 and Qr.ndim == 1:
            # drop v when both dims use diagonal preconditioners
            A = Ql[:, None] * dG * Qr.conj()
            invQQl, invQQr = 1 / (Ql * Ql.conj()), 1 / (Qr * Qr.conj())

            AAc1, BBc1 = jnp.sum(A * A.conj(), axis=1), jnp.sum(invQQr) * invQQl
            AAc2, BBc2 = jnp.sum(A * A.conj(), axis=0), jnp.sum(invQQl) * invQQr
            grad1 = AAc1 - BBc1
            grad2 = AAc2 - BBc2

            if step_normalizer == "2nd":
                step1 = precond_lr / _add_eps(jnp.max(jnp.real(AAc1 + BBc1)))
                step2 = precond_lr / _add_eps(jnp.max(jnp.real(AAc2 + BBc2)))
            else:
                step1 = precond_lr / _add_eps(jnp.max(jnp.abs(grad1)))
                step2 = precond_lr / _add_eps(jnp.max(jnp.abs(grad2)))

            Ql = Ql - step1 * grad1 * Ql
            Qr = Qr - step2 * grad2 * Qr
        elif Ql.ndim == 1 and Qr.ndim == 2 and Ql.shape[0] >= Qr.shape[0]:
            # drop v when left is diagonal, right is dense, and gradient is a tall matrix
            A = (Ql[:, None] * dG) @ Qr.conj().T
            invQQl = 1 / (Ql * Ql.conj())
            invQr = _solve_triangular(
                Qr, jnp.eye(Qr.shape[0], dtype=Qr.dtype), upper=True
            )
            invQQr = invQr.conj().T @ invQr

            AAc, BBc = jnp.sum(A * A.conj(), axis=1), jnp.trace(invQQr) * invQQl
            AhA, BBh = A.conj().T @ A, jnp.sum(invQQl) * invQQr
            grad1 = AAc - BBc
            grad2 = jnp.triu(AhA - BBh)

            if step_normalizer == "2nd":
                step1 = precond_lr / _add_eps(jnp.max(jnp.real(AAc + BBc)))
                step2 = precond_lr / _add_eps(_norm_lower_bound(AhA + BBh))
            else:
                step1 = precond_lr / _add_eps(jnp.max(jnp.abs(grad1)))
                step2 = precond_lr / _add_eps(_norm_lower_bound(grad2))

            Ql -= step1 * grad1 * Ql
            Qr -= step2 * grad2 @ Qr
        elif Qr.ndim == 1 and Ql.ndim == 2 and Qr.shape[0] >= Ql.shape[0]:
            # drop v when right is diagonal, left is dense, and gradient is a short matrix
            A = Ql @ (dG * Qr.conj())
            invQl = _solve_triangular(
                Ql, jnp.eye(Ql.shape[0], dtype=Ql.dtype), upper=True
            )
            invQQl = invQl.conj().T @ invQl
            invQQr = 1 / (Qr * Qr.conj())

            AAh, BhB = A @ A.conj().T, jnp.sum(invQQr) * invQQl
            AAc, BBc = jnp.sum(A * A.conj(), axis=0), jnp.trace(invQQl) * invQQr
            grad1 = jnp.triu(AAh - BhB)
            grad2 = AAc - BBc

            if step_normalizer == "2nd":
                step1 = precond_lr / _add_eps(_norm_lower_bound(AAh + BhB))
                step2 = precond_lr / _add_eps(jnp.max(jnp.real(AAc + BBc)))
            else:
                step1 = precond_lr / _add_eps(_norm_lower_bound(grad1))
                step2 = precond_lr / _add_eps(jnp.max(jnp.abs(grad2)))

            Ql -= step1 * grad1 @ Ql
            Qr -= step2 * grad2 * Qr
        else:
            # keeping v as an auxiliary variable could save computations (tradeoff of performance, similar to Hutchinson’s trick) when
            #   1) gradient is a tall matrix, but left side is a dense preconditioner, right side is diagonal
            #   2) gradient is a short matrix, but left side is a diagonal preconditioner, right side is dense
            #   3) both sides use dense preconditioner, but gradient is skewed (no saving for square shape gradient)
            key, subkey = jax.random.split(key)
            # JAX does an ok job at sharding this so let's leave alone for now
            v = jax.random.normal(subkey, dG.shape, dtype=dG.dtype)
            key, subkey = jax.random.split(key)
            return _update_precond_affine_math_(
                subkey, Ql, Qr, v, dG, precond_lr, precision
            )

        def _balance():
            max_l = jnp.max(jnp.abs(Ql))
            max_r = jnp.max(jnp.abs(Qr))

            rho = jnp.sqrt(max_l / max_r)
            new_Ql = Ql / rho
            new_Qr = Qr * rho
            return new_Ql, new_Qr

        key, subkey = jax.random.split(key)
        Ql, Qr = _efficient_cond(jax.random.uniform(subkey) < 0.01, _balance, (Ql, Qr))

        return [Ql, Qr]


def _precond_grad_affine_math(Ql, Qr, grad):
    if Ql.ndim == 2:
        if Qr.ndim == 2:  # Ql.ndim=2 and Qr.ndim=2:
            return jnp.linalg.multi_dot([Ql.conj().T, Ql, grad, Qr.conj().T, Qr])
        else:  # Ql.ndim=2 and Qr.ndim=1:
            return jnp.linalg.multi_dot([Ql.conj().T, Ql, grad * (Qr * Qr.conj())])
    else:
        if Qr.ndim == 2:  # Ql.ndim=1 and Qr.ndim=2:
            return jnp.linalg.multi_dot(
                [(Ql * Ql.conj())[:, None] * grad, Qr.conj().T, Qr]
            )
        else:  # Ql.ndim=1 and Qr.ndim=1:
            return (Ql * Ql.conj())[:, None] * grad * (Qr * Qr.conj())


def _sort_and_group_matrices(matrix_shapes: List[Tuple[int, ...]]):
    """Sorts and groups matrices by shape."""
    indexed_list = list(enumerate(matrix_shapes))
    sorted_indexed = sorted(indexed_list, key=lambda x: x[1])

    sorted_shapes = [shape for _, shape in sorted_indexed]
    change_indices = [original_index for original_index, _ in sorted_indexed]
    revert_indices = [0] * len(matrix_shapes)
    for new_pos, (original_index, _) in enumerate(sorted_indexed):
        revert_indices[original_index] = new_pos

    shape_groups = defaultdict(list)
    for i, shape in enumerate(sorted_shapes):
        shape_groups[shape].append(i)

    unique_sorted_shapes = list(shape_groups.keys())

    return unique_sorted_shapes, dict(shape_groups), change_indices, revert_indices


def _stack_matrices(gs: List[jnp.ndarray], Qs: List[Tuple[jnp.ndarray, jnp.ndarray]]):
    """Sorts and stacks grads and preconditioners for scan."""
    shapes = [g.shape for g in gs]

    unique_shapes, shape_groups, change_indices, revert_indices = (
        _sort_and_group_matrices(shapes)
    )

    sorted_gs = [gs[i] for i in change_indices]
    sorted_Qs = [Qs[i] for i in change_indices]

    stacked_gs = []
    stacked_Qs = []

    for shape in unique_shapes:
        indices = shape_groups[shape]
        stacked_g = jnp.stack([sorted_gs[i] for i in indices])
        stacked_gs.append(stacked_g)
        stacked_Q = [
            jnp.stack([sorted_Qs[i][0] for i in indices]),
            jnp.stack([sorted_Qs[i][1] for i in indices]),
        ]
        stacked_Qs.append(stacked_Q)

    return stacked_gs, stacked_Qs, revert_indices


def _unstack_matrices(
    stacked_gs: List[jnp.ndarray],
    stacked_Qs: List[Tuple[jnp.ndarray, jnp.ndarray]],
    revert_indices: List[int],
):
    """Unstacks and reverts matrices to their original order."""
    unstacked_gs = []
    unstacked_Qs = []

    for g, Q in zip(stacked_gs, stacked_Qs):
        unstacked_gs.extend(jnp.split(g, g.shape[0]))
        unstacked_Qs.extend(
            zip(jnp.split(Q[0], Q[0].shape[0]), jnp.split(Q[1], Q[1].shape[0]))
        )

    gs = [jnp.squeeze(unstacked_gs[i], axis=0) for i in revert_indices]
    Qs = [
        [jnp.squeeze(q[0], axis=0), jnp.squeeze(q[1], axis=0)]
        for q in [unstacked_Qs[i] for i in revert_indices]
    ]

    return gs, Qs


def _efficient_cond(
    predicate: bool, compute_fn: Callable, init_state: Any, *args, **kwargs
) -> Tuple:
    """Avoids wasteful buffer allocation with XLA.

    Useful when the False branch of the cond is only a pass-through. `init_state`
    must be same structure as the output of `compute_fn`."""

    def _iter_body(_):
        return False, compute_fn(*args, **kwargs)

    def _iter_condition(state):
        return state[0]

    results = jax.lax.while_loop(_iter_condition, _iter_body, (predicate, init_state))

    return results[1]
