from typing import Any, Optional, Union, Callable, NamedTuple, List

import jax
from jax import numpy as jnp
from jax.random import PRNGKey

from optax import tree_utils as otu
from optax._src import base, transform, clipping
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

from psgd_jax.utils import add_eps, apply_momentum


class PSGDAffineState(NamedTuple):
    count: jax.Array
    key: PRNGKey
    mu: Optional[base.Updates]
    Qs: List[List[jax.Array]]


def scale_by_affine(
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 128,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
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
        affine_reshapers = [_shape_as_matrix(x) for x in jax.tree.leaves(params)]
        Qs = [
            _initQ(s[2], max_size_triangular, max_skew_triangular, jnp.float32)
            for s in affine_reshapers
        ]

        # initial state
        return PSGDAffineState(count=jnp.zeros([], jnp.int32), key=key, mu=mu, Qs=Qs)

    def update_fn(
        updates: base.Updates,
        state: PSGDAffineState,
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
        affine_reshapers = [_shape_as_matrix(x) for x in jax.tree.leaves(updates)]

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        def _update_precond(key: PRNGKey, state: PSGDAffineState, Hvs, vs):
            Hvs = [r[0](x) for x, r in zip(jax.tree.leaves(Hvs), affine_reshapers)]

            if hessian_based_preconditioning:
                vs = [r[0](x) for x, r in zip(jax.tree.leaves(vs), affine_reshapers)]

                # init Qs
                def init_q(v, h):
                    if precond_init_scale is not None:
                        return precond_init_scale
                    else:
                        return (jnp.sum(v * v.conj()) / jnp.sum(h * h.conj())) ** 0.25

                Qs = jax.lax.cond(
                    state.count == 0,
                    lambda: [
                        [init_q(v, h) ** 0.5 * q for q in Qlr]
                        for v, h, Qlr in zip(vs, Hvs, state.Qs)
                    ],
                    lambda: state.Qs,
                )

                # update preconditioner
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, len(Qs))
                Qs = [
                    _update_precond_affine_math_(
                        k,
                        Qlr[0],
                        Qlr[1],
                        v,
                        h,
                        precond_lr_in,
                        step_normalizer_order,
                        precision,
                    )
                    for (k, Qlr, v, h) in zip(
                        keys, Qs, jax.tree.leaves(vs), jax.tree.leaves(Hvs)
                    )
                ]
            else:
                # init Qs
                def init_q(g):
                    if precond_init_scale is not None:
                        return precond_init_scale
                    else:
                        return (g.size / jnp.sum(g * g.conj())) ** 0.25

                Qs = jax.lax.cond(
                    state.count == 0,
                    lambda: [
                        [init_q(g) ** 0.5 * q for q in Qlr]
                        for g, Qlr in zip(Hvs, state.Qs)
                    ],
                    lambda: state.Qs,
                )

                # update preconditioner
                key, subkey = jax.random.split(key)
                keys = jax.random.split(subkey, len(Qs))
                Qs = [
                    _update_precond_affine_dropv_math(
                        k,
                        Qlr[0],
                        Qlr[1],
                        h,
                        precond_lr_in,
                        step_normalizer_order,
                        precision,
                    )
                    for (k, Qlr, h) in zip(keys, Qs, jax.tree.leaves(Hvs))
                ]

            return key, Qs

        def _dont_update_precond(key, state, Hvs, vs):
            return key, state.Qs

        if not hessian_based_preconditioning:
            # update cond not passed in, create here
            key, subkey = jax.random.split(key)
            update_preconditioner = jnp.logical_or(
                jax.random.uniform(subkey) < preconditioner_update_probability,
                state.count < 2,
            )
            # use grads as Hvp
            Hvp = updates

        key, Qs = jax.lax.cond(
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
        flat_updates = [
            r[0](u) for u, r in zip(jax.tree.leaves(updates), affine_reshapers)
        ]
        flat_updates = [
            _precond_grad_affine_math(Qlr[0], Qlr[1], g)
            for (Qlr, g) in zip(Qs, flat_updates)
        ]
        flat_updates = [r[1](u) for u, r in zip(flat_updates, affine_reshapers)]
        updates = jax.tree_unflatten(jax.tree.structure(updates), flat_updates)

        # clipping
        if update_global_norm_clip is not None:
            updates, _ = clipping.clip_by_global_norm(update_global_norm_clip).update(
                updates, base.EmptyState
            )

        mu = otu.tree_cast(mu, mu_dtype)
        state = PSGDAffineState(count=count_inc, key=key, mu=mu, Qs=Qs)
        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def affine(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.9,
    nesterov: bool = False,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    max_size_triangular: int = 4096,
    max_skew_triangular: int = 128,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    update_global_norm_clip: Optional[float] = None,
    step_normalizer_order: str = "2nd",
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
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
        scale_by_affine(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            nesterov=nesterov,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
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


def _shape_as_matrix(x: jax.Array) -> tuple:
    """Reshapes tensor x to a matrix with conditions to improve efficiency.

    From original pytorch version.

    Args:
        x: jax.Array, tensor to be reshaped.

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
    if x.ndim == 2:  # t already is a matrix, do nothing
        return lambda u: u, lambda v: v, x.shape
    elif x.ndim < 2:  # scalar or vector, simple reshape to matrix
        mtx_shape = (1, x.size)
        return (
            lambda u, shape=mtx_shape: u.reshape(shape),
            lambda v, shape=x.shape: v.reshape(shape),
            mtx_shape,
        )
    else:  # higher order tensor, a little complicated
        p0, s0 = tuple(range(x.ndim)), x.shape  # original permutation and shape
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


def _initQ(shape, max_size, max_skew, dtype=jnp.float32):
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
    return jax.lax.linalg.triangular_solve(a, b, left_side=left, lower=not upper)


def _update_precond_affine_math_(
    key, Ql, Qr, dX, dG, precond_lr, step_normalizer, precision
):
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
                    step1 = precond_lr / add_eps(_norm_lower_bound(AAh + BhB))
                    step2 = precond_lr / add_eps(_norm_lower_bound(AhA + BBh))
                else:
                    step1 = precond_lr / add_eps(_norm_lower_bound(grad1))
                    step2 = precond_lr / add_eps(_norm_lower_bound(grad2))

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
                    step1 = precond_lr / add_eps(_norm_lower_bound(AAh + BhB))
                    step2 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
                else:
                    step1 = precond_lr / add_eps(_norm_lower_bound(grad1))
                    step2 = precond_lr / add_eps(jnp.max(jnp.abs(grad2)))

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
                    step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
                    step2 = precond_lr / add_eps(_norm_lower_bound(AhA + BBh))
                else:
                    step1 = precond_lr / add_eps(jnp.max(jnp.abs(grad1)))
                    step2 = precond_lr / add_eps(_norm_lower_bound(grad2))

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
                    step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc1 + BBc1)))
                    step2 = precond_lr / add_eps(jnp.max(jnp.real(AAc2 + BBc2)))
                else:
                    step1 = precond_lr / add_eps(jnp.max(jnp.abs(grad1)))
                    step2 = precond_lr / add_eps(jnp.max(jnp.abs(grad2)))

                Ql -= step1 * grad1 * Ql
                Qr -= step2 * grad2 * Qr

        def _balance(Ql, Qr):
            max_l = jnp.max(jnp.abs(Ql))
            max_r = jnp.max(jnp.abs(Qr))

            rho = jnp.sqrt(max_l / max_r)
            Ql /= rho
            Qr *= rho
            return Ql, Qr

        key, subkey = jax.random.split(key)
        Ql, Qr = jax.lax.cond(
            jax.random.uniform(subkey) < 0.01, _balance, lambda ql, qr: (ql, qr), Ql, Qr
        )

        return [Ql, Qr]


def _update_precond_affine_dropv_math(
    key, Ql, Qr, dG, precond_lr, step_normalizer, precision
):
    with jax.default_matmul_precision(precision):

        def balance(key, Ql, Qr):
            def _balance(Ql, Qr):
                max_l = jnp.max(jnp.abs(Ql))
                max_r = jnp.max(jnp.abs(Qr))

                rho = jnp.sqrt(max_l / max_r)
                Ql /= rho
                Qr *= rho
                return Ql, Qr

            Ql, Qr = jax.lax.cond(
                jax.random.uniform(key) < 0.01,
                _balance,
                lambda ql, qr: (ql, qr),
                Ql,
                Qr,
            )
            return Ql, Qr

        if Ql.ndim == 1 and Qr.ndim == 1:
            # drop v when both dims use diagonal preconditioners
            A = Ql[:, None] * dG * Qr.conj()
            invQQl, invQQr = 1 / (Ql * Ql.conj()), 1 / (Qr * Qr.conj())

            AAc1, BBc1 = jnp.sum(A * A.conj(), axis=1), jnp.sum(invQQr) * invQQl
            AAc2, BBc2 = jnp.sum(A * A.conj(), axis=0), jnp.sum(invQQl) * invQQr
            grad1 = AAc1 - BBc1
            grad2 = AAc2 - BBc2

            if step_normalizer == "2nd":
                step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc1 + BBc1)))
                step2 = precond_lr / add_eps(jnp.max(jnp.real(AAc2 + BBc2)))
            else:
                step1 = precond_lr / add_eps(jnp.max(jnp.abs(grad1)))
                step2 = precond_lr / add_eps(jnp.max(jnp.abs(grad2)))

            Ql = Ql - step1 * grad1 * Ql
            Qr = Qr - step2 * grad2 * Qr

            key, subkey = jax.random.split(key)
            Ql, Qr = balance(subkey, Ql, Qr)

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
                step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
                step2 = precond_lr / add_eps(_norm_lower_bound(AhA + BBh))
            else:
                step1 = precond_lr / add_eps(jnp.max(jnp.abs(grad1)))
                step2 = precond_lr / add_eps(_norm_lower_bound(grad2))

            Ql -= step1 * grad1 * Ql
            Qr -= step2 * grad2 @ Qr

            key, subkey = jax.random.split(key)
            Ql, Qr = balance(subkey, Ql, Qr)

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
                step1 = precond_lr / add_eps(_norm_lower_bound(AAh + BhB))
                step2 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
            else:
                step1 = precond_lr / add_eps(_norm_lower_bound(grad1))
                step2 = precond_lr / add_eps(jnp.max(jnp.abs(grad2)))

            Ql -= step1 * grad1 @ Ql
            Qr -= step2 * grad2 * Qr

            key, subkey = jax.random.split(key)
            Ql, Qr = balance(subkey, Ql, Qr)

        else:
            # keeping v as an auxiliary variable could save computations (tradeoff of performance, similar to Hutchinson’s trick) when
            #   1) gradient is a tall matrix, but left side is a dense preconditioner, right side is diagonal
            #   2) gradient is a short matrix, but left side is a diagonal preconditioner, right side is dense
            #   3) both sides use dense preconditioner, but gradient is skewed (no saving for square shape gradient)
            key, subkey = jax.random.split(key)
            v = otu.tree_random_like(subkey, dG, jax.random.normal)
            key, subkey = jax.random.split(key)
            return _update_precond_affine_math_(
                subkey, Ql, Qr, v, dG, precond_lr, step_normalizer, precision
            )

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
