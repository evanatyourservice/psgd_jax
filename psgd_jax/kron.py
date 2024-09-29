from typing import Any, List, Optional, Union, Callable, Tuple
from functools import partial

import jax
from jax import vmap
import jax.numpy as jnp
import opt_einsum
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.linear_algebra import global_norm
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


def precond_update_prob_schedule(
    max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=200
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 200 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work very well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.minimum(
            jnp.maximum(max_prob * jnp.exp(-decay * (n - flat_start)), min_prob),
            max_prob,
        )

    return _schedule


def scale_by_kron(
    b1: float = 0.9,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        b1: float, momentum parameter.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        max_skew_triangular: int, max skew for dim's preconditioner to be triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of bool same structure as params
            indicating scanned layers. PSGD will vmap over the first dim.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype)

    # some hardcoded settings
    precond_init_scale = 0.1
    momentum_into_preconditioner = True
    integrate_out_v = False

    def map_fn(do_map, fn, *args):
        """Maybe map a fn along axes."""
        if do_map:
            if lax_map_scanned_layers:
                return jax.lax.map(
                    lambda xs: fn(*xs),
                    xs=args,
                    batch_size=lax_map_batch_size if lax_map_batch_size > 1 else None,
                )
            else:
                return vmap(fn)(*args)
        else:
            return fn(*args)

    def init_fn(params):
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)

        # momentum
        mu = None
        if b1 > 0:
            mu = otu.tree_zeros_like(params, mu_dtype)

        # preconditioners
        Qs = [
            _init_Q_exprs(
                t[0] if s else t,
                precond_init_scale,
                max_size_triangular,
                max_skew_triangular,
                precond_dtype,
            )[0]
            for t, s in zip(jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]
        # broadcast for scanned layers
        Qs = [
            (
                jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0), q
                )
                if s
                else q
            )
            for q, t, s in zip(
                Qs, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]
        Qs = jax.tree.structure(params).unflatten(Qs)

        # Calculate sizes for nu (preconditioner) and mu (momentum)
        Qs_n_elements = sum([q.size for q in jax.tree.leaves(Qs)])
        Qs_size_MB = sum(
            [q.size * q.dtype.itemsize / (2**20) for q in jax.tree.leaves(Qs)]
        )
        if jax.process_index() == 0:
            print(
                f"PSGD Preconditioners size: {Qs_n_elements} elements, "
                f"{Qs_size_MB:.2f} MB"
            )
        if mu is not None:
            mu_n_elements = sum([p.size for p in jax.tree.leaves(mu)])
            mu_size_MB = sum(
                [p.size * p.dtype.itemsize / (2**20) for p in jax.tree.leaves(mu)]
            )
            if jax.process_index() == 0:
                print(
                    f"PSGD Momentum size: {mu_n_elements} elements, {mu_size_MB:.2f} MB"
                )

        # initial state
        return dict(count=jnp.zeros([], jnp.int32), mu=mu, Qs_preconditioners=Qs)

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = jax.random.fold_in(jax.random.PRNGKey(5318008), state["count"])

        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        # momentum
        momentum_updates = updates
        mu = None
        if state["mu"] is not None:
            momentum_updates, mu = _apply_momentum(updates, state["mu"], count_inc, b1)

        # flatten pytrees
        updates, grads_structure = jax.tree.flatten(updates)
        momentum_updates = grads_structure.flatten_up_to(momentum_updates)
        Qs = grads_structure.flatten_up_to(state["Qs_preconditioners"])
        scanned_layers_ = grads_structure.flatten_up_to(scanned_layers_)

        # get einsum expressions
        expressions = [
            _init_Q_exprs(
                t[0] if s else t,
                precond_init_scale,
                max_size_triangular,
                max_skew_triangular,
                precond_dtype,
                existing_Q=jax.tree.map(lambda d: d[0], Q) if s else Q,
            )
            for t, s, Q in zip(updates, scanned_layers_, Qs)
        ]

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precision):
                if momentum_into_preconditioner:
                    precond_updates_in = momentum_updates
                else:
                    precond_updates_in = updates

                if integrate_out_v:
                    Vs = [None for _ in precond_updates_in]
                else:
                    # random vectors
                    key, subkey = jax.random.split(key)
                    Vs_keys = jax.random.split(subkey, len(precond_updates_in))
                    Vs = [
                        jax.random.normal(k, shape=g.shape, dtype=g.dtype)
                        for k, g in zip(Vs_keys, precond_updates_in)
                    ]

                # form conjB or invQhinvQ
                conjB_or_invQhinvQ = [
                    map_fn(s, _conjB_or_invQhinvQ, Q, g, v)
                    for s, Q, g, v in zip(scanned_layers_, Qs, precond_updates_in, Vs)
                ]

                # update Qs
                new_Qs = [
                    map_fn(
                        s,
                        partial(
                            _update_Q,
                            exprs=exprs,
                            precond_lr=precond_lr_in,
                            integrate_out_v=integrate_out_v,
                        ),
                        Q,
                        g,
                        c_or_i,
                    )
                    for s, exprs, Q, g, c_or_i in zip(
                        scanned_layers_,
                        expressions,
                        Qs,
                        precond_updates_in,
                        conjB_or_invQhinvQ,
                    )
                ]

                # maybe balance preconditioners (useful for quantization/low precision)
                def balance_Qs(Qs: List[List[jax.Array]]):
                    def _balance_Q(Q: List[jax.Array]):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.prod(norms) ** (1 / len(norms))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return [
                        map_fn(s, _balance_Q, Q) for Q, s in zip(Qs, scanned_layers_)
                    ]

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) < 0.01
                new_Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, new_Qs)

                new_Qs = otu.tree_cast(new_Qs, precond_dtype)
                return new_Qs

        key, subkey = jax.random.split(key)
        do_update = jax.random.uniform(subkey, dtype=jnp.float32) < update_prob_in
        key, subkey = jax.random.split(key)
        Qs = jax.lax.cond(
            do_update, update_preconditioner, lambda _, qs: qs, subkey, Qs
        )

        # precondition gradients
        precond_gs = [
            map_fn(s, partial(_precond_grad_kron_math, exprs=exprs), Q, g)
            for s, exprs, Q, g in zip(
                scanned_layers_, expressions, Qs, momentum_updates
            )
        ]

        # trust region
        # global clipping (sqrt n params)
        max_norm = jnp.sqrt(
            jnp.array(
                [p.size for p in jax.tree.leaves(precond_gs)], dtype=jnp.float32
            ).sum()
        )
        precond_gs = _global_clip(precond_gs, max_norm)
        # element-wise clipping (1.0)
        precond_gs = jax.tree.map(lambda x: jnp.clip(x, -1.0, 1.0), precond_gs)

        # unflatten pytrees
        updates = grads_structure.unflatten(precond_gs)
        Qs = grads_structure.unflatten(Qs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(count=count_inc, mu=mu, Qs_preconditioners=Qs)

        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    max_skew_triangular: int = 10,
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate.
        b1: float, momentum parameter.
        weight_decay: float, weight decay. PSGD does not need high weight decay.
        mask: optional Any or callable, mask to apply to parameters.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        max_skew_triangular: int, max skew for dim's preconditioner to be triangular.
        precond_lr: float or callable, learning rate for the preconditioner.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioner.
        precision: str, precision for matmul during preconditioner update,
            'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of bool same structure as params
            indicating scanned layers. PSGD will vmap over the first dim.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_kron(
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            max_size_triangular=max_size_triangular,
            max_skew_triangular=max_skew_triangular,
            precond_lr=precond_lr,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precision=precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
        )
    ]
    if weight_decay > 0:
        opt.append(transform.add_decayed_weights(weight_decay, mask=mask))
    opt.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*opt)


def _apply_momentum(
    updates: base.Updates, momentum: base.Updates, step, b1
) -> Tuple[base.Updates, base.Updates]:
    mu = otu.tree_update_moment(updates, momentum, b1, 1)
    updates = otu.tree_bias_correction(mu, b1, step)
    return updates, mu


def _add_eps(x):
    return jnp.clip(x, 1e-30, None)


def _global_clip(updates, max_norm):
    g_norm = global_norm(updates)
    g_norm = jnp.maximum(max_norm, g_norm)
    updates = jax.tree.map(
        lambda u: (u / g_norm.astype(u.dtype)) * max_norm.astype(u.dtype), updates
    )
    return updates


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Numerical results on random matrices with a wide range of distributions and
    sizes suggest, norm(A) <= sqrt(2) * norm_lower_bound(A). Looks to be a very
    tight lower bound.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        A_conj = A.conj()

        aa = jnp.real(A * A_conj)

        aa_sum0 = jnp.sum(aa, axis=0)
        aa_sum1 = jnp.sum(aa, axis=1)
        i = jnp.argmax(aa_sum0, 0)
        j = jnp.argmax(aa_sum1, 0)
        value0 = jax.lax.dynamic_index_in_dim(aa_sum0, i, 0, keepdims=False)
        value1 = jax.lax.dynamic_index_in_dim(aa_sum1, j, 0, keepdims=False)

        def gt_branch():
            x = jax.lax.dynamic_index_in_dim(A, i, 1, keepdims=False)
            x = x.conj() @ A
            return max_abs * jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A_conj.T)

        def le_branch():
            x = jax.lax.dynamic_index_in_dim(A, j, 0, keepdims=False)
            x = A @ x.conj()
            return max_abs * jnp.linalg.norm(A_conj.T @ (x / jnp.linalg.norm(x)))

        return jax.lax.cond(value0 > value1, gt_branch, le_branch)

    def no_calc(_):
        return max_abs

    return jax.lax.cond(max_abs > 0, calc, no_calc, A)


def _init_Q_exprs(t, scale, max_size, max_skew, dtype, existing_Q=None):
    """
    For a scalar or tensor `t`, we initialize its preconditioner `Q` and reusable
    contraction expressions for updating `Q` and preconditioning gradient.

    1, Preconditioner `Q` is initialized to
    `Q = scale * I = scale * kron(eye(t.shape[0]), eye(t.shape[1]), ...)`
    where the `eye(.)` may be replaced with `diag(ones(.))` if that dim is too large,
    determined by `max_size` and `max_skew`.

    2, A series of einsum contract expressions. The following subscript examples are for
    a 5th order tensor.
        2.1, `exprA` is the expression for calculating `A`, e.g.,
            `'aA,bB,cC,dD,eE,ABCDE->abcde'`
        2.2, `exprGs` is a list of expressions for calculating the gradients wrt `Q`
            on each dim, e.g., `'abCde,abγde->Cγ'` for the middle dim of a 5th order
            tensor `Q`.
        2.3, `exprP` is the expression for calculating the preconditioned gradient,
            e.g., `'aA,bB,cC,dD,eE,aα,bβ,cγ,dδ,eε,αβγδε->ABCDE'`

    If `existing_Q` is passed in, only expressions are returned.
    """
    shape = t.shape
    if len(shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones_like(t, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = opt_einsum.contract_expression(",->", Q[0].shape, t.shape)
        exprP = opt_einsum.contract_expression(",,->", Q[0].shape, Q[0].shape, t.shape)
        exprGs = [opt_einsum.contract_expression(",->", t.shape, t.shape)]
    else:  # tensor
        if len(shape) > 26:
            raise ValueError(
                f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters; "
                "Replace 26 with larger numbers!"
            )

        scale = scale ** (1 / len(shape))
        if len(shape) == 1:
            beta_size = 1  # 2nd largest size
        else:
            beta_size = sorted(list(shape))[-2]

        Q = [] if existing_Q is None else existing_Q
        exprGs = []
        # used for getting the subscripts for exprA
        piece1A, piece2A, piece3A = ([], "", "")
        # used for getting the subscripts for exprP
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, size in enumerate(shape):
            if size == 1 or size > max_size or size > max_skew * beta_size:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.ones(size, dtype=dtype))

                piece1A.append(opt_einsum.get_symbol(i))
                piece2A = piece2A + opt_einsum.get_symbol(i)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                piece1P.append(opt_einsum.get_symbol(i + 26))
                piece2P.append(opt_einsum.get_symbol(i + 26))
                piece3P = piece3P + opt_einsum.get_symbol(i + 26)
                piece4P = piece4P + opt_einsum.get_symbol(i + 26)

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1 + "," + piece1 + "->" + opt_einsum.get_symbol(i + 26)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )
            else:
                # use triangular matrix as preconditioner for this dim
                if existing_Q is None:
                    Q.append(scale * jnp.eye(size, dtype=dtype))

                piece1A.append(opt_einsum.get_symbol(i) + opt_einsum.get_symbol(i + 26))
                piece2A = piece2A + opt_einsum.get_symbol(i + 26)
                piece3A = piece3A + opt_einsum.get_symbol(i)

                a, b, c = (
                    opt_einsum.get_symbol(i),
                    opt_einsum.get_symbol(i + 26),
                    opt_einsum.get_symbol(i + 805),
                )
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

                piece1 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 26)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (
                            opt_einsum.get_symbol(i + 805)
                            if j == i
                            else opt_einsum.get_symbol(j)
                        )
                        for j in range(len(shape))
                    ]
                )
                subscripts = (
                    piece1
                    + ","
                    + piece2
                    + "->"
                    + opt_einsum.get_symbol(i + 26)
                    + opt_einsum.get_symbol(i + 805)
                )
                exprGs.append(
                    opt_einsum.contract_expression(subscripts, t.shape, t.shape)
                )

        subscripts = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprA = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], t.shape
        )

        subscripts = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )
        exprP = opt_einsum.contract_expression(
            subscripts, *[q.shape for q in Q], *[q.shape for q in Q], t.shape
        )

    exprGs = tuple(exprGs)

    if existing_Q is not None:
        return exprA, exprGs, exprP
    return [Q, (exprA, exprGs, exprP)]


def _solve_triangular(A, B, upper, left=True):
    """A triangular solve has roughly the same complexity as a matmul."""
    dtype_in = jnp.promote_types(A.dtype, B.dtype)
    A, B = A.astype(dtype_in), B.astype(dtype_in)
    leading_dims = 0
    if B.ndim > 2:
        leading_dims = B.ndim - 2
    solve_fn = partial(jax.lax.linalg.triangular_solve, left_side=left, lower=not upper)
    for _ in range(leading_dims):
        solve_fn = vmap(solve_fn, in_axes=(None, 0))
    return solve_fn(A, B)


def _triangular_inv(A):
    """Compute inv(A).

    A triangular solve has roughly the same complexity as a matmul.
    """
    I = jnp.eye(A.shape[0], dtype=A.dtype)
    return _solve_triangular(A, I, upper=True)


def _solve_triangular_right(X, A):
    """Compute X @ inv(A).

    A triangular solve has roughly the same complexity as a matmul.
    """
    if X.ndim > 1:
        return _solve_triangular(A, X, upper=True, left=False)
    else:
        return _solve_triangular(A, X[None, :], upper=True, left=False)[0]


def _conjB_or_invQhinvQ(Q, G, V):
    """Compute conjB or trace(inv(Q).H @ inv(Q)) depending on V."""
    if V is not None:
        order = G.ndim
        p = list(range(order))
        # permute dims like [0,1,2,3,4] -> [1,2,3,4,0]
        conjB = jnp.transpose(V.conj(), p[1:] + p[:1])
        for i, q in enumerate(Q):
            conjB = conjB / q if q.ndim < 2 else _solve_triangular_right(conjB, q)
            if i < order - 1:
                # transpose dims like
                # [1,2,3,4,0]->[0,2,3,4,1]->[0,1,3,4,2]->[0,1,2,4,3]->[0,1,2,3,4]
                conjB = jnp.swapaxes(conjB, i, order - 1)
        return conjB
    else:
        # V is integrated out, no need to form conjB
        invQ = [1 / q if q.ndim < 2 else _triangular_inv(q) for q in Q]
        invQhinvQ = [q.conj() * q if q.ndim < 2 else q.conj().T @ q for q in invQ]
        return invQhinvQ


def _update_Q(Q, G, conjB_or_invQhinvQ, exprs, precond_lr, integrate_out_v):
    """Compute A and update Q."""
    exprA, exprGs, _ = exprs

    A = exprA(*Q, G, backend="jax")
    A_conj = A.conj()
    if integrate_out_v:
        invQhinvQ = conjB_or_invQhinvQ
        trace_invQhinvQ = [
            jnp.sum(q) if q.ndim < 2 else jnp.trace(q) for q in invQhinvQ
        ]
    else:
        conjB = conjB_or_invQhinvQ
        conjB_conj = conjB.conj()

    def _update_single_q(i, q):
        term1 = exprGs[i](A, A_conj)
        if integrate_out_v:
            term2 = 1.0
            for j, trace in enumerate(trace_invQhinvQ):
                term2 = term2 * (trace if i != j else invQhinvQ[i])
        else:
            term2 = exprGs[i](conjB_conj, conjB)

        if q.ndim < 2:
            q -= (
                precond_lr
                / _add_eps(jnp.max(jnp.abs(term1 + term2)))
                * (term1 - term2)
                * q
            )
        else:
            q -= (
                precond_lr
                / _add_eps(_norm_lower_bound(term1 + term2))
                * jnp.triu(term1 - term2)
                @ q
            )
        return q

    return [_update_single_q(i, q) for i, q in enumerate(Q)]


def _precond_grad_kron_math(Q, G, exprs):
    """Precondition gradient G with preconditioner Q."""
    exprP = exprs[-1]
    return exprP(*[q.conj() for q in Q], *Q, G, backend="jax")
