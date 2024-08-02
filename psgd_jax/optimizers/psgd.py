from typing import Any, Optional, Union, Callable, Tuple

import jax
from jax import numpy as jnp
from jax.random import PRNGKey
from flax import struct

from optax import tree_utils as otu
from optax._src import base, transform, clipping
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain

from psgd_jax.optimizers.first_order_optimizers import scale_by_adam
from psgd_jax.optimizers.optimizer_utils import norm_grads


def add_eps(x):
    return jnp.where(x == 0, jnp.finfo(x.dtype).tiny, x)


def psgd_hvp_helper(
    key: PRNGKey,
    loss_fn: Callable,
    params: base.Params,
    loss_fn_extra_args: Tuple = (),
    has_aux: bool = False,
    pmap_axis_name: Optional[str] = None,
    preconditioner_update_probability: float = 1.0,
):
    """Helper function for computing hessian vector product for PSGD.

    This helps handle the calculation of a hessian vector product if wanting to use exact
    hvp instead of the default gradient whitening style preconditioner. It returns the
    loss fn output, gradients, hvp, vector, and a bool of whether we're updating the
    preconditioner this step. The hvp, vector, and update cond are then passed into PSGD's
    update fn. This fn is not needed if wanting to use the default gradient whitening style
    preconditioner.

    Args:
        key: PRNGKey, random key.
        loss_fn: callable, loss function.
        params: flax.Params, model parameters.
        loss_fn_extra_args: tuple, extra arguments for loss function to be used as
            `loss_fn(params, *loss_fn_extra_args)`.
        has_aux: bool, whether loss function has aux output.
        pmap_axis_name: optional str, provide axis name if in pmap mode.
        preconditioner_update_probability: float, probability of updating the preconditioner.

    Returns:
        loss_out: jnp.ndarray, output of loss function.
        grads: flax.Params, gradients.
        hvp: flax.Params, hessian vector product.
        vector: flax.Params, random vector.
        update_preconditioner: bool, whether we're updating preconditioner this step.
    """
    if pmap_axis_name is not None:
        # make sure same key is used on all devices
        key = jax.lax.all_gather(key, pmap_axis_name)[0]

    key1, key2 = jax.random.split(key)

    obj_fn = lambda params: loss_fn(params, *loss_fn_extra_args)

    def grad_fn(params):
        loss_out, grad = jax.value_and_grad(obj_fn, has_aux=has_aux)(params)
        return grad, loss_out

    def hvp_fn(params):
        # TODO sharding
        vector = otu.tree_random_like(key1, params, jax.random.normal)
        grad, hvp, loss_out = jax.jvp(grad_fn, (params,), (vector,), has_aux=True)

        if pmap_axis_name is not None:
            grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)
            hvp = jax.lax.pmean(hvp, axis_name=pmap_axis_name)

        return grad, loss_out, hvp, vector

    # TODO (evanatyourservice): finite difference hvp option

    def g_fn(params):
        grad, loss_out = grad_fn(params)
        dummy_hvp = jax.tree.map(jnp.zeros_like, params)
        dummy_vector = jax.tree.map(jnp.zeros_like, params)

        if pmap_axis_name is not None:
            grad = jax.lax.pmean(grad, axis_name=pmap_axis_name)

        return grad, loss_out, dummy_hvp, dummy_vector

    update_precond = jax.random.uniform(key2) < preconditioner_update_probability

    grad, loss_out, hvp, vector = jax.lax.cond(update_precond, hvp_fn, g_fn, params)

    return loss_out, grad, hvp, vector, update_precond


@struct.dataclass
class PSGDState:
    count: jax.Array
    mu: Optional[base.Updates]
    U: Optional[jax.Array]
    V: Optional[jax.Array]
    d: Optional[jax.Array]
    Qs: Optional[jax.Array]
    key: PRNGKey
    diag_opt_state: Optional[base.OptState]
    affine_reshapers: Optional[Tuple[Callable, Callable, tuple]] = struct.field(
        pytree_node=False
    )


def scale_by_psgd(
    preconditioner_type: str = "xmat",
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.0,
    heavyball: bool = False,
    nesterov: bool = False,
    gradient_clip: Optional[float] = None,
    uvd_rank_of_approximation: int = 10,
    affine_max_size_triangular: int = 4096,
    affine_max_skew_triangular: int = 128,
    affine_scanned_layers: Optional[Any] = None,  # TODO (evanatyourservice)
    step_normalizer_order: str = "2nd",
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    feed_into_adam: bool = False,
    graft_adam_lr: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_norm_grads: Optional[str] = None,
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements UVd, XMat, and Affine PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        preconditioner_type: str, 'xmat', 'uvd', or 'affine'.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        heavyball: bool, whether to use Heavyball momentum.
        nesterov: bool, whether to use Nesterov momentum.
        gradient_clip: optional float, global gradient norm clipping.
        uvd_rank_of_approximation: int, rank of approximation for uvd preconditioner.
        affine_max_size_triangular: int, max size for affine preconditioner to be
            triangular.
        affine_max_skew_triangular: int, max skew for affine preconditioner to be
            triangular.
        affine_scanned_layers: optional Any, pytree same structure as params denoting
            which layers should be vmapped over for affine preconditioner (set scanned
            layers to True).
        step_normalizer_order: str, '1st' or '2nd'.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        feed_into_adam: bool, whether to feed the preconditioned gradients into
            adam optimizer.
        graft_adam_lr: bool, whether to graft adam step size for updates.
        adam_b1: float, beta1 parameter for the grafting optimizer.
        adam_b2: float, beta2 parameter for the grafting optimizer.
        adam_norm_grads: optional str, 'global', 'layer', or None. Whether to
            normalize gradients to unit norm before grafting optimizer.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    if feed_into_adam and graft_adam_lr:
        print(
            "WARNING: both feed_into_adam and grafting are set to True for psgd. "
            "Disabling grafting and keeping feed_into_adam."
        )
        graft_adam_lr = False
    mu_dtype = canonicalize_dtype(mu_dtype)

    diag_opt = None
    if graft_adam_lr or feed_into_adam:
        print("PSGD: Using diagonal optimizer.")
        diag_opt = []
        if adam_norm_grads is not None:
            diag_opt += [
                norm_grads(layerwise=adam_norm_grads in ["layer", "layerwise"])
            ]
        if gradient_clip:
            diag_opt += [clipping.clip_by_global_norm(gradient_clip)]
        diag_opt += [
            scale_by_adam(
                b1=adam_b1, b2=adam_b2, eps=1e-8, mu_dtype=mu_dtype, nesterov=nesterov
            )
        ]
        diag_opt = chain(*diag_opt)

    def init_fn(params):
        key = seed if seed is not None else jax.random.PRNGKey(36)

        # preliminary
        n_params = sum([x.size for x in jax.tree.leaves(params)])
        affine_reshapers = None
        if preconditioner_type == "affine":
            affine_reshapers = [_shape_as_matrix(x) for x in jax.tree.leaves(params)]

        # momentum
        mu = None
        if b1 > 0 and not feed_into_adam:
            print("PSGD: Using momentum.")
            if preconditioner_type == "xmat":
                mu = jnp.zeros((n_params,), mu_dtype)
            elif preconditioner_type == "uvd":
                mu = jnp.zeros((n_params, 1), mu_dtype)
            else:  # affine
                mu = otu.tree_zeros_like(params, mu_dtype)
                mu = [r[0](x) for x, r in zip(jax.tree.leaves(mu), affine_reshapers)]

        # preconditioners
        if preconditioner_type == "xmat":
            U = jnp.ones((n_params,), jnp.float32)
            V = jnp.zeros((n_params,), jnp.float32)
            d, Qs = None, None
        elif preconditioner_type == "uvd":
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
            Qs = None
        elif preconditioner_type == "affine":
            Qs = [
                _initQ(
                    s[2],
                    affine_max_size_triangular,
                    affine_max_skew_triangular,
                    jnp.float32,
                )
                for s in affine_reshapers
            ]
            U, V, d = None, None, None
        else:
            raise ValueError(f"Unknown preconditioner type: {preconditioner_type}")

        # grafting
        diag_state = None
        if diag_opt is not None:
            diag_state = diag_opt.init(params)

        # initial state
        return PSGDState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            U=U,
            V=V,
            d=d,
            Qs=Qs,
            key=key,
            diag_opt_state=diag_state,
            affine_reshapers=affine_reshapers,
        )

    def update_fn(
        updates: base.Updates,
        state: PSGDState,
        params: base.Params = None,
        Hvp: Optional[base.Updates] = None,
        vector: Optional[base.Updates] = None,
        update_preconditioner: Optional[bool] = None,
    ):
        # use hessian preconditioning if hessian provided
        # otherwise use gg^T whitening type preconditioning
        hessian_based_preconditioning = Hvp is not None

        count_inc = safe_int32_increment(state.count)
        key = state.key
        grads = updates
        goal_shape = (-1,) if preconditioner_type == "xmat" else (-1, 1)

        precond_lr_in = precond_lr
        if isinstance(precond_lr, Callable):
            precond_lr_in = precond_lr(count_inc)

        if preconditioner_type in ["xmat", "uvd"]:
            flat_grads = jnp.concatenate(
                [jnp.reshape(x, goal_shape) for x in jax.tree.leaves(grads)], 0
            )
        else:
            flat_grads = [
                r[0](x) for x, r in zip(jax.tree.leaves(grads), state.affine_reshapers)
            ]

        def _update_precond(key: PRNGKey, state: PSGDState, Hvs, vs):
            if preconditioner_type == "xmat":
                v = jnp.concatenate(
                    [jnp.reshape(x, goal_shape) for x in jax.tree.leaves(vs)], 0
                )
                if hessian_based_preconditioning:
                    h = jnp.concatenate(
                        [jnp.reshape(x, goal_shape) for x in jax.tree.leaves(Hvs)], 0
                    )
                else:
                    h = flat_grads

                # init U
                if precond_init_scale is not None:
                    init_scale = precond_init_scale
                else:
                    if hessian_based_preconditioning:
                        init_scale = (jnp.sum(v * v) / jnp.sum(h * h)) ** 0.25
                    else:
                        init_scale = (
                            len(flat_grads) / jnp.sum(jnp.square(flat_grads))
                        ) ** 0.25
                U = jax.lax.cond(
                    state.count == 0, lambda: state.U * init_scale, lambda: state.U
                )

                # update preconditioner
                U, V = _update_precond_Xmat_math_(
                    U, state.V, v, h, precond_lr_in, step_normalizer_order, precision
                )
                d, Qs = None, None
            elif preconditioner_type == "uvd":
                v = jnp.concatenate(
                    [jnp.reshape(x, goal_shape) for x in jax.tree.leaves(vs)], 0
                )
                if hessian_based_preconditioning:
                    h = jnp.concatenate(
                        [jnp.reshape(x, goal_shape) for x in jax.tree.leaves(Hvs)], 0
                    )
                else:
                    h = flat_grads

                # init d
                if precond_init_scale is not None:
                    init_scale = precond_init_scale
                else:
                    if hessian_based_preconditioning:
                        init_scale = (jnp.sum(v * v) / jnp.sum(h * h)) ** 0.25
                    else:
                        init_scale = (
                            len(flat_grads) / jnp.sum(jnp.square(flat_grads))
                        ) ** 0.25
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
                Qs = None
            else:  # affine
                if hessian_based_preconditioning:
                    # reshape vs and Hvs to matrices
                    vs = [
                        r[0](x)
                        for x, r in zip(jax.tree.leaves(vs), state.affine_reshapers)
                    ]
                    Hvs = [
                        r[0](x)
                        for x, r in zip(jax.tree.leaves(Hvs), state.affine_reshapers)
                    ]

                    # init Qs
                    def init_q(v, h):
                        if precond_init_scale is not None:
                            return precond_init_scale
                        else:
                            return (
                                jnp.sum(v * v.conj()) / jnp.sum(h * h.conj())
                            ) ** 0.25

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
                    Hvs = flat_grads

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

                U, V, d = None, None, None

            return key, U, V, d, Qs

        def _dont_update_precond(key, state, Hvs, vs):
            return key, state.U, state.V, state.d, state.Qs

        if not hessian_based_preconditioning:
            # update cond and vector not passed in, create here
            key, subkey = jax.random.split(key)
            update_preconditioner = jnp.logical_or(
                jax.random.uniform(subkey) < preconditioner_update_probability,
                state.count == 0,
            )
            key, subkey = jax.random.split(key)
            # TODO sharding
            vector = otu.tree_random_like(subkey, params, jax.random.normal)

        key, U, V, d, Qs = jax.lax.cond(
            update_preconditioner,
            _update_precond,
            _dont_update_precond,
            key,
            state,
            Hvp,
            vector,
        )

        updates = flat_grads

        # momentum
        mu = None
        if state.mu is not None:
            if heavyball:
                # heavyball momentum
                f = lambda g, t: g + b1 * t
                mu = jax.tree.map(f, updates, state.mu)
                # optional nesterov
                updates = jax.tree.map(f, updates, mu) if nesterov else mu
            else:
                # ema
                mu = otu.tree_update_moment(updates, state.mu, b1, 1)
                if nesterov:
                    # nesterov for ema with bias correction
                    # https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
                    updates = jax.tree.map(
                        lambda m, g: b1 * m + (1 - b1) * g,
                        otu.tree_bias_correction(
                            mu, b1, safe_int32_increment(count_inc)
                        ),
                        otu.tree_bias_correction(updates, b1, count_inc),
                    )
                else:
                    # bias correction
                    # torch psgd inits mu first step, we init w 0 and bias-correct
                    updates = otu.tree_bias_correction(mu, b1, count_inc)

        # preconditioning
        if preconditioner_type == "xmat":
            updates = _precond_grad_Xmat_math(U, V, updates, precision)
        elif preconditioner_type == "uvd":
            updates = _precond_grad_UVd_math(U, V, d, updates, precision)
        else:  # affine
            updates = [
                _precond_grad_affine_math(Qlr[0], Qlr[1], g, precision)
                for (Qlr, g) in zip(Qs, updates)
            ]

        with jax.ensure_compile_time_eval():
            params_struct = jax.tree.structure(params)
            param_sizes = [x.size for x in jax.tree.leaves(params)]
            param_cumsizes = [x.item() for x in jnp.cumsum(jnp.array(param_sizes))]
            param_shapes = [x.shape for x in jax.tree.leaves(params)]

        # permute and reshape back to original structure
        if preconditioner_type in ["xmat", "uvd"]:
            updates = [
                jnp.reshape(updates[idx - size : idx], s)
                for idx, size, s in zip(param_cumsizes, param_sizes, param_shapes)
            ]
            updates = jax.tree.unflatten(params_struct, updates)
        else:  # affine
            updates = [r[1](u) for u, r in zip(updates, state.affine_reshapers)]
            updates = jax.tree_unflatten(params_struct, updates)

        # optional diagonal optimizer
        # TODO (evanatyourservice) if scale_by_ema
        if feed_into_adam:
            # feed preconditioned gradients into adam
            updates, diag_state = diag_opt.update(updates, state.diag_opt_state)
        elif graft_adam_lr:
            # grafting adam lr onto psgd layer-wise
            diag_updates, diag_state = diag_opt.update(grads, state.diag_opt_state)
            updates = jax.tree.map(
                lambda u, du: u * jnp.linalg.norm(du) / add_eps(jnp.linalg.norm(u)),
                updates,
                diag_updates,
            )
        else:
            # psgd update only
            if gradient_clip:
                updates, _ = clipping.clip_by_global_norm(gradient_clip).update(
                    updates, base.EmptyState
                )
            diag_state = None

        # cast momentum back to mu_dtype
        mu = otu.tree_cast(mu, mu_dtype)

        state = PSGDState(
            count=count_inc,
            mu=mu,
            U=U,
            V=V,
            d=d,
            Qs=Qs,
            affine_reshapers=state.affine_reshapers,
            key=key,
            diag_opt_state=diag_state,
        )

        return updates, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def psgd(
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    preconditioner_type: str = "xmat",
    preconditioner_update_probability: float = 1.0,
    b1: float = 0.0,
    heavyball: bool = False,
    nesterov: bool = False,
    gradient_clip: Optional[float] = None,
    weight_decay: float = 0.0,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    uvd_rank_of_approximation: int = 10,
    affine_max_size_triangular: int = 4096,
    affine_max_skew_triangular: int = 128,
    step_normalizer_order: str = "2nd",
    precond_lr: Union[float, Callable[[int], float]] = 0.1,
    precond_init_scale: Optional[float] = None,
    feed_into_adam: bool = False,
    graft_adam_lr: bool = False,
    adam_b1: float = 0.9,
    adam_b2: float = 0.999,
    adam_norm_grads: Optional[str] = None,
    seed: Optional[PRNGKey] = None,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precision: str = "tensorfloat32",
) -> base.GradientTransformationExtraArgs:
    """
    Implements UVd, XMat, and Affine PSGD from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate for the optimizer.
        preconditioner_type: str, 'xmat', 'uvd', or 'affine'.
        preconditioner_update_probability: float, probability of updating the
            preconditioner.
        b1: float, momentum parameter.
        heavyball: bool, whether to use Heavyball momentum.
        nesterov: bool, whether to use Nesterov momentum.
        gradient_clip: optional float, global gradient norm clipping.
        weight_decay: float, weight decay.
        mask: optional mask for weight decay.
        uvd_rank_of_approximation: int, rank of approximation for uvd preconditioner.
        affine_max_size_triangular: int, max size for affine preconditioner to be
            triangular.
        affine_max_skew_triangular: int, max skew for affine preconditioner to be
            triangular.
        step_normalizer_order: str, '1st' or '2nd'.
        precond_lr: float or callable, learning rate for the preconditioner.
        precond_init_scale: optional float, initial scale for the preconditioner.
        feed_into_adam: bool, whether to feed the preconditioned gradients into
            adam optimizer.
        graft_adam_lr: bool, whether to graft adam step size for updates.
        adam_b1: float, beta1 parameter for the grafting optimizer.
        adam_b2: float, beta2 parameter for the grafting optimizer.
        adam_norm_grads: optional str, 'global', 'layer', or None. Whether to
            normalize gradients to unit norm before grafting optimizer.
        seed: Optional PRNGKey, random seed.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum accumulator.
            Defaults to the same dtype as the parameters.
        precision: str, precision for matmul, 'bfloat16', 'tensorfloat32', 'float32'.

    Returns:
        optax.GradientTransformationExtraArgs
    """
    opt = [
        scale_by_psgd(
            preconditioner_type=preconditioner_type,
            preconditioner_update_probability=preconditioner_update_probability,
            b1=b1,
            heavyball=heavyball,
            nesterov=nesterov,
            gradient_clip=gradient_clip,
            uvd_rank_of_approximation=uvd_rank_of_approximation,
            affine_max_size_triangular=affine_max_size_triangular,
            affine_max_skew_triangular=affine_max_skew_triangular,
            step_normalizer_order=step_normalizer_order,
            precond_lr=precond_lr,
            precond_init_scale=precond_init_scale,
            feed_into_adam=feed_into_adam,
            graft_adam_lr=graft_adam_lr,
            adam_b1=adam_b1,
            adam_b2=adam_b2,
            adam_norm_grads=adam_norm_grads,
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


def _precond_grad_UVd_math(U, V, d, g, precision):
    """
    Preconditioning gradient g with Q = (I + U*V')*diag(d).

    All variables here are either matrices or column vectors.
    """
    # with jax.default_matmul_precision(precision):
    g = _IpUVtmatvec(U, V, d * g)
    g = d * _IpUVtmatvec(V, U, g)
    return g


def _update_precond_Xmat_math_(a, b, v, h, precond_lr, step_normalizer, precision):
    """
    Update preconditioner Q = diag(a) + adiag(b) with (vector, Hessian-vector product) = (v, h).
    """
    with jax.default_matmul_precision(precision):
        Qh = a * h + b * jnp.flip(h, 0)
        aflip, bflip = jnp.flip(a, 0), jnp.flip(b, 0)
        invQtv = (aflip * v - bflip * jnp.flip(v, 0)) / (a * aflip - b * bflip)

        u, v = Qh * Qh, invQtv * invQtv
        nablaA = u - v
        nablaB = Qh * jnp.flip(Qh, 0) - invQtv * jnp.flip(invQtv, 0)
        q, r = jnp.divmod(len(nablaB), 2)
        nablaB = jnp.where(r == 1, nablaB.at[q].set(0), nablaB)

        if step_normalizer == "2nd":
            mu = precond_lr / add_eps(jnp.max(u + v))
        else:
            mu = precond_lr / add_eps(
                jnp.maximum(jnp.max(jnp.abs(nablaA)), jnp.max(jnp.abs(nablaB)))
            )

        a -= mu * (nablaA * a + nablaB * bflip)
        b -= mu * (nablaA * b + nablaB * aflip)

        return a, b


def _precond_grad_Xmat_math(a, b, g, precision):
    """
    Preconditioning gradient g with Q = diag(a) + adiag(b).

    All variables here are either matrices or column vectors.
    """
    # with jax.default_matmul_precision(precision):
    ab = a * b
    return (a * a + jnp.flip(b * b, 0)) * g + (ab + jnp.flip(ab, 0)) * jnp.flip(g, 0)


def _norm_lower_bound(A: jax.Array):
    """
    Returns a cheap lower bound for the spectral norm of A.
    Numerical results on random matrices with a wide range of distributions and sizes suggest,
        norm(A) <= sqrt(2) * norm_lower_bound(A)
    Looks to be a very tight lower bound.
    """
    aa = jnp.real(A * A.conj())
    value0, i = jnp.max(jnp.sum(aa, axis=0), 0)
    value1, j = jnp.max(jnp.sum(aa, axis=1), 0)

    def gt_branch():
        x = A[:, i].conj() @ A
        return jnp.linalg.norm((x / jnp.linalg.norm(x)) @ A.conj().T)

    def le_branch():
        x = A @ A[j].conj()
        normx = jnp.linalg.norm(x)
        return jax.lax.cond(
            normx > 0, lambda: jnp.linalg.norm(A.conj().T @ (x / normx)), lambda: normx
        )

    return jax.lax.cond(value0 > value1, gt_branch, le_branch)


def _triu01(A):
    # it is useful as for a small A, the R of QR decomposition qr(I + A) is about
    # I + triu(A, 0) + triu(A, 1)
    return jnp.triu(A, 0) + jnp.triu(A, 1)


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
                grad1 = _triu01(A @ A.conj().T - BhB)
                grad2 = _triu01(AhA - Bh.conj().T @ Bh)

                if step_normalizer == "2nd":
                    step1 = precond_lr / add_eps(jnp.trace(AhA) + jnp.trace(BhB))
                    step2 = step1
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
                grad1 = _triu01(AAh - BhB)
                grad2 = AAc - BBc

                if step_normalizer == "2nd":
                    step1 = precond_lr / add_eps(jnp.trace(AAh + BhB))
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
                grad2 = _triu01(AhA - BBh)

                if step_normalizer == "2nd":
                    step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
                    step2 = precond_lr / add_eps(jnp.trace(AhA + BBh))
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
            grad2 = _triu01(AhA - BBh)

            if step_normalizer == "2nd":
                step1 = precond_lr / add_eps(jnp.max(jnp.real(AAc + BBc)))
                step2 = precond_lr / add_eps(jnp.trace(AhA + BBh))
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
            grad1 = _triu01(AAh - BhB)
            grad2 = AAc - BBc

            if step_normalizer == "2nd":
                step1 = precond_lr / add_eps(jnp.trace(AAh + BhB))
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
            # TODO sharding
            v = otu.tree_random_like(subkey, dG, jax.random.normal)
            key, subkey = jax.random.split(key)
            return _update_precond_affine_math_(
                subkey, Ql, Qr, v, dG, precond_lr, step_normalizer, precision
            )

        return [Ql, Qr]


def _precond_grad_affine_math(Ql, Qr, grad, precision):
    # with jax.default_matmul_precision(precision):
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
