from typing import Any, List, Optional, Sequence, Union, Callable, Tuple
from functools import partial
import string
import numpy as np

import chex
import jax
from jax import numpy as jnp, vmap
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint
import flax.linen as nn
from optax import tree_utils as otu
from optax._src import base, transform
from optax._src.numerics import safe_int32_increment
from optax._src.utils import canonicalize_dtype
from optax._src.combine import chain


def precond_update_prob_schedule(
    max_prob=1.0,
    min_prob=0.03,
    decay=0.001,
    flat_start=500,
):
    """Anneal preconditioner update probability during beginning of training.

    PSGD benefits from more preconditioner updates at the beginning of training,
    but once the preconditioner is learned the update probability can drop low.

    This schedule is an exponential anneal with a flat start. Default settings keep
    update probability at 1.0 for 500 steps then exponentially anneal down to
    `min_prob` by 4000 steps. Default settings work well for most models and
    training regimes.
    """

    def _schedule(n):
        """Exponential anneal with flat start."""
        return jnp.clip(
            max_prob * jnp.exp(-decay * (n - flat_start)), min_prob, max_prob
        )

    return _schedule


def scale_by_kron(
    b1: float = 0.9,
    normalize_grads: bool = False,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 256,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec[str, str]] = None,
    **kwargs,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformationExtraArgs
    """
    mu_dtype = canonicalize_dtype(mu_dtype)
    precond_dtype = canonicalize_dtype(precond_dtype or jnp.float32)
    lax_map = lax_map_scanned_layers
    bs = lax_map_batch_size

    def init_fn(params, return_partition_specs_only=False):
        have_params_sharding = params_sharding is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        params = jax.tree.map(
            lambda x: x.unbox() if isinstance(x, nn.Partitioned) else x,
            params,
            is_leaf=lambda x: isinstance(x, nn.Partitioned),
        )

        # check that there is a PartitionSpec for every param
        if params_sharding is not None:
            assert len(jax.tree.leaves(params_sharding)) == len(
                jax.tree.leaves(params)
            ), "There must be a PartitionSpec for every parameter in PSGD Kron."
        # check that preconditioner sharding length is at least 1
        if preconditioner_sharding is not None:
            assert len(preconditioner_sharding) > 0, (
                "preconditioner_sharding must have length > 0. For example, "
                "PartitionSpec(None) or PartitionSpec('fsdp', None) are valid."
            )

        # extend partition specs
        params_sharding_ = params_sharding
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda p, sh: PartitionSpec(*(sh + (None,) * (len(p.shape) - len(sh)))),
                params,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(
                    preconditioner_sharding[0], None
                )

        # reshape params shaped () to (1,) to make things simpler
        params = jax.tree.map(lambda p: p[None] if len(p.shape) == 0 else p, params)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, params)
        scanned_sizes = jax.tree.map(
            lambda p, s: p.shape[0] if s else 0, params, scanned_layers_
        )

        # momentum
        mu = None
        mu_sharding = params_sharding_
        if b1 > 0 and not return_partition_specs_only:
            mu = jax.tree.map(lambda x: jnp.zeros_like(x, dtype=mu_dtype), params)
            # apply params sharding to momentum buffer
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda p, s: _get_preconditioner_types(
                p.shape[int(s) :],
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
            ),
            params,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        nones = jax.tree.map(lambda _: None, params)
        merged_shapes = jax.tree.map(
            lambda p, s: p.shape[int(s) :], params, scanned_layers_
        )
        if merge_small_dims:
            output = jax.tree.map(
                lambda p, s, dd, sh: _merge_small_dims(
                    p.shape[int(s) :], target_merged_dim_size, dd, sh
                ),
                params,
                scanned_layers_,
                dim_diag,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]

        # partition grads into blocks
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                params,
                merged_shapes,
                dim_diag,
            )
            # we can grab resulting shapes from partitioners
            partitioned_shapes = jax.tree.map(
                lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners
            )

        # initialize preconditioners
        output = jax.tree.map(
            lambda _, ps, dd, sh: list(
                _init_Q_exprs(
                    ps[1:] if partition_grads_into_blocks else ps,
                    preconditioner_init_scale,
                    dd,
                    precond_dtype,
                    existing_Q=True if return_partition_specs_only else None,
                    precond_sharding=preconditioner_sharding_,
                    param_sharding=sh,
                )
            ),
            params,
            partitioned_shapes,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
        )
        if return_partition_specs_only:
            exprs, Qs_sharding_no_leading_dims = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(2)
            ]
        else:
            Qs, exprs, Qs_sharding_no_leading_dims = [
                jax.tree.map(lambda _, x: x[i], params, output) for i in range(3)
            ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                params,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        if not return_partition_specs_only:
            # broadcast Qs for stacks and scans
            def broadcast_qs(_, ps, q, s):
                stack_n = ps[0]
                if partition_grads_into_blocks:
                    # add leading dim for stacked partitions
                    q = jax.tree.map(
                        lambda x: jnp.repeat(jnp.expand_dims(x, 0), stack_n, axis=0), q
                    )
                if s > 0:
                    # add leading dim if we're scanning this layer
                    q = jax.tree.map(
                        lambda d: jnp.repeat(jnp.expand_dims(d, 0), s, axis=0), q
                    )
                return q

            Qs = jax.tree.map(
                broadcast_qs, params, partitioned_shapes, Qs, scanned_sizes
            )
            if have_qs_sharding:
                Qs = _safe_sharding_constraint(Qs, Qs_sharding)

            # Calculate and print sizes for preconditioners and momentum
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

        if return_partition_specs_only:
            return dict(
                count=PartitionSpec(),
                mu=mu_sharding,
                Qs_preconditioners=Qs_sharding,
                update_counter=PartitionSpec(),
            )

        return dict(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=jnp.zeros([], jnp.int32),
        )

    def update_fn(updates: base.Updates, state: dict, params: base.Params = None):
        del params
        count_inc = safe_int32_increment(state["count"])
        key = jax.random.fold_in(jax.random.PRNGKey(42), state["count"])

        have_params_sharding = params_sharding is not None
        have_qs_sharding = have_params_sharding or preconditioner_sharding is not None

        # unbox if flax style partitioned
        boxed_updates, grads_structure = jax.tree.flatten(
            updates,
            is_leaf=lambda g: isinstance(
                g, (chex.Array, nn.Partitioned, jax.ShapeDtypeStruct)
            ),
        )
        flax_partitioned = False
        if isinstance(boxed_updates[0], nn.Partitioned):
            flax_partitioned = True
            updates = [g.unbox() for g in boxed_updates]
            updates = grads_structure.unflatten(updates)

        # extend partition specs
        params_sharding_ = params_sharding
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda g, sh: PartitionSpec(*(sh + (None,) * (len(g.shape) - len(sh)))),
                updates,
                params_sharding_,
            )
        preconditioner_sharding_ = preconditioner_sharding
        if preconditioner_sharding is not None:
            if len(preconditioner_sharding) < 2:
                preconditioner_sharding_ = PartitionSpec(
                    preconditioner_sharding[0], None
                )

        # reshape params shaped () to (1,) to make things simpler
        input_shapes = jax.tree.map(lambda g: g.shape, updates)
        updates = jax.tree.map(lambda g: g[None] if len(g.shape) == 0 else g, updates)
        if have_params_sharding:
            params_sharding_ = jax.tree.map(
                lambda sh: PartitionSpec(None) if sh == PartitionSpec() else sh,
                params_sharding_,
            )

        # scanned layers
        scanned_layers_ = scanned_layers
        if scanned_layers is None:
            scanned_layers_ = jax.tree.map(lambda _: False, updates)

        # update probability can be scheduled
        update_prob_in = preconditioner_update_probability
        if isinstance(preconditioner_update_probability, Callable):
            update_prob_in = preconditioner_update_probability(count_inc)

        # normalize grads
        def norm_grads(g):
            return g / (jnp.linalg.norm(g) + 1e-16)

        if normalize_grads:
            updates = jax.tree.map(norm_grads, updates)

        # momentum
        mu = None
        momentum_updates = updates
        if state["mu"] is not None:
            mu = otu.tree_update_moment(updates, state["mu"], b1, 1)
            if have_params_sharding:
                mu = _safe_sharding_constraint(mu, params_sharding_)
            momentum_updates = otu.tree_bias_correction(mu, b1, count_inc)

        # which preconditioners will be diagonal
        dim_diag = jax.tree.map(
            lambda g, s: _get_preconditioner_types(
                g.shape[int(s) :],
                max_size_triangular,
                min_ndim_triangular,
                memory_save_mode,
            ),
            momentum_updates,
            scanned_layers_,
        )

        # split sharding specs
        scanned_dim_sharding = None
        sharding_without_scan = None
        if have_params_sharding:
            scanned_dim_sharding = jax.tree.map(
                lambda sh, s: PartitionSpec(sh[0]) if s else None,
                params_sharding_,
                scanned_layers_,
            )
            sharding_without_scan = jax.tree.map(
                lambda sh, s: PartitionSpec(*(sh[int(s) :])),
                params_sharding_,
                scanned_layers_,
            )

        # merge small dimensions
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        nones = jax.tree.map(lambda _: None, momentum_updates)
        merged_params_sharding = params_sharding_
        original_shapes = None
        if merge_small_dims:
            original_shapes = jax.tree.map(
                lambda g, s: g.shape[int(s) :], momentum_updates, scanned_layers_
            )
            output = jax.tree.map(
                lambda g, dd, s, sh: _merge_small_dims(
                    g.shape[int(s) :], target_merged_dim_size, dd, sh
                ),
                momentum_updates,
                dim_diag,
                scanned_layers_,
                sharding_without_scan if have_params_sharding else nones,
            )
            merged_shapes, dim_diag, sharding_without_scan = [
                jax.tree.map(lambda _, x: x[i], momentum_updates, output)
                for i in range(3)
            ]
            # reshape
            momentum_updates = jax.tree.map(
                lambda g, s, ns: _map_fn(
                    False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g
                ),
                momentum_updates,
                scanned_layers_,
                merged_shapes,
            )
            if have_params_sharding:
                # scanned dim sharding + new merged sharding
                merged_params_sharding = jax.tree.map(
                    lambda sws, sds: PartitionSpec(
                        *(sds + sws if sds is not None else sws)
                    ),
                    sharding_without_scan,
                    scanned_dim_sharding,
                )
                # constrain sharding
                momentum_updates = _safe_sharding_constraint(
                    momentum_updates, merged_params_sharding
                )

        # partition grads into blocks
        partitioned_sharding = merged_params_sharding
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        partitioners = None
        partitioned_shapes = None
        if partition_grads_into_blocks:
            partitioners = jax.tree.map(
                lambda g, dd, s: BlockPartitioner(g.shape[int(s) :], block_size, dd),
                momentum_updates,
                dim_diag,
                scanned_layers_,
            )
            # layers become tuples each containing layer's partitions
            momentum_updates = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                momentum_updates,
                partitioners,
                scanned_layers_,
            )
            partitioned_shapes = jax.tree.map(
                lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # constrain partitions to same sharding as entire layer
                momentum_updates = jax.tree.map(
                    lambda _, g, mps: jax.tree.map(
                        lambda x: _safe_sharding_constraint(x, mps), g
                    ),
                    dummy_updates_tree,
                    momentum_updates,
                    merged_params_sharding,
                )
            # pad and stack partitions, tuples become arrays with new leading dim
            momentum_updates = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                momentum_updates,
                scanned_layers_,
            )
            if have_params_sharding:
                # add dim to sharding specs for new stacked dim
                partitioned_sharding = jax.tree.map(
                    lambda mps, s: PartitionSpec(*(mps[: int(s)] + (None,) + mps[1:])),
                    merged_params_sharding,
                    scanned_layers_,
                )
                # constrain sharding
                momentum_updates = _safe_sharding_constraint(
                    momentum_updates, partitioned_sharding
                )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)

        # get einsum expressions and Qs sharding
        Qs = state["Qs_preconditioners"]
        Qs_sharding = None
        exprs_and_sharding = jax.tree.map(
            lambda g, dd, sh, nm: _init_Q_exprs(
                g.shape[nm:],
                preconditioner_init_scale,
                dd,
                precond_dtype,
                existing_Q=True,
                precond_sharding=preconditioner_sharding_,
                param_sharding=sh,
            ),
            momentum_updates,
            dim_diag,
            sharding_without_scan if have_params_sharding else nones,
            n_dims_to_map,
        )
        exprs, Qs_sharding_no_leading_dims = [
            jax.tree.map(lambda _, x: x[i], dummy_updates_tree, exprs_and_sharding)
            for i in range(2)
        ]
        Qs_sharding = None
        if have_qs_sharding:
            # add scan and stack dims to Qs sharding
            def add_dims_to_spec(_, qss, sds):
                if partition_grads_into_blocks:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*((None,) + qs)), qss)
                if sds is not None:
                    qss = jax.tree.map(lambda qs: PartitionSpec(*(sds + qs)), qss)
                return qss

            Qs_sharding = jax.tree.map(
                add_dims_to_spec,
                dummy_updates_tree,
                Qs_sharding_no_leading_dims,
                scanned_dim_sharding,
            )

        # maybe update preconditioner
        def update_preconditioner(key, Qs):
            with jax.default_matmul_precision(precond_update_precision):
                # balance preconditioners about every 100 updates
                def balance_Qs(Qs_to_bal):
                    def _balance_Q(Q):
                        norms = jnp.array(
                            [jnp.max(jnp.abs(q)) for q in Q], dtype=jnp.float32
                        )
                        gmean = jnp.exp(jnp.mean(jnp.log(norms)))
                        to_mul = gmean / norms
                        return [q * x.astype(q.dtype) for q, x in zip(Q, to_mul)]

                    return jax.tree.map(
                        lambda _, Q, nm: _map_fn(False, 0, nm, _balance_Q, Q),
                        dummy_updates_tree,
                        Qs_to_bal,
                        n_dims_to_map,
                    )

                key, subkey = jax.random.split(key)
                do_balances = jax.random.uniform(subkey) <= 0.01
                Qs = jax.lax.cond(do_balances, balance_Qs, lambda qs: qs, Qs)
                if have_qs_sharding:
                    Qs = _safe_sharding_constraint(Qs, Qs_sharding)

                # create random vectors
                key, subkey = jax.random.split(key)
                Vs = _tree_random_like(subkey, momentum_updates)
                # apply params sharding to random vectors
                if have_params_sharding:
                    Vs = _safe_sharding_constraint(Vs, partitioned_sharding)

                # damp based on machine precision (f32 probably enough)
                damp_eps = jnp.sqrt(jnp.finfo(jnp.float32).eps)
                grads_in = jax.tree.map(
                    lambda g, v: g
                    + damp_eps.astype(g.dtype) * jnp.mean(jnp.abs(g)) * v,
                    momentum_updates,
                    Vs,
                )

                # form conjB
                conjBs = jax.tree.map(
                    lambda g, Q, v, nm: _map_fn(lax_map, bs, nm, _conjB, Q, g, v),
                    grads_in,
                    Qs,
                    Vs,
                    n_dims_to_map,
                )
                if have_params_sharding:
                    conjBs = _safe_sharding_constraint(conjBs, partitioned_sharding)

                # update Qs and constrain sharding
                new_Qs = jax.tree.map(
                    lambda g, Q, conjb, expr, nm, qss, sh: _map_fn(
                        lax_map,
                        bs,
                        nm,
                        partial(
                            _update_precond,
                            exprs=expr,
                            precond_lr=preconditioner_lr,
                            qs_sharding=qss,
                            params_sharding=sh,
                        ),
                        Q,
                        g,
                        conjb,
                    ),
                    grads_in,
                    Qs,
                    conjBs,
                    exprs,
                    n_dims_to_map,
                    Qs_sharding_no_leading_dims if have_qs_sharding else nones,
                    sharding_without_scan if have_params_sharding else nones,
                )
                if have_qs_sharding:
                    new_Qs = _safe_sharding_constraint(new_Qs, Qs_sharding)

                new_Qs = otu.tree_cast(new_Qs, precond_dtype)
                return new_Qs

        # update preconditioner deterministically
        update_counter_inc = safe_int32_increment(state["update_counter"])
        do_update = update_counter_inc >= 1 / update_prob_in
        update_counter_inc = jnp.where(do_update, 0, update_counter_inc)
        key, subkey = jax.random.split(key)
        Qs = jax.lax.cond(
            do_update, update_preconditioner, lambda _, qs: qs, subkey, Qs
        )
        if have_qs_sharding:
            Qs = _safe_sharding_constraint(Qs, Qs_sharding)

        # precondition gradients
        with jax.default_matmul_precision(precond_grads_precision):
            # precondition with stale Qs
            precond_gs = jax.tree.map(
                lambda g, Q, expr, nm: _map_fn(
                    lax_map, bs, nm, partial(_precond_grad, exprs=expr), Q, g
                ),
                momentum_updates,
                Qs,
                exprs,
                n_dims_to_map,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, partitioned_sharding)

        # unpartition grads
        if partition_grads_into_blocks:
            precond_gs = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                precond_gs,
                scanned_layers_,
                partitioned_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(
                    precond_gs, merged_params_sharding
                )
            precond_gs = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(
                    False, 0, int(s), p_cls.merge_partitions, g
                ),
                dummy_updates_tree,
                precond_gs,
                scanned_layers_,
                partitioners,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(
                    precond_gs, merged_params_sharding
                )

        # un-merge dimensions
        if merge_small_dims:
            precond_gs = jax.tree.map(
                lambda g, s, os: _map_fn(
                    False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g
                ),
                precond_gs,
                scanned_layers_,
                original_shapes,
            )
            if have_params_sharding:
                precond_gs = _safe_sharding_constraint(precond_gs, params_sharding_)

        # return scalars to original shape
        precond_gs = jax.tree.map(
            lambda g, s: jnp.reshape(g, s), precond_gs, input_shapes
        )

        # box preconditioned grads
        if flax_partitioned:
            flat_precond_gs, _ = jax.tree.flatten(precond_gs)
            precond_gs = [
                bu.replace_boxed(g) for bu, g in zip(boxed_updates, flat_precond_gs)
            ]
            precond_gs = grads_structure.unflatten(precond_gs)

        # dtypes and new state
        mu = otu.tree_cast(mu, mu_dtype)
        Qs = otu.tree_cast(Qs, precond_dtype)
        state = dict(
            count=count_inc,
            mu=mu,
            Qs_preconditioners=Qs,
            update_counter=update_counter_inc,
        )

        return precond_gs, state

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def kron(
    learning_rate: Union[float, Callable[[int], float]] = 0.001,
    b1: float = 0.9,
    weight_decay: float = 0.0,
    weight_decay_mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    normalize_grads: bool = False,
    preconditioner_update_probability: Union[
        float, Callable[[int], float]
    ] = precond_update_prob_schedule(),
    max_size_triangular: int = 8192,
    min_ndim_triangular: int = 2,
    memory_save_mode: Optional[str] = None,
    preconditioner_lr: float = 0.1,
    preconditioner_init_scale: float = 1.0,
    mu_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_dtype: Optional[Union[str, jnp.dtype]] = None,
    precond_update_precision: Optional[str] = "tensorfloat32",
    precond_grads_precision: Optional[str] = None,
    scanned_layers: Optional[base.Params] = None,
    lax_map_scanned_layers: bool = False,
    lax_map_batch_size: int = 8,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    partition_grads_into_blocks: bool = False,
    block_size: int = 256,
    params_sharding: Optional[Any] = None,
    preconditioner_sharding: Optional[PartitionSpec[str, str]] = None,
) -> base.GradientTransformationExtraArgs:
    """
    Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        learning_rate: float or callable, learning rate schedule.
        b1: float, momentum parameter. 0.9 or 0.95 are common values.
        weight_decay: float, weight decay coefficient.
        weight_decay_mask: optional pytree same structure as params, or callable
            returning a pytree, that masks weight decay. Weight decay is applied to
            leaves that are True.
        normalize_grads: bool, whether to normalize the incoming gradients to unit
            norm layer-wise. Can help with stability.
        preconditioner_update_probability: float, probability of updating the
            preconditioner. Default anneals from 1.0 to 0.03 by 4000 steps.
        max_size_triangular: int, max size for dim's preconditioner to be triangular.
        min_ndim_triangular: int, minimum number of dimensions a layer needs to have
            triangular preconditioners.
        memory_save_mode: optional str, None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        preconditioner_lr: float, learning rate for preconditioner.
        preconditioner_init_scale: float, scale for preconditioner initialization.
        mu_dtype: optional str or jnp.dtype, dtype of the momentum buffer. Defaults to
            same dtype as the parameters.
        precond_dtype: optional str or jnp.dtype, dtype of the preconditioners. Defaults
            to 'float32'.
        precond_update_precision: str, precision for matmul during preconditioner update,
             'bfloat16', 'tensorfloat32', 'float32'.
        precond_grads_precision: str, precision for matmul during preconditioning grads,
             'bfloat16', 'tensorfloat32', 'float32'.
        scanned_layers: optional base.Params, tree of booleans same structure as
            params indicating scanned dimensions for each layer. PSGD will vmap over
            leading dimension.
        lax_map_scanned_layers: bool, whether to use lax.map for scanned layers
            instead of vmap. Useful to save memory with large models.
        lax_map_batch_size: int, batch size for lax.map, see JAX docs for more info.
        merge_small_dims: bool, whether to merge small dimensions to improve
            preconditioner efficiency.
        target_merged_dim_size: int, target size of merged dimensions.
        partition_grads_into_blocks: bool, whether to partition grads into chunks of
            size `block_size` for efficiency.
        block_size: int, block size to use for partitioning grads.
        params_sharding: pytree same structure as params of jax.sharding.PartitionSpec.
        preconditioner_sharding: `None` or `PartitionSpec(str | None, str | None)`,
            PartitionSpec for preconditioner matrices. `None` infers a strategy
            from params_sharding that matches first preconditioner axis to
            corresponding axis in params. Best practice, though, is to shard the first
            dimension across fsdp-like mesh axis, or the largest, most common axis in
            params. For example, PartitionSpec('fsdp') or PartitionSpec('fsdp', 'tp').

    Returns:
        optax.GradientTransformationExtraArgs
    """
    optimizer = [
        scale_by_kron(
            b1=b1,
            normalize_grads=normalize_grads,
            preconditioner_update_probability=preconditioner_update_probability,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            preconditioner_lr=preconditioner_lr,
            preconditioner_init_scale=preconditioner_init_scale,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            precond_update_precision=precond_update_precision,
            precond_grads_precision=precond_grads_precision,
            scanned_layers=scanned_layers,
            lax_map_scanned_layers=lax_map_scanned_layers,
            lax_map_batch_size=lax_map_batch_size,
            merge_small_dims=merge_small_dims,
            target_merged_dim_size=target_merged_dim_size,
            partition_grads_into_blocks=partition_grads_into_blocks,
            block_size=block_size,
            params_sharding=params_sharding,
            preconditioner_sharding=preconditioner_sharding,
        )
    ]
    if weight_decay > 0.0:
        optimizer.append(transform.add_decayed_weights(weight_decay, weight_decay_mask))
    optimizer.append(transform.scale_by_learning_rate(learning_rate))
    return chain(*optimizer)


def get_opt_state_partition_specs(
    params: base.Params, scale_by_kron_only: bool = False, **kwargs
) -> Any:
    """Get optimized tree of PartitionSpecs for kron optimizer state."""
    # Optimize tree flattening with manual caching
    params_flat, params_struct = jax.tree.flatten(params)

    # Vectorized parameter processing
    if isinstance(params_flat[0], nn.Partitioned):
        params_flat = [p.unbox(p) for p in params_flat]

    if not isinstance(params_flat[0], jax.ShapeDtypeStruct):
        params_flat = [jax.ShapeDtypeStruct(p.shape, p.dtype) for p in params_flat]

    # Reconstruct params efficiently
    params = params_struct.unflatten(params_flat)
    specs = scale_by_kron(**kwargs).init(params, return_partition_specs_only=True)

    if not scale_by_kron_only:
        specs = (specs,)
        if kwargs.get("weight_decay", 0.0) > 0.0:
            specs += (None,)
        specs += (None,)

    return specs


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_size: int, min_ndim: int, mem_save_mode: Optional[str]
) -> List[bool]:
    """Get optimized preconditioner types with vectorized operations."""
    shape_array = np.array(shape)
    dim_count = len(shape)

    if mem_save_mode == "one_diag":
        largest_dim_idx = np.argmax(shape_array)
        dim_diag = np.zeros(dim_count, dtype=bool)
        dim_diag[largest_dim_idx] = True
    elif mem_save_mode == "all_diag":
        dim_diag = np.ones(dim_count, dtype=bool)
    else:
        dim_diag = np.zeros(dim_count, dtype=bool)

    condition_mask = (shape_array == 1) | (shape_array > max_size)
    if len(shape) < min_ndim:
        condition_mask |= True

    return (dim_diag | condition_mask).tolist()


def _init_Q_exprs(
    t_shape,
    scale,
    dim_diag,
    dtype,
    existing_Q=None,
    precond_sharding=None,
    param_sharding=None,
):
    have_qs_sharding = precond_sharding is not None or param_sharding is not None
    letters = string.ascii_lowercase + string.ascii_uppercase
    if len(t_shape) == 0:  # scalar
        Q = (
            [scale * jnp.ones(t_shape, dtype=dtype)]
            if existing_Q is None
            else existing_Q
        )
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"

        sharding_out = [None]
        if have_qs_sharding:
            sharding_out = [PartitionSpec()]
    else:  # tensor
        if len(t_shape) > 13:
            raise ValueError(
                f"Got tensor with dim {len(t_shape.shape)}; Einstein runs out of letters!"
            )
        scale = scale ** (1 / len(t_shape))
        Q = [] if existing_Q is None else existing_Q
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")

        params_specs = param_sharding
        if param_sharding is None:
            params_specs = PartitionSpec(*((None,) * len(t_shape)))
        sharding_out = [None] * len(t_shape)
        if have_qs_sharding:
            sharding_out = [PartitionSpec(None)] * len(t_shape)

        for i, (size, dim_d, dim_sh) in enumerate(zip(t_shape, dim_diag, params_specs)):
            if dim_d:
                # use diagonal matrix as preconditioner for this dim
                if existing_Q is None:
                    q = scale * jnp.ones(size, dtype=dtype)
                    Q.append(q)

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(t_shape))
                    ]
                )
                exprGs.append(piece1 + "," + piece1 + "->" + letters[i + 13])

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                q_sharding = None
                if have_qs_sharding:
                    # infer a so-so sharding scheme from params if nothing specified
                    # (first dim of q will match corresponding dim in params)
                    q_sharding = (
                        precond_sharding
                        if precond_sharding is not None
                        else PartitionSpec(dim_sh, None)
                    )
                    sharding_out[i] = q_sharding
                if existing_Q is None:
                    q = scale * jnp.eye(size, dtype=dtype)
                    if have_qs_sharding:
                        q = _safe_sharding_constraint(q, q_sharding)
                    Q.append(q)

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join(
                    [
                        (letters[i + 13] if j == i else letters[j])
                        for j in range(len(t_shape))
                    ]
                )
                piece2 = "".join(
                    [
                        (letters[i + 26] if j == i else letters[j])
                        for j in range(len(t_shape))
                    ]
                )
                exprGs.append(
                    piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                )

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = (
            ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
        )

    exprGs = tuple(exprGs)
    if existing_Q is not None:
        return (exprA, exprGs, exprP), sharding_out
    return Q, (exprA, exprGs, exprP), sharding_out


def _norm_lower_bound(A: jax.Array):
    """Returns a cheap lower bound for the spectral norm of A.

    Uses einsum for better XLA optimization and removes unnecessary intermediate
    arrays. The function is optimized for hermitian matrices.
    """
    max_abs = jnp.max(jnp.abs(A))

    def calc(A):
        A = A / max_abs
        # Replace A * A and sum with einsum for better optimization
        aa_sum0 = jnp.einsum("ij,ij->j", A, A)
        i = jnp.argmax(aa_sum0)

        # Replace matrix multiplications with einsum
        x = A[:, i]
        x_A = jnp.einsum("i,ij->j", x, A)
        x_norm = jnp.sqrt(jnp.einsum("i,i->", x_A, x_A))
        normalized_x = x_A / x_norm

        # Final matrix multiplication using einsum
        result = jnp.sqrt(jnp.einsum("i,ij,j->", normalized_x, A, normalized_x))
        return max_abs * result

    return jnp.where(max_abs > 0, calc(A), max_abs)


def _solve_triangular_right(X, A):
    """Compute X @ inv(A) optimized for XLA.

    Uses better type handling and simplified vmapping for better performance.
    A triangular solve has roughly the same complexity as a matmul.
    """
    # Handle input dimensionality
    X_ndim = X.ndim
    X = jnp.atleast_2d(X)

    # Promote types once
    dtype_out = jnp.promote_types(A.dtype, X.dtype)
    A = A.astype(dtype_out)
    X = X.astype(dtype_out)

    # Calculate leading dimensions once
    leading_dims = max(0, X.ndim - 2)

    # Create solve function with fixed parameters
    solve_fn = partial(
        jax.lax.linalg.triangular_solve, left_side=False, lower=False, transpose_a=False
    )

    # Apply vmap if needed using a more efficient approach
    if leading_dims > 0:
        for axis in range(leading_dims):
            solve_fn = vmap(solve_fn, in_axes=(None, 0))

    # Solve and reshape result
    solution = solve_fn(A, X)
    return solution[0] if X_ndim < 2 else solution


def _conjB(Q, G, V):
    """Compute conjB with optimized transpositions and solves.

    Uses more efficient array operations and minimizes temporary allocations.
    """
    order = G.ndim
    # Optimize initial transpose by directly creating the new axes order
    conjB = jnp.transpose(V, [*range(1, order), 0])

    for i, q in enumerate(Q):
        if q.ndim < 2:
            # Scalar division case
            conjB = conjB / q
        else:
            # Matrix solve case
            conjB = _solve_triangular_right(conjB, q)

        if i < order - 1:
            # Use moveaxis instead of swapaxes for potentially better optimization
            conjB = jnp.moveaxis(conjB, i, order - 1)

    return conjB


def _update_precond(Q, G, conjB, exprs, precond_lr, qs_sharding, params_sharding):
    """Compute A and update Q with optimized matrix operations."""
    exprA, exprGs, _ = exprs

    # Compute A using the provided einsum expression
    A = jnp.einsum(exprA, *Q, G)

    def _update_single_q(i, q):
        # Compute terms using einsum expressions
        term1 = jnp.einsum(exprGs[i], A, A)
        term2 = jnp.einsum(exprGs[i], conjB, conjB)

        if q.ndim < 2:
            # Scalar case optimization
            max_abs = jnp.max(jnp.abs(term1 + term2))
            update = (precond_lr / _add_tiny(max_abs)) * (term1 - term2) * q
            return q - update
        else:
            # Matrix case optimization
            if qs_sharding is not None:
                sharding = qs_sharding[i]
                # Handle sharding for terms
                if len(sharding) < 2:
                    sharding = PartitionSpec(*((None,) + sharding))
                else:
                    assert len(sharding) == 2
                    sharding = PartitionSpec(*(sharding[1:] + sharding[:1]))
                term1 = _safe_sharding_constraint(term1, sharding)
                term2 = _safe_sharding_constraint(term2, sharding)

            # Optimize the matrix update computation
            norm_factor = _add_tiny(_norm_lower_bound(term1 + term2))
            diff_term = jnp.triu(term1 - term2)
            update = (precond_lr / norm_factor) * jnp.einsum("ij,jk->ik", diff_term, q)
            return q - update

    return [_update_single_q(i, q) for i, q in enumerate(Q)]


def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Map a function along multiple leading axes with improved handling."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        # Optimize map case using partial for better closure handling
        mapped_fn = partial(_map_fn, lax_map, bs, n_maps - 1, fn)
        return jax.lax.map(
            lambda xs: mapped_fn(*xs), xs=args, batch_size=bs if bs > 1 else None
        )
    else:
        # Optimize vmap case
        mapped_fn = vmap(partial(_map_fn, lax_map, bs, n_maps - 1, fn), in_axes=0)
        return mapped_fn(*args)


def _precond_grad(Q, G, exprs):
    """Precondition gradient G with preconditioner Q."""
    exprP = exprs[-1]
    return jnp.einsum(exprP, *Q, *Q, G)


def _safe_sharding_constraint(x, sharding):
    if sharding is None:
        return x
    else:
        return with_sharding_constraint(x, sharding)


def _add_tiny(x):
    return x + jnp.finfo(x.dtype).tiny


def _tree_random_like(
    rng_key: chex.PRNGKey, target_tree: chex.ArrayTree, dtype=None
) -> chex.ArrayTree:
    # adopted from optax
    tree_def = jax.tree.structure(target_tree)
    keys = jax.random.split(rng_key, tree_def.num_leaves)
    keys_tree = jax.tree.unflatten(tree_def, keys)
    return jax.tree.map(
        lambda L, k: jax.random.normal(
            k, L.shape, dtype if dtype is not None else L.dtype
        ),
        target_tree,
        keys_tree,
    )


@chex.dataclass
class SplitInfo:
    """Stores split information for a dimension."""

    axis: int
    indices: np.ndarray
    sizes: np.ndarray


@jax.jit
def _split_tensor(tensor: jax.Array, axis: int, indices: jax.Array) -> List[jax.Array]:
    """JIT-compiled tensor splitting."""
    return jnp.split(tensor, indices_or_sections=indices, axis=axis)


@jax.jit
def _merge_tensors(tensors: Sequence[jax.Array], axis: int) -> jax.Array:
    """JIT-compiled tensor merging."""
    return jnp.concatenate(tensors, axis=axis)


class BlockPartitioner:
    """Partitions a tensor into smaller tensors with optimized operations.

    Modified from distributed_shampoo with performance improvements.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    """

    def __init__(
        self, param_shape: Tuple[int, ...], block_size: int, dim_diag: Sequence[bool]
    ):
        """Initialize the partitioner with optimized preprocessing.

        Args:
            param_shape: Shape of the parameter tensor to partition
            block_size: Size of blocks to partition into
            dim_diag: Boolean flags indicating diagonal dimensions
        """
        assert len(dim_diag) == len(
            param_shape
        ), "dim_diag must match param_shape length"

        self._shape = param_shape
        self._splits: List[SplitInfo] = []
        split_sizes: List[np.ndarray] = []

        # Vectorized split calculation
        for i, (d, is_diag) in enumerate(zip(param_shape, dim_diag)):
            if 0 < block_size < d and not is_diag:
                # Compute splits efficiently using numpy
                nsplit = (d - 1) // block_size
                indices = np.arange(1, nsplit + 1, dtype=np.int32) * block_size

                # Calculate sizes with vectorized operations
                sizes = np.full(nsplit + 1, block_size, dtype=np.int32)
                sizes[-1] = d - indices[-1]

                self._splits.append(SplitInfo(i, indices, sizes))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))

        self._split_sizes = split_sizes

        # Precompute shapes for efficiency
        single_shape = np.array([s[0] for s in split_sizes])
        padded_single_shape = (
            np.ceil(single_shape / block_size).astype(np.int32) * block_size
        )

        # Compute stack size efficiently
        split_lengths = np.array([len(s) for s in split_sizes])
        stack_size = max(1, np.prod(np.maximum(1, split_lengths)))

        self._padded_stacked_shape = tuple([stack_size] + padded_single_shape.tolist())

        # Cache frequently used values
        self._nsplits = len(self._splits)
        self._reverse_splits = list(reversed(self._splits))

    @property
    def split_sizes(self) -> List[np.ndarray]:
        """Return the sizes of splits."""
        return self._split_sizes

    def partition(self, tensor: jax.Array) -> Tuple[jax.Array, ...]:
        """Partition tensor into blocks with optimized operations."""
        assert (
            tensor.shape == self._shape
        ), f"Expected shape {self._shape}, got {tensor.shape}"

        if not self._splits:
            return (tensor,)

        tensors = [tensor]
        for split_info in self._splits:
            # Process each split level
            tensors_local = []
            for t in tensors:
                # Use JIT-compiled split function
                split_results = _split_tensor(t, split_info.axis, split_info.indices)
                tensors_local.extend(split_results)
            tensors = tensors_local

        return tuple(tensors)

    def merge_partitions(self, partitions: Sequence[jax.Array]) -> jax.Array:
        """Merge partitions back to original shape with optimized operations."""
        if not self._splits:
            assert len(partitions) == 1
            return partitions[0]

        parts = list(partitions)
        for split_info in self._reverse_splits:
            n = len(split_info.indices) + 1
            merged = []

            # Process chunks in parallel where possible
            for start_idx in range(0, len(parts), n):
                chunk = parts[start_idx : start_idx + n]
                # Use JIT-compiled merge function
                merged.append(_merge_tensors(chunk, split_info.axis))

            parts = merged

        assert len(parts) == 1, f"Expected 1 partition, got {len(parts)}"
        return parts[0]


def _partitions(lst):
    """Generate all partitions of a list."""
    if not lst:
        yield [[]]
    else:
        for i in range(len(lst)):
            for part in _partitions(lst[i + 1 :]):
                yield [lst[: i + 1]] + part


def _merge_small_dims(
    shape_to_merge: Tuple[int, ...],
    max_dim: int,
    dim_diag: List[bool],
    sharding_to_merge: Optional[Tuple] = None,
) -> Tuple[List[int], List[bool], Optional[Tuple]]:
    """Merge small dimensions with optimized computations."""
    # Handle special cases efficiently
    if not shape_to_merge:
        return [], [True], PartitionSpec() if sharding_to_merge is not None else None

    shape_array = np.array(shape_to_merge)
    if np.all(shape_array == 1):
        return (
            [1],
            [True],
            PartitionSpec(None) if sharding_to_merge is not None else None,
        )

    @partial(jax.jit, static_argnums=(1,))
    def dim2loss(d: float, dim0: float = max_dim) -> float:
        """Vectorized heuristic for dimension loss calculation."""
        too_small = dim0 / 8
        too_large = dim0 * 8

        # Compute losses using vectorized operations
        small_loss = jnp.where(
            d < dim0,
            jnp.log2(dim0 / d)
            + jnp.where(d < too_small, 100 * jnp.log2(too_small / d), 0.0),
            0.0,
        )

        large_loss = jnp.where(
            d >= dim0,
            10 * jnp.log2(d / dim0)
            + jnp.where(d > too_large, 1000 * jnp.log2(d / too_large), 0.0),
            0.0,
        )

        return small_loss + large_loss

    # Pre-compute products for efficiency
    shape_indices = list(range(len(shape_to_merge)))
    best_loss = float("inf")
    best_partition = None

    # Use numpy for partition calculations
    for partition in _partitions(shape_indices):
        if not partition:
            continue

        # Vectorized group product calculation
        group_products = [
            np.prod([shape_to_merge[i] for i in group], dtype=np.float64)
            for group in partition
            if group
        ]

        # Compute total loss efficiently
        total_loss = sum(dim2loss(d) for d in group_products)

        if total_loss < best_loss:
            best_loss = total_loss
            best_partition = [group for group in partition if group]

    # Compute final results efficiently
    merged_shape = [
        np.prod([shape_to_merge[i] for i in group], dtype=np.int64)
        for group in best_partition
    ]

    merged_diag = [all(dim_diag[i] for i in group) for group in best_partition]

    if not sharding_to_merge:
        return merged_shape, merged_diag, None

    # Handle sharding efficiently
    merged_sharding = []
    for group in best_partition:
        group_shardings = [sharding_to_merge[i] for i in group]
        valid_shardings = [s for s in group_shardings if s is not None]

        if len(valid_shardings) > 1:
            merged_sharding.append(tuple(valid_shardings))
        elif valid_shardings:
            merged_sharding.append(valid_shardings[0])
        else:
            merged_sharding.append(None)

    return merged_shape, merged_diag, PartitionSpec(*merged_sharding)


def _pad_and_stack_matrices(array_list: List[jax.Array], block_size: int) -> jax.Array:
    """Efficiently pad and stack matrices with minimized memory operations."""
    # Handle scalar case efficiently
    is_scalar = len(array_list[0].shape) == 0
    if is_scalar:
        array_list = [arr[None] for arr in array_list]

    # Compute shapes and padding efficiently
    shapes = [arr.shape for arr in array_list]
    max_dims = jnp.array(
        [max(shape[i] for shape in shapes) for i in range(len(shapes[0]))]
    )

    # Calculate padded shape using vectorized operations
    padded_shape = jnp.ceil(max_dims / block_size).astype(int) * block_size

    # Optimize padding operation
    def pad_array(arr):
        pad_width = [(0, padded_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        return jnp.pad(arr, pad_width)

    # Use vmap for parallel padding
    padded_arrays = jax.vmap(pad_array)(jnp.array(array_list))
    return padded_arrays


def _unstack_and_unpad_matrices(
    stacked_array: jax.Array,
    original_shapes: Sequence[Tuple[int, ...]],
) -> Tuple[jax.Array, ...]:
    """Efficiently unstack and unpad matrices with vectorized operations.

    Args:
        stacked_array: A stacked array of padded matrices
        original_shapes: Sequence of original shapes before padding

    Returns:
        Tuple of arrays with their original shapes restored
    """
    is_scalar = len(original_shapes[0]) == 0

    # Optimize unstacking using array_split instead of split
    # This avoids creating unnecessary views
    unstacked = jnp.array_split(stacked_array, stacked_array.shape[0], axis=0)

    if is_scalar:
        # Handle scalar case efficiently with vectorized operations
        return tuple(arr.squeeze(axis=0)[0] for arr in unstacked)

    # Create efficient slicing function
    @partial(jax.vmap, in_axes=(0, 0))
    def unpad_array(arr, shape):
        # Create dynamic slices for each dimension
        return jax.lax.slice(arr, (0,) * arr.ndim, shape)

    # Remove the extra dimension from unstacking
    arrays = jnp.concatenate([arr[None, ...] for arr in unstacked], axis=0)

    # Convert shapes to array for vectorized operations
    max_dims = max(len(shape) for shape in original_shapes)
    shapes_array = jnp.array(
        [shape + (1,) * (max_dims - len(shape)) for shape in original_shapes]
    )

    # Apply unpadding to all arrays at once
    unpadded = unpad_array(arrays, shapes_array)

    # Convert back to the expected output format
    return tuple(arr for arr in unpadded)
