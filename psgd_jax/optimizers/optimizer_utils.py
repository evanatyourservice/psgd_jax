import jax
import optax
from jax import numpy as jnp, vmap
import flax
from optax._src import base


def norm_grads(layerwise: bool = False) -> base.GradientTransformation:
    """Gradient transformation that normalizes gradients to unit norm.

    Args:
        layerwise (bool): Whether to normalize gradients layer-wise, otherwise
            the gradients are normalized globally.

    Returns:
        base.GradientTransformation: The gradient transformation
    """

    def init_fn(params):
        del params
        return base.EmptyState()

    def update_fn(updates, state, params):
        del params

        if layerwise:

            def update_fn(g):
                norm = jnp.linalg.norm(g)
                norm = jnp.where(norm == 0, 1, norm)
                return g / norm

            g_regular = flax.traverse_util.ModelParamTraversal(
                lambda path, param: "scan" not in path
            ).update(update_fn, updates)
            updates = flax.traverse_util.ModelParamTraversal(
                lambda path, param: "scan" in path
            ).update(vmap(update_fn), g_regular)
        else:
            global_norm = optax.global_norm(updates)
            global_norm = jnp.where(global_norm == 0, 1, global_norm)
            updates = jax.tree.map(lambda g: g / global_norm, updates)

        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
