import jax
from jax import numpy as jnp, vmap
import flax
from optax._src import base


def hvp(obj_fn, model, vector):
    return jax.jvp(jax.grad(obj_fn), (model,), (vector,))[1]


def norm_grad_layerwise() -> base.GradientTransformation:
    def init_fn(params):
        del params
        return base.EmptyState()

    def update_fn(updates, state, params):
        del params

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
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
