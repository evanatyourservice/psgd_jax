import argparse

import jax
from jax import numpy as jnp
import flax


def to_half(pytree):
    """Cast to bfloat16. Ignores booleans, integers."""
    return jax.tree.map(
        lambda t: (
            t.astype(jnp.bfloat16)
            if t.dtype not in [jnp.bool_, jnp.int32, jnp.int64]
            else t
        ),
        pytree,
    )


def to_full(pytree):
    """Cast to float32. Ignores booleans, integers."""
    return jax.tree.map(
        lambda t: (
            t.astype(jnp.float32)
            if t.dtype not in [jnp.bool_, jnp.int32, jnp.int64]
            else t
        ),
        pytree,
    )


def z_loss(logits: jax.Array) -> jax.Array:
    """Compute palm style z-loss from logits.

    Args:
        logits: [batch_size, n_class] float tensor

    Returns:
        [batch_size, 1] z-loss of each example
    """
    log_z = jax.scipy.special.logsumexp(logits, axis=-1)
    return jnp.expand_dims(log_z**2, axis=-1)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "batch"), "batch")


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    return state._replace(batch_stats=cross_replica_mean(state.batch_stats))
