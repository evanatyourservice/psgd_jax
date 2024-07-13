from typing import Optional

import jax
import numpy as np
from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp

from psgd_jax.image_classification.network_utils import normal_init, flax_scan


class LearnablePositionalEncoding(nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array):
        assert x.ndim == 3, "Input to LearnablePositionalEncoding must be 3D"
        pe = self.param("pe", normal_init, (x.shape[-2], x.shape[-1]))
        return x + jnp.expand_dims(pe, axis=0)


class SwiGLU(nn.Module):
    @nn.compact
    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        gates = nn.silu(gates)
        return x * gates


class TransformerBlock(nn.Module):
    n_heads: int
    dropout_rate: float = 0.0
    is_training: Optional[bool] = None

    @nn.compact
    def __call__(self, a, is_training: Optional[bool] = None):
        is_training = nn.merge_param("is_training", self.is_training, is_training)

        n_tokens, enc_dim = a.shape[-2:]

        # https://arxiv.org/abs/2302.05442 style without parallel blocks
        a2 = nn.LayerNorm(use_bias=False)(a)
        a2 = nn.SelfAttention(
            num_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            kernel_init=normal_init,
            broadcast_dropout=False,
            use_bias=False,
            normalize_qk=True,
        )(a2, deterministic=not is_training)
        b = self.param("att_bias", nn.initializers.zeros, (enc_dim,))
        a2 = a2 + jnp.reshape(b, (1, 1, enc_dim))
        a2 = nn.Dropout(rate=self.dropout_rate)(a2, deterministic=not is_training)
        a = a + a2

        a2 = nn.LayerNorm(use_bias=False)(a)
        a2 = nn.Dense(features=int(enc_dim * 8), kernel_init=normal_init)(a2)
        a2 = SwiGLU()(a2)
        a2 = nn.Dropout(rate=self.dropout_rate)(a2, deterministic=not is_training)
        a2 = nn.Dense(features=enc_dim, kernel_init=normal_init)(a2)
        a2 = nn.Dropout(rate=self.dropout_rate)(a2, deterministic=not is_training)
        a = a + a2

        return a


class Transformer(nn.Module):
    n_layers: int
    enc_dim: int
    n_heads: int
    n_empty_registers: int
    dropout_rate: float = 0.0
    output_dim: int = 1000

    @nn.compact
    def __call__(self, x, is_training: bool):
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)", p1=16, p2=16)
        x = nn.Dense(features=self.enc_dim, kernel_init=normal_init, use_bias=False)(x)
        x *= jnp.sqrt(self.enc_dim).astype(x.dtype)
        x = LearnablePositionalEncoding()(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        cls_token = self.param(
            "cls_token", nn.initializers.zeros_init(), (self.enc_dim,)
        )
        cls_token = jnp.tile(jnp.reshape(cls_token, (1, 1, -1)), (x.shape[0], 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)

        if self.n_empty_registers > 0:
            # https://arxiv.org/abs/2309.16588
            empty_registers = self.param(
                "registers",
                nn.initializers.normal(1 / np.sqrt(self.enc_dim)),
                (self.n_empty_registers, x.shape[-1]),
            )
            empty_registers = jnp.tile(
                jnp.expand_dims(empty_registers, axis=0), (x.shape[0], 1, 1)
            )
            x = jnp.concatenate([x, empty_registers], axis=1)

        x = flax_scan(TransformerBlock, length=self.n_layers, unroll=2)(
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            is_training=is_training,
        )(x)

        x = nn.LayerNorm(use_bias=False)(x[:, 0])  # take cls token

        return nn.Dense(
            features=self.output_dim, kernel_init=normal_init, use_bias=False
        )(x)
