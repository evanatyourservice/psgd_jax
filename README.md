# PSGD (Preconditioned Stochastic Gradient Descent)

Implementation of [PSGD optimizer](https://github.com/lixilinx/psgd_torch) in JAX (optax-style). 
PSGD is a second-order optimizer originally created by Xi-Lin Li that uses a hessian-based 
preconditioner and lie groups to improve convergence, generalization, and efficiency.


## Installation

```bash
pip install psgd-jax
```

## Usage

PSGD defaults to a gradient whitening type preconditioner (gg^T). In this case, you can use PSGD 
like any other optax optimizer:

```python
import jax
import jax.numpy as jnp
import optax
from psgd_jax.xmat import xmat  # or low_rank_approximation, affine


def loss_fn(params, x):
    return jnp.sum((params - x) ** 2)


params = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([0.0, 0.0, 0.0])

# make optimizer and init state
opt = xmat(
    learning_rate=1.0,
    b1=0.0,
    preconditioner_update_probability=1.0,  # preconditioner update frequency
)
opt_state = opt.init(params)


def step(params, x, opt_state):
    loss_val, grad = jax.value_and_grad(loss_fn)(params, x)
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val


while True:
    params, opt_state, loss_val = step(params, x, opt_state)
    print(loss_val)
    if loss_val < 1e-4:
        print("yay")
        break

# Expected output:
# 14.0
# 5.1563816
# 1.7376599
# 0.6118454
# 0.18457186
# 0.056664664
# 0.014270116
# 0.0027846962
# 0.00018843572
# 4.3836744e-06
# yay
```

However, PSGD is best used with a hessian vector product. If values are provided for PSGD's extra 
update function arguments `Hvp`, `vector`, and `update_preconditioner`, PSGD automatically 
uses hessian-based preconditioning. `Hvp` is the hessian vector product, `vector` is the random 
vector used to calculate the hessian vector product, and `update_preconditioner` is a boolean 
that tells PSGD whether we're updating the preconditioner this step (passed in real hvp and 
vector) or not (passed in dummy hvp and vector).

The `hessian_helper` function can help with this and generally replace `jax.value_and_grad`:

```python
import jax
import jax.numpy as jnp
import optax
from psgd_jax.xmat import xmat  # or low_rank_approximation, affine
from psgd_jax import hessian_helper


def loss_fn(params, x):
    return jnp.sum((params - x) ** 2)


params = jnp.array([1.0, 2.0, 3.0])
x = jnp.array([0.0, 0.0, 0.0])

# make optimizer and init state
# no need to set 'preconditioner_update_probability' here, it's handled by hessian_helper
opt = xmat(
    learning_rate=1.0,
    b1=0.0,
)
opt_state = opt.init(params)


def step(key, params, x, opt_state):
    # replace jax.value_and_grad with the hessian_helper:
    key, subkey = jax.random.split(key)
    loss_fn_out, grad, hvp, vector, update_precond = hessian_helper(
        subkey,
        loss_fn,
        params,
        loss_fn_extra_args=(x,),
        has_aux=False,
        preconditioner_update_probability=1.0,  # update frequency handled in hessian_helper
    )
    loss_val = loss_fn_out

    # Pass hvp, random vector, and whether we're updating the preconditioner 
    # this step into the update function. PSGD will automatically switch to 
    # hessian-based preconditioning when these are provided.
    updates, opt_state = opt.update(
        grad,
        opt_state,
        Hvp=hvp,
        vector=vector,
        update_preconditioner=update_precond
    )

    params = optax.apply_updates(params, updates)
    return key, params, opt_state, loss_val


key = jax.random.PRNGKey(0)
while True:
    key, params, opt_state, loss_val = step(key, params, x, opt_state)
    print(loss_val)
    if loss_val < 1e-4:
        print("yay")
        break

# Expected output:
# 14.0
# 7.460699e-14
# yay
```

If `preconditioner_update_probability` is lowered, time is saved by calculating the hessian less 
often, but convergence could be slower.

## PSGD variants

`psgd_jax.xmat` `psgd_jax.low_rank_approximation` `psgd_jax.affine`

There are three variants of PSGD: XMat, which uses an x-shaped global preconditioner, LRA, which 
uses a low-rank approximation global preconditioner, and Affine, which uses block diagonal or 
diagonal preconditioners.

**XMat:**

XMat is very simple to use, uses global hessian information for its preconditioner, and has 
memory use of only n_params * 3 (including momentum which is optional, set b1 to 0 to disable).

**LRA:**

Low rank approximation uses a low rank hessian for its preconditioner and can give very strong 
results. It has memory use of n_params * (2 * rank + 1) (n_params * (2 * rank) without momentum).

**Affine:**

Affine does not use global hessian information, but can be powerful nonetheless and possibly use 
less memory than xmat or LRA. `max_size_triangular` and `max_skew_triangular` determine whether 
a dimension's preconditioner is either block diagonal or diagonal.

For example, if `max_size_triangular` is set to 512 and a layer's is shape (1024, 16, 64), the 
preconditioner shapes will be [diag, block_diag, block_diag] or [(1024,), (16, 16), (64, 64)] 
because 1024 > 512.

If `max_skew_triangular` is set to 32 and a layer's shape is (1024, 3), 
the preconditioner shapes will be [diag, block_diag] or [(1024,), (3, 3)] because 1024/3 is 
greater than 32.

If `max_size_triangular` and `max_skew_triangular` are set to 0, the affine preconditioners
will be entirely diagonal and would use less memory than adam even with momentum.


## Notes on sharding:

For now PSGD does not explicitly handle any sharding, so intermediates would be handled naively by 
JAX based on how users define in and out shardings. Our goal is to improve preconditioner shapes 
and explicitly handle sharding for PSGD, especially for XMat and LRA, to make it more efficient
in distributed settings.

**Optimizer state shapes:**

Momentum is always same shape as params.

Affine might be the most out-of-the-box sharding friendly as it uses block diagonal or diagonal 
preconditioners. For example, if a layer has shape (1024, 16, 64) and `max_size_triangular` is set 
to 512, the preconditioner shapes will be `[(1024,), (16, 16), (64, 64)]`, which could be sharded as 
the user sees fit.

XMat's preconditioners `a` and `b` are both of shape `(n_params,)`. If n_params is odd, or not divisible 
by number of devices, dummy params could be added before optimizer init and update.

LRA's preconditioner shapes are `U=(n_params, rank)`, `V=(n_params, rank)`, and `d=(n_params, 1)`.


## Resources

PSGD papers and resources listed from Xi-Lin's repo

1) Xi-Lin Li. Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Xi-Lin Li. Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Xi-Lin Li. Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. See [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Xi-Lin Li. Stochastic Hessian fittings on Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD. The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Omead Pooladzandi, Xi-Lin Li. Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)


## License

[![CC BY 4.0][cc-by-image]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

2024 Evan Walters, Omead Pooladzandi, Xi-Lin Li


[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://licensebuttons.net/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
