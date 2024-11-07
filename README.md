# PSGD (Preconditioned Stochastic Gradient Descent)

For original PSGD repo, see [psgd_torch](https://github.com/lixilinx/psgd_torch).

For PyTorch Kron version, see [kron_torch](https://github.com/evanatyourservice/kron_torch).

Implementations of [PSGD optimizers](https://github.com/lixilinx/psgd_torch) in JAX (optax-style). 
PSGD is a second-order optimizer originally created by Xi-Lin Li that uses either a hessian-based 
or whitening-based (gg^T) preconditioner and lie groups to improve training convergence, 
generalization, and efficiency. I highly suggest taking a look at Xi-Lin's PSGD repo's readme linked
to above for interesting details on how PSGD works and experiments using PSGD. There are also 
paper resources listed near the bottom of this readme.

### `kron`:

The most versatile and easy-to-use PSGD optimizer is `kron`, which uses a Kronecker-factored 
preconditioner. It has less hyperparameters that need tuning than adam, and can generally act as a 
drop-in replacement.

## Installation

```bash
pip install psgd-jax
```

## Basic Usage (Kron)

Kron schedules the preconditioner update probability by default to start at 1.0 and anneal to 0.03 
at the beginning of training, so training will be slightly slower at the start but will speed up 
by around 4k steps.

For basic usage, use `kron` optimizer like any other optax optimizer:

```python
from psgd_jax.kron import kron

optimizer = kron()
opt_state = optimizer.init(params)

updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

**Basic hyperparameters:**

TLDR: Learning rate and weight decay act similarly to adam's, start with adam-like settings and go 
from there. Maybe use slightly lower learning rate (like /2). There is no b2 or epsilon.

These next 3 settings control whether a dimension's preconditioner is diagonal or triangular. 
For example, for a layer with shape (256, 128), triagular preconditioners would be shapes (256, 256)
and (128, 128), and diagonal preconditioners would be shapes (256,) and (128,). Depending on how 
these settings are chosen, `kron` can balance between memory/speed and effectiveness. Defaults lead
to most precoditioners being triangular except for 1-dimensional layers and very large dimensions.

`max_size_triangular`: Any dimension with size above this value will have a diagonal preconditioner.

`min_ndim_triangular`: Any tensor with less than this number of dims will have all diagonal 
preconditioners. Default is 2, so single-dim layers like bias and scale will use diagonal
preconditioners.

`memory_save_mode`: Can be None, 'one_diag', or 'all_diag'. None is default and lets all 
preconditioners be triangular. 'one_diag' sets the largest or last dim per layer as diagonal 
using `np.argsort(shape)[::-1][0]`. 'all_diag' sets all preconditioners to be diagonal.

`preconditioner_update_probability`: Preconditioner update probability uses a schedule by default 
that works well for most cases. It anneals from 1 to 0.03 at the beginning of training, so training 
will be slightly slower at the start but will speed up by around 4k steps. PSGD generally benefits
from more preconditioner updates at the start of training, but once the preconditioner is learned 
it's okay to do them less often. An easy way to adjust update frequency is to define your own schedule
using the `precond_update_prob_schedule` function in kron.py (just changing the `min_prob` value 
is easiest) and pass this into kron through the `preconditioner_update_probability` hyperparameter.

This is the default schedule defined in the `precond_update_prob_schedule` function at the top of kron.py:

<img src="assets/default_schedule.png" alt="Default Schedule" width="800" style="max-width: 100%; height: auto;" />


**Sharding:**

Kron contains einsums, and in general the first axis of the preconditioner matrices is the 
contracting axis.

If using only FSDP, I usually shard the last axis of each preconditioner matrix and call it good.

However, if using tensor parallelism in addition to FSDP, you may think more carefully about how 
the preconditioners are sharded in train_state. For example, with grads of shape (256, 128) and kron 
preconditioners of shapes (256, 256) and (128, 128), if the grads are sharded as (fsdp, tensor), 
then you may want to shard the (256, 256) preconditioner as (fsdp, tensor) and the (128, 128) 
preconditioner as (tensor, fsdp) so the grads and its preconditioners have similar contracting axes.


**Scanned layers:**

If you are scanning layers in your network, you can also have kron scan over these layers while 
updating and applying the preconditioner. Simply pass in a pytree through `scanned_layers` with 
the same structure as your params with bool values indicating which layers are scanned. PSGD will 
vmap over the first dims of those layers. If you need a more advanced scanning setup, please open 
an issue.

For very large models, the preconditioner update may use too much memory all at once when scanning, 
in which case you can set `lax_map_scanned_layers` to `True` and set `lax_map_batch_size` to a 
reasonable batch size for your setup (`lax.map` scans over batches of vmap, see JAX docs). If 
your net is 32 layers and you're hitting OOM during the optimizer step, you can break the model into
2 or 4 and set `lax_map_batch_size` to 16 or 8 respectively.


## Advanced Usage (XMat, LRA, Affine)

Other forms of PSGD include XMat, LRA, and Affine. PSGD defaults to a gradient 
whitening type preconditioner (gg^T). In this case, you can use PSGD like any other 
optax optimizer:

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

However, PSGD can also be used with a hessian vector product. If values are provided for PSGD's extra 
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

`psgd_jax.kron` - `psgd_jax.xmat` - `psgd_jax.low_rank_approximation` - `psgd_jax.affine`

There are four variants of PSGD: Kron, which uses Kronecker-factored preconditioners for tensors
of any number of dimensions, XMat, which uses an x-shaped global preconditioner, LRA, which uses 
a low-rank approximation global preconditioner, and Affine, which uses kronecker-factored 
preconditioners for matrices.

**Kron:**

Kron uses Kronecker-factored preconditioners for tensors of any number of dimensions. It's very 
versatile, has less hyperparameters that need tuning than adam, and can generally act as a drop-in 
replacement for adam.

**XMat:**

XMat is very simple to use, uses global hessian information for its preconditioner, and has 
memory use of only n_params * 3 (including momentum which is optional, set b1 to 0 to disable).

**LRA:**

Low rank approximation uses a low rank hessian for its preconditioner and can give very strong 
results. It has memory use of n_params * (2 * rank + 1) (n_params * (2 * rank) without momentum).

**Affine:**

Affine does not use global hessian information, but can be powerful nonetheless and possibly use 
less memory than xmat, LRA, or adam. `max_size_triangular` and `max_skew_triangular` determine whether 
a dimension's preconditioner is triangular or diagonal. Affine and Kron are nearly identical for matrices.


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
