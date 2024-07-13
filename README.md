# PSGD (Preconditioned Stochastic Gradient Descent)

Implementations of [PSGD optimizer](https://github.com/lixilinx/psgd_torch) variants (XMat, UVd, Affine) along with a 
couple experiments for optimizer testing/comparison including temporal XOR problem and image classification.

This repo is in early stages and will be updated soon with more features and documentation. Please feel free to 
open issues with any bug reports, feature requests, or questions! PyPI package coming soon.


Available optimizers from psgd_jax/optimizers/create_optimizer.py:
- PSGD
- shampoo
- CASPR
- sophia
- lion
- adam
- adamw
- sgd
- sign_sgd
- lamb
- adagrad
- adagrad_momentum
- rmsprop
- radam
- sm3
- adabelief
- novograd
- adam3 (https://github.com/wyzjack/AdaM3)
- Any of the above can be wrapped with schedule-free (https://github.com/facebookresearch/schedule_free)


## Installation

Use python 3.10-3.12 (mostly tested on 3.11)

Clone the repository:
```bash
git clone https://github.com/evanatyourservice/psgd_jax.git
```

Install for...

CPU:
```bash
pip install -U pip && pip install psgd_jax/
```

GPU:
```bash
pip install -U pip && pip install -U "jax[cuda12]" && pip install psgd_jax/
```

TPU:
```bash
pip install -U pip && pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && pip install psgd_jax/
```


## Usage

Add your wandb key to environment variables if you want to log results to wandb:
```bash
export WANDB_API_KEY=<your_key>
```

Try the image classification example (set the --wandb_entity flag to your wandb username to log results):
```bash
cd psgd_jax/psgd_jax/image_classification && python3 train.py --dataset cifar10 --model resnet18 --optimizer psgd --psgd_precond_type xmat
```


## Roadmap

The goals of this project include implementing PSGD and its variants in JAX, and also to provide a scalable 
distributed framework with which to test optimizers on larger-scale problems to help further optimization research.

- [x] jax.pmap psgd version
- [x] simple image classification training
- [x] Add other optimizers for comparisons
- [x] Add XOR experiment
- [ ] Add larger vision and LLM experiments
- [ ] Add other PSGD features from torch version
- [ ] Create distributed version of PSGD for jax.jit sharding
- [ ] Add MLCommons algorithmic efficiency benchmarks (https://github.com/mlcommons/algorithmic-efficiency)
- [ ] Allow passing custom optimizers into experiments for easier experimentation


## Resources

Listed from Xi-Lin's repo

1) Preconditioned stochastic gradient descent, [arXiv:1512.04202](https://arxiv.org/abs/1512.04202), 2015. (General ideas of PSGD, preconditioner fitting losses and Kronecker product preconditioners.)
2) Preconditioner on matrix Lie group for SGD, [arXiv:1809.10232](https://arxiv.org/abs/1809.10232), 2018. (Focus on preconditioners with the affine Lie group.)
3) Black box Lie group preconditioners for SGD, [arXiv:2211.04422](https://arxiv.org/abs/2211.04422), 2022. (Mainly about the LRA preconditioner. See [these supplementary materials](https://drive.google.com/file/d/1CTNx1q67_py87jn-0OI-vSLcsM1K7VsM/view) for detailed math derivations.)
4) Stochastic Hessian fittings on Lie groups, [arXiv:2402.11858](https://arxiv.org/abs/2402.11858), 2024. (Some theoretical works on the efficiency of PSGD. The Hessian fitting problem is shown to be strongly convex on set ${\rm GL}(n, \mathbb{R})/R_{\rm polar}$.)
5) Curvature-informed SGD via general purpose Lie-group preconditioners, [arXiv:2402.04553](https://arxiv.org/abs/2402.04553), 2024. (Plenty of benchmark results and analyses for PSGD vs. other optimizers.)


## License

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

2024 Evan Walters, Omead Pooladzandi, Xi-Lin Li


[cc-by-sa]: https://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
