"""
Simple image classification script for training on CIFAR-10, CIFAR-100, ImageNet, and
Imagenette datasets. Supports ResNetTiny, ResNet18, ResNet50, and Vision Transformer
models.

Uses jax.pmap for parallelization across devices, so data is sharded but the model and
optimizer are replicated. With pmap, the params and optimizer states have leading
dimensions with size equal to the number of devices, i.e. [n_devices, ...].
"""

import argparse
import os
import random
from functools import partial
from pprint import pprint
from typing import Optional, Any, NamedTuple
import wandb
import numpy as np

import jax
import jax.numpy as jnp
from jax import pmap
import flax
from flax import core
import optax
from optax.contrib._schedule_free import schedule_free_eval_params
import tensorflow_datasets as tfds
import tensorflow as tf

from psgd_jax.image_classification.imagenet_pipeline import (
    create_split,
    _add_tpu_host_options,
    split_batch,
)
from psgd_jax.image_classification.models.ViT import Transformer
from psgd_jax.image_classification.models.resnet import ResNetTiny, ResNet18, ResNet50
from psgd_jax.image_classification.tf_preprocessing_tools import CifarPreprocess
from psgd_jax.optimizers.create_optimizer import create_optimizer
from psgd_jax.optimizers.psgd import psgd_hvp_helper
from psgd_jax.image_classification.training_utils import (
    to_full,
    to_half,
    z_loss,
    str2bool,
    sync_batch_stats,
)


wandb.require("core")
tf.config.experimental.set_visible_devices([], "GPU")
tf.config.experimental.set_visible_devices([], "TPU")
jax.config.update("jax_default_matmul_precision", jax.lax.Precision.DEFAULT)


parser = argparse.ArgumentParser()


# wandb
parser.add_argument("--log_to_wandb", type=str2bool, default=True)
parser.add_argument("--wandb_entity", type=str, default="")
parser.add_argument("--wandb_project", type=str, default="opt_image_classification")

# Training
parser.add_argument("--global_seed", type=int, default=100)
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100", "imagenet", "imagenette"],
)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--bfloat16", type=str2bool, default=False)
parser.add_argument("--l2_regularization", type=float, default=1e-6)
parser.add_argument(
    "--randomize_l2_reg",
    type=str2bool,
    default=False,
    help="random uniform * l2_reg (PSGD style).",
)
parser.add_argument(
    "--apply_z_loss",
    type=str2bool,
    default=True,
    help="https://arxiv.org/abs/2309.14322",
)

# Model
parser.add_argument(
    "--model_type",
    type=str,
    default="resnet18",
    choices=["resnettiny", "resnet18", "resnet50", "vit"],
)
"""ViT-Base 12 768 12
ViT-Large 24 1024 16
ViT-Huge 32 1280 16"""
parser.add_argument("--n_layers", type=int, default=12, help="ViT only.")
parser.add_argument("--enc_dim", type=int, default=768, help="ViT only.")
parser.add_argument("--n_heads", type=int, default=12, help="ViT only.")
parser.add_argument(
    "--n_empty_registers",
    type=int,
    default=0,
    help="https://arxiv.org/abs/2309.16588, ViT only.",
)
parser.add_argument("--dropout_rate", type=float, default=0.0, help="ViT only.")

# Optimizer
parser.add_argument(
    "--optimizer",
    type=str,
    default="psgd",
    choices=[
        "adam",
        "adamw",
        "lamb",
        "adagrad",
        "adagrad_momentum",
        "sgd",
        "rmsprop",
        "radam",
        "sm3",
        "adabelief",
        "novograd",
        "adam3",
        "lion",
        "shampoo",
        "caspr",
        "psgd",
    ],
)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--min_learning_rate", type=float, default=0.0)
parser.add_argument(
    "--lr_schedule",
    type=str,
    default="linear",
    choices=["cosine", "linear", "flat_w_warmup", "rsqrt", "trapezoidal"],
    help="Trapezoidal uses 1-sqrt cooldown from https://arxiv.org/abs/2405.18392.",
)
parser.add_argument("--warmup_steps", type=int, default=512)
parser.add_argument(
    "--cooldown_steps",
    type=int,
    default=10000,
    help="Used for rsqrt and trapezoidal schedules.",
)
parser.add_argument(
    "--schedule_free",
    type=str2bool,
    default=False,
    help=(
        "https://arxiv.org/abs/2405.15682. Schedule-free removes momentum from "
        "optimizer and replaces with schedule-free momentum/interpolation."
    ),
)
parser.add_argument(
    "--norm_grads",
    type=str,
    default=None,
    help=(
        "Normalize the gradients to unit norm before optimizer either globally or "
        "per layer (None for off). This is not applied before second-order optimizers, "
        "only first-order optimizers and the grafting optimizers in second-order "
        "optimizers."
    ),
    choices=[None, "global", "layer"],
)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--epsilon", type=float, default=1e-8)
parser.add_argument(
    "--nesterov",
    type=str2bool,
    default=False,
    help=(
        "Modifies sgd and adam[w]. Also modifies momentum in psgd, shampoo, "
        "caspr, and their grafting optimizers."
    ),
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.1,
    help="Adamw style weight decay, 0 to turn off.",
)
parser.add_argument(
    "--gradient_clip",
    type=float,
    default=1.0,
    help="Clips gradient by global norm, None to turn off.",
)
parser.add_argument(
    "--graft",
    type=str2bool,
    default=False,
    help=(
        "Whether to graft from adam in psgd, shampoo, caspr. "
        "If False, shampoo/caspr grafts from SGD instead of None."
    ),
)
parser.add_argument(
    "--shampoo_precondition_every_n", type=int, default=32, help="Shampoo only."
)
parser.add_argument(
    "--shampoo_precond_block_size", type=int, default=128, help="Shampoo only."
)
parser.add_argument(
    "--psgd_precond_type", type=str, default="xmat", choices=["xmat", "uvd", "affine"]
)
parser.add_argument("--psgd_use_hessian", type=str2bool, default=True)
parser.add_argument(
    "--psgd_feed_into_adam",
    type=str2bool,
    default=False,
    help="Feeds preconditioned gradient into an adam optimizer for psgd.",
)
parser.add_argument(
    "--psgd_heavyball",
    type=str2bool,
    default=False,
    help="Use heavyball momentum in psgd, otherwise bias-corrected ema (adam-like).",
)
parser.add_argument("--psgd_rank", type=int, default=4, help="For psgd LRA/UVd.")
parser.add_argument("--psgd_update_probability", type=float, default=0.1)
parser.add_argument("--psgd_precond_lr", type=float, default=0.1)
parser.add_argument("--psgd_precond_init_scale", type=float, default=None)
parser.add_argument(
    "--mu_dtype",
    type=str,
    default="float32",
    help="Dtype for momentum buffers.",
    choices=["float32", "bfloat16"],
)


class TrainState(NamedTuple):
    """Train state.

    Everything in JAX is functional, so we can't use a class and keep mutable states
    in 'self', we instead pass state in and out of functions explicitly, usually using
    a dataclass, flax.struct.dataclass, or just a NamedTuple like this.
    """

    step: jax.Array
    params: core.FrozenDict[str, Any]
    batch_stats: Optional[core.FrozenDict[str, Any]]
    opt_state: optax.OptState


def main(
    log_to_wandb: bool,
    wandb_entity: str,
    wandb_project: str,
    global_seed: int,
    dataset: str,
    batch_size: int,
    n_epochs: int,
    bfloat16: bool,
    l2_regularization: float,
    randomize_l2_reg: bool,
    apply_z_loss: bool,
    model_type: str,
    n_layers: int,
    enc_dim: int,
    n_heads: int,
    n_empty_registers: int,
    dropout_rate: float,
    # optimizer
    user_defined_optimizer: Optional[optax.GradientTransformation] = None,
    optimizer: str = "psgd",
    learning_rate: float = 0.01,
    min_learning_rate: float = 0.0,
    lr_schedule: str = "linear",
    warmup_steps: int = 512,
    cooldown_steps: int = 10000,
    schedule_free: bool = False,
    norm_grads: str = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    nesterov: bool = False,
    weight_decay: float = 0.1,
    gradient_clip: float = 1.0,
    graft: bool = False,
    shampoo_precondition_every_n: int = 32,
    shampoo_precond_block_size: int = 128,
    psgd_precond_type: str = "xmat",
    psgd_use_hessian: bool = True,
    psgd_feed_into_adam: bool = False,
    psgd_heavyball: bool = False,
    psgd_rank: int = 4,
    psgd_update_probability: float = 0.1,
    psgd_precond_lr: float = 0.1,
    psgd_precond_init_scale: Optional[float] = None,
    mu_dtype: str = "float32",
):
    # take a look at the devices and see if we're on CPU, GPU, or TPU
    devices = jax.local_devices()
    print(f"JAX Devices: {devices}")
    platform = devices[0].platform

    # set seeds
    rng = jax.random.PRNGKey(global_seed)  # jax uses explicit seed handling
    tf.random.set_seed(global_seed)
    np.random.seed(global_seed)
    random.seed(global_seed)

    # dataset
    if dataset == "imagenette":
        dataset_name = "imagenette/full-size-v2"
        n_classes = 10
    elif dataset == "imagenet":
        dataset_name = "imagenet2012"
        n_classes = 1000
    elif dataset == "cifar10":
        dataset_name = "cifar10"
        n_classes = 10
    elif dataset == "cifar100":
        dataset_name = "cifar100"
        n_classes = 100
    else:
        raise ValueError(
            f"Unknown dataset: {dataset}, must be one of "
            f"'imagenet', 'imagenette', 'cifar10', 'cifar100'"
        )

    # wandb setup
    if log_to_wandb:
        if not wandb_entity:
            print(
                "WARNING: No wandb entity provided, running without logging to wandb."
            )
            log_to_wandb = False
        else:
            locals_ = locals()
            if not os.environ["WANDB_API_KEY"]:
                raise ValueError(
                    "No WANDB_API_KEY found in environment variables, see readme "
                    "for instructions on setting wandb API key."
                )
            wandb.login(key=os.environ["WANDB_API_KEY"])
            opt_name = f"{optimizer}_SF" if schedule_free else optimizer
            name = (
                f"{opt_name}_{model_type}_{dataset}_bs{batch_size}_ep{n_epochs}_"
                f"lr{learning_rate}_wd{weight_decay}"
            )
            wandb.init(
                entity=wandb_entity, project=wandb_project, name=name, config=locals_
            )

    def get_datasets():
        """Download and prepare tensorflow datasets."""
        ds_builder = tfds.builder(dataset_name)
        print("Downloading and preparing dataset.", flush=True)
        ds_builder.download_and_prepare()

        if dataset in ["imagenette", "imagenet"]:
            print("Using imagenet style data pipeline.")
            train_ds = create_split(
                ds_builder,
                batch_size,
                train=True,
                platform=platform,
                dtype=tf.float32,
                shuffle_buffer_size=250 if dataset == "imagenette" else 2000,
                prefetch=4,
            )
            test_ds = create_split(
                ds_builder,
                batch_size,
                train=False,
                platform=platform,
                dtype=tf.float32,
                shuffle_buffer_size=250 if dataset == "imagenette" else 2000,
                prefetch=4,
            )
        else:
            print("Using cifar style data pipeline.")
            train_ds = ds_builder.as_dataset(split="train", shuffle_files=True)
            test_ds = ds_builder.as_dataset(split="test", shuffle_files=True)

            if platform == "tpu":
                train_ds = _add_tpu_host_options(train_ds)
                test_ds = _add_tpu_host_options(test_ds)

            train_ds = (
                train_ds.repeat()
                .shuffle(2000)
                .map(
                    CifarPreprocess(True, dataset),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .batch(
                    batch_size,
                    drop_remainder=True,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(
                    split_batch,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .prefetch(8)
                .as_numpy_iterator()
            )
            test_ds = (
                test_ds.repeat()
                .shuffle(2000)
                .map(
                    CifarPreprocess(False, dataset),
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .batch(
                    batch_size,
                    drop_remainder=True,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .map(
                    split_batch,
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False,
                )
                .prefetch(4)
                .as_numpy_iterator()
            )

        if platform in ["gpu", "tpu"]:
            # prefetch 1 (instead of 0) on TPU in case snag with
            # JAX async dispatch in train loop
            train_ds = flax.jax_utils.prefetch_to_device(
                train_ds, 2 if platform == "gpu" else 1
            )
            test_ds = flax.jax_utils.prefetch_to_device(
                test_ds, 2 if platform == "gpu" else 1
            )
        return train_ds, test_ds

    # download datasets and create data iterators
    train_ds, test_ds = get_datasets()
    if dataset == "imagenette":
        train_ds_size = 9469
    elif dataset == "imagenet":
        train_ds_size = 1281167
    else:
        train_ds_size = 50000
    steps_per_epoch = train_ds_size // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    n_steps = steps_per_epoch * n_epochs
    print(f"Total train steps: {n_steps}")
    print(f"Total epochs: {n_epochs}")

    # create optimizer (and lr function for logging)
    if user_defined_optimizer is not None:
        print("Using user-defined optimizer.")
        if optimizer == "psgd" and psgd_use_hessian:
            print(
                "WARNING: psgd_use_hessian is True while using user defined "
                "optimizer. If not using PSGD, consider setting psgd_use_hessian "
                "to False to avoid unnecessary calculations."
            )
        tx = user_defined_optimizer
        lr_fn = None
    else:
        tx, lr_fn = create_optimizer(
            optimizer=optimizer,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            norm_grads=norm_grads,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            nesterov=nesterov,
            weight_decay=weight_decay,
            lr_schedule=lr_schedule,
            schedule_free=schedule_free,
            warmup_steps=warmup_steps,
            total_train_steps=n_steps,
            gradient_clip=gradient_clip,
            pmap_axis_name="batch",
            graft=graft,
            shampoo_precond_every_n=shampoo_precondition_every_n,
            shampoo_precond_block_size=shampoo_precond_block_size,
            psgd_precond_type=psgd_precond_type,
            psgd_update_prob=psgd_update_probability,
            psgd_rank=psgd_rank,
            psgd_heavyball=psgd_heavyball,
            psgd_feed_into_adam=psgd_feed_into_adam,
            psgd_precond_lr=psgd_precond_lr,
            psgd_precond_init_scale=psgd_precond_init_scale,
            cooldown_steps=cooldown_steps,
            mu_dtype=mu_dtype,
        )

    # create model
    if "resnet" in model_type:
        print("Creating ResNet model.")
        if dataset in ["imagenet", "imagenette"]:
            fl_kernel_size, fl_stride, fl_pool = 7, 2, True
        else:
            fl_kernel_size, fl_stride, fl_pool = 3, 1, False
        if model_type == "resnettiny":
            rn_class = ResNetTiny
        elif model_type == "resnet18":
            rn_class = ResNet18
        elif model_type == "resnet50":
            rn_class = ResNet50
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model = rn_class(
            num_classes=n_classes,
            first_layer_kernel_size=fl_kernel_size,
            first_layer_stride=fl_stride,
            first_layer_max_pool=fl_pool,
        )
    elif model_type == "vit":
        print("Creating ViT model.")
        model = Transformer(
            n_layers=n_layers,
            enc_dim=enc_dim,
            n_heads=n_heads,
            n_empty_registers=n_empty_registers,
            dropout_rate=dropout_rate,
            output_dim=n_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    def loss_fn(params, batch_stats, rng, images, labels):
        """Computes loss for a single batch.

        Args:
            params: dict, model parameters.
            batch_stats: dict, batch statistics.
            rng: PRNGKey
            images: jnp.ndarray, batch of images.
            labels: jnp.ndarray, batch of labels.

        Returns:
            loss: float, mean loss.
            aux: tuple of new model state and logits.
        """
        # optionally carry out calculations in bfloat16
        if bfloat16:
            params = to_half(params)
            images = to_half(images)

        rng, subkey = jax.random.split(rng)
        if "resnet" in model_type:
            logits, new_model_state = model.apply(
                {"params": params, "batch_stats": batch_stats},
                images,
                rngs={"dropout": subkey},
                mutable=["batch_stats"],
                is_training=True,
            )
        else:
            logits = model.apply(
                {"params": params}, images, rngs={"dropout": subkey}, is_training=True
            )
            new_model_state = {"batch_stats": batch_stats}
        # back to float32 for loss calculation
        logits = to_full(logits)
        one_hot = jax.nn.one_hot(labels, n_classes)
        loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)

        # z-loss, https://arxiv.org/pdf/2309.14322
        if apply_z_loss:
            loss += z_loss(logits) * 1e-4

        if l2_regularization > 0:
            # randomized l2 regularization psgd style
            if randomize_l2_reg:
                rng, subkey = jax.random.split(rng)
                multiplier = jax.random.uniform(subkey)
            else:
                multiplier = 1.0
            loss += multiplier * l2_regularization * optax.global_norm(params) ** 2

        return loss.mean(), (new_model_state, logits)

    @partial(pmap, axis_name="batch", donate_argnums=(1,))
    def train_step(rng, state, batch):
        """Applies an update to parameters and returns new state.

        Args:
            rng: PRNGKey, random number generator.
            state: TrainState, current state.
            batch: dict, batch of data.

        Returns:
            rng: PRNGKey, random number generator.
            new_state: TrainState, new state.
            loss: float, mean loss.
            accuracy: float, mean accuracy.
            grad_norm: float, mean gradient
        """
        rng, subkey = jax.random.split(rng)
        if optimizer == "psgd" and psgd_use_hessian:
            # use helper function to calc hvp and pass into psgd
            loss_out, grads, hvp, vector, update_precond = psgd_hvp_helper(
                subkey,
                loss_fn,
                state.params,
                loss_fn_extra_args=(
                    state.batch_stats,
                    subkey,
                    batch["image"],
                    batch["label"],
                ),
                has_aux=True,
                pmap_axis_name="batch",
                preconditioner_update_probability=psgd_update_probability,
            )

            loss, aux = loss_out

            updates, new_opt_state = tx.update(
                grads,
                state.opt_state,
                state.params,
                Hvp=hvp,
                vector=vector,
                update_preconditioner=update_precond,
            )
        else:
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, state.batch_stats, subkey, batch["image"], batch["label"]
            )
            # mean gradients across devices
            grads = jax.lax.pmean(grads, axis_name="batch")

            updates, new_opt_state = tx.update(grads, state.opt_state, state.params)

        new_model_state, logits = aux
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])

        # apply updates to model params
        new_params = optax.apply_updates(state.params, updates)

        # create new state
        new_state = state._replace(
            step=state.step + 1,
            params=new_params,
            batch_stats=new_model_state["batch_stats"],
            opt_state=new_opt_state,
        )

        # mean stats across devices
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        # grad norm metric
        grad_norm = optax.global_norm(grads)

        return rng, new_state, loss, accuracy, grad_norm

    @partial(pmap, axis_name="batch")
    def inference(state, batch):
        """Computes gradients, loss and accuracy for a single batch."""

        variables = {
            "params": (
                schedule_free_eval_params(state.opt_state, state.params)
                if schedule_free
                else state.params
            )
        }
        if "resnet" in model_type:
            variables["batch_stats"] = state.batch_stats
        images, labels = batch["image"], batch["label"]

        logits = model.apply(variables, images, is_training=False)
        one_hot = jax.nn.one_hot(labels, n_classes)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch["label"])

        # mean stats across devices
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        return loss, accuracy

    @pmap
    def create_train_state(rng):
        """Creates initial `TrainState`.

        Decorated with `pmap` so train state is automatically replicated across devices.
        """
        image_size = 224 if dataset in ["imagenet", "imagenette"] else 32
        dummy_image = jnp.ones([1, image_size, image_size, 3])  # batch size 1 for init
        variables = model.init(rng, dummy_image, is_training=False)

        opt_state = tx.init(variables["params"])

        print("Network params:")
        pprint(
            jax.tree.map(lambda x: x.shape, variables["params"]),
            width=120,
            compact=True,
        )

        # create initial train state
        state = TrainState(
            step=jnp.zeros([], jnp.int32),
            params=variables["params"],
            batch_stats=variables.get("batch_stats"),
            opt_state=opt_state,
        )
        return state

    print("Creating train state.")
    state = create_train_state(jax.device_put_replicated(rng, jax.local_devices()))
    rng = jax.random.split(rng, len(jax.local_devices()))  # split rng for pmap
    print("Train state created.")

    # print number of parameters
    total_params = jax.tree.map(
        lambda x: jnp.prod(jnp.array(x.shape)),
        jax.tree.map(lambda x: x[0], state.params),
    )
    total_params = sum(jax.tree.leaves(total_params))
    print(f"Total number of parameters: {total_params}")

    # test inference real quick
    _ = inference(state, next(test_ds))

    print(f"Training for {n_steps} steps...")
    all_test_losses = []
    all_test_accs = []
    train_losses = []
    train_accuracies = []
    grad_norms = []
    for e in range(n_epochs):
        for i in range(steps_per_epoch):
            rng, state, train_loss, train_accuracy, grad_norm = train_step(
                rng, state, next(train_ds)
            )
            train_losses.append(train_loss[0].item())
            train_accuracies.append(train_accuracy[0].item())
            grad_norms.append(grad_norm[0].item())

            if state.step[0].item() % 100 == 0 or (
                i == steps_per_epoch - 1 and e == n_epochs - 1
            ):
                # sync batch stats before evaluation
                if state.batch_stats is not None:
                    state = sync_batch_stats(state)
                test_losses = []
                test_accuracies = []
                for j in range(10):
                    test_loss, test_accuracy = inference(state, next(test_ds))
                    test_losses.append(test_loss[0].item())
                    test_accuracies.append(test_accuracy[0].item())
                mean_test_loss = np.mean(test_losses)
                all_test_losses.append(mean_test_loss)
                mean_test_acc = np.mean(test_accuracies)
                all_test_accs.append(mean_test_acc)
                mean_grad_norm = np.mean(grad_norms)
                single_params = jax.tree.map(lambda x: x[0], state.params)
                params_norm = optax.global_norm(single_params)

                to_log = {
                    "train_loss": np.mean(train_losses),
                    "train_accuracy": np.mean(train_accuracies) * 100,
                    "test_loss": mean_test_loss,
                    "test_accuracy": mean_test_acc * 100,
                    "grad_norm": mean_grad_norm,
                    "params_norm": params_norm,
                }
                if lr_fn is not None:
                    to_log["lr"] = lr_fn(state.step[0].item())
                if log_to_wandb:
                    wandb.log(to_log, step=state.step[0].item())
                    wandb.summary["min_loss"] = min(all_test_losses)
                    wandb.summary["max_accuracy"] = max(all_test_accs) * 100
                if state.step[0].item() % 1000 == 0 or (
                    i == steps_per_epoch - 1 and e == n_epochs - 1
                ):
                    print(
                        "step:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, "
                        "test_accuracy: %.2f, grad_norm: %.2f, params_norm: %.2f"
                        % (
                            state.step[0].item(),
                            to_log["train_loss"],
                            to_log["train_accuracy"],
                            to_log["test_loss"],
                            to_log["test_accuracy"],
                            to_log["grad_norm"],
                            to_log["params_norm"],
                        )
                    )

                train_losses = []
                train_accuracies = []

    print(f"Min loss: {min(all_test_losses):.4f}")
    print(f"Max accuracy: {max(all_test_accs) * 100:.2f}%")

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = vars(parser.parse_args())

    tiny_test = False
    if tiny_test:
        args["n_epochs"] = 4
        args["batch_size"] = 64
        args["dataset"] = "cifar10"
        args["model_type"] = "vit"
        args["n_layers"] = 3
        args["enc_dim"] = 32
        args["n_heads"] = 2
        args["optimizer"] = "psgd"
        args["learning_rate"] = 0.003
        args["lr_schedule"] = "linear"
        args["warmup_steps"] = 100
        args["weight_decay"] = 0.001
        args["gradient_clip"] = 1.0
        args["psgd_precond_type"] = "xmat"
        args["psgd_use_hessian"] = True
        args["psgd_feed_into_adam"] = True

    main(**args)
