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
parser.add_argument(
    "--l2_regularization",
    type=float,
    default=0.0,
    help="Psgd-style with random strength per param.",
)
parser.add_argument(
    "--apply_z_loss",
    type=str2bool,
    default=True,
    help="https://arxiv.org/abs/2309.14322",
)

# Model
"""
ViT-Base 12 768 12
ViT-Large 24 1024 16
ViT-Huge 32 1280 16
"""
parser.add_argument(
    "--model_type",
    type=str,
    default="resnet18",
    choices=["resnettiny", "resnet18", "resnet50", "vit"],
)
parser.add_argument("--n_layers", type=int, default=12, help="ViT only.")
parser.add_argument("--enc_dim", type=int, default=768, help="ViT only.")
parser.add_argument("--n_heads", type=int, default=12, help="ViT only.")
parser.add_argument(
    "--n_empty_registers",
    type=int,
    default=0,
    help="https://arxiv.org/abs/2309.16588, ViT only.",
)
parser.add_argument("--dropout_rate", type=float, default=0.05, help="ViT only.")

# Learning rate
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--min_learning_rate", type=float, default=0.0)
parser.add_argument(
    "--lr_schedule",
    type=str,
    default="linear",
    choices=["cosine", "linear", "linear_warmup", "rsqrt", "trapezoidal"],
    help=(
        "linear_warmup is a linear warmup followed by flat. trapezoidal uses "
        "1-sqrt cooldown from https://arxiv.org/abs/2405.18392."
    ),
)
parser.add_argument("--warmup_steps", type=int, default=1000)
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
        "sophia",
        "shampoo",
        "caspr",
        "psgd",
    ],
)
parser.add_argument(
    "--norm_grads",
    type=str2bool,
    default=False,
    help=(
        "Normalize the gradients to unit norm before optimizer either globally or "
        "per layer."
    ),
)
parser.add_argument(
    "--norm_grad_type",
    type=str,
    default="global",
    choices=["global", "layer"],
    help=(
        "'global' or 'layer'. 'layer' scales gradient for each layer to have "
        "unit norm i.e. layer/||layer||_2. 'layer' is applied before first-ord "
        "optimizers, or before grafting optimizer in second-ord optimizers, but "
        "never before second-ord optimizers directly because this negatively "
        "affects the preconditioner. 'global' is applied before all optimizers, "
        "and scales the entire gradient to have unit norm i.e. grad/||grad||_2."
    ),
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
    "--precondition_every_n", type=int, default=32, help="For shampoo and sophia."
)
parser.add_argument("--precond_block_size", type=int, default=128, help="Shampoo only.")
parser.add_argument("--sophia_gamma", type=float, default=0.05)
parser.add_argument(
    "--psgd_precond_type", type=str, default="xmat", choices=["xmat", "uvd", "affine"]
)
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
parser.add_argument("--psgd_precond_lr", type=float, default=0.01)
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
    apply_z_loss: bool,
    model_type: str,
    n_layers: int,
    enc_dim: int,
    n_heads: int,
    n_empty_registers: int,
    dropout_rate: float,
    learning_rate: float,
    min_learning_rate: float,
    lr_schedule: str,
    warmup_steps: int,
    cooldown_steps: int,
    schedule_free: bool,
    optimizer: str,
    norm_grads: bool,
    norm_grad_type: str,
    beta1: float,
    beta2: float,
    epsilon: float,
    nesterov: bool,
    weight_decay: float,
    gradient_clip: float,
    graft: bool,
    precondition_every_n: int,
    precond_block_size: int,
    sophia_gamma: float,
    psgd_precond_type: str,
    psgd_feed_into_adam: bool,
    psgd_heavyball: bool,
    psgd_rank: int,
    psgd_update_probability: float,
    psgd_precond_lr: float,
    mu_dtype: str,
):
    # TODO (evanatyourservice): allow for custom optimizer pass in
    lr_schedule = "linear_warmup" if schedule_free else lr_schedule
    if norm_grads and norm_grad_type == "global" and gradient_clip is not None:
        gradient_clip = None
        print("Global gradient normalization is on, turning off gradient clipping.")

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
                image_size=224,
                shuffle_buffer_size=250 if dataset == "imagenette" else 2000,
                prefetch=4,
            )
            test_ds = create_split(
                ds_builder,
                batch_size,
                train=False,
                platform=platform,
                dtype=tf.float32,
                image_size=224,
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
    tx, lr_fn = create_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        norm_grads_layerwise=norm_grads and norm_grad_type == "layer",
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
        precondition_every_n=precondition_every_n,
        precond_block_size=precond_block_size,
        sophia_gamma=sophia_gamma,
        psgd_precond_type=psgd_precond_type,
        psgd_feed_into_adam=psgd_feed_into_adam,
        psgd_heavyball=psgd_heavyball,
        psgd_rank=psgd_rank,
        psgd_update_prob=psgd_update_probability,
        psgd_precond_lr=psgd_precond_lr,
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
            flat_params, _ = jax.tree.flatten(params)
            rngs = jax.random.split(rng, len(flat_params))
            noisy_params = [
                p * jax.random.uniform(k, p.shape, p.dtype)
                for p, k in zip(flat_params, rngs)
            ]
            loss += l2_regularization * optax.global_norm(noisy_params)

        return loss.mean(), (new_model_state, logits)

    def apply_model(rng, state, images, labels):
        """Computes gradients, loss and accuracy for a single batch.

        Args:
            rng: PRNGKey, random number generator.
            state: TrainState, current state.
            images: jnp.ndarray, batch of images.
            labels: jnp.ndarray, batch of labels.

        Returns:
            grads: jnp.ndarray, gradients.
            loss: float, mean loss.
            accuracy: float, mean accuracy.
            new_model_state: dict, updated model state.
        """
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(
            state.params, state.batch_stats, rng, images, labels
        )
        new_model_state, logits = aux
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return grads, loss, accuracy, new_model_state

    @partial(pmap, axis_name="batch", donate_argnums=(1,))
    def train_step(rng, state, batch):
        """Applies an update to parameters and returns new state.

        This function is jit-compiled and runs on multiple devices in parallel (pmap).
        Therefore, each argument has a leading batch dim equal to number of
        devices and is split across devices when passed to the function. Within the
        function, they do not have this leading batch dim. When values
        are returned from the function, they are recombined and again have the leading
        batch dim of size number of devices.

        For example, with 8 devices, a batched image input would have shape
        [8, 128, 32, 32, 3] with a leading device dim, within the function it would
        have shape [128, 32, 32, 3] without a leading device dim, and returned values
        would again have a leading device dim of 8, [8, ...].

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
        rng, rng2 = jax.random.split(rng)
        grads, loss, accuracy, new_model_state = apply_model(
            rng2, state, batch["image"], batch["label"]
        )

        # mean gradients across devices
        grads = jax.lax.pmean(grads, axis_name="batch")

        grad_norm = optax.global_norm(grads)
        if norm_grads and norm_grad_type == "global":
            # norm grads by global norm
            grads = jax.tree.map(
                lambda x: x / jnp.where(grad_norm == 0, 1, grad_norm), grads
            )

        if optimizer in ["sophia", "psgd"]:
            # sophia, psgd need loss function to compute Hessian diagonal

            def temp_loss_fn(params):
                return loss_fn(
                    params, state.batch_stats, rng2, batch["image"], batch["label"]
                )[0]

            updates, new_opt_state = tx.update(
                grads, state.opt_state, state.params, obj_fn=temp_loss_fn
            )
        else:
            updates, new_opt_state = tx.update(grads, state.opt_state, state.params)

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
                    "lr": lr_fn(state.step[0].item()),
                }
                if log_to_wandb:
                    wandb.log(to_log, step=state.step[0].item())
                    wandb.summary["min_loss"] = min(all_test_losses)
                    wandb.summary["max_accuracy"] = max(all_test_accs) * 100
                if state.step[0].item() % 1000 == 0 or (
                    i == steps_per_epoch - 1 and e == n_epochs - 1
                ):
                    print(
                        "step:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, "
                        "test_accuracy: %.2f, grad_norm: %.2f, params_norm: %.2f, lr: %.6f"
                        % (
                            state.step[0].item(),
                            to_log["train_loss"],
                            to_log["train_accuracy"],
                            to_log["test_loss"],
                            to_log["test_accuracy"],
                            to_log["grad_norm"],
                            to_log["params_norm"],
                            to_log["lr"],
                        )
                    )

                train_losses = []
                train_accuracies = []

    print(f"Min loss: {min(all_test_losses):.4f}")
    print(f"Max accuracy: {max(all_test_accs) * 100:.2f}%")

    if log_to_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(**vars(parser.parse_args()))
