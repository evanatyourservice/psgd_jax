import math

import jax
import tensorflow as tf
import tensorflow_datasets as tfds


@tf.function
def random_erasing(
    image: tf.Tensor,
    probability: float = 0.5,
    min_area: float = 0.02,
    max_area: float = 1 / 3,
    min_aspect: float = 0.3,
    fill_value: float = 127.5,
    min_count=1,
    max_count=1,
    trials=10,
) -> tf.Tensor:
    uniform_random = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    mirror_cond = tf.less(uniform_random, probability)
    image = tf.cond(
        mirror_cond,
        lambda: _erase(
            image,
            min_area,
            max_area,
            min_aspect,
            fill_value,
            min_count,
            max_count,
            trials,
        ),
        lambda: image,
    )
    return image


def _erase(
    image: tf.Tensor,
    min_area: float,
    max_area: float,
    min_aspect: float,
    fill_value: float,
    min_count=1,
    max_count=1,
    trials=10,
) -> tf.Tensor:
    """Erase an area."""
    _min_log_aspect = math.log(min_aspect)
    _max_log_aspect = math.log(1 / min_aspect)

    if min_count == max_count:
        count = min_count
    else:
        count = tf.random.uniform(
            shape=[],
            minval=int(min_count),
            maxval=int(max_count - min_count + 1),
            dtype=tf.int32,
        )

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    area = tf.cast(image_width * image_height, tf.float32)

    for _ in range(count):
        # Work around since break is not supported in tf.function
        is_trial_successfull = False
        for _ in range(trials):
            if not is_trial_successfull:
                erase_area = tf.random.uniform(
                    shape=[], minval=area * min_area, maxval=area * max_area
                )
                aspect_ratio = tf.math.exp(
                    tf.random.uniform(
                        shape=[], minval=_min_log_aspect, maxval=_max_log_aspect
                    )
                )

                half_height = tf.cast(
                    tf.math.round(tf.math.sqrt(erase_area * aspect_ratio) / 2),
                    dtype=tf.int32,
                )
                half_width = tf.cast(
                    tf.math.round(tf.math.sqrt(erase_area / aspect_ratio) / 2),
                    dtype=tf.int32,
                )

                if 2 * half_height < image_height and 2 * half_width < image_width:
                    center_height = tf.random.uniform(
                        shape=[],
                        minval=0,
                        maxval=int(image_height - 2 * half_height),
                        dtype=tf.int32,
                    )
                    center_width = tf.random.uniform(
                        shape=[],
                        minval=0,
                        maxval=int(image_width - 2 * half_width),
                        dtype=tf.int32,
                    )

                    image = _fill_rectangle(
                        image,
                        center_width,
                        center_height,
                        half_width,
                        half_height,
                        fill_value,
                    )

                    is_trial_successfull = True

    return image


def _fill_rectangle(
    image, center_width, center_height, half_width, half_height, fill_value
):
    """Fills blank area."""
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    lower_pad = tf.maximum(0, center_height - half_height)
    upper_pad = tf.maximum(0, image_height - center_height - half_height)
    left_pad = tf.maximum(0, center_width - half_width)
    right_pad = tf.maximum(0, image_width - center_width - half_width)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad),
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1
    )
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])

    image = tf.where(tf.equal(mask, 0), fill_value, image)

    return image


class CifarPreprocess(tf.keras.Model):
    def __init__(self, training_ds: bool, dataset_type: str):
        super().__init__()

        if dataset_type == "cifar10":
            self.means = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
            self.stds = tf.constant([0.2023, 0.1994, 0.2010], dtype=tf.float32)
        elif dataset_type == "cifar100":
            self.means = tf.constant([0.5071, 0.4867, 0.4408], dtype=tf.float32)
            self.stds = tf.constant([0.2675, 0.2565, 0.2761], dtype=tf.float32)
        else:
            self.means = None
            self.stds = None

        if self.means is not None:
            self.means = tf.reshape(self.means, (1, 1, 3))
            self.stds = tf.reshape(self.stds, (1, 1, 3))

        self.training_ds = training_ds

        self.n_devices = jax.local_device_count()

    @tf.function
    def call(self, x):
        image = x["image"]
        label = x["label"]

        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)

        if self.training_ds:
            # augment
            image = tf.pad(
                image, [[4, 4], [4, 4], [0, 0]], mode="CONSTANT", constant_values=127.5
            )  # pad with gray
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)
            image = random_erasing(image, probability=0.75, min_area=0.02, max_area=0.1)

        # scale
        image = image / 255.0
        if self.means is not None:
            image = (image - self.means) / self.stds

        batch = {"image": image, "label": label}

        return batch


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds_builder = tfds.builder("cifar10")
    ds_builder.download_and_prepare()
    ds = ds_builder.as_dataset(split="train")
    preprocess = CifarPreprocess(training_ds=True, dataset_type="cifar10")
    ds = ds.map(preprocess)

    for batch in ds.take(8):
        image = batch["image"]
        label = batch["label"]

        image = image - tf.reduce_min(image)
        image = image / tf.reduce_max(image)
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.show()
