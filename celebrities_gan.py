from gan import GAN
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt


@tf.function
def tf_load_celebrities_dataset(
        pattern,
        to_grayscale,
        img_shape
):
    all_image_paths = tf.io.gfile.glob(pattern=pattern)
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    @tf.function
    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cond(
            pred=tf.equal(to_grayscale, tf.constant(True)),
            true_fn=(lambda: tf.image.rgb_to_grayscale(image)),
            false_fn=(lambda: tf.identity(image))
        )
        image = tf.image.resize(image, img_shape)
        return image

    image_ds = path_ds.map(load_and_preprocess_image)
    return image_ds
