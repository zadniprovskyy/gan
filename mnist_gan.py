from gan import GAN
import tensorflow as tf


BATCH_SIZE = 256
BUFFER_SIZE = 60000
EPOCHS = 1


if __name__=='__main__':

    # Initialize GAN model (create generator, discriminator and their optimizers)
    gan = GAN()

    # Load training data
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    # reshape training images
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # normalize training images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    gan.train(train_dataset, EPOCHS)
