from gan import GAN
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

def load_celebrities_dataset(preprocess_image, path="/Users/Yegor/Downloads/celebrities-100k/100k/"):
    data_root = pathlib.Path(path)
    all_image_paths = list(data_root.glob('*.jpg'))
    all_image_paths = [str(path) for path in all_image_paths]
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    image_ds = path_ds.map(load_and_preprocess_image)
    return image_ds

#
# if __name__=='__main__':
#     dataset = load_celebrities_dataset()
#
#
#     train_dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
#
#     # Plot input images
#     # for image_batch in train_images:
#     #     plt.imshow((image_batch * 127.5 + 127.5)[0, :, :, 0], cmap='gray')
#
#
#     gan = GAN()
#     # set batch size before training
#     gan.set_batch_size(batch_size=BATCH_SIZE)
#     gan.train(train_dataset, EPOCHS)
