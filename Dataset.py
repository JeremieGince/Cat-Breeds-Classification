import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import color, transform
from PIL import Image, ImageFile
import util
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ColorizerDataGenerator:
    def __init__(self, image_data_generator, **kwargs):
        self.image_data_generator = image_data_generator
        self.virtual_length = kwargs.get("virtual_length", None)

    @property
    def labels(self):
        return self.image_data_generator.labels

    def __len__(self):
        return len(self.image_data_generator) if self.virtual_length is None else self.virtual_length

    def __call__(self):
        while True:
            rgb_batch = self.image_data_generator.next()
            lab_batch = color.rgb2lab(rgb_batch).astype(np.float32)
            l_batch = util.normalize_l_channel(lab_batch[:, :, :, 0][..., np.newaxis])
            ab_batch = util.normalize_ab_channels(lab_batch[:, :, :, 1:])
            yield l_batch, ab_batch

    def __next__(self):
        return next(self())

    def __iter__(self):
        return self()

    def next(self):
        return next(self())


class CatColorizerDataset:
    _URL = r"https://www.kaggle.com/ma7555/cat-breeds-dataset/download/"
    _DIR = r'C:\Users\gince\Documents\GitHub\Cat-Breeds-Classification\Data\cat-breeds-dataset/'

    BATCH_SIZE = 256
    IMG_SIZE = 80  # must be more than 80, 80 is recommended
    VAL_SPLIT = 0.5

    def __init__(self, **kwargs):
        self.IMG_SIZE = kwargs.get("img_size", CatColorizerDataset.IMG_SIZE)
        self.BATCH_SIZE = kwargs.get("batch_size", CatColorizerDataset.BATCH_SIZE)

        # self.path_to_zip = tf.keras.utils.get_file('cat-breeds-dataset.zip', origin=self._URL, extract=True)
        # print(f"Data saved to: {self.path_to_zip}")
        # self.PATH = os.path.join(os.path.dirname(self.path_to_zip), '/cat-breeds-dataset/images')
        self.PATH = os.path.join(os.path.dirname(self._DIR), 'images')

        self.image_generator = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=45,
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True,
            zoom_range=0.5,
            validation_split=self.VAL_SPLIT,
        )  # Generator for our training data

        self.train_data_gen = ColorizerDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                shuffle=True,
                seed=3,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode=None,
                subset="training"
            )
        )

        self.val_data_gen = ColorizerDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode=None,
                seed=3,
                subset="validation"
            ),
            virtual_length=10_000 // self.BATCH_SIZE
        )

    @property
    def train_length(self):
        return len(self.train_data_gen)

    @property
    def val_length(self):
        return len(self.val_data_gen)

    @property
    def labels(self):
        return set(list(self.train_data_gen.labels)+list(self.val_data_gen.labels))

    @property
    def number_of_train_call(self):
        return self.train_length // self.BATCH_SIZE

    @property
    def number_of_val_call(self):
        return self.val_length // self.BATCH_SIZE

    def plot_samples(self, nb_samples=5):
        nb_samples = min(nb_samples, self.BATCH_SIZE)
        l_batch, ab_batch = next(self.train_data_gen)

        lab_batch = np.zeros((l_batch.shape[0], l_batch.shape[1], l_batch.shape[2], 3))
        lab_batch[:, :, :, 0] = util.unnormalize_l_channel(tf.squeeze(l_batch))
        lab_batch[:, :, :, [1, 2]] = util.unnormalize_ab_channels(ab_batch)

        rgb_batch = color.lab2rgb(lab_batch)

        fig, axes = plt.subplots(nb_samples, 2, figsize=(10, 10))
        for i in range(nb_samples):
            axes[i][0].imshow(tf.squeeze(l_batch[i]), cmap='gray')
            axes[i][1].imshow(rgb_batch[i])

            axes[i][0].axis('off')
            axes[i][1].axis('off')
        plt.tight_layout(pad=0.0)
        plt.show()


if __name__ == '__main__':
    dataset = CatColorizerDataset(img_size=256, batch_size=100)
    dataset.plot_samples(5)

