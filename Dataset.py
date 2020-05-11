import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CatDataset:
    _URL = r"https://www.kaggle.com/ma7555/cat-breeds-dataset/download/"
    _DIR = r'C:\Users\gince\Documents\GitHub\Cat-Breeds-Classification\Data\cat-breeds-dataset/'

    BATCH_SIZE = 256
    IMG_HEIGHT = 150
    IMG_WIDTH = 150

    def __init__(self):
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
            validation_split=0.2,
            # preprocessing_function=tf.image.rgb_to_grayscale,
        )  # Generator for our training data

        self.train_data_gen = self.image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                       directory=self.PATH,
                                                                       shuffle=True,
                                                                       target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                       class_mode='input',
                                                                       subset="training")

        self.val_data_gen = self.image_generator.flow_from_directory(batch_size=self.BATCH_SIZE,
                                                                     directory=self.PATH,
                                                                     target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                     class_mode='input',
                                                                     subset="validation")

        self.inception_resnet_v2_model = tf.keras.applications.InceptionResNetV2(
            input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3),
            include_top=False,
            weights='imagenet'
        )
        self.inception_resnet_v2_model.trainable = False

    def generate_training_data(self):
        while True:
            input_batch, output_batch = self.train_data_gen.next()
            grayscale_batch = tf.image.rgb_to_grayscale(input_batch)
            rgb_batch = tf.repeat(grayscale_batch, 3, -1)
            resnet_features = self.inception_resnet_v2_model(rgb_batch)
            yield [grayscale_batch, resnet_features[:, 0]], output_batch

    def generate_validation_data(self):
        while True:
            input_batch, output_batch = self.val_data_gen.next()
            grayscale_batch = tf.image.rgb_to_grayscale(input_batch)
            rgb_batch = tf.repeat(grayscale_batch, 3, -1)
            resnet_features = self.inception_resnet_v2_model(rgb_batch)
            yield [grayscale_batch, resnet_features], output_batch

    def plot_samples(self, nb_samples=5):
        # This function will plot images in the form of a grid
        # with 1 row and 5 columns where images are placed in each column.
        nb_samples = min(nb_samples, self.BATCH_SIZE)
        [sample_input, sample_resnet_features], sample_output = next(self.generate_training_data())
        print(f"sample_resnet_features.shape: {sample_resnet_features.shape}")

        fig, axes = plt.subplots(nb_samples, 2, figsize=(10, 10))
        for i in range(nb_samples):
            axes[i][0].imshow(tf.squeeze(sample_input[i]), cmap='gray')
            axes[i][1].imshow(sample_output[i])

            axes[i][0].axis('off')
            axes[i][1].axis('off')
        plt.tight_layout(pad=0.0)
        plt.show()


def plotHistory(history, epochs):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    dataset = CatDataset()
    dataset.plot_samples(3)
    # _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    #
    # path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    #
    # PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
    # train_dir = os.path.join(PATH, 'train')
    # validation_dir = os.path.join(PATH, 'validation')
    #
    # train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    # train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    # validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    # validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures
    #
    # num_cats_tr = len(os.listdir(train_cats_dir))
    # num_dogs_tr = len(os.listdir(train_dogs_dir))
    #
    # num_cats_val = len(os.listdir(validation_cats_dir))
    # num_dogs_val = len(os.listdir(validation_dogs_dir))
    #
    # total_train = num_cats_tr + num_dogs_tr
    # total_val = num_cats_val + num_dogs_val
    #
    # print('total training cat images:', num_cats_tr)
    # print('total training dog images:', num_dogs_tr)
    #
    # print('total validation cat images:', num_cats_val)
    # print('total validation dog images:', num_dogs_val)
    # print("--")
    # print("Total training images:", total_train)
    # print("Total validation images:", total_val)
    #
    # batch_size = 128
    # epochs = 15
    # IMG_HEIGHT = 150
    # IMG_WIDTH = 150
    #
    # train_image_generator = ImageDataGenerator(
    #     rescale=1. / 255,
    #     rotation_range=45,
    #     width_shift_range=.15,
    #     height_shift_range=.15,
    #     horizontal_flip=True,
    #     zoom_range=0.5
    # )  # Generator for our training data
    #
    # validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    #
    # train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
    #                                                            directory=train_dir,
    #                                                            shuffle=True,
    #                                                            target_size=(IMG_HEIGHT, IMG_WIDTH),
    #                                                            class_mode='binary')
    #
    # val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
    #                                                               directory=validation_dir,
    #                                                               target_size=(IMG_HEIGHT, IMG_WIDTH),
    #                                                               class_mode='binary')
    #
    # sample_training_images, _ = next(train_data_gen)
    # plotImages(sample_training_images[:5])
    #
    # model = Sequential([
    #     Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #     MaxPooling2D(),
    #     Conv2D(32, 3, padding='same', activation='relu'),
    #     MaxPooling2D(),
    #     Conv2D(64, 3, padding='same', activation='relu'),
    #     MaxPooling2D(),
    #     Flatten(),
    #     Dense(512, activation='relu'),
    #     Dense(1)
    # ])
    #
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    #
    # model.summary()
    #
    # history = model.fit_generator(
    #     train_data_gen,
    #     steps_per_epoch=total_train // batch_size,
    #     epochs=epochs,
    #     validation_data=val_data_gen,
    #     validation_steps=total_val // batch_size
    # )
    #
    # plotHistory(history, epochs)
