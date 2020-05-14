import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from skimage import color, transform
import numpy as np
import util
from Dataset import CatColorizerDataset
from models import CatColorizer

print(tf.__version__)


def plot_cat_colorization_prediction_samples(model_manager, nb_samples=3):
    image_generator = ImageDataGenerator(rescale=1. / 255)

    data_gen = image_generator.flow_from_directory(
        batch_size=nb_samples,
        directory="Figures/Images_to_predict/",
        shuffle=True,
        target_size=(model_manager.img_size, model_manager.img_size),
        class_mode=None,
    )

    rgb_batch = next(data_gen)

    lab_batch = color.rgb2lab(rgb_batch).astype(np.float32)
    l_batch = util.normalize_l_channel(lab_batch[:, :, :, 0][..., np.newaxis])
    ab_batch = util.normalize_ab_channels(lab_batch[:, :, :, 1:])

    rgb_batch_prediction = model_manager.predict_rgb(l_batch)

    fig, axes = plt.subplots(nb_samples, 3, figsize=(10, 10))
    for i in range(nb_samples):
        axes[i][0].imshow(tf.squeeze(l_batch[i]), cmap='gray')
        axes[i][1].imshow(rgb_batch_prediction[i])
        axes[i][2].imshow(rgb_batch[i])

        axes[i][0].axis('off')
        axes[i][1].axis('off')
        axes[i][2].axis('off')

    axes[0][0].set_title("Input")
    axes[0][1].set_title("Prediction")
    axes[0][2].set_title("Ground truth")

    plt.tight_layout(pad=0.0)
    plt.show()


if __name__ == '__main__':
    BATCH_SIZE = 100
    IMG_SIZE = 80
    EPOCHS = 1
    FUSION_DEPTH = 256

    dataset = CatColorizerDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # dataset.plot_samples(3)
    print(f"dataset labels: {dataset.labels}")

    model_manager = CatColorizer(depth_after_fusion=FUSION_DEPTH, img_size=dataset.IMG_SIZE, name="CatColorizer")
    model_manager.build_and_compile()
    model_manager.load()

    model_manager.model.summary()
    plot_model(model_manager.model, to_file='Figures/CatColorizer.png', show_shapes=True)
    util.plotHistory(model_manager.history, model_manager.current_epoch)
    plot_cat_colorization_prediction_samples(model_manager)
