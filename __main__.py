import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from skimage import color, transform
import numpy as np
import util
from Dataset import CatColorizerDataset, CatBreedsClassifierDataset
from models import CatColorizer, CatBreedsClassifier

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


def plot_history_comparison(pre_trained_history, no_pre_trained_history):
    pass


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # hyper-paramètres
    # -----------------------------------------------------------------------------------------------------------------
    BATCH_SIZE = 100
    IMG_SIZE = 80
    EPOCHS = 1
    FUSION_DEPTH = 256

    # -----------------------------------------------------------------------------------------------------------------
    # colorization
    # -----------------------------------------------------------------------------------------------------------------

    col_dataset = CatColorizerDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    # col_dataset.plot_samples(3)
    print(f"dataset labels: {col_dataset.labels}")

    col_model_manager = CatColorizer(depth_after_fusion=FUSION_DEPTH, img_size=col_dataset.IMG_SIZE,
                                     name="CatColorizer")
    col_model_manager.build_and_compile()
    col_model_manager.load()

    col_model_manager.model.summary()
    plot_model(col_model_manager.model, to_file='Figures/CatColorizer.png', show_shapes=True)
    util.plotHistory(col_model_manager.history, col_model_manager.current_epoch)
    plot_cat_colorization_prediction_samples(col_model_manager)

    plot_model(col_model_manager.model, to_file=f"Figures/{col_model_manager.name}.png",
               show_layer_names=True, show_shapes=True)

    # -----------------------------------------------------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------------------------------------------------

    cls_dataset = CatBreedsClassifierDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # ---------------------------------------------------------
    # classification with pre-trained features
    # ---------------------------------------------------------
    cls_model_pretrained_manager = CatBreedsClassifier(
        depth_after_fusion=FUSION_DEPTH,
        img_size=IMG_SIZE,
        output_size=cls_dataset.nb_cls,
        cat_col_manager=col_model_manager,
        name="CatBreedsClassifier_withPretrainedFeatures",
        pretrained_head=True,
    )

    plot_model(cls_model_pretrained_manager.model,
               to_file=f"Figures/{cls_model_pretrained_manager.name}.png",
               show_layer_names=True, show_shapes=True)

    cls_model_pretrained_manager.build_and_compile()
    cls_model_pretrained_manager.load_history()
    util.plotHistory(cls_model_pretrained_manager.history, cls_model_pretrained_manager.current_epoch)

    # ---------------------------------------------------------
    # classification without pre-trained features
    # ---------------------------------------------------------
    cls_model_no_pretrained_manager = CatBreedsClassifier(
        depth_after_fusion=FUSION_DEPTH,
        img_size=IMG_SIZE,
        output_size=cls_dataset.nb_cls,
        cat_col_manager=col_model_manager,
        name="CatBreedsClassifier_withoutPretrainedFeatures",
        pretrained_head=False,
    )

    plot_model(cls_model_no_pretrained_manager.model,
               to_file=f"Figures/{cls_model_no_pretrained_manager.name}.png",
               show_layer_names=True, show_shapes=True)

    cls_model_no_pretrained_manager.build_and_compile()
    cls_model_no_pretrained_manager.load_history()
    util.plotHistory(cls_model_no_pretrained_manager.history, cls_model_no_pretrained_manager.current_epoch)

    # ---------------------------------------------------------
    # comparison
    # ---------------------------------------------------------
    plot_history_comparison(cls_model_pretrained_manager.history, cls_model_no_pretrained_manager.history)
