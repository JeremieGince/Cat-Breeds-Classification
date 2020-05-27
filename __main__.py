import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from skimage import color, transform
import numpy as np
import util
from Dataset import CatColorizerDataset, CatBreedsClassifierDataset
from models import CatColorizer, CatBreedsClassifier

print(tf.__version__)


def plot_history_comparison(pre_trained_history, no_pre_trained_history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    axes[0].plot(pre_trained_history["val_loss"], label="Pre-trained model")
    axes[0].plot(no_pre_trained_history["val_loss"], label="No pre-trained model")
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("epochs")

    axes[1].plot(pre_trained_history["val_accuracy"], label="Pre-trained model")
    axes[1].plot(no_pre_trained_history["val_accuracy"], label="No pre-trained model")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epochs")

    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # -----------------------------------------------------------------------------------------------------------------
    # hyper-parameters
    # -----------------------------------------------------------------------------------------------------------------
    from hyperparameters import SEED, \
                                BATCH_SIZE,\
                                IMG_SIZE, \
                                FEATURES_TRAINING_EPOCHS, \
                                CLASSIFIER_EPOCHS, \
                                FUSION_DEPTH, \
                                GAMUT_SIZE, \
                                BINS

    # setting the seed
    tf.random.set_seed(SEED)

    # -----------------------------------------------------------------------------------------------------------------
    # colorization
    # -----------------------------------------------------------------------------------------------------------------

    col_dataset = CatColorizerDataset(
        gamut_size=GAMUT_SIZE,
        bins=BINS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    col_dataset.plot_samples(3)
    col_dataset.show_gamut_probabilities()
    col_dataset.show_gamut_probabilities(rebin=True)
    print(f"dataset length: {len(col_dataset)}")
    print(f"dataset labels: {col_dataset.labels}")

    col_model_manager = CatColorizer(
        col_dataset.gamut_instances,
        fusion_depth=FUSION_DEPTH,
        img_size=col_dataset.IMG_SIZE,
        name=f"CatColorizer_gamut-{col_dataset.GAMUT_SIZE}",
    )
    col_model_manager.build_and_compile()
    col_model_manager.load()

    col_model_manager.model.summary()
    plot_model(col_model_manager.model, to_file=f"Figures/{col_model_manager.name}.png",
               show_layer_names=True, show_shapes=True)
    util.plotHistory(col_model_manager.history)
    util.plot_cat_colorization_prediction_samples(col_model_manager)

    # -----------------------------------------------------------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------------------------------------------------------

    raise NotImplementedError("The classification part is not updated with the new colorizer")

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
    cls_model_pretrained_manager.build_and_compile()
    plot_model(cls_model_pretrained_manager.model,
               to_file=f"Figures/{cls_model_pretrained_manager.name}.png",
               show_layer_names=True, show_shapes=True)

    cls_model_pretrained_manager.load_history()
    util.plotHistory(cls_model_pretrained_manager.history)

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
    cls_model_no_pretrained_manager.build_and_compile()
    plot_model(cls_model_no_pretrained_manager.model,
               to_file=f"Figures/{cls_model_no_pretrained_manager.name}.png",
               show_layer_names=True, show_shapes=True)

    cls_model_no_pretrained_manager.load_history()
    util.plotHistory(cls_model_no_pretrained_manager.history)

    # ---------------------------------------------------------
    # comparison
    # ---------------------------------------------------------
    plot_history_comparison(cls_model_pretrained_manager.history, cls_model_no_pretrained_manager.history)
