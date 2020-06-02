from models import NetworkModelManager, CatColorizer, NetworkManagerCallback
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os


class Trainer:
    def __init__(self, model_manager: NetworkModelManager, dataset,
                 network_callback_args=None,
                 **kwargs):
        if network_callback_args is None:
            network_callback_args = {
                "verbose": True,
                "save_freq": 1
            }
        self.BATCH_SIZE = kwargs.get("batch_size", 256)
        self.IMG_SIZE = kwargs.get("img_size", 80)
        self.FUSION_DEPTH = kwargs.get("fusion_depth", 128)
        self.modelManager = model_manager
        self.model = model_manager.model
        self.dataset = dataset
        self.use_saving_callback = kwargs.get("use_saving_callback", True)
        self.load_on_start = kwargs.get("load_on_start", True)
        self.verbose = kwargs.get("verbose", 1)

        self.network_callback = NetworkManagerCallback(self.modelManager, **network_callback_args)

    def train(self, epochs=1):
        if self.load_on_start:
            self.modelManager.load()
        history = self.model.fit(
            self.dataset.train_data_gen(),
            steps_per_epoch=len(self.dataset.train_data_gen),
            epochs=epochs,
            validation_data=self.dataset.val_data_gen() if self.dataset.val_data_gen is not None else None,
            validation_steps=len(self.dataset.val_data_gen) if self.dataset.val_data_gen is not None else None,
            callbacks=[self.network_callback] if self.use_saving_callback else [],
            verbose=self.verbose,
            initial_epoch=self.modelManager.current_epoch,
        )
        self.modelManager.save_weights()
        return history


if __name__ == '__main__':
    from util import plotHistory
    from models import CatBreedsClassifier
    from Dataset import CatBreedsClassifierDataset, CatColorizerOverfittingDataset, CatColorizerDataset
    from hyperparameters import *
    import time

    # -----------------------------------------------------------------------------------------------------------------
    # hyper-parameters
    # -----------------------------------------------------------------------------------------------------------------
    tf.random.set_seed(SEED)
    print(get_str_repr_for_hyper_params())

    # -----------------------------------------------------------------------------------------------------------------
    # Colorization dataset
    # -----------------------------------------------------------------------------------------------------------------
    col_dataset = CatColorizerDataset(
        gamut_size=GAMUT_SIZE,
        bins=BINS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        VAL_SPLIT=COL_VAL_SPLIT,
        val_virtual_length=None,
    )
    col_dataset.show_gamut_probabilities()
    col_dataset.show_gamut_probabilities(rebin=True)

    col_model_manager = CatColorizer(
        *col_dataset.get_gamut_params(),
        fusion_depth=FUSION_DEPTH,
        img_size=col_dataset.IMG_SIZE,
        name=f"CatColorizer_gamut-{col_dataset.GAMUT_SIZE}",
    )
    col_model_manager.build_and_compile()

    # -----------------------------------------------------------------------------------------------------------------
    # Training the features
    # -----------------------------------------------------------------------------------------------------------------
    col_trainer = Trainer(
        col_model_manager,
        col_dataset,
        network_callback_args={
            "verbose": True,
            "save_freq": 1
        }
    )
    start_time = time.time()
    col_trainer.train(FEATURES_TRAINING_EPOCHS)
    end_feature_training_time = time.time() - start_time
    print(f"--- Elapse feature training time: {end_feature_training_time} [s] ---")

    plotHistory(col_model_manager.history)

    # -----------------------------------------------------------------------------------------------------------------
    # Classifier dataset
    # -----------------------------------------------------------------------------------------------------------------
    cls_dataset = CatBreedsClassifierDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # -----------------------------------------------------------------------------------------------------------------
    # classification with pre-trained features
    # -----------------------------------------------------------------------------------------------------------------
    cls_model_pretrained_manager = CatBreedsClassifier(
        depth_after_fusion=FUSION_DEPTH,
        img_size=IMG_SIZE,
        output_size=cls_dataset.nb_cls,
        cat_col_manager=col_model_manager,
        name="CatBreedsClassifier_withPretrainedFeatures",
        pretrained_head=True,
    )
    cls_model_pretrained_manager.build_and_compile()

    cls_trainer = Trainer(cls_model_pretrained_manager, cls_dataset)

    start_time = time.time()
    cls_trainer.train(CLASSIFIER_EPOCHS)
    end_cls_pt_training_time = time.time() - start_time
    print(f"--- Elapse classification with pre-trained features training time: {end_cls_pt_training_time} [s] ---")

    cls_model_pretrained_manager.load_history()
    plotHistory(cls_model_pretrained_manager.history)

    # -----------------------------------------------------------------------------------------------------------------
    # classification without pre-trained features
    # -----------------------------------------------------------------------------------------------------------------
    cls_model_no_pretrained_manager = CatBreedsClassifier(
        depth_after_fusion=FUSION_DEPTH,
        img_size=IMG_SIZE,
        output_size=cls_dataset.nb_cls,
        cat_col_manager=col_model_manager,
        name="CatBreedsClassifier_withoutPretrainedFeatures",
        pretrained_head=False,
    )
    cls_model_no_pretrained_manager.build_and_compile()

    cls_trainer = Trainer(cls_model_no_pretrained_manager, cls_dataset)

    start_time = time.time()
    cls_trainer.train(CLASSIFIER_EPOCHS)
    end_cls_npt_training_time = time.time() - start_time
    print(f"--- Elapse classification without pre-trained features training time: {end_cls_npt_training_time} [s] ---")

    cls_model_no_pretrained_manager.load_history()
    plotHistory(cls_model_no_pretrained_manager.history)
