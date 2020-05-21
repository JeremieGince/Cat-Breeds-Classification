from Dataset import CatColorizerDataset
from models import NetworkModelManager, CatColorizer, NetworkManagerCallback
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import os


class Trainer:
    def __init__(self, model_manager: NetworkModelManager, dataset, **kwargs):
        self.BATCH_SIZE = kwargs.get("batch_size", 256)
        self.IMG_SIZE = kwargs.get("img_size", 80)
        self.FUSION_DEPTH = kwargs.get("fusion_depth", 128)
        self.modelManager = model_manager
        self.model = model_manager.model
        self.dataset = dataset

        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.modelManager.checkpoint_path,
            verbose=1,
            save_weights_only=False,
            save_freq='epoch'
        )

        self.network_callback = NetworkManagerCallback(self.modelManager)

    def train(self, epochs=1):
        self.modelManager.load()
        history = self.model.fit(
            self.dataset.train_data_gen(),
            steps_per_epoch=len(self.dataset.train_data_gen),
            epochs=epochs,
            validation_data=self.dataset.val_data_gen(),
            validation_steps=len(self.dataset.val_data_gen),
            callbacks=[self.cp_callback, self.network_callback],
            verbose=1,
            initial_epoch=self.modelManager.current_epoch,
        )
        self.modelManager.update_history(history.history)


if __name__ == '__main__':
    from util import plotHistory
    from models import CatBreedsClassifier
    from Dataset import CatBreedsClassifierDataset
    BATCH_SIZE = 64
    IMG_SIZE = 80
    FEATURES_TRAINING_EPOCHS = 5
    CLASSIFIER_EPOCHS = 5
    FUSION_DEPTH = 256

    col_dataset = CatColorizerDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    col_model_manager = CatColorizer(
        depth_after_fusion=FUSION_DEPTH,
        img_size=col_dataset.IMG_SIZE,
        name="CatColorizer",
    )
    col_model_manager.build_and_compile()

    # Training of the features
    col_trainer = Trainer(col_model_manager, col_dataset)
    for e in range(FEATURES_TRAINING_EPOCHS):
        col_trainer.train(e)
    plotHistory(col_model_manager.history, col_model_manager.current_epoch)

    cls_dataset = CatBreedsClassifierDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    # classification with pretrained features
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
    for e in range(CLASSIFIER_EPOCHS):
        cls_trainer.train(e)
    cls_model_pretrained_manager.load_history()
    plotHistory(cls_model_pretrained_manager.history, cls_model_pretrained_manager.current_epoch)

    # classification without pretrained features
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
    for e in range(CLASSIFIER_EPOCHS):
        cls_trainer.train(e)
    cls_model_no_pretrained_manager.load_history()
    plotHistory(cls_model_no_pretrained_manager.history, cls_model_no_pretrained_manager.current_epoch)
