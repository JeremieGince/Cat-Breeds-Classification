from Dataset import CatColorizerDataset
from models import NetworkModelManager, CatColorizer
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
            save_weights_only=True,
            save_freq='epoch'
        )

    def train(self, epochs=1):
        self.modelManager.load()
        history = self.model.fit(
            self.dataset.train_data_gen(),
            steps_per_epoch=len(self.dataset.train_data_gen),
            epochs=self.modelManager.current_epoch+epochs,
            validation_data=self.dataset.val_data_gen(),
            validation_steps=len(self.dataset.val_data_gen),
            callbacks=[self.cp_callback],
            verbose=1,
            initial_epoch=self.modelManager.current_epoch,
        )
        self.modelManager.update_history(history.history)


if __name__ == '__main__':
    from util import plotHistory
    BATCH_SIZE = 100
    IMG_SIZE = 80
    EPOCHS = 100
    FUSION_DEPTH = 256

    dataset = CatColorizerDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    model_manager = CatColorizer(depth_after_fusion=FUSION_DEPTH, img_size=dataset.IMG_SIZE, name="CatColorizer")
    model_manager.build_and_compile()

    trainer = Trainer(model_manager, dataset)
    for _ in range(EPOCHS):
        trainer.train()
    plotHistory(model_manager.history, model_manager.current_epoch)
