from tensorflow.keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate,\
    UpSampling2D, Flatten, Conv2DTranspose, Layer
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import backend as Kbackend
from tensorflow.keras.optimizers import Adam
import util
import numpy as np
from skimage import color
import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin/'


class NetworkModelManager:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "network_model")
        os.makedirs("training_data/"+self.name, exist_ok=True)
        self.checkpoint_path = "training_data/"+self.name+"/cp-{epoch:04d}.ckpt"
        self.history_path = f"training_data/{self.name}/cp-history.json"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.history = dict()
        self.model = None
        self.current_epoch = 0

    def load_weights(self):
        latest_cp: str = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.current_epoch = int(latest_cp[latest_cp.find("cp-")+3:latest_cp.find(".ckpt")])
        self.model.load_weights(latest_cp)

    def save_weights(self, epoch=0):
        self.model.save_weights(self.checkpoint_path.format(epoch=epoch))

    def load_history(self):
        import json
        if os.path.exists(self.history_path):
            self.history = json.load(open(self.history_path, 'r'))

    def save_history(self):
        import json
        json.dump(self.history, open(self.history_path, 'w'))

    def update_curr_epoch(self):
        latest_cp: str = tf.train.latest_checkpoint(self.checkpoint_dir)
        self.current_epoch = int(latest_cp[latest_cp.find("cp-") + 3:latest_cp.find(".ckpt")])

    def update_history(self, other: dict):
        temp = {**self.history, **other}
        for key, value in temp.items():
            if key in self.history and key in other:
                if isinstance(value, list):
                    temp[key] = self.history[key] + value
                elif isinstance(value, np.ndarray):
                    temp[key] = list(self.history[key]).extend(list(value))
                else:
                    temp[key] = [self.history[key], value]
        self.history = temp
        self.save_history()
        self.update_curr_epoch()

    def load(self):
        self.load_weights()
        self.load_history()

    def save(self, epoch=0):
        self.save_weights(epoch)
        self.save_history()


class CatColorizer(NetworkModelManager):
    def __init__(self, depth_after_fusion=256, img_size=128, learning_rate=1e-3, **kwargs):
        super(CatColorizer, self).__init__(**kwargs)
        self.depth_after_fusion = depth_after_fusion
        self.img_size = img_size
        self.lr = learning_rate

        self.inception_resnet_v2_model = tf.keras.applications.InceptionResNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        self.inception_resnet_v2_model.trainable = False

    def build(self):
        # encoder
        encoder_input = Input(shape=(self.img_size, self.img_size, 1), name="encoder_input")
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', name="encoder_output")(x)

        batch, height, width, channels = Kbackend.int_shape(encoder_output)

        # Fusion Layer
        repeat_input = Kbackend.concatenate([encoder_input, encoder_input, encoder_input], axis=-1)
        embs = self.inception_resnet_v2_model(repeat_input)
        embs = tf.keras.layers.GlobalAveragePooling2D()(embs)

        embs = RepeatVector(height * width)(embs)
        embs = Reshape((height, width, embs.shape[-1]))(embs)
        embs = concatenate([encoder_output, embs], axis=-1)

        # decoder
        decoder_input = Conv2D(self.depth_after_fusion, (1, 1), activation='relu', name="decoder_input")(embs)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder_input)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
        decoder_output = UpSampling2D((2, 2), name="decoder_output")(x)
        self.model = Model(encoder_input, decoder_output, name=self.name)

        if len(os.listdir(self.checkpoint_dir)) == 0:
            self.save_weights(epoch=0)
        return self.model

    def build_and_compile(self):
        self.model = self.build()
        self.model = self.compile()
        return self.model

    def compile(self):
        assert self.model is not None
        self.model.compile(
            optimizer=Adam(self.lr),
            loss='mae',
            metrics=[
                'accuracy',
                'mse',
                'mae',
                'mape',
                 # y_true_max,
                 # y_true_min,
                 # y_pred_max,
                 # y_pred_min,
            ]
        )
        return self.model

    def predict_rgb(self, l_batch):
        ab_batch = self.model.predict(l_batch)
        lab_batch = np.zeros((l_batch.shape[0], l_batch.shape[1], l_batch.shape[2], 3))
        lab_batch[:, :, :, 0] = util.unnormalize_l_channel(tf.squeeze(l_batch))
        lab_batch[:, :, :, [1, 2]] = util.unnormalize_ab_channels(ab_batch)
        rgb_batch = color.lab2rgb(lab_batch)

        return rgb_batch


# shows the minimum value of the AB channels
def y_true_min(yt, yp):
    return Kbackend.min(yt)


# shows the maximum value of the RGB AB channels
def y_true_max(yt, yp):
    return Kbackend.max(yt)


# shows the minimum value of the predicted AB channels
def y_pred_min(yt, yp):
    return Kbackend.min(yp)


# shows the maximum value of the predicted AB channels
def y_pred_max(yt, yp):
    return Kbackend.max(yp)


if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model
    model_manager = CatColorizer(img_size=80, name="CatColorizer")


    new_history = {'loss': [1076.923828125], 'accuracy': [0.4869177043437958], 'mse': [0.010041841305792332], 'mae': [0.06346332281827927], 'mape': [1076.924072265625], 'val_loss': [330.6434020996094], 'val_accuracy': [0.7349596619606018], 'val_mse': [0.010745665989816189], 'val_mae': [0.06535135209560394], 'val_mape': [330.6432189941406]}
    model_manager.load_history()
    print(model_manager.history)
    model_manager.update_history(new_history)
    print(model_manager.history)


    # model.summary()
    # plot_model(model, to_file='Figures/CatColorizer.png', show_shapes=True)

