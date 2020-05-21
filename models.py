import os

import numpy as np
import tensorflow as tf
from skimage import color
from tensorflow.keras import backend as Kbackend
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.layers import Conv2D, Input, Reshape, RepeatVector, concatenate, \
    UpSampling2D, Flatten, Dense, BatchNormalization, Softmax
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

import util

os.environ["PATH"] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin/'


def get_latest_checkpoint(checkpoint_dir, ext=".h5", head="cp-"):
    latest = None
    cp_max = -1
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(ext) and filename.startswith(head):
            cp = int(filename[filename.find(head) + len(head):filename.find(ext)])
            if cp > cp_max:
                cp_max = cp
                latest = filename
    return checkpoint_dir + "/" + latest


class NetworkModelManager:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "network_model")
        os.makedirs("training_data/" + self.name, exist_ok=True)
        self.checkpoint_path = "training_data/" + self.name + "/cp-{epoch:04d}.h5"
        self.history_path = f"training_data/{self.name}/cp-history.json"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)
        self.history = dict()
        self.model = None
        self.current_epoch = 0

    @property
    def summary(self):
        return self.model.summary()

    def load_weights(self):
        latest_cp: str = get_latest_checkpoint(self.checkpoint_dir, ext=".h5", head="cp-")
        self.current_epoch = int(latest_cp[latest_cp.find("cp-") + 3:latest_cp.find(".h5")])
        self.model.load_weights(latest_cp)

    def save_weights(self, epoch=0):
        self.model.save(self.checkpoint_path.format(epoch=epoch))

    def load_history(self):
        import json
        if os.path.exists(self.history_path):
            self.history = json.load(open(self.history_path, 'r'))
        self.update_curr_epoch()

    def save_history(self):
        import json
        json.dump(self.history, open(self.history_path, 'w'))

    def update_curr_epoch(self):
        latest_cp: str = get_latest_checkpoint(self.checkpoint_dir, ext=".h5", head="cp-")
        self.current_epoch = int(latest_cp[latest_cp.find("cp-") + 3:latest_cp.find(".h5")])

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

    def load(self):
        self.load_weights()
        self.load_history()

    def save(self, epoch=0):
        self.save_weights(epoch)
        self.save_history()

    def build(self):
        raise NotImplementedError

    def compile(self):
        raise NotImplementedError

    def build_and_compile(self):
        self.model = self.build()
        self.model = self.compile()

        if len(os.listdir(self.checkpoint_dir)) == 0:
            self.save_weights(epoch=0)
        return self.model


class NetworkManagerCallback(tf.keras.callbacks.Callback):
    def __init__(self, network_manager: NetworkModelManager):
        super().__init__()
        self.network_manager = network_manager

    def on_epoch_end(self, epoch, logs=None):
        self.network_manager.current_epoch = epoch


class CatColorizer(NetworkModelManager):
    def __init__(self, depth_after_fusion=256, img_size=128, learning_rate=1e-3, **kwargs):
        super(CatColorizer, self).__init__(**kwargs)
        self.depth_after_fusion = depth_after_fusion
        self.img_size = img_size
        self.lr = learning_rate

        self.encoder_input = Input(shape=(self.img_size, self.img_size, 1), name="encoder_input_1c")
        self.inception_resnet_v2_model = tf.keras.applications.InceptionResNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        self.inception_resnet_v2_model.trainable = False
        self.initializer = GlorotNormal()

    def _build_encoder(self, encoder_input):
        # encoder
        x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, kernel_initializer=self.initializer,
                   name="conv2d_0")(encoder_input)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_1")(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer, strides=2,
                   name="conv2d_2")(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_3")(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   strides=2, name="conv2d_4")(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_5")(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_6")(x)
        encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                                name="encoder_output")(x)
        return encoder_output

    def _build_fusion(self, encoder_input, encoder_output):
        batch, height, width, channels = Kbackend.int_shape(encoder_output)

        # Fusion Layer
        if encoder_input.shape[-1] == 1:
            repeat_input = Kbackend.concatenate([encoder_input, encoder_input, encoder_input], axis=-1)
            embs = self.inception_resnet_v2_model(repeat_input)
        else:
            embs = self.inception_resnet_v2_model(encoder_input)

        embs = tf.keras.layers.GlobalAveragePooling2D()(embs)

        embs = RepeatVector(height * width)(embs)
        embs = Reshape((height, width, embs.shape[-1]))(embs)
        embs = concatenate([encoder_output, embs], axis=-1)
        return embs

    def _build_decoder(self, fusion_output):
        # decoder
        decoder_input = Conv2D(self.depth_after_fusion, (1, 1), activation='relu', kernel_initializer=self.initializer,
                               name="decoder_input")(fusion_output)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_7")(decoder_input)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_8")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_9")(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_10")(x)
        x = Conv2D(2, (3, 3), activation='tanh', padding='same', kernel_initializer=self.initializer,
                   name="conv2d_11")(x)
        decoder_output = UpSampling2D((2, 2), name="decoder_output")(x)
        return decoder_output

    def build(self):
        decoder_output = self._build_decoder(
            self._build_fusion(
                self.encoder_input, self._build_encoder(self.encoder_input)
            )
        )
        self.model = Model(self.encoder_input, decoder_output, name=self.name)
        return self.model

    def get_head_with_pretrained_weights(self, encoder_input, pretrained=True):
        head = self._build_fusion(
            encoder_input, self._build_encoder(encoder_input)
        )
        head_model = Model(encoder_input, head, name=self.name + "_head")

        if not pretrained:
            # for layer in head_model.layers:
            #     print(layer.name, layer.trainable)
            return head_model

        # load weights if pretrained is True
        latest_cp: str = get_latest_checkpoint(self.checkpoint_dir, ext=".h5", head="cp-")
        self.current_epoch = int(latest_cp[latest_cp.find("cp-") + 3:latest_cp.find(".h5")])
        head_model.load_weights(latest_cp, by_name=True, skip_mismatch=True)
        for layer in head_model.layers:
            if layer.name in encoder_input.name or layer.name == "conv2d_0":
                layer.trainable = True
            else:
                layer.trainable = False
            # print(layer.name, layer.trainable)
        return head_model

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


class CatBreedsClassifier(NetworkModelManager):
    def __init__(self, img_size=128, learning_rate=1e-3, output_size=1, **kwargs):
        super(CatBreedsClassifier, self).__init__(**kwargs)
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.depth_after_fusion = kwargs.get("depth_after_fusion", 256)
        self.pretrained_head = kwargs.get("pretrained_head", True)

        self.encoder_input = Input(shape=(self.img_size, self.img_size, 3), name="encoder_input_3c")
        self.cat_col_manager = kwargs.get(
            "cat_col_manager",
            CatColorizer(self.img_size, name=kwargs.get("cat_col_name", "CatColorizer"))
        )
        self.head = self.cat_col_manager.get_head_with_pretrained_weights(self.encoder_input,
                                                                          pretrained=self.pretrained_head)

    def build(self):
        self.model = Sequential(
            [
                self.head,
                Conv2D(self.depth_after_fusion, (1, 1), activation='relu'),
                Flatten(),
                BatchNormalization(),
                Dense(1024),
                BatchNormalization(),
                Dense(self.output_size),
                Softmax(),

            ]
        )

        return self.model

    def compile(self):
        assert self.model is not None
        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[
                'accuracy',
            ]
        )
        return self.model


if __name__ == '__main__':
    from tensorflow.keras.utils import plot_model

    col_model_manager = CatColorizer(img_size=80, name="CatColorizer")
    # col_model_manager.build()
    # print(col_model_manager.summary)
    # print(col_model_manager.model.get_weights()[-1])
    # col_model_manager.load_weights()
    # print(col_model_manager.model.get_weights()[-1])
    # col_model_manager.save_weights(col_model_manager.current_epoch)
    # col_model_manager.save_model()

    breeds_classifier_manager = CatBreedsClassifier(
        img_size=80,
        output_size=51,
        name="CatBreedsClassifier",
        cat_col_manager=col_model_manager,
        pretrained_head=False,
    )
    breeds_classifier_manager.build_and_compile()
    print(breeds_classifier_manager.summary)
    plot_model(breeds_classifier_manager.model, to_file="Figures/CatBreedsClassifier.png",
               show_layer_names=True, show_shapes=True)

    # new_history = {'loss': [1076.923828125], 'accuracy': [0.4869177043437958], 'mse': [0.010041841305792332], 'mae': [0.06346332281827927], 'mape': [1076.924072265625], 'val_loss': [330.6434020996094], 'val_accuracy': [0.7349596619606018], 'val_mse': [0.010745665989816189], 'val_mae': [0.06535135209560394], 'val_mape': [330.6432189941406]}
    # col_model_manager.load_history()
    # print(col_model_manager.history)
    # col_model_manager.update_history(new_history)
    # print(col_model_manager.history)

    # model.summary()
    # plot_model(model, to_file='Figures/CatColorizer.png', show_shapes=True)
