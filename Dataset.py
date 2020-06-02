import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import color, transform
from PIL import Image, ImageFile
from scipy.special import softmax as scipy_softmax
import util
import warnings
import hyperparameters
from hyperparameters import SEED
ImageFile.LOAD_TRUNCATED_IMAGES = True

tf.random.set_seed(SEED)


class ColorizerDataGenerator:
    def __init__(self, image_data_generator, **kwargs):
        self.image_data_generator = image_data_generator
        self.virtual_length = kwargs.get("virtual_length", None)

        self.bins = kwargs.get("bins")
        self.classes = kwargs.get("classes")

    @property
    def labels(self):
        return self.image_data_generator.labels

    def __len__(self):
        return len(self.image_data_generator) if self.virtual_length is None else self.virtual_length

    def __call__(self):
        while True:
            rgb_batch = self.image_data_generator.next()
            lab_batch = color.rgb2lab(rgb_batch).astype(np.float32)

            l_batch = lab_batch[:, :, :, 0][..., np.newaxis]
            ab_batch = lab_batch[:, :, :, 1:]

            one_hot_batch = util.ab_batch2one_hot_batch(ab_batch, self.classes, self.bins)
            yield l_batch, one_hot_batch

    def __next__(self):
        return next(self())

    def __iter__(self):
        return self()

    def get_l_ab(self):
        while True:
            rgb_batch = self.image_data_generator.next()
            lab_batch = color.rgb2lab(rgb_batch).astype(np.float32)
            # l_batch = util.normalize_l_channel(lab_batch[:, :, :, 0][..., np.newaxis])
            # ab_batch = util.normalize_ab_channels(lab_batch[:, :, :, 1:])
            # yield l_batch, ab_batch

            l_batch = lab_batch[:, :, :, 0][..., np.newaxis]
            ab_batch = lab_batch[:, :, :, 1:]
            yield l_batch, ab_batch

    def set_bins(self, bins: int):
        self.bins = bins

    def set_classes(self, classes: list):
        self.classes = classes

    def next(self):
        return next(self())


class CatBreedsClassifierDataGenerator:
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
            yield self.image_data_generator.next()

    def __next__(self):
        return next(self())

    def __iter__(self):
        return self()

    def next(self):
        return next(self())


class CatColorizerDataset:
    _URL = r"https://www.kaggle.com/ma7555/cat-breeds-dataset/download/"
    _DIR = r'C:\Users\gince\Documents\GitHub\Cat-Breeds-Classification\Data\cat-breeds-dataset/images'

    BATCH_SIZE = hyperparameters.BATCH_SIZE
    IMG_SIZE = hyperparameters.IMG_SIZE  # must be more than 80, 80 is recommended
    VAL_SPLIT = hyperparameters.COL_VAL_SPLIT
    GAMUT_SIZE = hyperparameters.GAMUT_SIZE  # number of gamut classes

    BINS = hyperparameters.BINS
    GAMUT_DOMAIN_SHAPE = (220, 220)
    NEW_GAMUT_DOMAIN_SHAPE = (GAMUT_DOMAIN_SHAPE[0] // BINS, GAMUT_DOMAIN_SHAPE[1] // BINS)
    GAMUT_DOMAIN = [-110, 110]
    SHIFT_GAMUT_DOMAIN = abs(GAMUT_DOMAIN[0])

    def __init__(self, **kwargs):
        self.IMG_SIZE = kwargs.get("img_size", CatColorizerDataset.IMG_SIZE)
        self.BATCH_SIZE = kwargs.get("batch_size", CatColorizerDataset.BATCH_SIZE)

        # self.path_to_zip = tf.keras.utils.get_file('cat-breeds-col_dataset.zip', origin=self._URL, extract=True)
        # print(f"Data saved to: {self.path_to_zip}")
        # self.PATH = os.path.join(os.path.dirname(self.path_to_zip), '/cat-breeds-dataset/images')
        self.PATH = kwargs.get("path", self._DIR)
        self.use_augmented_data = kwargs.get("use_augmented_data", True)
        self.VAL_SPLIT = kwargs.get("val_split", CatColorizerDataset.VAL_SPLIT)

        self.GAMUT_SIZE = kwargs.get("gamut_size", CatColorizerDataset.GAMUT_SIZE)
        self.BINS = kwargs.get("bins", CatColorizerDataset.BINS)
        self.NEW_GAMUT_DOMAIN_SHAPE = (self.GAMUT_DOMAIN_SHAPE[0] // self.BINS, self.GAMUT_DOMAIN_SHAPE[1] // self.BINS)

        img_gen_base_args = {
            "rescale": 1. / 255,
            "validation_split": self.VAL_SPLIT
        }

        img_gen_augmented_args = {
            "rotation_range": 45,
            "width_shift_range": .15,
            "height_shift_range": .15,
            "horizontal_flip": True,
            "zoom_range": 0.5,
        } if self.use_augmented_data else {}

        args = {**img_gen_base_args, **img_gen_augmented_args}
        self.image_generator = ImageDataGenerator(**args)  # Generator for our training data

        self.train_data_gen = ColorizerDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                shuffle=True,
                seed=hyperparameters.SEED,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode=None,
                subset="training"
            )
        )

        val_virtual_length = kwargs.get("val_virtual_length", None)
        if val_virtual_length is not None:
            val_virtual_length //= self.BATCH_SIZE

        self.val_data_gen = ColorizerDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode=None,
                seed=hyperparameters.SEED,
                subset="validation"
            ),
            virtual_length=val_virtual_length
        ) if self.VAL_SPLIT > 0.0 else None

        self.gamut_probabilities = np.zeros(self.GAMUT_DOMAIN_SHAPE, dtype=np.int32)
        self.gamut_probabilities_rebin = None
        self.nb_ab_instance = 0
        self.gamut_instances = None
        self.instance_probabilities = None

        self._initialize_data_gen()

    @property
    def train_length(self):
        return len(self.train_data_gen)

    @property
    def val_length(self):
        return len(self.val_data_gen)

    def __len__(self):
        return self.train_length + self.val_length

    @property
    def labels(self):
        return set(list(self.train_data_gen.labels)+list(self.val_data_gen.labels))

    @property
    def number_of_train_call(self):
        return self.train_length // self.BATCH_SIZE

    @property
    def number_of_val_call(self):
        return self.val_length // self.BATCH_SIZE

    def _initialize_data_gen(self):
        for _ in range(self.number_of_train_call):
            l_batch, ab_batch_raw = next(self.train_data_gen.get_l_ab())

            ab_batch = ab_batch_raw.astype(np.int32) + self.SHIFT_GAMUT_DOMAIN

            a_batch = ab_batch[:, :, :, 0]
            b_batch = ab_batch[:, :, :, 1]

            a_batch_flatten = a_batch.flatten()
            b_batch_flatten = b_batch.flatten()

            ab_zip = list(zip(a_batch_flatten, b_batch_flatten))
            self.nb_ab_instance += len(ab_zip)

            for t in ab_zip:
                self.gamut_probabilities[t[0], t[1]] = self.gamut_probabilities[t[0], t[1]] + 1

        self.gamut_probabilities = self.gamut_probabilities / self.nb_ab_instance
        self.gamut_probabilities_rebin = util.rebin(self.gamut_probabilities, self.NEW_GAMUT_DOMAIN_SHAPE)
        self.gamut_instances = self._create_gamut_instances(self.gamut_probabilities_rebin)

        assert len(self.gamut_instances) == self.GAMUT_SIZE
        self.train_data_gen.set_bins(self.BINS)
        self.train_data_gen.set_classes(self.gamut_instances)
        if self.val_data_gen is not None:
            self.val_data_gen.set_bins(self.BINS)
            self.val_data_gen.set_classes(self.gamut_instances)

    def _create_gamut_instances(self, gamut_probabilities_rebin):
        self.gamut_instances = list(zip(*np.where(gamut_probabilities_rebin > 0.0)))
        self.gamut_instances = sorted(self.gamut_instances, key=lambda t: gamut_probabilities_rebin[t[0], t[1]])

        if len(self.gamut_instances) < self.GAMUT_SIZE:
            warnings.warn(f"The GAMUT_SIZE is reduce to {len(self.gamut_instances)} cause of the gamut probabilities")
            self.GAMUT_SIZE = len(self.gamut_instances)

        nb_to_remove = len(self.gamut_instances) - self.GAMUT_SIZE
        for _ in range(nb_to_remove):
            self.gamut_instances.pop(0)

        self.instance_probabilities = scipy_softmax(np.array([self.gamut_probabilities_rebin[t]
                                                              for t in self.gamut_instances]))

        return self.gamut_instances

    def get_gamut_params(self):
        return self.gamut_instances, self.instance_probabilities

    def show_gamut_probabilities(self, **kwargs):
        rebin = kwargs.get("rebin", False)
        log = kwargs.get("log", True)

        p_raw = self.gamut_probabilities_rebin if rebin else self.gamut_probabilities
        p = np.log(p_raw) if log else p_raw

        title = f"{'$log$('if log else ''}Gamut Probabilities" \
                f"{f' rebin with bins: {self.BINS}' if rebin else ''}" \
                f"{')' if log else ''}"
        plt.figure(figsize=(12, 12))
        plt.title(title)
        plt.imshow(p, cmap="coolwarm", extent=[self.GAMUT_DOMAIN[0], self.GAMUT_DOMAIN[1] + 1,
                                               self.GAMUT_DOMAIN[0], self.GAMUT_DOMAIN[1] + 1])
        plt.colorbar()
        plt.xticks(range(self.GAMUT_DOMAIN[0], self.GAMUT_DOMAIN[1] + 1, 10))
        plt.yticks(range(self.GAMUT_DOMAIN[0], self.GAMUT_DOMAIN[1] + 1, 10))

        if kwargs.get("savefig", True):
            plt.savefig(f"Figures/gamut_prob"
                        f"{f'_rebinOf{self.BINS}'if rebin else ''}"
                        f"{'_log' if log else ''}"
                        f".png", dpi=500)
        plt.show()

    def plot_samples(self, nb_samples=5):
        nb_samples = min(nb_samples, self.BATCH_SIZE)
        l_batch, ab_batch = next(self.train_data_gen.get_l_ab())

        one_hot_batch = util.ab_batch2one_hot_batch(ab_batch, self.gamut_instances, self.BINS)
        ab_batch_rebin = util.one_hot_batch2ab_batch(one_hot_batch, self.gamut_instances, self.BINS)

        rgb_batch_rebin = util.l_and_ab_to_rgb(l_batch, ab_batch_rebin)
        rgb_batch = util.l_and_ab_to_rgb(l_batch, ab_batch)

        fig, axes = plt.subplots(nb_samples, 3, figsize=(10, 10))
        for i in range(nb_samples):
            axes[i][0].imshow(tf.squeeze(l_batch[i]), cmap='gray')
            axes[i][1].imshow(rgb_batch_rebin[i])
            axes[i][2].imshow(rgb_batch[i])

            axes[i][0].axis('off')
            axes[i][1].axis('off')
            axes[i][2].axis('off')

        axes[0][0].set_title("Input")
        axes[0][1].set_title("Expected output")
        axes[0][2].set_title("Real image")

        plt.tight_layout(pad=0.0)
        plt.savefig(f"Figures/samples_of_colorization_dataset.png", dpi=500)
        plt.show()


class CatColorizerOverfittingDataset(CatColorizerDataset):
    def __init__(self, **kwargs):
        self._DIR = "training_data/overfitted_data/"
        self.VAL_SPLIT = 0.0
        kwargs["val_split"] = self.VAL_SPLIT
        kwargs["use_augmented_data"] = False
        super().__init__(**kwargs)


class CatBreedsClassifierDataset:
    _URL = r"https://www.kaggle.com/ma7555/cat-breeds-dataset/download/"
    _DIR = r'C:\Users\gince\Documents\GitHub\Cat-Breeds-Classification\Data\cat-breeds-dataset/'

    BATCH_SIZE = 256
    IMG_SIZE = 80  # must be more than 80, 80 is recommended
    VAL_SPLIT = hyperparameters.CLS_VAL_SPLIT

    def __init__(self, **kwargs):
        self.IMG_SIZE = kwargs.get("img_size", CatColorizerDataset.IMG_SIZE)
        self.BATCH_SIZE = kwargs.get("batch_size", CatColorizerDataset.BATCH_SIZE)

        # self.path_to_zip = tf.keras.utils.get_file('cat-breeds-col_dataset.zip', origin=self._URL, extract=True)
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

        self.train_data_gen = CatBreedsClassifierDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                shuffle=True,
                seed=hyperparameters.SEED,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode="categorical",
                subset="training"
            )
        )

        self.val_data_gen = CatBreedsClassifierDataGenerator(
            self.image_generator.flow_from_directory(
                batch_size=self.BATCH_SIZE,
                directory=self.PATH,
                target_size=(self.IMG_SIZE, self.IMG_SIZE),
                class_mode="categorical",
                seed=hyperparameters.SEED,
                subset="validation"
            ),
            virtual_length=hyperparameters.CLS_VAL_VIRTUAL_LENGTH // self.BATCH_SIZE
        )

    @property
    def nb_cls(self):
        return len(self.labels)

    @property
    def train_length(self):
        return len(self.train_data_gen)

    @property
    def val_length(self):
        return len(self.val_data_gen)

    @property
    def labels(self):
        return set(list(self.train_data_gen.labels) + list(self.val_data_gen.labels))

    @property
    def number_of_train_call(self):
        return self.train_length // self.BATCH_SIZE

    @property
    def number_of_val_call(self):
        return self.val_length // self.BATCH_SIZE

    def plot_samples(self, nb_samples=5):
        nb_samples = min(nb_samples, self.BATCH_SIZE)
        rgb_batch, lbl_batch = next(self.train_data_gen)

        fig, axes = plt.subplots(1, nb_samples, figsize=(10, 10))
        for i in range(nb_samples):
            axes[i].imshow(tf.squeeze(rgb_batch[i]))
            axes[i].set_title(str(np.argmax(lbl_batch[i])))
            axes[i].axis('off')

        plt.tight_layout(pad=0.0)
        plt.show()


if __name__ == '__main__':
    col_dataset = CatColorizerDataset(img_size=80, batch_size=64, gamut_size=50)
    col_dataset.plot_samples(5)
    col_dataset.show_gamut_probabilities(rebin=False, log=True, savefig=False)
    col_dataset.show_gamut_probabilities(rebin=True, log=True, savefig=False)

    # cls_dataset = CatBreedsClassifierDataset(img_size=256, batch_size=100)
    # cls_dataset.plot_samples(5)
    # print(f"cls_dataset.nb_cls: {cls_dataset.nb_cls}")

