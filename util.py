from skimage import color
import numpy as np
from keras import backend as K

from copy import deepcopy


def normalize_l_channel(l_batch):
    return l_batch / 50 - 1


def unnormalize_l_channel(l_batch):
    return (l_batch + 1) * 50


def normalize_ab_channels(ab_batch):
    return ab_batch / 127


def unnormalize_ab_channels(ab_batch):
    return ab_batch * 127


def l_to_rgb(img_l):
    """
    Convert a numpy array (l channel) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.squeeze(255 * (img_l + 1) / 2)
    return color.gray2rgb(lab) / 255


def lab_to_rgb(img_l, img_ab):
    """
    Convert a pair of numpy arrays (l channel and ab channels) into an rgb image
    :param img_l:
    :return:
    """
    lab = np.empty([*img_l.shape[0:2], 3])
    lab[:, :, 0] = np.squeeze(((img_l + 1) * 50))
    lab[:, :, 1:] = img_ab * 127
    return color.lab2rgb(lab)


def l_and_ab_to_rgb(l_batch, ab_batch):
    lab_batch = np.zeros((l_batch.shape[0], l_batch.shape[1], l_batch.shape[2], 3))
    lab_batch[:, :, :, 0] = np.squeeze(l_batch)
    lab_batch[:, :, :, [1, 2]] = ab_batch
    rgb_batch = color.lab2rgb(lab_batch)
    return rgb_batch


def plotHistory(history: dict, **kwargs):
    import matplotlib.pyplot as plt
    # print(history.keys())
    acc = history['accuracy']
    val_acc = history.get('val_accuracy')

    loss = history['loss']
    val_loss = history.get('val_loss')

    epochs_range = range(1, len(acc)+1)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    if val_acc is not None:
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Training {"and Validation" if val_acc is not None else ""} Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if val_loss is not None:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Training {"and Validation" if val_loss is not None else ""} Loss')

    if kwargs.get("savefig", True):
        plt.savefig(f"Figures/{kwargs.get('savename', 'training_curve')}.png", dpi=500)
    plt.show()


def plot_cat_colorization_prediction_samples(model_manager, nb_samples=3, **kwargs):
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt

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
    l_batch = lab_batch[:, :, :, 0][..., np.newaxis]
    ab_batch = lab_batch[:, :, :, 1:]
    one_hot_batch = ab_batch2one_hot_batch(ab_batch, model_manager.gamut_instances, model_manager.bins)
    ab_batch_rebin = one_hot_batch2ab_batch(one_hot_batch, model_manager.gamut_instances, model_manager.bins)
    rgb_batch_rebin = l_and_ab_to_rgb(l_batch, ab_batch_rebin)

    rgb_batch_prediction = model_manager.predict_rgb(l_batch)

    fig, axes = plt.subplots(nb_samples, 4, figsize=(10, 10))
    for i in range(nb_samples):
        axes[i][0].imshow(tf.squeeze(l_batch[i]), cmap='gray')
        axes[i][1].imshow(rgb_batch_prediction[i])
        axes[i][2].imshow(rgb_batch_rebin[i])
        axes[i][3].imshow(rgb_batch[i])

        axes[i][0].axis('off')
        axes[i][1].axis('off')
        axes[i][2].axis('off')
        axes[i][3].axis('off')

    axes[0][0].set_title("Input")
    axes[0][1].set_title("Prediction")
    axes[0][2].set_title("Output expected")
    axes[0][3].set_title("Ground truth")

    plt.tight_layout(pad=0.0)
    if kwargs.get("savefig", True):
        plt.savefig(f"Figures/{kwargs.get('savename', f'{model_manager.name}_prediction_samples')}.png", dpi=500)
    plt.show()


def get_latest_checkpoint(checkpoint_dir, ext=".h5", head="cp-"):
    import os
    latest = None
    cp_max = -1
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(ext) and filename.startswith(head):
            cp = int(filename[filename.find(head) + len(head):filename.find(ext)])
            if cp > cp_max:
                cp_max = cp
                latest = filename
    if latest is None:
        return None
    return checkpoint_dir + "/" + latest


def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)


def ab_vector2one_hot(ab_vector, classes: list, bins: int) -> np.ndarray:
    # TODO: use dict
    ab_vector = deepcopy(ab_vector) + 110
    ab_vector = tuple((ab_vector//bins).astype(np.int32))
    try:
        cls = classes.index(ab_vector)
    except ValueError:
        nearest_ab_idx = int(np.argmin(np.sum(np.abs(np.array(classes) - ab_vector), axis=1)))
        nearest_ab = np.array(classes[nearest_ab_idx])
        cls = classes.index(tuple(nearest_ab))
    one_hot = np.zeros((len(classes, )), dtype=np.int32)
    one_hot[cls] = 1
    return one_hot


def one_hot2ab_vector(on_hot_vector: np.ndarray, classes: list, bins: int):
    cls = np.where(on_hot_vector == 1)[0][0]
    ab_vector = (np.array(classes[cls], dtype=np.int32) * bins) - 110
    return ab_vector


def ab_batch2one_hot_batch(ab_batch: np.ndarray, classes: list, bins: int):
    return np.apply_along_axis(ab_vector2one_hot, -1, ab_batch, *[classes, bins])


def one_hot_batch2ab_batch(one_hot_batch: np.ndarray, classes: list, bins: int):
    return np.apply_along_axis(one_hot2ab_vector, -1, one_hot_batch, *[classes, bins])


def probability_vector2one_hot_vector(probability_vector: np.ndarray) -> np.ndarray:
    one_hot_vector = np.zeros(probability_vector.shape, dtype=np.int32)
    one_hot_vector[np.argmax(probability_vector)] = 1
    return one_hot_vector


def probability_batch2one_hot_batch(probability_batch: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(probability_vector2one_hot_vector, -1, probability_batch)


def WeightedCategoricalCrossentropy(weights: np.ndarray):
    """
    Reference: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def _call(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = weights * y_true * K.log(y_pred)
        loss = -K.sum(loss, -1)
        return loss

    return _call


def normalize(vector: np.ndarray):
    norm = np.linalg.norm(vector)
    if norm == 0:
       return vector
    return vector / norm


