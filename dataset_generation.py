import numpy as np
from Dataset import CatColorizerDataset
import util
import matplotlib.pyplot as plt
from skimage import color
import tensorflow as tf

if __name__ == '__main__':
    BATCH_SIZE = 64
    IMG_SIZE = 80
    BINS = 10
    gamut_domain_shape = (220, 220)
    new_gamut_domain_shape = (gamut_domain_shape[0] // BINS, gamut_domain_shape[1] // BINS)
    gamut_domain = [-110, 110]
    shift_gamut_domain = abs(gamut_domain[0])

    col_dataset = CatColorizerDataset(img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.5)

    p = np.zeros(gamut_domain_shape, dtype=int)
    nb_tuple = 0

    l_batch_test, ab_batch_test = None, None

    for _ in range(col_dataset.number_of_train_call):
        l_batch, ab_batch_raw = next(col_dataset.train_data_gen)

        ab_batch = ab_batch_raw.astype(int) + shift_gamut_domain

        a_batch = ab_batch[:, :, :, 0]
        b_batch = ab_batch[:, :, :, 1]

        a_batch_flatten = a_batch.flatten()
        b_batch_flatten = b_batch.flatten()

        ab_zip = list(zip(a_batch_flatten, b_batch_flatten))
        nb_tuple += len(ab_zip)

        for t in ab_zip:
            p[t[0], t[1]] = p[t[0], t[1]] + 1

        l_batch_test = l_batch
        ab_batch_test = ab_batch_raw

    p_normalized = p / nb_tuple
    p_normalized_log = np.log(p_normalized)
    p_normalized_log_bins = util.rebin(p_normalized_log, new_gamut_domain_shape)
    p_normalized_bins = util.rebin(p_normalized, new_gamut_domain_shape)

    plt.title("p")
    plt.imshow(p, cmap="coolwarm", extent=[gamut_domain[0], gamut_domain[1]+1, gamut_domain[0], gamut_domain[1]+1])
    plt.colorbar()
    plt.xticks(range(gamut_domain[0], gamut_domain[1]+1, 10))
    plt.yticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.show()

    plt.title("log(p_normalized)")
    plt.imshow(np.log(p_normalized), cmap="coolwarm", extent=[gamut_domain[0], gamut_domain[1]+1, gamut_domain[0], gamut_domain[1]+1])
    plt.colorbar()
    plt.xticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.yticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.show()

    plt.title("P bins")
    plt.imshow(util.rebin(p, new_gamut_domain_shape), extent=[gamut_domain[0], gamut_domain[1]+1, gamut_domain[0], gamut_domain[1]+1])
    plt.colorbar()
    plt.xticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.yticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.show()

    plt.title("log(P) bins")
    plt.imshow(p_normalized_log_bins, extent=[gamut_domain[0], gamut_domain[1]+1, gamut_domain[0], gamut_domain[1]+1])
    plt.colorbar()
    plt.xticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.yticks(range(gamut_domain[0], gamut_domain[1] + 1, 10))
    plt.show()

    plt.imshow(util.l_and_ab_to_rgb(l_batch_test, ab_batch_test)[0])
    plt.axis('off')
    plt.show()

    cls_vector = list(zip(*np.where(p_normalized_bins > 0.0)))
    print(f"number of classes: {len(cls_vector)}")

    one_hot_batch = util.ab_batch2one_hot_batch(ab_batch_test, cls_vector, BINS)

    ab_batch_test_re = util.one_hot_batch2ab_batch(one_hot_batch, cls_vector, BINS)
    plt.imshow(util.l_and_ab_to_rgb(l_batch_test, ab_batch_test_re)[0])
    plt.axis('off')
    plt.show()


