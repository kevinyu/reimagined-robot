import matplotlib
matplotlib.use('Agg')
import os

import numpy as np
import matplotlib.pyplot as plt
import config

from tasks.mnist.generate_scenes import make_one, make_batch
from tasks.mnist.parameters import D_table
from tasks.mnist.query_scene import save_directions, directions, D_directions, _direction_keys
from network import glimpse_network_output, glimpse_features, networks, predict
from parameters import save_params, S0
from position_encoding import L
from train import train
from utils.complex import ComplexTuple
from visualize import raster_dict


def plot_belief(filename_base):
    scene, glimpses, glimpse_xy, sample_xy, digit_labels, color_labels = make_one()
    glimpses = glimpses / 255.0

    # gather three glimpses
    S = S0.get_value()
    nnnn = config.GLIMPSES - 3
    for i in range(config.GLIMPSES - 3):
        g = ComplexTuple(*predict(glimpses[i][None]))
        S = (S + g * L.encode_numeric(config.POS_SCALE(glimpse_xy[i]))).reshape((1024,))

    for i in range(3):
        g = ComplexTuple(*predict(glimpses[nnnn + i][None]))
        S = (S + g * L.encode_numeric(config.POS_SCALE(glimpse_xy[nnnn + i]))).reshape((1024,))

        D_digits = D_table["Digits"].get_value()
        belief = raster_dict(D_digits.real, D_digits.imag, S.real, S.imag)

        plt.figure(figsize=(10, 14))
        for glimpse_id in range(i+1):
            plt.subplot(6, 3, 3 + glimpse_id + 1)
            plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH, 3))

        for digit_id in range(11):
            plt.subplot(6, 3, 6 + digit_id + 1)
            plt.imshow(belief[:, :, digit_id], vmin=0.0, vmax=1.0)
            plt.text(10, 10, "BG" if digit_id == 10 else str(digit_id), fontsize=20, color="white")

        plt.subplot(6, 3, 1)
        plt.imshow(scene.img.astype(np.uint8))
        for x, y in glimpse_xy[:nnnn + i+1]:
            bounds_x = config.GLIMPSE_ON(x)
            bounds_y = config.GLIMPSE_ON(y)
            plt.hlines(bounds_x, bounds_y[0], bounds_y[1], color="red")
            plt.vlines(bounds_y, bounds_x[0], bounds_x[1], color="red")

        # FIXME: why is x and y backwards... does this mess up our directions?
        for digit_id, y, x, _ in scene.contents:
            plt.scatter(x, y)
            plt.text(x+5, y+5, str(digit_id), color="white")

        plt.subplot(6, 3, 3)
        entropy = - np.sum(belief * np.log2(belief), axis=2)
        plt.imshow(entropy, vmin=0.0, vmax=np.log2(11.0))
        plt.text(10, 10, "entropy", fontsize=14, color="white")
        plt.savefig(filename_base.format(i), format="png", dpi=400)
        plt.close()


if __name__ == "__main__":
    # TODO Allow for loading a specific config file...
    for iteration in range(config.TRAINING_ITERATIONS):
        if config.TRAIN_TYPE == "query-based":
            _, glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels = make_batch(config.BATCH_SIZE)
        else:
            _, glimpses, glimpse_xy, sample_xy, digit_labels, color_labels = make_batch(config.BATCH_SIZE)

        glimpses = glimpses / 255.0
        a, b, c = glimpses.shape
        glimpses = glimpses.reshape(a * b, c)

        if config.TRAIN_TYPE == "query-based":
            cost = train(glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels)
        else:
            cost = train(glimpses, glimpse_xy, sample_xy, digit_labels, color_labels)
        print cost

        if iteration % config.SAVE_EVERY == 0:
            print "rendering"
            plot_belief(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster.png".format(iteration)))
            for net in networks:
                net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
            save_params()
            save_directions()

    for net in networks:
        net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
    save_params()
    save_directions()
