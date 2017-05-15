import os

import matplotlib
matplotlib.use('Agg')

import theano
import matplotlib.pyplot as plt
import numpy as np

import config
assert config.TASK == "MNIST"

from tasks.mnist.generate_scenes import make_one, make_batch
from tasks.mnist.parameters import D_table
from tasks.mnist.query_scene import save_directions, directions, D_directions, _direction_keys
from network import glimpse_network_output, glimpse_features, networks
from parameters import S0, save_params
from position_encoding import L
from train_queries import train
from visualize import raster
from utils.complex import ComplexTuple
from position_encoding import get_U
from properties import Color


predict = theano.function(
        inputs=[glimpse_features],   # _ x N dimensional matrix
        outputs=glimpse_network_output,
        allow_input_downcast=True)


def plot_belief_colors(filename_base):
    """Make 3 plots for the state of belief after each glimpse"""
    scene, glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels = make_one()
    glimpses = glimpses / 255.0

    S = S0.get_value()
    nnnn = config.GLIMPSES - 3
    for i in range(config.GLIMPSES - 3):
        g = ComplexTuple(*predict(glimpses[i][None]))
        S = (S + g * L.encode_numeric(2 * (glimpse_xy[i] / config.POS_SCALE) - 1)).reshape((1024,))

    for i in range(3):
        g = ComplexTuple(*predict(glimpses[nnnn + i][None]))
        S = (S + g * L.encode_numeric(2 * (glimpse_xy[nnnn + i] / config.POS_SCALE) - 1)).reshape((1024,))

        D_color = D_table["Color"].get_value()
        belief = raster(D_color.real, D_color.imag, S.real, S.imag)

        plt.figure(figsize=(10, 14))

        for glimpse_id in range(i+1):
            plt.subplot(4, 3, 3 + glimpse_id + 1)
            plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH, 3))

        for color_id in range(5):
            plt.subplot(4, 3, 6 + color_id + 1)
            plt.imshow(belief[:, :, color_id], vmin=0.0, vmax=1.0)
            plt.text(10, 10, "BG" if color_id == 4 else Color.params[color_id], fontsize=20, color="white")

        plt.subplot(4, 3, 1)
        plt.imshow(scene.img / 255.0)
        for x, y in glimpse_xy[:nnnn + i+1]:
            bounds_x = config.GLIMPSE_ON(x)
            bounds_y = config.GLIMPSE_ON(y)
            plt.hlines(bounds_x, bounds_y[0], bounds_y[1], color="red")
            plt.vlines(bounds_y, bounds_x[0], bounds_x[1], color="red")

        # FIXME: why is x and y backwards... does this mess up our directions?
        for _, y, x, props in scene.contents:
            color_name = Color.params[props[0][1]]
            plt.scatter(x, y)
            plt.text(x+5, y+5, color_name, color="white")

        plt.subplot(6, 3, 3)
        entropy = - np.sum(belief * np.log2(belief), axis=2)
        plt.imshow(entropy, vmin=0.0, vmax=np.log2(11.0))
        plt.text(10, 10, "entropy", fontsize=14, color="white")
        plt.savefig(filename_base.format(i), format="png", dpi=400)
        plt.close()




def plot_belief(filename_base):
    """Make 3 plots for the state of belief after each glimpse"""
    scene, glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels = make_one()
    glimpses = glimpses / 255.0

    S = S0.get_value()
    nnnn = config.GLIMPSES - 3
    for i in range(config.GLIMPSES - 3):
        g = ComplexTuple(*predict(glimpses[i][None]))
        S = (S + g * L.encode_numeric(2 * (glimpse_xy[i] / config.POS_SCALE) - 1)).reshape((1024,))

    for i in range(3):
        g = ComplexTuple(*predict(glimpses[nnnn + i][None]))
        S = (S + g * L.encode_numeric(2 * (glimpse_xy[nnnn + i] / config.POS_SCALE) - 1)).reshape((1024,))

        D_digits = D_table["Digits"].get_value()
        belief = raster(D_digits.real, D_digits.imag, S.real, S.imag)

        # Six rows x three columns
        # Row one: original image w/ glimpse boxes, and entropy
        # Row two: glimpses
        # Row three-six: probability map for each digit
        plt.figure(figsize=(10, 14))

        for glimpse_id in range(i+1):
            plt.subplot(6, 3, 3 + glimpse_id + 1)
            plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH, 3))

        for digit_id in range(11):
            plt.subplot(6, 3, 6 + digit_id + 1)
            plt.imshow(belief[:, :, digit_id], vmin=0.0, vmax=1.0)
            plt.text(10, 10, "BG" if digit_id == 10 else str(digit_id), fontsize=20, color="white")

        plt.subplot(6, 3, 1)
        plt.imshow(scene.img / 255.0)
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


def plot_prior(filename):
    D_digits = D_table["Digits"].get_value()
    S = S0.get_value()
    belief = raster(D_digits.real, D_digits.imag, S.real, S.imag)
    plt.figure(figsize=(10, 14))

    for digit_id in range(11):
        plt.subplot(4, 3, digit_id + 1)
        plt.imshow(belief[:, :, digit_id], vmin=0.0, vmax=1.0)
        plt.text(10, 10, "BG" if digit_id == 10 else str(digit_id), fontsize=20, color="white")

    plt.savefig(filename, format="png", dpi=400)
    plt.close()


if __name__ == "__main__":
    # TODO Allow for loading a specific config file...
    for iteration in range(config.TRAINING_ITERATIONS):
        _, glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels = make_batch(config.BATCH_SIZE)
        glimpses = glimpses / 255.0

        a, b, c = glimpses.shape
        glimpses = glimpses.reshape(a * b, c)

        cost = train(glimpses, glimpse_xy, query_directions, query_digits, query_colors, digit_labels, color_labels)
        print cost
        print get_U()

        if iteration % config.SAVE_EVERY == 0:
            print "rendering"
            plot_belief(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster.png".format(iteration)))
            plot_belief_colors(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster_colors.png".format(iteration)))
            plot_prior(os.path.join(config.SAVE_DIR, "s0_raster.png"))
            # plot_directions(os.path.join(config.SAVE_DIR, "directions_raster.png"))
            for net in networks:
                net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
            save_params()
            save_directions()

    for net in networks:
        net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
    save_params()
    save_directions()
    plot_belief(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster.png".format(iteration)))
    plot_belief_colors(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster_colors.png".format(iteration)))
    # plot_directions(os.path.join(config.SAVE_DIR, "directions_raster.png"))
    plot_prior(os.path.join(config.SAVE_DIR, "s0_raster.png"))

