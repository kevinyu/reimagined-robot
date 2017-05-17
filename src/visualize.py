
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from network import predict
from position_encoding import L
from properties import Color
from raster import raster_scene, raster_dict
from tasks.mnist.query_scene import directions
from tasks.mnist.generate_scenes import make_one
from utils.complex import ComplexTuple
from words import D_table, S0


def plot_belief_digits(filename_base):
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
        belief = raster_scene(D_digits.real, D_digits.imag, S.real, S.imag)

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


def plot_belief_colors(filename_base):
    """Make 3 plots for the state of belief after each glimpse"""
    scene, glimpses, glimpse_xy, sample_xy, digit_labels, color_labels = make_one()
    glimpses = glimpses / 255.0

    S = S0.get_value()
    nnnn = config.GLIMPSES - 3
    for i in range(config.GLIMPSES - 3):
        g = ComplexTuple(*predict(glimpses[i][None]))
        S = (S + g * L.encode_numeric(config.POS_SCALE(glimpse_xy[i]))).reshape((1024,))

    for i in range(3):
        g = ComplexTuple(*predict(glimpses[nnnn + i][None]))
        S = (S + g * L.encode_numeric(config.POS_SCALE(glimpse_xy[nnnn + i]))).reshape((1024,))

        D_color = D_table["Color"].get_value()
        belief = raster_scene(D_color.real, D_color.imag, S.real, S.imag)

        plt.figure(figsize=(10, 14))

        for glimpse_id in range(i+1):
            plt.subplot(4, 3, 3 + glimpse_id + 1)
            plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH, 3))

        for color_id in range(5):
            plt.subplot(4, 3, 6 + color_id + 1)
            plt.imshow(belief[:, :, color_id], vmin=0.0, vmax=1.0)
            plt.text(10, 10, "BG" if color_id == 4 else Color.params[color_id], fontsize=20, color="white")

        plt.subplot(4, 3, 1)
        plt.imshow(scene.img.astype(np.uint8))
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


def plot_directions(filename):
    D_directions = D_table["Directions"].get_value()
    belief = raster_dict(D_directions.real, D_directions.imag)

    plt.figure(figsize=(10, 14))
    for i, direction in enumerate(directions):
        plt.subplot(2, 4, i + 1)
        plt.imshow(belief[:, :, i], vmin=0.0, vmax=1.0)
        plt.text(10, 10, direction, fontsize=20, color="white")

    plt.savefig(filename, format="png", dpi=400)
    plt.close()
