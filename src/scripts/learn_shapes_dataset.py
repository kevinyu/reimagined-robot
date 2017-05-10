import os

import matplotlib
matplotlib.use('Agg')

import theano
import matplotlib.pyplot as plt
import numpy as np

import config
from shapes.properties import N_objects
from datagen.shapes_data import make_batch, make_one, super_make_one
from network import glimpse_network, glimpse_features
from parameters import S0, save_params, separated_D
from position_encoding import L
from train import train
from visualize import raster
from utils.complex import ComplexTuple

predict = theano.function(
        inputs=[glimpse_features],   # _ x N dimensional matrix
        outputs=glimpse_network.output,
        allow_input_downcast=True)


def plot_belief(filename_base):
    scene, scene_label, glimpses, glimpse_xy, sample_labels, sample_xy = super_make_one(glimpse_strategy="mixed")
    glimpses = glimpses # / 255.0

    S = S0.get_value()
    for i in range(5):
        g = ComplexTuple(*predict(glimpses[i][None]))
        S = (S + g * L.encode_numeric(glimpse_xy[i] / config.POS_SCALE)).reshape((1024,))
        belief = raster(S.real, S.imag)

        # Six rows x three columns
        # Row one: original image w/ glimpse boxes, and entropy
        # Row two: glimpses
        # Row three-six: probability map for each digit
        plt.figure(figsize=(14, 14))

        for glimpse_id in range(i+1):
            plt.subplot(6, 5, 5 + glimpse_id + 1)
            plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH, config.COLOR_CHANNELS))

        for digit_id in range(N_objects + 1):
            plt.subplot(6, 5, 10 + digit_id + 1)
            plt.imshow(belief[:, :, digit_id], vmin=0.0, vmax=1.0, cmap="afmhot")

        plt.subplot(6, 5, 1)
        plt.imshow(scene.img)
        for x, y in glimpse_xy[:i+1]:
            bounds_x = config.GLIMPSE_ON(x)
            bounds_y = config.GLIMPSE_ON(y)
            plt.hlines(bounds_x, bounds_y[0], bounds_y[1], color="red")
            plt.vlines(bounds_y, bounds_x[0], bounds_x[1], color="red")

        plt.subplot(6, 5, 1)
        plt.imshow(scene.img)
        plt.savefig(filename_base.format(i), format="png", dpi=400)
        plt.close()


def plot_prior(filename):
    S = S0.get_value()
    belief = raster(S.real, S.imag)
    plt.figure(figsize=(5, 6))
    plt.imshow(belief[:, :, -1], vmin=0.0, vmax=1.0)
    plt.savefig(filename, format="png", dpi=400)
    plt.close()




if __name__ == "__main__":
    # TODO Allow for loading a specific config file...
    for iteration in range(config.TRAINING_ITERATIONS):
        _, glimpses, glimpse_xy, sample_labels, sample_xy = make_batch(config.BATCH_SIZE)
        glimpses = glimpses # / 255.0

        a, b, c = glimpses.shape
        glimpses = glimpses.reshape(a * b, c)

        cost = train(glimpses, glimpse_xy, sample_xy, sample_labels)
        print cost

        if iteration % config.SAVE_EVERY == 0:
            print "rendering"
            plot_belief(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster.png".format(iteration)))
            glimpse_network.save(os.path.join(config.SAVE_DIR, "saved_params.npy"))
            plot_prior(os.path.join(config.SAVE_DIR, "s0_raster.png"))

    plot_belief(os.path.join(config.SAVE_DIR, "iter_{}_glimpse_{{}}_raster.png").format("final"))
    plot_prior(os.path.join(config.SAVE_DIR, "s0_raster.png"))
    glimpse_network.save(os.path.join(config.SAVE_DIR, "saved_params.npy"))
