import numpy as np
import matplotlib.pyplot as plt

import config

import theano
from datasets.mnist_scene import generate_n_digit_scene
from datagen.sampling import take_glimpses, take_samples
from network import glimpse_network, glimpse_features
from parameters import S0
from position_encoding import L
from visualize import raster
from utils.complex import ComplexTuple


predict = theano.function(
        inputs=[glimpse_features],   # _ x N dimensional matrix
        outputs=glimpse_network.output,
        allow_input_downcast=True)


CMAP= "afmhot" # "bone" # "afmhot"

def render_scene(scene):
    plt.figure()
    plt.imshow(scene.img, cmap="gray")
    plt.xticks([])
    plt.yticks([])


def render_belief(belief, cmap="bone"):
    plt.figure(figsize=(8, 10))
    for digit_id in range(11):
        plt.subplot(4, 3, digit_id + 1)
        plt.imshow(belief[:, :, digit_id], vmin=0.1, vmax=0.9, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.text(12, 20, "BG" if digit_id == 10 else str(digit_id), fontsize=30, color="white")
    plt.tight_layout()


def new_scene():
    scene = generate_n_digit_scene(
            (config.IMG_WIDTH, config.IMG_HEIGHT),
            np.random.choice(config.N_DIGITS, p=config.P_DIGITS),
            scale_range=config.SCALE_RANGE)

    scene.add_fragment_noise(config.NOISE_FRAGMENTS, config.MAX_NOISE_SIZE)

    # glimpse locs is 2 x n_glimpses
    glimpse_data, glimpse_locs = take_glimpses(
            scene,
            glimpse_width=config.GLIMPSE_WIDTH,
            n_glimpses=8,
            strategy="mixed")

    return scene, glimpse_data, glimpse_locs.T


def test_scene(scene, glimpses, glimpse_xy):
    plt.figure()
    plt.imshow(scene.img, cmap="gray")
    plt.xticks([])
    plt.yticks([])

    for i in range(len(glimpse_xy)):
        for x, y in glimpse_xy[i:i+1]:
            plt.figure()
            plt.imshow(scene.img, cmap="gray")

            plt.hlines([x - 12, x+12], y-12, y+12, colors="red")
            plt.vlines([y - 12, y+12], x-12, x+12, colors="red")
            plt.xticks([])
            plt.yticks([])
        plt.show()


    for i in range(len(glimpses)):
        plt.figure()
        plt.imshow(glimpses[i].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH), vmin=0.0, vmax=255.0, cmap="gray")
        plt.xticks([])
        plt.yticks([])
    plt.show()


def iterate_over_glimpses(scene, glimpses, glimpse_xy, cmap="afmhot"):
    S = S0.get_value()
    belief = raster(S.real, S.imag)
    render_belief(belief, cmap=cmap)

    for glimpse_id in range(len(glimpses)):
        g = ComplexTuple(*predict(glimpses[glimpse_id][None]))
        S = (S + g * L.encode_numeric(glimpse_xy[glimpse_id] / config.POS_SCALE)).reshape((1024,))
        belief = raster(S.real, S.imag)
        """
        plt.imshow(glimpses[glimpse_id].reshape(config.GLIMPSE_WIDTH, config.GLIMPSE_WIDTH), cmap="gray")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        """
        render_belief(belief, cmap=cmap)

