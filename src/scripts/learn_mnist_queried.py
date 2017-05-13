import os

import matplotlib
matplotlib.use('Agg')

import theano
import matplotlib.pyplot as plt
import numpy as np

import config
assert config.TASK == "MNIST"

from tasks.mnist.generate_scenes import make_one, make_batch
from network import glimpse_network_output, glimpse_features, networks
from parameters import S0, save_params
from position_encoding import L
from train import train
from visualize import raster
from utils.complex import ComplexTuple


if __name__ == "__main__":
    # TODO Allow for loading a specific config file...
    for iteration in range(config.TRAINING_ITERATIONS):
        _, glimpses, glimpse_xy, queries, digit_labels, color_labels = make_batch(config.BATCH_SIZE)
        glimpses = glimpses / 255.0

        a, b, c = glimpses.shape
        glimpses = glimpses.reshape(a * b, c)

        cost = train(glimpses, glimpse_xy, queries, digit_labels, color_labels)
        print cost
        if iteration % config.SAVE_EVERY == 0:
            print "rendering"
            for net in networks:
                net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
            save_params()

    for net in networks:
        net.save(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
    save_params()

