import os

import config

from tasks.mnist.generate_scenes import make_one, make_batch
from tasks.mnist.query_scene import directions, _direction_keys
from network import glimpse_network_output, predict, save_nets
from position_encoding import L
from train import train
from utils.complex import ComplexTuple
from visualize import plot_belief_digits, plot_belief_colors, plot_directions
from words import D_table, S0, save_params


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
            plot_belief_digits(os.path.join(config.SAVE_DIR, "digits_iter_{}_glimpse_{{}}.png".format(iteration)))
            plot_belief_colors(os.path.join(config.SAVE_DIR, "colors_iter_{}_glimpse_{{}}.png".format(iteration)))
            plot_directions(os.path.join(config.SAVE_DIR, "directions.png"))
            save_nets()
            save_params()

    plot_belief_digits(os.path.join(config.SAVE_DIR, "digits_iter_{}_glimpse_{{}}.png".format(iteration)))
    plot_belief_colors(os.path.join(config.SAVE_DIR, "colors_iter_{}_glimpse_{{}}.png".format(iteration)))
    plot_directions(os.path.join(config.SAVE_DIR, "directions.png"))
    save_nets()
    save_params()
