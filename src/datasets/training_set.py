"""
Code for generating training set

Combines mnist scene generation, glimpsing, and sampling
"""

import numpy as np

from glimpse import glimpse
from mnist_scene import generate_several_digit_scene
from position_encoding import L
from utils import cartesian


def take_glimpses(scene, glimpse_width=None, n_glimpses=1, strategy="smart"):
    """Build an array of glimpses from a scene

    Params:
    scene (mnist_scene.MNISTScene)
    glimpse_width (int, default=scene.DIGIT_SIZE + 1)
        Width of glimpse window to use
    n_glimpses (int, default=1)
        Number of glimpses to take
    strategy (str, "smart" or "uniform")
        Take glimpses near digits, or just uniformly throughout image

    Returns:
    glimpses (np.array, n_glimpses x glimpse_width**2)
    glimpse_locations (np.array, n_glimpses x 2)
    """
    if glimpse_width is None:
        glimpse_width = scene.DIGIT_SIZE + (0 if scene.DIGIT_SIZE % 2 else 1) # welp?

    glimpses = np.zeros((n_glimpses, np.power(glimpse_width, 2)))

    if strategy == "smart":
        sample_locations = scene.sample_near_digits(n=n_glimpses, padding=glimpse_width / 2)
    elif strategy == "uniform":
        # FIXME: technically this should vary based on glimpse_size (valid___max is fn of digit size)
        choices = cartesian(
                (scene.DIGIT_SIZE / 2, scene.valid_x_max + scene.DIGIT_SIZE / 2),
                (scene.DIGIT_SIZE / 2, scene.valid_y_max + scene.DIGIT_SIZE / 2)
        )
        sample_locations = [choices[np.random.choice(len(choices))] for _ in range(n_glimpses)]
    elif strategy == "mixed":
        choices = cartesian(
                (scene.DIGIT_SIZE / 2, scene.valid_x_max + scene.DIGIT_SIZE / 2),
                (scene.DIGIT_SIZE / 2, scene.valid_y_max + scene.DIGIT_SIZE / 2)
        )
        shuffler = [0, 1] if n_glimpses % 2 else [0, 0]
        np.random.shuffle(shuffler)
        sample_locations = (
                scene.sample_near_digits(n=n_glimpses/2 + shuffler[0], padding=glimpse_width / 2) +
                [choices[np.random.choice(len(choices))] for _ in range(n_glimpses/2 + shuffler[1])]
        )
        np.random.shuffle(sample_locations)

    for i, (x, y) in enumerate(sample_locations):
        glimpses[i] = glimpse(scene.img, x, y, glimpse_width).reshape(glimpse_width ** 2)

    return glimpses, np.array(sample_locations).T


def take_samples(scene, n_samples=1, strategy="smart", within=None):
    """Sample one hot vectors at various points in a scene

    Params:
    scene (mnist_scene.MNISTScene)
    n_samples (int, default=1)
        Number of samples to take
    strategy (str, "smart" or "uniform")
        Take samples near digits only, or just uniformly throughout image

    Returns:
    one_hot (np.array, n_samples x 11)
        one hot vectors corresponding to digit identities in image
    sample_locations (np.array, n_samples x 2)
    """
    one_hot = np.zeros((n_samples, 11))

    if strategy == "smart":
        sample_locations = scene.sample_near_digits(n=n_samples, within=within)
    elif strategy == "uniform":
        choices = cartesian(scene.img.shape[0], scene.img.shape[1])
        sample_locations = [choices[np.random.choice(len(choices))] for _ in range(n_samples)]
    elif strategy == "mixed":
        choices = cartesian(scene.img.shape[0], scene.img.shape[1])
        odd = n_samples % 2
        sample_locations = (
                scene.sample_near_digits(n=n_samples/2 + (1 if odd else 0), within=within) +
                [choices[np.random.choice(len(choices))] for _ in range(n_samples/2)]
        )

    for i, (x, y) in enumerate(sample_locations):
        one_hot[i] = scene.get_label_at(x, y)

    return one_hot, np.array(sample_locations).T


def generate_training_set(
        img_shape=(64, 64),
        n_scenes=1000,
        n_glimpses=6,
        n_samples=400,
        min_digits=1,
        max_digits=2,
        glimpse_strategy="smart",
        sample_strategy="smart",
    ):

    datas = []
    for i in range(n_scenes):
        scene = generate_several_digit_scene(img_shape, (min_digits, max_digits + 1))
        glimpse_data, glimpse_locs = take_glimpses(scene, n_glimpses=n_glimpses, strategy=glimpse_strategy)
        sample_data, sample_locs = take_samples(scene, n_samples=n_samples, strategy=sample_strategy)

        # map the location data to the hypervector space before saving, and scale it to (0, 1)
        r_glimpse_real, r_glimpse_imag = L(glimpse_locs / np.max(img_shape))
        r_sample_real, r_sample_imag = L(sample_locs / np.max(img_shape))

        datas.append((
            scene.img,
            glimpse_data,
            glimpse_locs.T,
            sample_data,
            sample_locs.T,
            r_glimpse_real.T,
            r_glimpse_imag.T,
            r_sample_real.T,
            r_sample_imag.T,
        ))
        if i % 100 == 0:
            print "Generated {} scenes".format(i)
    return np.array(datas)


if __name__ == "__main__":
    import sys
    target = sys.argv[1]
    result = generate_training_set(
            img_shape=(64, 64),
            n_scenes=3000,
            n_glimpses=3,
            n_samples=200,
            max_digits=2,
            sample_strategy="uniform")
    np.save(target, result)


