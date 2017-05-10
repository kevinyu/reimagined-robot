import numpy as np

import config
from shapes.properties import generate, N_objects
from glimpse import glimpse
from position_encoding import L
from utils import cartesian
from shapes.queries import label_at


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
    glimpses = np.zeros((n_glimpses, config.COLOR_CHANNELS * np.power(glimpse_width, 2)))

    # FIXME just hardcoding
    choices = cartesian(
            (15, config.IMG_WIDTH - 15),
            (15, config.IMG_HEIGHT - 15)
    )

    sample_locations = [choices[np.random.choice(len(choices))] for _ in range(n_glimpses)]

    for i, (x, y) in enumerate(sample_locations):
        glimpses[i] = glimpse(scene.img, x, y, glimpse_width).reshape(config.COLOR_CHANNELS * glimpse_width ** 2)

    return glimpses, np.array(sample_locations).T


def near_digits(scene, scene_labels, within=None):
    """Return a grid the same shape as image with 1's indicating they are near a digit"""
    result = np.zeros(scene.img.shape[:2])
    y, x = np.meshgrid(np.arange(scene.img.shape[0]), np.arange(scene.img.shape[1]))

    for _, center_x, center_y, _ in scene_labels:
        condition = (np.power(x - center_x, 2) + np.power(y - center_y, 2)) <= np.power(within, 2)
        result[np.where(condition)] = 1
    return result


def sample_near_digits(scene, scene_labels, n=1, within=13):
    """Return a sample of coordinates that are near digit locations"""
    choices = zip(*np.where(near_digits(scene, scene_labels, within=within)))
    picked = np.random.choice(len(choices), size=n)
    return [choices[pick] for pick in picked]
    

def take_samples(scene, scene_label, n_samples=1, strategy="smart", within=None):
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
    one_hot = np.zeros((n_samples * 2, N_objects + 1))

    choices = cartesian(scene.img.shape[0], scene.img.shape[1])
    # sample_locations = [choices[np.random.choice(len(choices))] for _ in range(n_samples)]
    sample_locations = sample_near_digits(scene, scene_label, n=n_samples, within=13)
    sample_locations += [choices[np.random.choice(len(choices))] for _ in range(n_samples)]

    for i, (x, y) in enumerate(sample_locations):
        one_hot[i] = label_at(scene_label, x, y)

    return one_hot, np.array(sample_locations).T


def make_one(glimpse_strategy=None):
    scene, scene_label = generate()

    # glimpse locs is 2 x n_glimpses
    glimpse_data, glimpse_locs = take_glimpses(
            scene,
            glimpse_width=config.GLIMPSE_WIDTH,
            n_glimpses=config.GLIMPSES,
            strategy=glimpse_strategy or config.GLIMPSE_STRATEGY)
    # sample locs is 2 x n_samples
    sample_data, sample_locs = take_samples(
            scene,
            scene_label,
            n_samples=config.SAMPLES,
            strategy=config.SAMPLE_STRATEGY,
            within=config.SAMPLE_RADIUS)

    return scene, glimpse_data, glimpse_locs.T, sample_data, sample_locs.T


def super_make_one(glimpse_strategy=None):
    scene, scene_label = generate()

    # glimpse locs is 2 x n_glimpses
    glimpse_data, glimpse_locs = take_glimpses(
            scene,
            glimpse_width=config.GLIMPSE_WIDTH,
            n_glimpses=config.GLIMPSES,
            strategy=glimpse_strategy or config.GLIMPSE_STRATEGY)
    # sample locs is 2 x n_samples
    sample_data, sample_locs = take_samples(
            scene,
            scene_label,
            n_samples=config.SAMPLES,
            strategy=config.SAMPLE_STRATEGY,
            within=config.SAMPLE_RADIUS)

    return scene, scene_label, glimpse_data, glimpse_locs.T, sample_data, sample_locs.T



def make_batch(n):
    datas = []
    for _ in range(n):
        datas.append(make_one())
    return [np.array(a) for a in zip(*datas)]


