import numpy as np

import config
from datasets.mnist_scene import generate_n_digit_scene
from datasets.training_set import take_glimpses, take_samples


def make_one(glimpse_strategy=None):
    scene = generate_n_digit_scene(
            (config.IMG_WIDTH, config.IMG_HEIGHT),
            np.random.choice(config.N_DIGITS, p=config.P_DIGITS))
    if config.NOISE_FRAGMENTS:
        scene.add_fragment_noise(config.NOISE_FRAGMENTS, config.MAX_NOISE_SIZE)
    # glimpse locs is 2 x n_glimpses
    glimpse_data, glimpse_locs = take_glimpses(
            scene,
            glimpse_width=config.GLIMPSE_WIDTH,
            n_glimpses=config.GLIMPSES,
            strategy=glimpse_strategy or config.GLIMPSE_STRATEGY)
    # sample locs is 2 x n_samples
    sample_data, sample_locs = take_samples(
            scene,
            n_samples=config.SAMPLES,
            strategy=config.SAMPLE_STRATEGY,
            within=config.SAMPLE_RADIUS)
    return scene, glimpse_data, glimpse_locs.T, sample_data, sample_locs.T


def make_batch(n):
    datas = []
    for _ in range(n):
        datas.append(make_one())
    return [np.array(a) for a in zip(*datas)]


