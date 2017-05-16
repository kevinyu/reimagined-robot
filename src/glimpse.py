import numpy as np
import theano
import theano.tensor as T

import config
from utils import cartesian
from utils.complex import ComplexTuple


def accumulate_glimpses(S_0, glimpse_vectors, glimpse_positions):
    """Accumulate information into scene representation S over multiple glimpses

    Will build up and return intermediate states of the scene representation
    after each glimpse: {S_1, ... , S_n}

    Args:
    S_0 (complex tuple)
        0, 1: (T.fvector, N_hyper)
            initial scene representation (real, imag)
    glimpse_vectors (complex_tuple)
        0, 1: (T.fmatrix, N_glimpses x N_hyper)
            hypervector representations of glimpse contents (real, imag)
    glimpse_positions (complex_tuple)
        0, 1: (T.fmatrix, N_glimpses x N_hyper)
            hypervector representations of glimpse locations (real, imag)

    Returns:
    S (complex tuple)
        0, 1: (T.fmatrix, N_glimpses x N_hyper)
            real and imaginary parts of scene representations over glimpses
            each column of matrix is state of scene after n glimpses, S[n]
    """
    def add_glimpse(g_real, g_imag, r_real, r_imag, S_real, S_imag):
        g = ComplexTuple(g_real, g_imag)
        r = ComplexTuple(r_real, r_imag)
        S = ComplexTuple(S_real, S_imag)
        return S + g * r

    return ComplexTuple(*theano.scan(
            fn=add_glimpse,
            outputs_info=S_0,
            sequences=glimpse_vectors + glimpse_positions)[0])


def accumulate_glimpses_over_batch(S_0, glimpse_vectors, glimpse_positions):
    """Accumulate information into scene representation S over multiple glimpses

    Will build up and return intermediate states of the scene representation
    after each glimpse: {S_1, ... , S_n}

    Args:
    S_0 (complex tuple)
        0, 1: (T.fvector, N_hyper)
            initial scene representation (real, imag)
    glimpse_vectors (complex_tuple)
        0, 1: (T.fmatrix, N_batch_size x N_glimpses x N_hyper)
            hypervector representations of glimpse contents (real, imag)
    glimpse_positions (complex_tuple)
        0, 1: (T.fmatrix, N_batch_size x N_glimpses x N_hyper)
            hypervector representations of glimpse locations (real, imag)

    Returns:
    S (complex tuple)
        0, 1: (T.fmatrix, N_batch_size x N_glimpses x N_hyper)
            real and imaginary parts of scene representations over glimpses
            each column of matrix is state of scene after n glimpses, S[n]
    """
    return ComplexTuple(*theano.scan(
            fn=lambda gv_re, gv_im, gp_re, gp_im, s0_re, s0_im: accumulate_glimpses((s0_re, s0_im), (gv_re, gv_im), (gp_re, gp_im)),
            sequences=[
                glimpse_vectors.real,
                glimpse_vectors.imag,
                glimpse_positions.real,
                glimpse_positions.imag
            ],
            non_sequences=S_0)[0])
    

def glimpse(img, x, y, width):
    """Take a glimpse of image

    Try to pick an odd width; makes it easier on all of us
    """
    if width % 2 == 0:
        x_min, x_max = x - width / 2, x + width / 2
        y_min, y_max = y - width / 2, y + width / 2
    else:
        x_min, x_max = x - width / 2, x + 1 + width / 2
        y_min, y_max = y - width / 2, y + 1 + width / 2

    if x_min < 0 or x_max > img.shape[0] or y_min < 0 or y_max > img.shape[1]:
        raise Exception("Can't take glimpse of width {} at ({}, {}) for image size {}".format(
            width, x, y, img.shape))

    return img[x_min:x_max, y_min:y_max]


def take_glimpses(scene, glimpse_width, n_glimpses=1, strategy="smart"):
    """Build an array of glimpses from a scene

    Params:
    scene (mnist_scene.MNISTScene)
    glimpse_width (int, default=28 + 1)
        Width of glimpse window to use
    n_glimpses (int, default=1)
        Number of glimpses to take
    strategy (str, "smart" or "uniform")
        Take glimpses near digits, or just uniformly throughout image

    Returns:
    glimpses (np.array, n_glimpses x glimpse_width**2)
    glimpse_locations (np.array, n_glimpses x 2)
    """
    glimpses = np.zeros((n_glimpses, config.GLIMPSE_SIZE))

    # FIXME: technically this should vary based on glimpse_size (valid___max is fn of digit size)
    half_glimpse = glimpse_width / 2
    choices = cartesian(
        (half_glimpse, scene.img.shape[0] - half_glimpse - 1),
        (half_glimpse, scene.img.shape[1] - half_glimpse - 1)
    )

    # FIXME how to make it so we alwyas sample near digits
    '''
    near_digits = scene.near_digits(14)
    choices = [(x, y) for x, y in choices if
            near_digits[x + half_glimpse, y+half_glimpse] > 0]
    '''

    sample_locations = [choices[np.random.choice(len(choices))] for _ in range(n_glimpses)]

    locs = []
    for i, (x, y) in enumerate(scene.digit_locations[:n_glimpses]):
        locs.append([x, y])
        glimpses[i] = glimpse(scene.img, x, y, glimpse_width).reshape(config.GLIMPSE_SIZE)


    for i, (x, y) in enumerate(sample_locations[len(scene.digit_locations):]):
        locs.append([x, y])
        glimpses[len(locs) - 1] = glimpse(scene.img, x, y, glimpse_width).reshape(config.GLIMPSE_SIZE)

    return glimpses, np.array(locs).T


