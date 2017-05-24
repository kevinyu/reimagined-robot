import numpy as np
import theano

import config
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
    def _fn(gv_re, gv_im, gp_re, gp_im, s0_re, s0_im):
        return accumulate_glimpses(
                (s0_re, s0_im),
                (gv_re, gv_im),
                (gp_re, gp_im))

    return ComplexTuple(*theano.scan(
            fn=_fn,
            sequences=[
                glimpse_vectors.real,
                glimpse_vectors.imag,
                glimpse_positions.real,
                glimpse_positions.imag
            ],
            non_sequences=S_0)[0])


def glimpse(img, x, y, width):
    """Take a glimpse of image, with coordinates indicating center of glimpse

    Any portions glimpsed outside of scene frame will be set to zero
    """
    if width % 2 == 0:
        x_min, x_max = x - width / 2, x + width / 2
        y_min, y_max = y - width / 2, y + width / 2
    else:
        x_min, x_max = x - width / 2, x + 1 + width / 2
        y_min, y_max = y - width / 2, y + 1 + width / 2

    if img.ndim == 2:
        padded = np.zeros((img.shape[0] + width, img.shape[1] + width))
    elif img.ndim == 3:
        padded = np.zeros((img.shape[0] + width, img.shape[1] + width, 3))

    x1 = width / 2
    x2 = x1 + img.shape[0]
    y1 = width / 2
    y2 = y1 + img.shape[1]

    padded[x1:x2, y1:y2] = img

    return padded[x1 + x_min:x1 + x_max, y1 + y_min:y1 + y_max]


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

    locs = []
    for i, (x, y) in enumerate(scene.digit_locations[:n_glimpses]):
        # allow up to 10 pixels of jitter in either direction
        MAX_JITTER = config.GLIMPSE_JITTER
        if not MAX_JITTER:
            jittered_x = x
            jittered_y = y
        else:
            x_dist_to_low_edge = np.min([x, MAX_JITTER])
            x_dist_to_high_edge = np.min([scene.img.shape[0] - x, MAX_JITTER])
            y_dist_to_low_edge = np.min([y, MAX_JITTER])
            y_dist_to_high_edge = np.min([scene.img.shape[1] - y, MAX_JITTER])
            jittered_x = x + np.random.choice(
                    range(-x_dist_to_low_edge, x_dist_to_high_edge))
            jittered_y = y + np.random.choice(
                    range(-y_dist_to_low_edge, y_dist_to_high_edge))

        locs.append([jittered_x, jittered_y])
        glimpses[i] = glimpse(
                scene.img,
                jittered_x,
                jittered_y,
                glimpse_width
        ).reshape(config.GLIMPSE_SIZE)

    sample_locations = [
            [np.random.randint(scene.img.shape[0]), np.random.randint(scene.img.shape[1])]
            for _ in range(n_glimpses)]

    for i, (x, y) in enumerate(sample_locations[len(scene.digit_locations):]):
        locs.append([x, y])
        glimpses[len(locs) - 1] = glimpse(
                scene.img,
                x,
                y,
                glimpse_width
        ).reshape(config.GLIMPSE_SIZE)

    return glimpses, np.array(locs).T
