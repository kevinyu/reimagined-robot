import numpy as np
import theano

import config
from utils import float_x
from .encoder import PositionEncoder


if config.RANDOMIZE_POSITION_ENCODING:
    rs = np.random.RandomState(seed=config.SEED)

    K = theano.shared(float_x(
        rs.choice([-1, 1], size=(config.DIM, 2)) *
        rs.uniform(config.MIN_K, config.MAX_K, size=(config.DIM, 2))
    ))
else:
    nx = np.sqrt(config.DIM)
    assert (nx - int(nx)) == 0.0
    nx = int(nx)

    Kx = np.fft.fftfreq(nx, 1./float(nx))
    Kx, Ky = np.meshgrid(Kx, Kx)
    K = -1.0 * np.pi * np.array([Kx.flatten(), Ky.flatten()]).T / float(config.PLANCK_LENGTH)
    K = theano.shared(float_x(K))  # (N x 2)


L = PositionEncoder(K)
