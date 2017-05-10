import theano
import theano.tensor as T

import numpy as np


def float_x(x):
    return np.asarray(x, dtype=theano.config.floatX)


def cartesian(*x):
    if isinstance(x[0], int):
        base = np.ones(x)
    else:
        starts, ends = zip(*x)
        base = np.zeros(ends)
        base[[slice(s, None) for s in starts]] = 1
    return np.array(zip(*np.where(base)))
