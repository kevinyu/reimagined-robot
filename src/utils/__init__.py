import theano
import theano.tensor as T

import numpy as np

import config


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


def init_hypervectors(shape, zeros=False):
    if zeros:
        return theano.shared(float_x(np.zeros(shape)))
    else:
        return theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=shape)))


def tensor_softmax(t, axis=-1):
    t = t.dimshuffle(t.ndim - 1, *range(t.ndim - 1))
    original_shape = t.shape
    t = t.flatten(ndim=2)
    t = T.nnet.softmax(t.T).T.reshape(original_shape)
    return t.dimshuffle(*(range(1, t.ndim) + [0]))
