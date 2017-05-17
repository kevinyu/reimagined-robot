import numpy as np
import theano
import theano.tensor as T

import config
from utils.complex import ComplexTuple
from .fft import fft_keepdims


class PositionEncoder(object):
    def __init__(self, K):
        self.K = K
        # IMG_WIDTH x IMG_HEIGHT * DIM
    def encode(self, X):
        phi = T.dot(X, self.K.T)
        return ComplexTuple(T.cos(phi), T.sin(phi))

    def encode_numeric(self, X):
        phi = np.dot(X, self.K.get_value().T)
        return ComplexTuple(np.cos(phi), np.sin(phi))

    def FFT(self, X):
        """Convert a spatial map Y (n x m x ...) into a hypervector

        (first two dims must be spatial)
        """

        original_shape = X.shape  # (X, Y, ... )
        ndim = X.real.ndim
        X = X.flatten(ndim=3)     # (X, Y, a)
        result = fft_keepdims(X.dimshuffle(2, 0, 1))  # (a, X, Y) -> (a, sqrt(N), sqrt(N))

        # (sqrt(N), sqrt(N), a) -> (X, Y, ...) -> (N, ...)
        return result.dimshuffle(1, 2, 0).reshape(original_shape).reshape((config.DIM), ndim=ndim-1)

    def IFFT(self, Y):
        """Convert a hypervector (N x ...) into a spatial map
        
        first dim must but hyperdim
        """
        original_shape = Y.real.shape            # (N, ...)
        ndim = Y.real.ndim
        n = int(np.sqrt(config.DIM))
        Y.reshape((n, n), ndim=ndim + 1)  # (sqrt(N), sqrt(N), ...)
        Y = Y.flatten(ndim=3)                    # (sqrt(N), sqrt(N), a)
        Y = Y.dimshuffle(2, 0, 1)                # (a, sqrt(N), sqrt(N))
        result = fft_keepdims(Y.real, Y.imag, inverse=True)
        result = result[0]     # (a, X, Y)

        # (X, Y, a) -> (N, ...) -> (X, Y, ...)
        return result.dimshuffle(1, 2, 0).reshape(original_shape).reshape((n, n), ndim=ndim + 1)
   
