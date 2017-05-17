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
        """Convert a spatial map of dimension (x, y, _) into a hypervector

        (first two dims must be spatial)

        Returns a hypervector array of dimension (N, _)
        """
        _, _, z  = X.shape
        result = fft_keepdims(X.dimshuffle(2, 0, 1)).dimshuffle(1, 2, 0)
        return result.reshape((config.DIM, z))

    def IFFT(self, Y):
        """Convert a hypervector (N, _) into a spatial map
        
        first dim must but hyperdim

        Returns a spatial map of dimension (x, y, _)
        """
        if Y.real.ndim == 1:
            Y = Y.reshape((Y.real.shape[0], 1))
        n = int(np.sqrt(config.DIM))
        _, z  = Y.real.shape
        result = fft_keepdims(Y.T.reshape((z, n, n)), inverse=True).dimshuffle(1, 2, 0)
        return result
