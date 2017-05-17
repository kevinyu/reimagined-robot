import numpy as np
import theano
import theano.tensor as T
import theano.gpuarray.fft as Tfft

import config
from utils import float_x
from utils.complex import ComplexTuple
from utils.unitary import U, get_U



def fft_keepdims(xr, xi=None, inverse=False):
    '''
    behaves just like numpy.fft.fft2 or numpy.fft.ifft2
    :param xr, xi: real and imaginary inputs. Shape is assumed to be (batchsize, x, y)
    :return: real and imaginary parts of , both of shape (batchsize, x, y)
    '''
    if xi is None:
        rfft = Tfft.curfft(xr, norm='ortho') #(batch, x, y, 2)
        rfft_rev=rfft[:, :, ::-1, :]
        rfft_rev=T.roll(rfft_rev[:, ::-1, :, :], 1, axis=1)
        rfft_r = T.concatenate([rfft[:, :, :, 0], rfft_rev[:, :, 1:-1, 0]], axis=2)
        rfft_i = T.concatenate([rfft[:, :, :, 1], -rfft_rev[:, :, 1:-1, 1]], axis=2)
        return rfft_r, rfft_i
    else:
        xfrr, xfri = fft_keepdims(xr)
        if inverse:
            xi*=-1.
        xfir, xfii = fft_keepdims(xi)
        if inverse:
            return xfrr-xfii, -(xfri+xfir)
        else:
            return xfrr-xfii, xfri+xfir



if config.RANDOMIZE_POSITION_ENCODING:
    rs = np.random.RandomState(seed=config.SEED)

    K = theano.shared(float_x(
        rs.choice([-1, 1], size=(config.DIM, 2)) *
        rs.uniform(config.MIN_K, config.MAX_K, size=(config.DIM, 2))
    ))
else:
    nx=np.sqrt(config.DIM)
    assert (nx-int(nx)) == 0.0
    nx = int(nx)

    Kx = np.fft.fftfreq(nx, 1./float(nx))
    Kx, Ky = np.meshgrid(Kx,Kx)
    K = -1.0 * np.pi * np.array([Kx.flatten(), Ky.flatten()]).T / float(config.PLANCK_LENGTH)

    K=theano.shared(float_x(K.T))


class PositionEncoder(object):
    def __init__(self, K):
        self.K = K

    def encode(self, X):
        # X = T.dot(X, U)
        phi = T.dot(X, self.K.T)
        return ComplexTuple(T.cos(phi), T.sin(phi))

    def encode_numeric(self, X):
        # X = np.dot(X, get_U())
        phi = np.dot(X, self.K.get_value().T)
        return ComplexTuple(np.cos(phi), np.sin(phi))


L = PositionEncoder(K)

