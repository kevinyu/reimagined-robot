import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
from utils.complex import ComplexTuple
from utils.unitary import U, get_U


rs = np.random.RandomState(seed=config.SEED)

K = theano.shared(float_x(
    rs.choice([-1, 1], size=(config.DIM, 2)) *
    rs.uniform(config.MIN_K, config.MAX_K, size=(config.DIM, 2))
))
'''

nx=np.sqrt(config.DIM)
assert (nx-int(nx)) == 0.0
nx = int(nx)

temp=np.zeros((nx, nx),dtype=complex)
temp[1,0]=1.0; temp_dim0=temp*1.; temp*=0.
temp[0,1]=1.0; temp_dim1=temp*1.

ft0=(np.fft.fft2(temp_dim0)).flatten()
ft1=(np.fft.fft2(temp_dim1)).flatten()

K=theano.shared(float_x(np.array([np.angle(ft0), np.angle(ft1)]).T))
'''


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
