import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import theano
import theano.tensor as T

import config
from tasks.mnist.parameters import D_table
from position_encoding import L
from utils import float_x
from utils.complex import ComplexTuple
from tasks.mnist.query_scene import D_directions


# IMG_WIDTH x IMG_HEIGHT * DIM
X = L.encode_numeric(float_x(np.array(
    np.meshgrid(
        np.linspace(0, 1, config.IMG_WIDTH),
        np.linspace(0, 1, config.IMG_HEIGHT)
    )
).swapaxes(0, 2)))
X = ComplexTuple(theano.shared(X.real), theano.shared(X.imag))
D = ComplexTuple(T.fmatrix("D_real"), T.fmatrix("D_imag"))

S = ComplexTuple(*T.fvectors("scene_real", "scene_imag"))
similarity = (X.conj * S.dimshuffle("x", "x", 0)).dot(D).real
_n_glimpses, _n_samples, _n_digits = similarity.shape

belief = T.nnet.softmax(
        similarity.reshape((_n_glimpses * _n_samples, _n_digits))
).reshape((_n_glimpses, _n_samples, _n_digits ))

raster = theano.function(
        inputs=[D.real, D.imag, S.real, S.imag],
        outputs=belief,
        allow_input_downcast=True)

dir_sim = X.dot(D_directions).real
a, b, c = dir_sim.shape
dir_belief = T.nnet.softmax(
        dir_sim.reshape((a * b, c))
).reshape((a, b, c))

direction_raster = theano.function(
        inputs=[],
        outputs=dir_belief,
        allow_input_downcast=True)


