import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import theano
import theano.tensor as T

import config
from position_encoding import L
from tasks.mnist.parameters import D_table
from tasks.mnist.query_scene import D_directions
from utils.complex import ComplexTuple


D = ComplexTuple(T.fmatrix("D_real"), T.fmatrix("D_imag"))
S = ComplexTuple(*T.fvectors("scene_real", "scene_imag"))

# render belief over dictionary components
raster_dict = theano.function(
        inputs=[D.real, D.imag, S.real, S.imag],
        outputs=L.IFFT(S.dimshuffle(0, "x") * D.conj),
        allow_input_downcast=True)

# render a single hypervector
raster = theano.function(
        inputs=[S.real, S.imag],
        outputs=L.IFFT(S),
        allow_input_downcast=True)

