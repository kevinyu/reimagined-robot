import theano
import theano.tensor as T

from position_encoding import L
from utils.complex import ComplexTuple
from utils import tensor_softmax


D = ComplexTuple(T.fmatrix("D_real"), T.fmatrix("D_imag"))
S = ComplexTuple(*T.fvectors("scene_real", "scene_imag"))


# render belief over dictionary components in a scene
raster_scene = theano.function(
        inputs=[D.real, D.imag, S.real, S.imag],
        outputs=tensor_softmax(L.IFFT(S.dimshuffle(0, "x") * D.conj).real),
        allow_input_downcast=True)


# render a single hypervector
raster = theano.function(
        inputs=[S.real, S.imag],
        outputs=L.IFFT(S).real,
        allow_input_downcast=True)


# render a matrix whose columns are hypervectors
raster_dict = theano.function(
        inputs=[D.real, D.imag],
        outputs=L.IFFT(D).real,
        allow_input_downcast=True)
