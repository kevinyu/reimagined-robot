import theano
import theano.tensor as T

from position_encoding import L
from utils.complex import ComplexTuple


D = ComplexTuple(T.fmatrix("D_real"), T.fmatrix("D_imag"))
S = ComplexTuple(*T.fvectors("scene_real", "scene_imag"))


# render belief over dictionary components in a scene
raster_scene = theano.function(
        inputs=[D.real, D.imag, S.real, S.imag],
        outputs=L.IFFT(S.dimshuffle(0, "x") * D.conj),
        allow_input_downcast=True)


# render a single hypervector
raster = theano.function(
        inputs=[S.real, S.imag],
        outputs=L.IFFT(S),
        allow_input_downcast=True)


# render a matrix whose columns are hypervectors
raster_dict = theano.function(
        inputs=[D.real, D.imag],
        outputs=L.IFFT(D),
        allow_input_downcast=True)
