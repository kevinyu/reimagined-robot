import os

import numpy as np
import theano
import theano.tensor as T

import config
from properties import properties
from utils import float_x, init_hypervectors
from utils.complex import ComplexTuple


if os.path.exists(os.path.join(config.SAVE_DIR, "S0.npy")):
    S0_array = np.load(os.path.join(config.SAVE_DIR, "S0.npy"))
    S0 = ComplexTuple(
        theano.shared(float_x(S0_array[0])),
        theano.shared(float_x(S0_array[1]))
    )
else:
    S0 = ComplexTuple(
        theano.shared(float_x(np.zeros(config.DIM))),
        theano.shared(float_x(np.zeros(config.DIM)))
    )

D_table = {}

if os.path.exists(os.path.join(config.SAVE_DIR, "D_Digits.npy")):
    digits_array = np.load(os.path.join(config.SAVE_DIR, "D_Digits.npy"))
    D_table["Digits"] = ComplexTuple(
        theano.shared(float_x(digits_array[0])),
        theano.shared(float_x(digits_array[1]))
    )
else:
    D_table["Digits"] = ComplexTuple(
        init_hypervectors(11),
        init_hypervectors(11)
    )

for prop in properties:
    stream = prop.__name__
    filename = os.path.join(config.SAVE_DIR, "D_{}.npy".format(stream))

    if os.path.exists(filename):
        darray = np.load(filename)
        D_table[stream] = ComplexTuple(
            theano.shared(float_x(darray[0])),
            theano.shared(float_x(darray[1]))
        )
    else:
        D_table[stream] = ComplexTuple(
            init_hypervectors(len(prop.params) + 1),
            init_hypervectors(len(prop.params) + 1)
        )


nothing_vector_file = os.path.join(config.SAVE_DIR, "D_nothing.npy")
if os.path.exists(nothing_vector_file):
    darray = np.load(nothing_vector_file)
    nothing_vector = ComplexTuple(
        theano.shared(float_x(darray[0])),
        theano.shared(float_x(darray[1]))
    )
else:
    nothing_vector = ComplexTuple(
        init_hypervectors(1),
        init_hypervectors(1)
    )

learn_params = [S0.real, S0.imag]
for stream in config.STREAMS:
    learn_params += [D_table[stream].real, D_table[stream].imag]
learn_params += [nothing_vector.real, nothing_vector.imag]


# TODO delete this block when we no longer need this
_D_combined = D_table["Digits"]
for i, prop in enumerate(properties):
    # each iteration increases the dimensionality of D_combined by one
    # the last dimension corresponds to the ith property
    stream = prop.__name__
    i += 1
    _D_combined = (
        _D_combined.dimshuffle([0] + range(1, i+1) + ["x"]) *
        D_table[stream].dimshuffle(*[[0] + (["x"] * i) + [1]])
    )
D = _D_combined.flatten(2)
D = ComplexTuple(
    T.concatenate([D.real, nothing_vector.real], axis=1),
    T.concatenate([D.imag, nothing_vector.imag], axis=1)
)



def save_params():
    s0_filename = os.path.join(config.SAVE_DIR, "S0")
    np.save(s0_filename, np.array(list(S0.get_value())))

    for stream in config.STREAMS:
        D_filename = os.path.join(config.SAVE_DIR, "D_{}".format(stream))
        np.save(D_filename, np.array(list(D_table[stream].get_value())))

    np.save(
        os.path.join(config.SAVE_DIR, "D_nothing"),
        np.array(list(nothing_vector.get_value()))
    )
