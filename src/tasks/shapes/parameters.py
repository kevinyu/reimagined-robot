import os

import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
from utils.complex import ComplexTuple

from shapes.properties import properties
from shapes.objects import shapes


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



# D_table is a dict mapping categories to dictionaries of hypervectors
D_table = {}

filename = os.path.join(config.SAVE_DIR, "D_Shapes.npy")
if os.path.exists(filename):
    darray = np.load(filename)
    D_table["Shapes"] = ComplexTuple(
        theano.shared(float_x(darray[0])),
        theano.shared(float_x(darray[1]))
    )
else:
    D_table["Shapes"] = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(shapes))))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(shapes)))))
    )

for prop in properties:
    filename = os.path.join(config.SAVE_DIR, "D_{}.npy".format(prop.__name__))

    if os.path.exists(filename):
        darray = np.load(filename)
        D_table[prop.__name__] = ComplexTuple(
            theano.shared(float_x(darray[0])),
            theano.shared(float_x(darray[1]))
        )
    else:
        D_table[prop.__name__] = ComplexTuple(
            theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(prop.params))))),
            theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(prop.params)))))
        )


# Generate all bound combinations of available objects with properties
_D_combined = D_table["Shapes"]

for i, prop in enumerate(properties):
    # each iteration increases the dimensionality of D_combined by one
    # the last dimension corresponds to the ith property
    i += 1
    _D_combined = (
        _D_combined.dimshuffle([0] + range(1, i+1) + ["x"]) *
        D_table[prop.__name__].dimshuffle(*[[0] + (["x"] * i) + [1]])
    )

D = _D_combined.flatten(2)

# Concatenate a single vector representing background to D
bg_filename = os.path.join(config.SAVE_DIR, "D_bg.npy")
if os.path.exists(bg_filename):
    darray = np.load(bg_filename)
    bg_vector = ComplexTuple(
        theano.shared(float_x(darray[0])),
        theano.shared(float_x(darray[1]))
    )
else:
    bg_vector = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 1)))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 1))))
    )

D = ComplexTuple(
    T.concatenate([D.real, bg_vector.real], axis=1),
    T.concatenate([D.imag, bg_vector.imag], axis=1)
)


learn_params = [bg_vector.real, bg_vector.imag]
for D_prop in D_table.values():
    learn_params += [D_prop.real, D_prop.imag]

learn_params = [S0.real, S0.imag] + learn_params


def save_params():
    s0_filename = os.path.join(config.SAVE_DIR, "S0")
    np.save(s0_filename, np.array(list(S0.get_value())))

    D_filename = os.path.join(config.SAVE_DIR, "D_Shapes")
    np.save(D_filename, np.array(list(D_table["Shapes"].get_value())))

    for prop in properties:
        D_filename = os.path.join(config.SAVE_DIR, "D_{}".format(prop.__name__))
        np.save(D_filename, np.array(list(D_table[prop.__name__].get_value())))

    np.save(os.path.join(config.SAVE_DIR, "D_bg"), np.array(list(bg_vector.get_value())))

