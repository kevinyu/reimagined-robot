import os

import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
from utils.complex import ComplexTuple


S0 = ComplexTuple(
    theano.shared(float_x(np.zeros(config.DIM))),
    theano.shared(float_x(np.zeros(config.DIM)))
)


if not config.SHAPES:
    D = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11)))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11))))
    )
else:
    from shapes.properties import properties
    from shapes.objects import shapes
    D_table = {}

    D_table["Shapes"] = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(shapes))))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(shapes)))))
    )
    for prop in properties:
        D_table[prop.__name__] = ComplexTuple(
            theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(prop.params))))),
            theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, len(prop.params)))))
        )

    D = D_table["Shapes"]
    i = 1
    for prop in properties:
        D = (
            D.dimshuffle([0] + range(1, i+1) + ["x"]) *
            D_table[prop.__name__].dimshuffle(*[[0] + (["x"] * i) + [1]])
        )
        i += 1
    separated_D = D
    D = D.flatten(2)

    bg_vector = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 1)))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 1))))
    )

    D = ComplexTuple(
        T.concatenate([D.real, bg_vector.real], axis=1),
        T.concatenate([D.imag, bg_vector.imag], axis=1)
    )


def save_params(S0_name=None, D_name=None):
    if S0_name:
        s0_filename = os.path.join(config.SAVE_DIR, "S0_{}".format(S0_name))
    else:
        s0_filename = os.path.join(config.SAVE_DIR, "S0")
    np.save(s0_filename, np.array(list(S0.get_value())))

    if D_name:
        D_filename = os.path.join(config.SAVE_DIR, "D_{}".format(D_name))
    else:
        D_filename = os.path.join(config.SAVE_DIR, "D")
    np.save(D_filename, np.array(list(D.get_value())))


if config.LOAD_PREEXISTING:
    try:
        S0_array = np.load(os.path.join(config.SAVE_DIR, "S0.npy"))
    except:
        print "Failed to load S0"
    else:
        S0 = ComplexTuple(
                theano.shared(float_x(S0_array[0])),
                theano.shared(float_x(S0_array[1]))
        )

if config.LOAD_PREEXISTING:
    try:
        D_array = np.load(os.path.join(config.SAVE_DIR, "D.npy"))
    except:
        print "Failed to load D"
    else:
        D = ComplexTuple(
                theano.shared(float_x(D_array[0])),
                theano.shared(float_x(D_array[1]))
        )
elif not config.LEARN_D:
    a = np.random.uniform(-1, 1, size=(config.DIM, 11))
    a = a / np.linalg.norm(a, axis=0)
    b = np.random.uniform(-1, 1, size=(config.DIM, 11))
    b = b / np.linalg.norm(b, axis=0)
    D = ComplexTuple(
        theano.shared(float_x(a)),
        theano.shared(float_x(b))
    )


