import os

import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
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

if os.path.exists(os.path.join(config.SAVE_DIR, "D_digits.npy")):
    digits_array = np.load(os.path.join(config.SAVE_DIR, "D_digits.npy"))
    D = ComplexTuple(
        theano.shared(float_x(digits_array[0])),
        theano.shared(float_x(digits_array[1]))
    )
else:
    D = ComplexTuple(
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11)))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11))))
    )

D_table["Digits"] = D

learn_params = [S0.real, S0.imag, D.real, D.imag]


def save_params():
    s0_filename = os.path.join(config.SAVE_DIR, "S0")
    np.save(s0_filename, np.array(list(S0.get_value())))

    D_filename = os.path.join(config.SAVE_DIR, "D_digits")
    np.save(D_filename, np.array(list(D.get_value())))

