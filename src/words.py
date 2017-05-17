"""Store the dicionaries vector-symbolic representations

(for digits, directions, colors, etc)
"""
import os

import numpy as np
import theano

import config
from properties import properties
from utils import float_x, init_hypervectors
from utils.complex import ComplexTuple


D_table = {}


def load_up_complex(filename, shape, zeros=False):
    """Try to load complex data at a path if it exists

    If it doesn't exist, initialize new data
    """
    if os.path.exists(filename):
        data_array = np.load(filename)
        return ComplexTuple(
            theano.shared(float_x(data_array[0])),
            theano.shared(float_x(data_array[1]))
        )
    else:
        return ComplexTuple(
            init_hypervectors(shape, zeros=zeros),
            init_hypervectors(shape, zeros=zeros)
        )


S0 = load_up_complex(
        os.path.join(config.SAVE_DIR, "S0.npy"),
        config.DIM,
        zeros=True)


D_table["Digits"] = load_up_complex(
        os.path.join(config.SAVE_DIR, "D_Digits.npy"),
        (config.DIM, 11),
        zeros=False)


D_table["Directions"] = load_up_complex(
        os.path.join(config.SAVE_DIR, "D_Directions.npy"),
        (config.DIM, 8),
        zeros=True)


for prop in properties:
    prop_name = prop.__name__
    filename = os.path.join(config.SAVE_DIR, "D_{}.npy".format(prop_name))

    D_table[prop_name] = load_up_complex(
            filename,
            (config.DIM, len(prop.params) + 1),
            zeros=False)


def save_params():
    np.save(
        os.path.join(config.SAVE_DIR, "S0"),
        np.array(list(S0.get_value())))

    for name, D in D_table.items():
        np.save(
            os.path.join(config.SAVE_DIR, "D_{}".format(name)),
            np.array(list(D.get_value())))
