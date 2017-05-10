import os

import numpy as np
import theano
import theano.tensor as T

import config
from utils import float_x
from utils.complex import ComplexTuple


if config.TASK == "MNIST":
    from tasks.mnist.parameters import (S0, D, learn_params, save_params)
elif config.TASK == "SHAPES":
    from tasks.shapes.parameters import (S0, D, learn_params, save_params)


