import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import theano
import theano.tensor as T
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import OneHotEncoder
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import config
from cost import compute_cost_of_batch
from utils import float_x
from utils.complex import complex_dot, complex_multiply, complex_map, complex_conj
from position_encoding import K
from optimizers import adam
from layers import Layer


# digits = theano.shared(float_x(np.real(np.array([vec(config.DIM) for _ in range(11)]).T)))
digits = (theano.shared(float_x(np.zeros((11, config.DIM)).T)), theano.shared(float_x(np.zeros((11, config.DIM)).T)))
srng = RandomStreams()

HIDDEN_LAYERS = [1024]

# Prior
S_0 = (theano.shared(float_x(np.zeros(config.DIM))), theano.shared(float_x(np.zeros(config.DIM))))

# Dictionary (N x 11)
D = (
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11)))),
        theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(config.DIM, 11))))
)

class RepresentationModel(object):
    """Model the representation of a glimpse using a multi-layer neural network"""

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.layers = []

        if not isinstance(n_hidden, list):
            n_hidden = [n_hidden]

        n_prev = n_in
        input_to_layer = input
        for n in n_hidden:
            layer = Layer(rng=rng, input=input_to_layer, n_in=n_prev, n_out=n, activation=T.tanh)
            n_prev = n
            input_to_layer = layer.output
            self.layers.append(layer)

        # output layer
        layer_real = Layer(rng=rng, input=input_to_layer, n_in=n_prev, n_out=n_out, activation=T.tanh)
        layer_imag = Layer(rng=rng, input=input_to_layer, n_in=n_prev, n_out=n_out, activation=T.tanh)

        self.layers.append(layer_real)
        self.layers.append(layer_imag)

        self.output = (layer_real.output, layer_imag.output)

        self.L1_weight_norm = T.mean([abs(layer.W).sum() for layer in self.layers])
        self.L2_weight_norm = T.mean([(layer.W ** 2).sum() for layer in self.layers])

    @property
    def params(self):
        all_params = []
        for layer in self.layers:
            all_params += layer.params
        return all_params

batch_raw_glimpses = T.ftensor3("raw_glimpses")
batch_glimpse_positions = (T.ftensor3("glimpse_positions_real"), T.ftensor3("glimpse_positions_imag"))
batch_sample_labels = T.ftensor3("sample_labels")
batch_sample_positions = (T.ftensor3("sample_positions_real"), T.ftensor3("sample_positions_imag"))

a, b, c = batch_raw_glimpses.shape
batch_raw_glimpses_reshaped = batch_raw_glimpses.reshape((a * b, c))

glimpse_model = RepresentationModel(
        srng,
        batch_raw_glimpses_reshaped,
        n_in=29 ** 2,
        n_hidden=HIDDEN_LAYERS,
        n_out=config.DIM)
glimpse_vectors = complex_map(glimpse_model.output, lambda s: s.reshape((a, b, config.DIM)))

reg = 1e-5
weight_reg = reg * glimpse_model.L1_weight_norm
cost = compute_cost_of_batch(S_0, glimpse_vectors,
        batch_glimpse_positions, batch_sample_labels, batch_sample_positions, digits)

cost_ratio = cost / weight_reg

updates = adam(cost, glimpse_model.params + list(S_0) + list(digits))

inputs = ((batch_raw_glimpses,) + batch_glimpse_positions +
        (batch_sample_labels,) + batch_sample_positions)

train = theano.function(
        inputs=inputs,
        outputs=[cost, cost_ratio],
        updates=updates,
        allow_input_downcast=True)

predict = theano.function(
        inputs=[batch_raw_glimpses],
        outputs=glimpse_model.output,
        allow_input_downcast=True)
