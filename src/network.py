import cPickle
import os

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import config

from layers import Layer
from utils import float_x
from utils.complex import ComplexTuple


srng = RandomStreams()


# Model variables (batch and normal)
glimpse_features_batch = T.ftensor3("glimpse_features_batch")
_batch_size, _n_glimpses, _glimpse_size = glimpse_features_batch.shape
glimpse_features = glimpse_features_batch.reshape((_batch_size * _n_glimpses, _glimpse_size))


class GlimpseModel(object):
    """Network for taking glimpse features, mapping it into hypervector"""

    def save(self, filename):
        params = []
        for param in self.params:
            params.append(param.get_value())
        np.save(filename, np.array(params))

    def load_params(self, filename):
        params = np.load(filename)
        for param, param_value in zip(self.params, params):
            param.set_value(float_x(param_value))

    def __init__(self, rng, input, n_in, n_hidden, n_out, name=None):
        self._name = name
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

        self.output = ComplexTuple(layer_real.output, layer_imag.output)

        self.L1_weight_norm = T.mean([abs(layer.W).sum() for layer in self.layers])
        self.L2_weight_norm = T.mean([(layer.W ** 2).sum() for layer in self.layers])

    @property
    def params(self):
        all_params = []
        for layer in self.layers:
            all_params += layer.params
        return all_params


networks = []
for stream in config.STREAMS:
    networks.append(GlimpseModel(
        srng,
        glimpse_features,
        n_in=config.GLIMPSE_SIZE,
        n_hidden=config.HIDDEN_LAYERS,
        n_out=config.DIM,
        name="{}_stream".format(stream)))

for net in networks:
    try:
        net.load_params(os.path.join(config.SAVE_DIR, "{}.npy".format(net._name)))
    except:
        print "couldn't load preexisting network weights"

output_sum = networks[0].output
network_params = networks[0].params
for net in networks[1:]:
    output_sum = output_sum + net.output
    network_params += net.params

glimpse_network_output = output_sum

