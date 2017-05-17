import numpy as np
import theano
import theano.tensor as T

from utils import float_x


class Layer(object):

    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.rng = rng
        self.input = input
        self.n_in = n_in
        self.n_out = n_out
        self.activation = activation

        self.W = W if W is not None else self._generate_weights(self.n_in, self.n_out)
        self.b = b if b is not None else self._generate_bias(self.n_out)

        z = T.dot(input, self.W) + self.b

        self.output = self.activation(z) if self.activation else z

    def _generate_weights(self, n_in, n_out):
        n_tot = n_in + n_out
        W = float_x(
            np.random.uniform(
                low=-np.sqrt(6.0 / n_tot),
                high=np.sqrt(6.0 / n_tot),
                size=(n_in, n_out)
            )
        )
        return theano.shared(value=W, name="W", borrow=True)

    def _generate_bias(self, n_out):
        b = float_x(np.zeros(n_out))
        return theano.shared(value=b, name="b", borrow=True)

    @property
    def params(self):
        return [self.W, self.b]

