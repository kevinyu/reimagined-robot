import numpy as np
import theano.tensor as T


def add(x, y):
    return ComplexTuple(x.real + y.real, x.imag + y.imag)


def subtract(x, y):
    return ComplexTuple(x.real - y.real, x.imag - y.imag)


def multiply(x, y):
    x_real, x_imag = x
    y_real, y_imag = y

    return ComplexTuple((x_real * y_real - x_imag * y_imag),
            (x_real * y_imag + x_imag * y_real))


def dot(x, y):
    y = y.conj
    return ComplexTuple((T.dot(x.real, y.real) - T.dot(x.imag, y.imag)),
            (T.dot(x.real, y.imag) + T.dot(x.imag, y.real)))


def _numpydot(x, y):
    y = y.conj
    return ComplexTuple((np.dot(x.real, y.real) - np.dot(x.imag, y.imag)),
            (np.dot(x.real, y.imag) + np.dot(x.imag, y.real)))


class ComplexTuple(tuple):

    def __new__(cls, real_part, imag_part):
        return super(ComplexTuple, cls).__new__(cls, (real_part, imag_part))

    @property
    def real(self):
        return self[0]

    @property
    def imag(self):
        return self[1]

    @property
    def conj(self):
        return ComplexTuple(self[0], -self[1])

    def dot(self, y):
        if isinstance(self[0], np.ndarray):
            return _numpydot(self, y)
        return dot(self, y)

    def __add__(self, y):
        if isinstance(y, ComplexTuple):
            return add(self, y)
        else:
            return ComplexTuple(self[0].__add__(y), self[1].__add__(y))

    def __sub__(self, y):
        if isinstance(y, ComplexTuple):
            return subtract(self, y)
        else:
            return ComplexTuple(self[0].__sub__(y), self[1].__sub__(y))

    def __mul__(self, y):
        if isinstance(y, ComplexTuple):
            return multiply(self, y)
        else:
            return ComplexTuple(self[0].__mul__(y), self[1].__mul__(y))

    def __getattr__(self, attr):
        try:
            return tuple.__getattr__(self, attr)
        except AttributeError:
            _a1 = getattr(self[0], attr)
            _a2 = getattr(self[1], attr)
            if type(_a1) != type(_a2):
                raise Exception("Attr {} not consistent in complex tuple".format(attr))
            if callable(_a1) and callable(_a2):
                def _fn(*args, **kwargs):
                    return ComplexTuple(
                        _a1(*args, **kwargs),
                        _a2(*args, **kwargs)
                    )
                return _fn 
            else:
                return ComplexTuple(_a1, _a2)
