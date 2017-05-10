"""Render digit similarity"""

import theano
import matplotlib.pyplot as plt
import numpy as np

import config
from parameters import D
from visualize import X, L
from utils import float_x
from utils.complex import ComplexTuple


X = L.encode_numeric(float_x(np.array(
    np.meshgrid(
        np.linspace(-0.14, 0.14, config.IMG_WIDTH),
        np.linspace(-0.14, 0.14, config.IMG_HEIGHT)
    )
).swapaxes(0, 2)))
X = ComplexTuple(theano.shared(X.real), theano.shared(X.imag))
X = X.get_value()
X = X.real + 1j * X.imag


def plot_digits(D):
    rastered = X.conj().dot(D * D.conj()).real
    rastered = X.conj().dot(D).real

    for digit_id in range(11):
        plt.subplot(4, 3, digit_id + 1)
        plt.imshow(rastered[:, :, digit_id], vmin=0.0, vmax=1.0)
        plt.xticks([])
        plt.yticks([])
        plt.text(20, 20, "BG" if digit_id == 10 else str(digit_id), fontsize=20, color="white")

    # plt.savefig(filename, format="png", dpi=400)
    # plt.close()


if __name__ == "__main__":
    import sys
    if sys.argv[1]:
        digits = np.load(sys.argv[1])
    else:
        from parameters import D
        digits = D.get_value()
    digits = digits[0] + 1j * digits[1]

    dotted = digits.T.dot(digits.conj())
    norm = np.linalg.norm(digits, axis=0)
    norm = np.outer(norm, norm)

    plt.imshow(dotted.real / norm, vmin=-1.0, vmax=1.0, cmap="RdBu")
    plt.xticks(range(11), range(10) + ["BG"])
    plt.xticks(range(11), range(10) + ["BG"])
    plt.colorbar()

    plt.figure()
    plot_digits(digits)
    plt.show()
