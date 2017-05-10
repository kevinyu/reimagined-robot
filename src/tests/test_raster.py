import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import numpy as np

from visualize import examine

from utils.complex import *

DIM = 1024

digits = theano.shared(np.random.uniform(-1, 1, size=(11, DIM)))
digits = (digits, T.zeros_like(digits))

K = theano.shared(np.random.uniform(-8 * np.pi, 8 * np.pi, size=(DIM, 2)))

S = T.fvectors("S_real", "S_imag")

softmax_output = examine(S, K, digits)

evaluate = theano.function(
        inputs=S,
        outputs=softmax_output,
        allow_input_downcast=True)

def run_test(S):
    result = evaluate(*(S.real, S.imag))
    for i in range(11):
        plt.subplot(3, 4, i+1)
        plt.imshow(result[:, :, i], vmin=0.0, vmax=1.0, cmap="gray")
        plt.title("{}".format(i if i < 10 else "BG"))
    plt.show()
        
if __name__ == "__main__":
    d1 = digits[0].get_value()[2]
    phi = np.dot(np.array([[1., 1.]]), K.get_value().T)
    r1 = np.exp(1j * phi)
    S1 = d1 * r1.flatten()
    S1 = d1

    run_test(S1)
