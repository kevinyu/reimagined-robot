"""
Demonstrate that the thing is capable of learning relative positions in a glimpse
"""


import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from model import RepresentationModel
from optimizers import adam
from utils.complex import *

from datasets.mnist_scene import generate_single_digit_scene, generate_n_digit_scene
from utils import float_x


N = 1024
srng = RandomStreams()
G = 28 ** 2
k = 2
A = 56


class PositionEncoder(object):
    def __init__(self):
        self.K = theano.shared(float_x(np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(N, 2))))

    def encode(self, X):
        phi = T.dot(X, self.K.T)
        return (T.cos(phi), T.sin(phi))

position_encoder = PositionEncoder()


raw_glimpses = T.ftensor3("raw_glimpses")   # raw glimpses are  n x k x G  (k glimpses, G glimpse size)
target = T.ftensor4("target")               # target (label) is n x a x b x 11  (a, b are image size)
I = T.fvector("mnist_image")                # single mnist image

D = (
    theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(N, 11)))),
    theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(N, 11))))
)

a, b, c = raw_glimpses.shape
raw_glimpses_reshaped = raw_glimpses.reshape((a * b, c))

model = RepresentationModel(
        srng,
        raw_glimpses_reshaped,
        n_in=28 ** 2,
        n_hidden=[1024],
        n_out=N)

s = complex_map(model.output, lambda s: s.reshape((a, b, N)))

# cast x,y coordinates in X into N-dimensional position vectors
X = theano.shared(
        float_x(np.array(
            np.meshgrid(
                np.linspace(0, 1, 56),
                np.linspace(0, 1, 56)
            )
        ).swapaxes(0, 2)))

# sample locations in hypervector form
R = position_encoder.encode(X)


# R is shaped a x b x N
# s is shaped n x k x N
# D is shaped N x 2

s = complex_map(s, lambda r: r[:, 0, :])
R_ = complex_map(R, lambda r: r.dimshuffle("x", 0, 1, 2))
s_ = complex_map(s, lambda r: r.dimshuffle(0, "x", "x", 1))
similarity = complex_dot(complex_multiply(complex_conj(R_), s_), D)[0]  # take real part

# similarilty is n x k x a x b x 11
a, b, c, d = similarity.shape
result = T.nnet.softmax(similarity.reshape((a * b * c, d))).reshape((a, b, c, d))

cost = T.nnet.categorical_crossentropy(result, target).mean()

params = model.params + list(D)  # learn both model weights and digit dictionary

updates = adam(cost, params)

train = theano.function(
        inputs=[raw_glimpses, target],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)

raster = theano.function(  # takes just a 1 x 1 x G glimpse
        inputs=[raw_glimpses],
        outputs=result[0],
        allow_input_downcast=True)


def generate_target(pos_x, pos_y, val, radius=10):
    targ = np.zeros((56, 56, 11))
    targ[:, :, -1] = 1.0
    _x, _y = np.meshgrid(np.arange(56), np.arange(56))
    _x, _y = np.where((np.power(_x - pos_x, 2) + np.power(_y - pos_y, 2)) <= np.power(radius, 2))
    targ[_y, _x] = val
    return targ


def generate_target(scene, radius=12):
    scene.label_radius = radius
    targ = np.zeros((56, 56, 11))
    for _, loc in scene.digit_locations:
        for x in range(np.maximum(loc[0]-radius, 0), np.minimum(56, loc[0]+radius)):
            for y in range(np.maximum(loc[1]-radius, 0), np.minimum(56, loc[1]+radius)):
                targ[x, y] = scene.get_label_at(x, y)
    return targ


def generate_batch(n):
    """Generate a batch of n datas and their labels"""
    imgs = np.zeros((n, 1, 56, 56))
    glimpses = np.zeros((n, 1, 28 ** 2))
    targets = np.zeros((n, 56, 56, 11))
    
    for _ in range(n):
        scene = generate_n_digit_scene((A, A), 2)
        # loc = scene.digit_locations[0][1]
        # label_vec = scene.get_label_at(loc[0] + 14, loc[1] + 14)
        targ = generate_target(scene, radius=12)

        imgs[_, 0] = scene.img
        glimpses[_, 0] = scene.img[14:14+28, 14:14+28].flatten() / 255.0
        targets[_] = targ
    return imgs, glimpses, targets


def render_belief(img, glimpse):
    meh = raster(glimpse.flatten()[None, None, :] / 255.0)

    plt.figure(figsize=(10,14))
    plt.subplot(5, 3, 1)

    plt.hlines([14, 14+28], 14, 14+28, color="red")
    plt.vlines([14, 14+28], 14, 14+28, color="red")

    plt.imshow(img)
    plt.subplot(5, 3, 2)
    plt.imshow(img[14:14+28, 14:14+28])

    for i in range(11):
        plt.subplot(5, 3, 3 + i+2)
        plt.imshow(meh[:, :, i], vmin=0.0, vmax=1.0)
        if i == 10:
            i = "BG"
        plt.text(10, 10, "{}".format(i), fontsize=20, color="white")


if __name__ == "__main__":
    for i in range(20):
        for j in range(100):
            imgs, glimpses, targets = generate_batch(100)
            costy = train(glimpses, targets)
            print costy

        print "rendering"

        img = imgs[-1, 0]
        glimpse = img[14:14+28, 14:14+28]
        render_belief(img, glimpse)
        plt.savefig("saved_images/{}_{}_raster.png".format(i, j), format="png", dpi=400)
        plt.close()

