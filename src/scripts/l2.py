import numpy as np
import theano
import theano.tensor as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from model import RepresentationModel
from optimizers import adam
from utils.complex import *

from datasets.mnist_scene import generate_single_digit_scene, generate_several_digit_scene
from glimpse import accumulate_glimpses_over_batch
from utils import float_x
from position_encoding import K
from datasets.training_set import take_glimpses, take_samples



N = 1024
srng = RandomStreams()
G = 29 ** 2
k = 2
A = 64


class PositionEncoder(object):
    def __init__(self):
        # self.K = theano.shared(float_x(np.random.uniform(-2.0 * np.pi, 2.0 * np.pi, size=(N, 2))))
        self.K = K

    def encode(self, X):
        phi = T.dot(X, self.K.T)
        return (T.cos(phi), T.sin(phi))

    def numpyencode(self, X):
        phi = np.dot(X, self.K.get_value().T)
        return (np.cos(phi), np.sin(phi))

position_encoder = PositionEncoder()


raw_glimpses = T.ftensor3("raw_glimpses")   # raw glimpses are  n x k x G  (k glimpses, G glimpse size)
glimpse_positions_hd = (
        T.ftensor3("glimpse_positions"),
        T.ftensor3("glimpse_positions_im")
)# raw glimpses are  n x k x N  (k glimpses, G glimpse size)
target = T.ftensor3("target")               # target (label) is n x s x 11  (a, b are image size)


D = (
    theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(N, 11)))),
    theano.shared(float_x(0.01 * np.random.uniform(-1, 1, size=(N, 11))))
)
S_0 = (
    theano.shared(float_x(np.zeros(N))),
    theano.shared(float_x(np.zeros(N)))
)

a, b, c = raw_glimpses.shape
raw_glimpses_reshaped = raw_glimpses.reshape((a * b, c))

model = RepresentationModel(
        srng,
        raw_glimpses_reshaped,
        n_in=29 ** 2,
        n_hidden=[1024],
        n_out=N)

s = complex_map(model.output, lambda s: s.reshape((a, b, N)))

# s is n x k x N wher k is the number of glimpses in a trial
# want to compute up S_0 + s_n0, S_0 + s_n0 + s_n1, ... , S_0 + s_n0 + ... + s_nk
# into a vector S: (n x k x N)

# accumulate glimpses for each n
# put back together into single n x k x N matrix
# positions sampled at is R: n x s x N
# S_nk * R_n: n x k x s x N  # for each trial, for each glimpse, for each sample
# do a broadcast multiply
# then dot with digit dictionary  N x 11
# to form n x k x s x 11


S = accumulate_glimpses_over_batch(S_0, s, glimpse_positions_hd)

# now S is n x k x N
# gotta unbind now
sample_positions_hd = (
        T.ftensor3("sample_positions"),
        T.ftensor3("sample_positions_imag")
)# n x s x N
S_unbound = complex_multiply(
        complex_map(S, lambda r: r.dimshuffle(0, 1, "x", 2)),
        complex_map(complex_conj(sample_positions_hd), lambda r: r.dimshuffle(0, "x", 1, 2))
)
similarity = complex_dot(S_unbound, D)[0]  # take real part, this is now n x k x s x 11
a, b, c, d = similarity.shape
result = T.nnet.softmax(similarity.reshape((a * b * c, d))).reshape((a, b, c, d))

target_expanded = target.dimshuffle(0, "x", 1, 2)
cost = (-T.sum(target_expanded * T.log(result), axis=-1)).mean()

# cast x,y coordinates in X into N-dimensional position vectors
X = theano.shared(
        float_x(np.array(
            np.meshgrid(
                np.linspace(0, 1, 64),
                np.linspace(0, 1, 64)
            )
        ).swapaxes(0, 2)))

# sample locations in hypervector form
R = position_encoder.encode(X)

# this is aside stuff for just the viewing pleasure you know
s_bleh_ = complex_map(S, lambda r: r[:, -1, :])  # take last glimpse 
R__ = complex_map(R, lambda r: r.dimshuffle("x", 0, 1, 2))  # _ x y N
s__ = complex_map(s_bleh_, lambda r: r.dimshuffle(0, "x", "x", 1)) # n _ _ N
# position_map = complex_dot(R, s_bleh_[0])[0]
raster_sim = complex_dot(complex_multiply(complex_conj(R__), s__), D)[0]  # take real part n x y N
a, b, c, d = raster_sim.shape
raster_result = T.nnet.softmax(raster_sim.reshape((a * b * c, d))).reshape((a, b, c, d))

params = model.params + list(S_0) + list(D)  # learn both model weights and digit dictionary

updates = adam(cost, params)

train = theano.function(
        inputs=[raw_glimpses] + list(glimpse_positions_hd) + list(sample_positions_hd) + [target],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True)

raster = theano.function(  # takes just a 1 x 1 x G glimpse and 1 x whatever x N glimpse locations
        inputs=[raw_glimpses] + list(glimpse_positions_hd),
        outputs=raster_result[0],
        allow_input_downcast=True)

raster_single = theano.function(
        inputs=s_bleh_,  # should be 1 x N
        outputs=raster_result[0],
        allow_input_downcast=True)

# digit map
# D is  N x 11
# R is x y N
# dot(R, N) -> x y 11

raster_map = theano.function(
        inputs=[],  # should be 1 x N
        outputs=complex_dot(R, D)[0],  # 
        allow_input_downcast=True)

def render_digits():
    meh = raster_map()

    for i in range(11):
        plt.subplot(4, 3, i+1)
        plt.imshow(meh[:, :, i], vmin=0.0, vmax=1.0)
        if i == 10:
            i = "BG"
        plt.text(10, 10, "{}".format(i), fontsize=20, color="white")

def render_belief(img, glimpse_xy, glimpses, glimpse_positions):
    meh = raster(glimpses, *glimpse_positions)

    plt.figure(figsize=(10,14))
    plt.subplot(6, 3, 1)

    for x, y in glimpse_xy:
        plt.hlines([x-14, x+15], y-14, y+15, color="red")
        plt.vlines([y-14, y+15], x-14, x+15, color="red")

    plt.imshow(img)
    for i, glimpse in enumerate(glimpses[0]):
        plt.subplot(6, 3, 3  + 1 + i)
        plt.imshow(glimpse.reshape(29, 29) * 255.0)

    for i in range(11):
        plt.subplot(6, 3, 6 + i+2)
        plt.imshow(meh[:, :, i], vmin=0.0, vmax=1.0)
        if i == 10:
            i = "BG"
        plt.text(10, 10, "{}".format(i), fontsize=20, color="white")

    # entropy 
    plt.subplot(6, 3, 3)
    ENTROPY = -np.sum(meh * np.log2(meh), axis=2)
    plt.imshow(ENTROPY)
    plt.text(10, 10, "entropy", fontsize=5, color="white")


if __name__ == "__main__":
    import os
    training_data = np.load(os.path.join(os.getenv("PROJECT_ROOT"), "src", "datasets", "training_data.npy"))
    def make_batch1(training_data, start, end):
        return [np.array(a) for a in zip(*training_data[start:end])]

    def make_batch(n, img_shape, max_digits, n_glimpses, glimpse_strategy, sample_strategy):
        datas = []
        for _ in range(n):
            scene = generate_several_digit_scene(img_shape, (0, max_digits + 1))
            glimpse_data, glimpse_locs = take_glimpses(scene, glimpse_width=29, n_glimpses=n_glimpses, strategy=glimpse_strategy)
            sample_data, sample_locs = take_samples(scene, n_samples=200, strategy=sample_strategy)
            # map the location data to the hypervector space before saving, and scale it to (0, 1)
            r_glimpse_real, r_glimpse_imag = position_encoder.numpyencode(glimpse_locs.T / np.max(img_shape))
            r_sample_real, r_sample_imag = position_encoder.numpyencode(sample_locs.T / np.max(img_shape))
            datas.append((
                scene.img,
                glimpse_data,
                glimpse_locs.T,
                sample_data,
                sample_locs.T,
                r_glimpse_real,
                r_glimpse_imag,
                r_sample_real,
                r_sample_imag
            ))
        return [np.array(a) for a in zip(*datas)]

    j = 0
    for i in range(6000):
        batch = make_batch(30, (64, 64), 2, 3, np.random.choice(["uniform", "smart"], p=[0.1, 0.9]), "uniform")
        # batch = make_batch1(training_data, 20 * (i % 100), 20 * (i % 100 + 1))
        (imgs, raw_glimpses, glimpse_xy, sample_labels, sample_xy,
                glimpse_positions_real, glimpse_positions_imag,
                sample_positions_real, sample_positions_imag) = batch

        raw_glimpses = raw_glimpses / 255.0
        # map the location data to the hypervector space before saving, and scale it to (0, 1)

        glimpse_positions_real, glimpse_positions_imag = position_encoder.numpyencode(glimpse_xy / 64.0)
        sample_positions_real, sample_positions_imag = position_encoder.numpyencode(sample_xy / 64.0)

        costy = train(raw_glimpses,
                glimpse_positions_real, glimpse_positions_imag,
                sample_positions_real, sample_positions_imag,
                sample_labels)
        print costy
        if i % 50 == 0:
            render_digits()
            plt.savefig("saved_images/digits.png", format="png", dpi=400)
            plt.close()

            how_many_glimpses = 3
            batch = make_batch(1, (64, 64), 3, how_many_glimpses, "uniform", "uniform")
            # batch = make_batch1(training_data, 20 * (i % 100), 20 * (i % 100 + 1))
            (imgs, raw_glimpses, glimpse_xy, sample_labels, sample_xy,
                    glimpse_positions_real, glimpse_positions_imag,
                    sample_positions_real, sample_positions_imag) = batch

            raw_glimpses = raw_glimpses / 255.0
            # map the location data to the hypervector space before saving, and scale it to (0, 1)

            glimpse_positions_real, glimpse_positions_imag = position_encoder.numpyencode(glimpse_xy / 64.0)
            sample_positions_real, sample_positions_imag = position_encoder.numpyencode(sample_xy / 64.0)

            I = np.random.choice(range(len(imgs)))
            img = imgs[I]
            for glimpse_k in range(how_many_glimpses):
                print "rendering"
                render_belief(img, glimpse_xy[I, :glimpse_k+1], raw_glimpses[I:I+1, :glimpse_k+1], (glimpse_positions_real[I:I+1, :glimpse_k+1], glimpse_positions_imag[I:I+1, :glimpse_k+1]))
                plt.savefig("saved_images_3/hooray_{}_{}_raster.png".format(i, glimpse_k), format="png", dpi=400)
                plt.close()

            render_digits()
            plt.savefig("saved_images_3/digits.png", format="png", dpi=400)
            plt.close()

            meh = raster_single(S_0[0].get_value()[None, :], S_0[1].get_value()[None, :])
            for i in range(11):
                plt.subplot(6, 3, 6 + i+2)
                plt.imshow(meh[:, :, i], vmin=0.0, vmax=1.0)
                if i == 10:
                    i = "BG"
                plt.text(10, 10, "{}".format(i), fontsize=20, color="white")
            plt.savefig("saved_images_3/s0.png".format(i, costy), format="png", dpi=400)
            plt.close()


    for i in range(30):
        batch = make_batch1(training_data, 2100+i, 2100+i+1)
        (imgs, raw_glimpses, glimpse_xy, sample_labels, sample_xy,
                glimpse_positions_real, glimpse_positions_imag,
                sample_positions_real, sample_positions_imag) = batch

        raw_glimpses = raw_glimpses / 255.0
        # map the location data to the hypervector space before saving, and scale it to (0, 1)

        glimpse_positions_real, glimpse_positions_imag = position_encoder.numpyencode(glimpse_xy / 64.0)
        sample_positions_real, sample_positions_imag = position_encoder.numpyencode(sample_xy / 64.0)

        print "rendering"
        I = np.random.choice(range(len(imgs)))
        img = imgs[I]
        render_belief(img, glimpse_xy[I], raw_glimpses[I:I+1], (glimpse_positions_real[I:I+1], glimpse_positions_imag[I:I+1]))
        plt.savefig("saved_images_3/{}_raster.png".format(i), format="png", dpi=400)
        plt.close()

    np.save("saved_images_3/D.npy", np.array(complex_map(D, lambda x: x.get_value())))
    np.save("saved_images_3/S_0.npy", np.array(complex_map(S_0, lambda x: x.get_value())))


    render_digits()
    plt.savefig("saved_images_3/digits.png", format="png", dpi=400)
    plt.close()

    meh = raster_single(S_0[0].get_value()[None, :], S_0[1].get_value()[None, :])
    for i in range(11):
        plt.subplot(6, 3, 6 + i+2)
        plt.imshow(meh[:, :, i], vmin=0.0, vmax=1.0)
        if i == 10:
            i = "BG"
        plt.text(10, 10, "{}".format(i), fontsize=20, color="white")
    plt.savefig("saved_images_3/s0.png".format(i, costy), format="png", dpi=400)
    plt.close()

