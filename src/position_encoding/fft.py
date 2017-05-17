import theano.tensor as T
import theano.gpuarray.fft as Tfft

from utils.complex import ComplexTuple


def fft_keepdims(xr, xi=None, inverse=False):
    """
    behaves just like numpy.fft.fft2 or numpy.fft.ifft2
    :param xr, xi: real and imaginary inputs. Shape is assumed to be (batchsize, x, y)
    :return: real and imaginary parts of , both of shape (batchsize, x, y)
    """
    if xi is None:
        rfft = Tfft.curfft(xr, norm='ortho')  # (batch, x, y, 2)
        rfft_rev = rfft[:, :, ::-1, :]
        rfft_rev = T.roll(rfft_rev[:, ::-1, :, :], 1, axis=1)
        rfft_r = T.concatenate([rfft[:, :, :, 0], rfft_rev[:, :, 1:-1, 0]], axis=2)
        rfft_i = T.concatenate([rfft[:, :, :, 1], -rfft_rev[:, :, 1:-1, 1]], axis=2)
        return rfft_r, rfft_i
    else:
        xfrr, xfri = fft_keepdims(xr)
        if inverse:
            xi *= -1.
        xfir, xfii = fft_keepdims(xi)
        if inverse:
            return xfrr-xfii, -(xfri+xfir)
        else:
            return ComplexTuple(xfrr-xfii, xfri+xfir)
