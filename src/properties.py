import numpy as np
from scipy.misc import imresize, imrotate

import config


class Property(object):
    @classmethod
    def sample_params(cls):
        return [np.random.choice(range(len(cls.params)))]


class Color(Property):
    params = [
        "red",
        "blue",
        "green",
        "yellow",
    ]

    @classmethod
    def transform(cls, obj, params):
        color = cls.params[params[0]]

        zero = np.zeros_like(obj)

        # if the thing is already 3 color channel, take the max intensity
        # intensity and then remap to the right color
        if obj.ndim == 3:
            obj = np.max(obj, axis=2)

        if color == "red":
            obj = np.array([obj * 255., obj * 0., obj * 0.]).swapaxes(0, 2)
        if color == "green":
            obj = np.array([obj * 0., obj * 255., obj * 0.]).swapaxes(0, 2)
        if color == "blue":
            obj = np.array([obj * 0., obj * 0., obj * 255.]).swapaxes(0, 2)
        if color == "yellow":
            obj = np.array([obj * 255., obj * 255., obj * 0.]).swapaxes(0, 2)
        else:
            raise Exception("Invalid color")


properties = []
if "Color" in config.STREAMS:
    properties.append(Color)


