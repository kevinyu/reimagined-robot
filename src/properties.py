import numpy as np
from scipy.misc import imresize, imrotate

import config


class Property(object):
    @classmethod
    def sample_param(cls):
        return np.random.choice(range(len(cls.params)))


class Color(Property):
    params = [
        "red",
        "blue",
        "green",
        "yellow",
    ]

    @classmethod
    def transform(cls, obj, param):
        color = cls.params[param]
        scale = np.max(obj)
        obj = obj > 0

        # if the thing is already 3 color channel, take the max intensity
        # intensity and then remap to the right color

        if color == "red":
            obj = np.array([obj * 255., obj * 0., obj * 0.]).swapaxes(0, 2)
        elif color == "green":
            obj = np.array([obj * 0., obj * 255., obj * 0.]).swapaxes(0, 2)
        elif color == "blue":
            obj = np.array([obj * 0., obj * 0., obj * 255.]).swapaxes(0, 2)
        elif color == "yellow":
            obj = np.array([obj * 255., obj * 255., obj * 0.]).swapaxes(0, 2)
        else:
            raise Exception("Invalid color")
        return obj


def rand_color(obj):
    """take a 1 color channel image, and turn it into a random color"""
    obj = obj > 0

    color = np.random.choice(["red", "green", "blue", "yellow"])
    if color == "red":
        obj = np.array([obj * 255., obj * 0., obj * 0.]).swapaxes(0, 2)
    elif color == "green":
        obj = np.array([obj * 0., obj * 255., obj * 0.]).swapaxes(0, 2)
    elif color == "blue":
        obj = np.array([obj * 0., obj * 0., obj * 255.]).swapaxes(0, 2)
    elif color == "yellow":
        obj = np.array([obj * 255., obj * 255., obj * 0.]).swapaxes(0, 2)

    return obj


properties = []
if "Color" in config.STREAMS:
    properties.append(Color)


