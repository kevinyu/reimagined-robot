import numpy as np
from scipy.misc import imresize, imrotate

import config
from shapes import objects


class Property(object):
    @classmethod
    def sample_params(cls):
        return [np.random.choice(range(len(cls.params)))]


class Color(Property):
    params = [
        "white",
        "red",
        "green",
        "blue",
        "something",
        "or another",
        "or another one",
        ][:config.COLOR_CHOICES]

    @classmethod
    def transform(cls, obj, params):
        color = cls.params[params[0]]

        zero = np.zeros_like(obj)

        if color == "white":
            return np.array([obj * 1, obj * 1, obj * 1]).swapaxes(0, 2)
        elif color == "red":
            return np.array([obj * 1, obj * 0, obj * 0]).swapaxes(0, 2)
        elif color == "green":
            return np.array([obj * 0, obj * 1, obj * 0]).swapaxes(0, 2)
        elif color == "blue":
            return np.array([obj * 0, obj * 0, obj * 1]).swapaxes(0, 2)
        elif color == "something":
            return np.array([obj * 1, obj * 0, obj * 1]).swapaxes(0, 2)
        elif color == "or another":
            return np.array([obj * 1, obj * 1, obj * 0]).swapaxes(0, 2)
        elif color == "or another one":
            return np.array([obj * 0, obj * 1, obj * 1]).swapaxes(0, 2)


class Scale(Property):
    params = [
        "0.75",
        "1.0",
        "1.25"
    ][:config.SCALE_CHOICES]

    @classmethod
    def transform(cls, obj, params):
        scale = float(cls.params[params[0]])
        return imresize(np.array(obj, dtype=np.uint8), scale)


class Rotation(Property):
    params = [
        "0.0",
        "20.0",
        "-20.0",
    ][:config.ROTATION_CHOICES]

    @classmethod
    def transform(cls, obj, params):
        rot = float(cls.params[params[0]])
        return imrotate(np.array(obj, dtype=np.uint8), rot)


properties = []
if config.COLOR_CHOICES:
    properties.append(Color)
if config.SCALE_CHOICES:
    properties.append(Scale)
if config.ROTATION_CHOICES:
    properties.append(Rotation)


property_combinations = []
N_objects = len(objects.shapes)
for prop in properties:
    N_objects *= len(prop.params)


class Scene(object):
    def __init__(self, size, color_channels=3):
        self.img = np.zeros((size, size, color_channels))

    def valid_xy(self, obj):
        """for object placement"""
        dx, dy = obj.shape[:2]
        x_max, y_max = self.img.shape[:2]
        return np.arange(x_max - dx), np.arange(y_max - dy)

    def place(self, obj, x_min, y_min):
        """Place an object at a position in the image
        """
        dx, dy = obj.shape[:2]
        what_used_to_be_there = self.img[x_min:x_min + dx, y_min:y_min + dy]
        self.img[x_min:x_min + dx, y_min:y_min + dy] = np.maximum(what_used_to_be_there, obj[:, :])


def generate():
    """
    Return two arrays

    metadata on object identities and properties
    actual data of images and object info
    """
    scene = Scene(config.IMG_SIZE[0], color_channels=config.COLOR_CHANNELS)

    labels = []

    prev_locations = []

    for i in range(20):
        obj_id = np.random.choice(len(objects.shapes))
        obj = objects.shapes[obj_id]

        props_info = []
        for prop in properties:
            params = prop.sample_params()
            obj = prop.transform(obj, params)
            props_info.append([prop.__name__, params])

        valid_x, valid_y = scene.valid_xy(obj)
        put_x = np.random.choice(valid_x)
        put_y = np.random.choice(valid_y)

        center_x, center_y = put_x + obj.shape[0] / 2.0, put_y + obj.shape[1] / 2.0
        if prev_locations:
            nearest_dist = np.sqrt(np.min([(center_x - x)**2 + (center_y - y)**2 for x, y in prev_locations]))
            if nearest_dist < config.OBJ_SIZE:
                continue

        prev_locations.append((center_x, center_y))

        labels.append([obj_id, center_x, center_y, props_info])
        scene.place(obj, put_x, put_y)

    return scene, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    a, labels = generate()
    print labels
    plt.imshow(a.img)
    plt.show()
