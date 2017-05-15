import os

import numpy as np
from sklearn.datasets import fetch_mldata
# from sklearn.preprocessing import OneHotEncoder

import config
from properties import properties, rand_color
from utils import cartesian
from query_scene import generate_queries
from glimpse import take_glimpses


data_home = os.path.join(os.getenv("PROJECT_ROOT"), "media", "datasets", "mnist")
mnist = fetch_mldata("MNIST original", data_home=data_home)
mnist_images = mnist["data"].reshape(len(mnist["data"]), 28, 28)
mnist_labels = mnist["target"]


class MNISTScene(object):
    """An mnist scene with ground truth labels
    
    Image coordinates are the min x and min y values (not the centers!)
    """

    def __init__(self, img_shape):
        self.x_max, self.y_max = img_shape
        self.img = np.zeros((self.x_max, self.y_max, 3))
        self.contents = []

    @property
    def digit_locations(self):
        return [(x, y) for _, x, y, _ in self.contents]

    def valid_xy(self, obj):
        """Return valid x and y (min x, min y) for placing object"""
        dx, dy = obj.shape[:2]
        x_max, y_max = self.img.shape[:2]
        return np.arange(x_max - dx), np.arange(y_max - dy)

    def add_thing_at(self, x, y, obj):
        obj= obj.swapaxes(0, 1)  # FIXME I DONT FUCKING GET THIS
        dx, dy = obj.shape[:2]
        if x < 0 or y < 0 or x + dx > self.x_max or y + dy > self.y_max:
            raise Exception("Cannot add fragment size {} at ({}, {}), image is {}".format(
                obj.shape, x, y, self.img.shape))
        previous_contents = self.img[x:x + dx, y:y + dy]
        self.img[x:x + dx, y:y + dy] = np.maximum(previous_contents, obj)

    def add_digit_at(self, x, y, obj, digit_id, digit_properties):
        """Add given mnist digit to the scene

        Params:
        x (int) : where to place digit (min corner)
        y (int)
        digit_image (np.array, obj_width x obj_height)
        digit_id (int, [0, 10])
        digit_properties (list):
            Tuple pairs for each property and index of paramter value
            e.g. [("Color", 0), ... ]
        """
        dx, dy = obj.shape[:2]
        self.add_thing_at(x, y, obj)
        self.contents.append((digit_id, x + dx /2, y + dy/2, digit_properties))

    def add_fragment_noise(self, n, max_fragment_size):
        choices = cartesian(self.x_max - max_fragment_size, self.y_max - max_fragment_size)
        for i in range(n):
            digit_idx = np.random.randint(mnist_images.shape[0])
            fragment_size = np.random.choice(np.arange(config.MIN_NOISE_SIZE, max_fragment_size))
            pick_x, pick_y = np.random.choice(np.arange(28 - fragment_size), size=2)
            fragment = rand_color(mnist_images[digit_idx][pick_x:pick_x + fragment_size, pick_y:pick_y + fragment_size])
            x, y = choices[np.random.choice(len(choices))]
            if np.random.random() > 0.5:
                self.add_thing_at(x, y, fragment)
            else:
                self.add_thing_at(x, y, fragment.swapaxes(0, 1))

    def near_digits(self, within=None):
        """Return a grid the same shape as image with 1's indicating they are near a digit"""
        result = np.zeros(self.img.shape[:2])
        if within is None:
            within = self.label_radius
        y, x = np.meshgrid(np.arange(self.x_max), np.arange(self.y_max))
        if not self.digit_locations:
            result = np.ones(self.img.shape[:2])
        for digit_x, digit_y in self.digit_locations:
            d_half = 28.0 / 2.0
            center_x, center_y = (digit_x + d_half, digit_y + d_half)
            condition = (np.power(x - center_x, 2) + np.power(y - center_y, 2)) <= np.power(within, 2)
            result[np.where(condition)] = 1
        return result

    def padded(self, padding=None):
        result = np.zeros(self.img.shape[:2])
        if padding is None:
            padding = self.DIGIT_SIZE / 2

        y, x = np.meshgrid(np.arange(self.x_max), np.arange(self.y_max))
        condition = (
                (padding <= x) *
                (x < (self.x_max - padding)) *
                (padding <= y) *
                (y < (self.y_max - padding))
        )
        result[np.where(condition)] = 1
        return result

    def sample_near_digits(self, n=1, within=None, padding=0):
        """Return a sample of coordinates that are near digit locations"""
        if padding:
            choices = zip(*np.where(self.near_digits(within=within) * self.padded(padding)))
        else:
            choices = zip(*np.where(self.near_digits(within=within)))
        picked = np.random.choice(len(choices), size=n)
        return [choices[pick] for pick in picked]


def generate_scene(img_shape):
    """Generate an image with multiple mnist digits located throughout
    """
    scene = MNISTScene(img_shape)

    # FIXME randomly select digits that dont overlap... ugh
    for i in range(10):
        idx = np.random.randint(mnist_images.shape[0])
        obj = mnist_images[idx]

        props_info = []
        for prop in properties:
            param = prop.sample_param()
            obj = prop.transform(obj, param)
            props_info.append([prop.__name__, param])

        dx, dy = obj.shape[:2]
        choices = cartesian(*[len(v) for v in scene.valid_xy(obj)])
        # TODO only use this when debugging, this just centers the image in every scene
        # choices = [(30, 60)]

        put_x, put_y = choices[np.random.choice(len(choices))]

        center_x, center_y = put_x + dx / 2.0, put_y + dy / 2.0

        if scene.digit_locations:
            nearest_dist = np.sqrt(np.min([(center_x - x)**2 + (center_y - y)**2 for x, y in scene.digit_locations]))
            if nearest_dist < 40.0:
                # skip becuase its too damn close to another digit
                continue

        scene.add_digit_at(put_x, put_y, obj, mnist_labels[idx], props_info)

    return scene


def make_one(glimpse_strategy=None):
    scene = generate_scene(config.IMG_SIZE)

    if config.NOISE_FRAGMENTS:
        scene.add_fragment_noise(config.NOISE_FRAGMENTS, config.MAX_NOISE_SIZE)

    # glimpse locs is 2 x n_glimpses
    # FIXME: need to fix the take_glimpses code to use the new MNISTScene object...
    # and to work with three color channels
    glimpse_data, glimpse_locs = take_glimpses(
            scene,
            glimpse_width=config.GLIMPSE_WIDTH,
            n_glimpses=config.GLIMPSES,
            strategy=glimpse_strategy or config.GLIMPSE_STRATEGY)

    query_directions, query_digits, query_colors, digit_labels, color_labels = generate_queries(
            scene, config.N_QUERIES)

    return scene, glimpse_data, glimpse_locs.T, query_directions, query_digits, query_colors, digit_labels, color_labels


def make_batch(n):
    datas = []
    for _ in range(n):
        datas.append(make_one())
    return [np.array(a) for a in zip(*datas)]


