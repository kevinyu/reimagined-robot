import os

import numpy as np
from sklearn.datasets import fetch_mldata
# from sklearn.preprocessing import OneHotEncoder

import config
from properties import properties, rand_color
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
        x_max = self.x_max - max_fragment_size
        y_max = self.y_max - max_fragment_size
        for i in range(n):
            digit_idx = np.random.randint(mnist_images.shape[0])
            fragment_size = np.random.randint(config.MIN_NOISE_SIZE, max_fragment_size)
            pick_x, pick_y = np.random.randint(28 - fragment_size, size=2)
            fragment = rand_color(mnist_images[digit_idx][pick_x:pick_x + fragment_size, pick_y:pick_y + fragment_size])
            x = np.random.randint(x_max)
            y = np.random.randint(y_max)
            if np.random.random() > 0.5:
                self.add_thing_at(x, y, fragment)
            else:
                self.add_thing_at(x, y, fragment.swapaxes(0, 1))

    def near_digits(self, within=None):
        """Return a grid the same shape as image with 1's indicating they are near a digit"""
        result = np.zeros(self.img.shape[:2])
        if within is None:
            within = config.LABEL_RADIUS
        # FIXME: x y flip flop?
        y, x = np.meshgrid(np.arange(self.x_max), np.arange(self.y_max))
        if not self.digit_locations:
            result = np.ones(self.img.shape[:2])
        for digit_x, digit_y in self.digit_locations:
            condition = (np.power(x - digit_x, 2) + np.power(y - digit_y, 2)) <= np.power(within, 2)
            result[np.where(condition)] = 1
        return result

    def padded(self, padding=None):
        result = np.zeros(self.img.shape[:2])
        if padding is None:
            padding = self.DIGIT_SIZE / 2

        x, y = np.meshgrid(np.arange(self.x_max), np.arange(self.y_max))
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
        picked = np.random.randint(len(choices), size=n)
        return [choices[pick] for pick in picked]

    def get_label_at(self, x, y):
        """Return a two one hot vectors for the digit/color
        whose presence is present at position (x, y)
        """
        # FIXME hardcoding digit and color sizes
        one_hot = np.zeros(11)
        one_hot_color = np.zeros(5)

        for digit_id, center_x, center_y, props in self.contents:
            color_idx = props[0][1]
            dist = np.power(x - center_x, 2) + np.power(y - center_y, 2)

            if dist < np.power(config.LABEL_RADIUS, 2):
                one_hot[int(digit_id)] = 1.0
                one_hot_color[color_idx] = 1.0

        if one_hot.sum() == 0.0:
            one_hot[-1] = 1.0

        if one_hot_color.sum() == 0.0:
            one_hot_color[-1] = 1.0

        return one_hot / one_hot.sum(), one_hot_color / one_hot_color.sum()



def generate_scene(img_shape):
    """Generate an image with multiple mnist digits located throughout
    """
    scene = MNISTScene(img_shape)

    put_x = None

    # FIXME randomly select digits that dont overlap... ugh
    for i in range(2 if config.SIMPLE_DATASET else 10):
        idx = np.random.randint(mnist_images.shape[0])
        obj = mnist_images[idx]

        props_info = []
        for prop in properties:
            param = prop.sample_param()
            obj = prop.transform(obj, param)
            props_info.append([prop.__name__, param])

        dx, dy = obj.shape[:2]

        if config.SIMPLE_DATASET:
            put_x = put_x or np.random.randint(scene.x_max - dx)
            if i == 0:
                put_y = np.random.randint(scene.y_max / 2 - dy)
            elif i == 1:
                put_y = np.random.randint(scene.y_max / 2, scene.y_max - dy)
        else:
            put_x = np.random.randint(scene.x_max - dx)
            put_y = np.random.randint(scene.y_max - dy)

        center_x, center_y = put_x + dx / 2.0, put_y + dy / 2.0

        if scene.digit_locations:
            nearest_dist = np.sqrt(np.min([(center_x - x)**2 + (center_y - y)**2 for x, y in scene.digit_locations]))
            if nearest_dist < 30.0:
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

    if config.TRAIN_TYPE == "query-based":
        query_directions, query_digits, query_colors, digit_labels, color_labels = generate_queries(
                scene, config.N_QUERIES)

        return scene, glimpse_data, glimpse_locs.T, query_directions, query_digits, query_colors, digit_labels, color_labels
    else:
        # sample locs is 2 x n_samples
        sample_data_digits, sample_data_color, sample_locs = take_samples(
                scene,
                n_samples=config.SAMPLES,
                strategy=config.SAMPLE_STRATEGY,
                within=config.SAMPLE_RADIUS)
        return scene, glimpse_data, glimpse_locs.T, sample_locs.T, sample_data_digits, sample_data_color


def make_batch(n):
    datas = []
    for _ in range(n):
        datas.append(make_one())
    return [np.array(a) for a in zip(*datas)]


def take_samples(scene, n_samples=1, strategy="smart", within=None):
    """Sample one hot vectors at various points in a scene

    Params:
    scene (mnist_scene.MNISTScene)
    n_samples (int, default=1)
        Number of samples to take
    strategy (str, "smart" or "uniform")
        Take samples near digits only, or just uniformly throughout image

    Returns:
    one_hot (np.array, n_samples x 11)
        one hot vectors corresponding to digit identities in image
    sample_locations (np.array, n_samples x 2)
    """
    # FIXME: hardcoding one hot lengths and numbers
    one_hot = np.zeros((n_samples + 30, 11))
    one_hot_colors = np.zeros((n_samples + 30, 5))

    x_max = scene.img.shape[0]
    y_max = scene.img.shape[1]
    if strategy == "smart":
        sample_locations = scene.sample_near_digits(n=n_samples + 30, within=within)
    elif strategy == "uniform":
        sample_locations = [
                [np.random.randint(x_max), np.random.randint(y_max)]
                for _ in range(n_samples + 30)]
    elif strategy == "mixed":
        odd = n_samples % 2
        sample_locations = (
                scene.sample_near_digits(n=n_samples, within=within) +
                [[np.random.randint(x_max), np.random.randint(y_max)] for _ in range(30)]
        )

    for i, (x, y) in enumerate(sample_locations):
        one_hot[i], one_hot_colors[i] = scene.get_label_at(x, y)

    return one_hot, one_hot_colors, np.array(sample_locations).T

