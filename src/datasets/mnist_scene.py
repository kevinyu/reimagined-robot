import numpy as np

from sklearn.datasets import fetch_mldata
# from sklearn.preprocessing import OneHotEncoder

import config
from utils import cartesian

import os

data_home = os.path.join(os.getenv("PROJECT_ROOT"), "media", "datasets", "mnist")
mnist = fetch_mldata("MNIST original", data_home=data_home)
mnist_images = mnist["data"].reshape(len(mnist["data"]), 28, 28)
mnist_labels = mnist["target"]


class MNISTScene(object):
    """An mnist scene with ground truth labels
    
    Image coordinates are the min x and min y values (not the centers!)
    
    Defines a region around the center of each digit with the correct label"""

    DIGIT_SIZE = 28

    def __init__(self, img_shape, label_radius=14.0):
        self.label_radius = label_radius
        self.x_max, self.y_max = img_shape

        self.img = np.zeros(img_shape)
        self.digit_locations = []

    @property
    def valid_x_max(self):
        return self.x_max - self.DIGIT_SIZE

    @property
    def valid_y_max(self):
        return self.y_max - self.DIGIT_SIZE

    def near_digits(self, within=None):
        """Return a grid the same shape as image with 1's indicating they are near a digit"""
        result = np.zeros_like(self.img)
        if within is None:
            within = self.label_radius
        y, x = np.meshgrid(np.arange(self.x_max), np.arange(self.y_max))
        if not self.digit_locations:
            result = np.ones_like(self.img)
        for _, (digit_x, digit_y) in self.digit_locations:
            d_half = self.DIGIT_SIZE / 2.0
            center_x, center_y = (digit_x + d_half, digit_y + d_half)
            condition = (np.power(x - center_x, 2) + np.power(y - center_y, 2)) <= np.power(within, 2)
            result[np.where(condition)] = 1
        return result

    def padded(self, padding=None):
        result = np.zeros_like(self.img)
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

    def get_label_at(self, x, y):
        """Return a one hot vector for the digits whose presence is present at position (x, y)
        """
        one_hot = np.zeros(11)
        for digit_id, (digit_x, digit_y) in self.digit_locations:
            d_half = self.DIGIT_SIZE / 2.0
            center = (digit_x + d_half, digit_y + d_half)
            dist = np.power(x - center[0], 2) + np.power(y - center[1], 2)

            if dist < np.power(self.label_radius, 2):
                one_hot[int(digit_id)] = 1.0

        if one_hot.sum() == 0.0:
            one_hot[-1] = 1.0

        return one_hot / one_hot.sum()

    def add_digit_at(self, x, y, digit_image, digit_label):
        """Add given mnist digit to the scene

        Params:
        x (int)
        y (int)
            location in image to place digit
        digit_image (np.array, 28 x 28)
        digit_label (int, [0, 11))
        """
        self.add_thing_at(x, y, digit_image)
        self.digit_locations.append((digit_label, (x, y)))

    def add_thing_at(self, x, y, fragment):
        dx, dy = fragment.shape
        if x < 0 or y < 0 or x + dx > self.x_max or y + dy > self.y_max:
            raise Exception("Cannot add fragment size {} at ({}, {}), image is {}".format(
                fragment.shape, x, y, self.img.shape))
        previous_contents = self.img[x:x + dx, y:y + dy]
        self.img[x:x + dx, y:y + dy] = np.maximum(previous_contents, fragment)

    def add_fragment_noise(self, n, max_fragment_size):
        choices = cartesian(self.x_max - max_fragment_size, self.y_max - max_fragment_size)
        for i in range(n):
            digit_idx = np.random.randint(mnist_images.shape[0])
            fragment_size = np.random.choice(np.arange(config.MIN_NOISE_SIZE, max_fragment_size))
            pick_x, pick_y = np.random.choice(np.arange(self.DIGIT_SIZE - fragment_size), size=2)
            fragment = mnist_images[digit_idx][pick_x:pick_x + fragment_size, pick_y:pick_y + fragment_size]
            x, y = choices[np.random.choice(len(choices))]
            if np.random.random() > 0.5:
                self.add_thing_at(x, y, fragment)
            else:
                self.add_thing_at(x, y, fragment.T)


def generate_single_digit_scene(img_shape):
    """Generate an image with one mnist digit
    """
    scene = MNISTScene(img_shape)
    # randomly select digit
    digit_idx = np.random.randint(mnist_images.shape[0])
    choices = cartesian(scene.valid_x_max, scene.valid_y_max)
    x, y = choices[np.random.choice(len(choices))]
    scene.add_digit_at(x, y, mnist_images[digit_idx], mnist_labels[digit_idx])
    return scene


def generate_n_digit_scene(img_shape, n_digits):
    """Generate an image with multiple mnist digits located throughout
    """
    scene = MNISTScene(img_shape)
    # randomly select digits
    digit_indices = np.random.randint(mnist_images.shape[0], size=n_digits)
    choices = cartesian(scene.valid_x_max, scene.valid_y_max)
    # choices = [(30, 60)]  # TODO remove, this just centers the image in every scene

    for digit_idx in digit_indices:
        x, y = choices[np.random.choice(len(choices))]
        scene.add_digit_at(x, y, mnist_images[digit_idx], mnist_labels[digit_idx])
    return scene


def generate_several_digit_scene(img_shape, n_digit_range):
    """Generate an image with a variable number within the range n_digit_range

    n_digit_range (tuple):
        0: lower bound on number of digits in scene (inclusive)
        1: upper bound on number of digits in scene (exclusive)
    """
    n_digits = np.random.choice(np.arange(*n_digit_range))
    return generate_n_digit_scene(img_shape, n_digits)


