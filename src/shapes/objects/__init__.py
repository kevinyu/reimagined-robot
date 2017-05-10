import numpy as np

import matplotlib.pyplot as plt

from shapes.config import OBJ_SIZE


z = 1.0 / np.sqrt(2)
PADDING = int(OBJ_SIZE - (OBJ_SIZE * z)) - 1
BASE_SIZE = OBJ_SIZE - PADDING


def pad(x, padding):
    return np.lib.pad(x, padding, mode="constant", constant_values=(0,))


def make_square():
    img = np.ones((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(0, 1, BASE_SIZE), np.linspace(0, 1, BASE_SIZE))

    return pad(img, PADDING)


def make_circle():
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(-1, 1, BASE_SIZE), np.linspace(-1, 1, BASE_SIZE))
    img[(x**2 + y**2) <= z] = 1
    return pad(img, PADDING)


def make_triangle():
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(-1, 1, BASE_SIZE), np.linspace(0, 1, BASE_SIZE))
    img[(x < 0) * (-x <= y)] = 1
    img[(0 < x) * (x <= y)] = 1
    return pad(img, PADDING)


def make_triangle_2():
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(0, 1, BASE_SIZE), np.linspace(0, 1, BASE_SIZE))
    img[x <= y] = 1
    return pad(img, PADDING)


def make_nub():
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(-1, 3, BASE_SIZE), np.linspace(0, 1, BASE_SIZE))
    img[(x < 0) * (y > 0.25) * (y < 0.75)] = 1
    img[x >= 0]=  1
    return pad(img, PADDING)


def make_hexagon():
    img = np.zeros((BASE_SIZE, BASE_SIZE))
    x, y = np.meshgrid(np.linspace(0, 2, BASE_SIZE), np.linspace(-1, 1, BASE_SIZE))
    img[(x < 0.5) * (y <= (.86 / .5) * x) * (y >= (-0.86 / 0.5) * x)] = 1
    img[(x > 0.5) * (x < 1.5) * (y > -1 + 0.134) * (y < 1.0 - 0.134)] = 1
    x, y = np.meshgrid(np.linspace(2, 0, BASE_SIZE), np.linspace(-1, 1, BASE_SIZE))
    img[(x < 0.5) * (y <= (.86 / .5) * x) * (y >= (-0.86 / 0.5) * x)] = 1
    return pad(img, PADDING)


shapes = [
        make_square(),
        make_circle(),
        make_triangle(),
        make_triangle_2(),
        make_nub(),
        make_hexagon()
]
