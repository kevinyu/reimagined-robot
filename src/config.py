import os

import numpy as np


SEED = 2

MAX_K = 18.0 * np.pi
MIN_K = 0.0

SHAPES = True
COLOR_CHANNELS = 3 if SHAPES else 1

# Hypervector dimensionality
DIM = 1024

# width of a glimpse window (not used if non-grid glimpses are used)
GLIMPSE_WIDTH = 29 
GLIMPSE_SIZE = COLOR_CHANNELS * 29 ** 2
GLIMPSE_STRATEGY = "mixed"
GLIMPSES = 5

if GLIMPSE_WIDTH % 2 == 1:
    GLIMPSE_ON = lambda x: [x - GLIMPSE_WIDTH / 2, x + 1 + GLIMPSE_WIDTH / 2]
else:
    GLIMPSE_ON = lambda x: [x - GLIMPSE_WIDTH / 2, x + GLIMPSE_WIDTH / 2]

# Size of images in dataset
IMG_WIDTH = 100
IMG_HEIGHT = 100
POS_SCALE = float(np.max([IMG_WIDTH, IMG_HEIGHT]))
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
NOISE_FRAGMENTS = 10
MAX_NOISE_SIZE = 11
MIN_NOISE_SIZE = 6

# how many samples when computing cost
SAMPLES = 300
SAMPLE_STRATEGY = "mixed"
SAMPLE_RADIUS = 13.0

# Outputs 1 [1024]
# Outputs 2 [1024]
# Outputs 3 [728, 728]
# Outputs 4 [728, 728]
# Outputs 5 [1024, 1024]
HIDDEN_LAYERS = [2048, 2048]

N_DIGITS = [0, 1, 2, 3, 4]
P_DIGITS = [0.1, 0.2, 0.3, 0.2, 0.2]
DICT_SIM_ALPHA = 0.0
LEARN_D = True

TRAINING_ITERATIONS = 6000
BATCH_SIZE = 30
SAVE_EVERY = 100

SAVE_DIR = os.path.join(os.getenv("PROJECT_ROOT"), "shapes_test")

LOAD_PREEXISTING = True
