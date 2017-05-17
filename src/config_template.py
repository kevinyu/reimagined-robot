import os
import numpy as np

TASK = "MNIST"
DIM = 1024

IMG_SIZE = (100, 100)

OBJ_SIZE = 30
LABEL_RADIUS = OBJ_SIZE

STREAMS = [
    "Digits" if TASK == "MNIST" else "Shapes",
    "Color",
]

COLOR_CHOICES = 2  # max 7
SCALE_CHOICES = 0
ROTATION_CHOICES = 0

N_QUERIES = 12

GLIMPSE_WIDTH = 29
GLIMPSES = 3

SAMPLES = 300  # training samples per scene
SAMPLE_RADIUS = 13.0

HIDDEN_LAYERS = [1024, 1024]

TRAINING_ITERATIONS = 6000
BATCH_SIZE = 30
SAVE_EVERY = 100

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

### Other stuff

SEED = 2

RANDOMIZE_POSITION_ENCODING = False

# This effectively adjusts how much bandwidth is devoted to representing positions.
PLANCK_LENGTH = 2

# PLANCK_LENGTH can be thought of as the smallest discernable distance:
# All location vectors with distance > PLANCK_LENGTH will have a very small dot product.
# It is specified in pixels


COLOR_CHANNELS = 3 if "Color" in STREAMS else 1

GLIMPSE_SIZE = COLOR_CHANNELS * 29 ** 2
GLIMPSE_STRATEGY = "mixed"

if GLIMPSE_WIDTH % 2 == 1:
    GLIMPSE_ON = lambda x: [x - GLIMPSE_WIDTH / 2, x + 1 + GLIMPSE_WIDTH / 2]
else:
    GLIMPSE_ON = lambda x: [x - GLIMPSE_WIDTH / 2, x + GLIMPSE_WIDTH / 2]

# Size of images in dataset
IMG_WIDTH = IMG_SIZE[0]
IMG_HEIGHT = IMG_SIZE[1]
POS_SCALEFACTOR = float(np.max([IMG_WIDTH, IMG_HEIGHT])) / 2.0
POS_SCALE = lambda x: -1 + (x / POS_SCALEFACTOR)

NOISE_FRAGMENTS = 10
MAX_NOISE_SIZE = 8
MIN_NOISE_SIZE = 6

# how many samples when computing cost
SAMPLE_STRATEGY = "mixed"

N_DIGITS = [0, 1, 2, 3, 4]
P_DIGITS = [0.1, 0.2, 0.3, 0.2, 0.2]

