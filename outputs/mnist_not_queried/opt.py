import os
import numpy as np

TASK = "MNIST"
DIM = 1024

IMG_SIZE = (120, 120)

OBJ_SIZE = 30
LABEL_RADIUS = 13.0

STREAMS = [
    "Digits" if TASK == "MNIST" else "Shapes",
    "Color",
]
N_QUERIES = 14
TRAIN_TYPE = "non-query-based"

COLOR_CHOICES = 2  # max 7
SCALE_CHOICES = 0
ROTATION_CHOICES = 0

GLIMPSE_WIDTH = 29
GLIMPSES = 6

SAMPLES = 400  # training samples per scene
SAMPLE_RADIUS = 12.0

HIDDEN_LAYERS = [4096, 2048]

TRAINING_ITERATIONS = 6000
BATCH_SIZE = 40
SAVE_EVERY = 100

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

### Other stuff

SEED = 2

MAX_K = 6.0 * np.pi
MIN_K = 0.0
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

NOISE_FRAGMENTS = 0 # 10
MAX_NOISE_SIZE = 8
MIN_NOISE_SIZE = 6

# how many samples when computing cost
SAMPLE_STRATEGY = "mixed"

N_DIGITS = [0, 1, 2, 3, 4]
P_DIGITS = [0.1, 0.2, 0.3, 0.2, 0.2]

