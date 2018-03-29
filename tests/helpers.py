import os

import numpy as np

TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'resources')

EPS = 1e-10
N_BINS = 10
N_EDGES = N_BINS + 1
N_DATA = 100
MIN_EDGE = 0
MAX_EDGE = 10

TEST_EDGES = np.linspace(MIN_EDGE, MAX_EDGE, num=N_EDGES)
TEST_BIN_CENTRES = np.linspace(0.5, 9.5, num=N_BINS)
TEST_ARRAY = np.linspace(MIN_EDGE + EPS, MAX_EDGE - EPS, num=N_DATA)
TEST_TRAIN_LABELS = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))

