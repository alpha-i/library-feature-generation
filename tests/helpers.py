import os
import shutil

import numpy as np

TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'resources',
)

sample_hdf5_file = os.path.join(TEST_DATA_PATH, 'sample_hdf5.h5')

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

RTOL = 1e-5
ATOL = 1e-8

SAMPLE_START_DATE = 20140101
SAMPLE_END_DATE = 20140301
SAMPLE_SYMBOLS = ['AAPL', 'INTC', 'MSFT']

TEST_TEMP_FOLDER = os.path.join(
    TEST_DATA_PATH,
    'temp'
)


def create_temp_folder():
    if not os.path.exists(TEST_TEMP_FOLDER):
        os.makedirs(TEST_TEMP_FOLDER)


def destroy_temp_folder():
    shutil.rmtree(TEST_TEMP_FOLDER)


ASSERT_NDECIMALS = 5
