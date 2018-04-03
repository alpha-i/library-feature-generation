import glob
import os

import numpy as np
import pandas as pd

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


def load_test_data(fixture_folder):
    sample_dict = {}

    for feature_file in glob.glob(os.path.join(TEST_DATA_PATH, fixture_folder, '*.csv')):

        data_frame = pd.read_csv(feature_file, index_col=0)
        data_frame.index = pd.to_datetime(data_frame.index, utc=True)
        feature_name = os.path.basename(feature_file).replace(".csv", "")
        sample_dict[feature_name] = data_frame

    return sample_dict
