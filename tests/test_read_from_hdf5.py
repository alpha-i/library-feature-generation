import numpy as np
import pandas as pd
import pytest

from tests.helpers import (
    SAMPLE_START_DATE,
    SAMPLE_END_DATE,
    SAMPLE_SYMBOLS,
    sample_hdf5_file,
)
from tests.hdf5_reader import read_symbol_data_dict_from_hdf5, read_feature_data_dict_from_hdf5, \
    get_all_table_names_in_hdf5


def test_symbol_not_present():
    with pytest.raises(KeyError):
        read_symbol_data_dict_from_hdf5(['CAT', 'DOG'], SAMPLE_START_DATE, SAMPLE_END_DATE, sample_hdf5_file)


def test_input_start_before_data():
    early_start = 20131201
    data_dict = read_symbol_data_dict_from_hdf5(SAMPLE_SYMBOLS, early_start, SAMPLE_END_DATE, sample_hdf5_file)
    assert data_dict[SAMPLE_SYMBOLS[0]].index[0] == pd.Timestamp('201401020400').tz_localize('America/New_York')


def test_input_end_after_data():
    late_end = 20140520
    data_dict = read_symbol_data_dict_from_hdf5(SAMPLE_SYMBOLS, SAMPLE_START_DATE, late_end, sample_hdf5_file)
    assert data_dict[SAMPLE_SYMBOLS[0]].index[-1] == pd.Timestamp('201402281958').tz_localize('America/New_York')


def test_input_end_before_input_start():
    late_start = 20140130
    early_end = 20140101
    with pytest.raises(AssertionError):
        read_symbol_data_dict_from_hdf5(SAMPLE_SYMBOLS, late_start, early_end, sample_hdf5_file)


def test_read_symbol_data_dict_from_hdf5_check_file_content():
    data_dict = read_symbol_data_dict_from_hdf5(SAMPLE_SYMBOLS, SAMPLE_START_DATE, SAMPLE_END_DATE, sample_hdf5_file)
    assert(set(data_dict.keys()) == set(SAMPLE_SYMBOLS))

    # check the summary statistics
    describe_symbol_0 = data_dict[SAMPLE_SYMBOLS[0]].describe().values[1]
    np.testing.assert_allclose(describe_symbol_0, [73.00,   73.03,   72.98,   73.01, 127526.52], atol=1e-2)

    describe_symbol_1 = data_dict[SAMPLE_SYMBOLS[1]].describe().values[2]
    np.testing.assert_allclose(describe_symbol_1, [0.616,   0.617,   0.616,   0.616, 171988.44], atol=1e-2)

    describe_symbol_2 = data_dict[SAMPLE_SYMBOLS[2]].describe().values[7]
    np.testing.assert_allclose(describe_symbol_2, [36.44, 36.45, 36.43, 36.44, 14024009.00], atol=1e-2)


def test_read_feature_data_dict_from_hdf5_check_file_content():
    data_dict = read_feature_data_dict_from_hdf5(SAMPLE_SYMBOLS, SAMPLE_START_DATE, SAMPLE_END_DATE, sample_hdf5_file)
    assert(set(data_dict.keys()) == {'close', 'high', 'low', 'open', 'volume'})

    # check the summary statistics
    # two checks on open
    open_describe_0 = data_dict['open'].describe().values[0]
    np.testing.assert_array_equal(open_describe_0, [24364.,  18665.,  19171.])

    open_describe_1 = data_dict['open'].describe().values[1]
    np.testing.assert_allclose(open_describe_1, [73.00616,  23.406957,  34.642669], atol=1e-6)

    # two check on low
    low_describe_0 = data_dict['low'].describe().values[2]
    np.testing.assert_allclose(low_describe_0, [2.352077,  0.615931,  0.793538], atol=1e-6)

    low_describe_1 = data_dict['low'].describe().values[3]
    np.testing.assert_allclose(low_describe_1, [67.53,  22.00, 32.58], atol=1e-6)

    # two check on high
    high_describe_0 = data_dict['high'].describe().values[7]
    np.testing.assert_allclose(high_describe_0, [76.89,  25.16,  36.45], atol=1e-6)

    high_describe_1 = data_dict['high'].describe().values[1]
    np.testing.assert_allclose(high_describe_1, [73.028583,  23.415788,  34.655273], atol=1e-6)

    # two tests on volume
    volume_describe_0 = data_dict['volume'].describe().values[2]
    np.testing.assert_allclose(volume_describe_0,
                               [265075.256502,  171988.439997,  247455.665898], atol=1e-6)

    volume_describe_1 = data_dict['volume'].describe().values[1]
    np.testing.assert_allclose(volume_describe_1,
                               [127526.523477,   74079.441682,   86996.518439], atol=1e-6)


def test_get_all_table_names_in_hdf5():
    assert get_all_table_names_in_hdf5(sample_hdf5_file) == SAMPLE_SYMBOLS
