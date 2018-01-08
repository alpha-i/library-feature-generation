import string
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from alphai_feature_generation.cleaning import (
    select_between_timestamps_data_frame,
    select_between_timestamps_data_dict,
    resample_data_frame,
    resample_data_dict,
    resample_ohlcv,
    select_above_floor_data_frame,
    select_above_floor_data_dict,
    select_below_ceiling_data_frame,
    select_below_ceiling_data_dict,
    fill_gaps_data_frame,
    fill_gaps_data_dict,
    interpolate_gaps_data_frame,
    convert_data_frame_to_utc,
    convert_data_dict_to_utc,
    select_trading_hours_data_frame,
    select_trading_hours_data_dict,
    sample_minutes_after_market_open_data_frame,
    select_columns_data_dict,
    find_duplicated_symbols_data_frame,
    remove_duplicated_symbols_ohlcv,
    swap_keys_and_columns,
)
from tests.helpers import (
    COLUMNS_OHLCV,
    sample_data_dict,
    sample_data_frame,
    sample_market_calendar,
    sample_hourly_ohlcv_data_dict,
    tmp_symbols,
)


def test_select_between_timestamps_data_frame():
    lower_bound_index = 10
    upper_bound_index = 10
    start_timestamp = sample_data_frame.index[lower_bound_index]
    end_timestamp = sample_data_frame.index[-upper_bound_index]
    selected_data_frame = select_between_timestamps_data_frame(sample_data_frame, start_timestamp, end_timestamp)
    assert len(selected_data_frame) == len(sample_data_frame) - (lower_bound_index + upper_bound_index - 1)


def test_select_between_timestamps_data_dict():
    lower_bound_index = 10
    upper_bound_index = 10
    dict_first_key = list(sample_data_dict.keys())[0]
    start_timestamp = sample_data_dict[dict_first_key].index[lower_bound_index]
    end_timestamp = sample_data_dict[dict_first_key].index[-upper_bound_index]
    selected_data_dict = select_between_timestamps_data_dict(sample_data_dict, start_timestamp, end_timestamp)
    for key in sample_data_dict.keys():
        assert len(selected_data_dict[key]) == len(sample_data_dict[key]) - (lower_bound_index + upper_bound_index - 1)


def test_resample_data_frame_wrong_resample_rule_type():
    with pytest.raises(ValueError):
        resample_data_frame(sample_data_frame, 15)
        resample_data_frame(sample_data_frame, '15')


def test_resample_data_frame_wrong_sampling_function_type():
    with pytest.raises(AssertionError):
        resample_data_frame(sample_data_frame, '15T', 12)
        resample_data_frame(sample_data_frame, '15T', ['mean', 'median'])


def test_resample_data_frame_wrong_sampling_function():
    with pytest.raises(AssertionError):
        resample_data_frame(sample_data_frame, '15T', 'wrong')


def test_resample_data_frame_rules():
    resample_rules = ['1T', '2T', '5T', '10T', '60T', '1H']
    expected_lengths = [782, 392, 158, 80, 14, 14]
    for resample_rule, expected_length in zip(resample_rules, expected_lengths):
        resampled_data_frame = resample_data_frame(sample_data_frame, resample_rule)
        assert len(resampled_data_frame) == expected_length


def test_resample_data_frame_functions():
    index = pd.date_range('1/1/2000', periods=9, freq='T')
    data_frame = pd.DataFrame(list(range(9)), index=index, columns=['col'])
    resample_rule = '3T'

    sampling_function = 'mean'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0.,  2.,  5.,  7.5])

    sampling_function = 'median'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0.,  2.,  5.,  7.5])

    sampling_function = 'sum'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0,  6, 15, 15])

    sampling_function = 'first'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0, 1, 4, 7])

    sampling_function = 'last'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0, 3, 6, 8])

    sampling_function = 'min'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0, 1, 4, 7])

    sampling_function = 'max'
    assert_almost_equal(resample_data_frame(data_frame, resample_rule, sampling_function)['col'], [0, 3, 6, 8])


def test_resample_data_dict_mean_for_all():
    resample_rules = ['1T', '2T', '5T', '10T', '60T', '1H']
    expected_lengths = [103609, 51841, 20737, 10369, 1729, 1729]
    for resample_rule, expected_length in zip(resample_rules, expected_lengths):
        resampled_data_dict = resample_data_dict(sample_data_dict, resample_rule)
        for key, resampled_data_frame in resampled_data_dict.items():
            assert len(resampled_data_frame) == expected_length


def test_resample_data_dict_sampling_functions():
    resample_rules = ['1T', '2T', '5T', '10T', '60T', '1H']
    expected_lengths = [103609, 51841, 20737, 10369, 1729, 1729]
    sampling_functions = {'dummy_key': 'sum'}
    for resample_rule, expected_length in zip(resample_rules, expected_lengths):
        resampled_data_dict = resample_data_dict(sample_data_dict, resample_rule,
                                                 sampling_functions)
        for key, resampled_data_frame in resampled_data_dict.items():
            assert len(resampled_data_frame) == expected_length


def test_select_above_floor_data_frame():
    floor = 0.
    select_above_floor_data_frame(sample_data_frame, floor).equals(
        sample_data_frame.drop('A1', axis=1))


def test_select_above_floor_data_dict():
    floor = 0.
    tmp_sample_data_dict = {'key1': sample_data_frame, 'key2': sample_data_frame}
    selected_data_dict = select_above_floor_data_dict(tmp_sample_data_dict, floor)
    for df in selected_data_dict.values():
        df.equals(sample_data_frame.drop('A1', axis=1))


def test_select_below_ceiling_data_frame():
    ceiling = 500.
    select_below_ceiling_data_frame(sample_data_frame, ceiling).equals(
        sample_data_frame.drop('A5', axis=1))


def test_select_below_ceiling_data_dict():
    ceiling = 500.
    tmp_sample_data_dict = {'key1': sample_data_frame, 'key2': sample_data_frame}
    selected_data_dict = select_below_ceiling_data_dict(tmp_sample_data_dict, ceiling)
    for df in selected_data_dict.values():
        df.equals(sample_data_frame.drop('A5', axis=1))


def test_fill_gaps_data_frame():
    fill_limit = 3
    fill_gaps_data_frame(sample_data_frame, fill_limit, dropna=False).equals(
        sample_data_frame.fillna(method='ffill', limit=3))

    fill_gaps_data_frame(sample_data_frame, fill_limit, dropna=True).equals(
        sample_data_frame.fillna(method='ffill', limit=3)
            .dropna(axis=1, how='any'))


def test_fill_gaps_data_dict():
    fill_limit = 3
    tmp_key_list = ['key1', 'key2']
    tmp_data_dict = {key: sample_data_frame for key in tmp_key_list}
    filled_gaps_drop_false_data_dict = fill_gaps_data_dict(tmp_data_dict, fill_limit, dropna=False)
    filled_gaps_drop_true_data_dict = fill_gaps_data_dict(tmp_data_dict, fill_limit, dropna=True)

    for key in tmp_key_list:
        filled_gaps_drop_false_data_dict[key].equals(
            sample_data_frame.fillna(method='ffill', limit=3))
        filled_gaps_drop_true_data_dict[key].equals(
            sample_data_frame.fillna(method='ffill', limit=3).dropna(axis=1, how='any'))


def make_simple_df():
    df = pd.DataFrame(np.array([np.stack(np.linspace(i, 10 * i, 10)) for i in range(10, 20)]).transpose())
    df.iloc[0:5, 0] = np.nan
    df.iloc[6, 1] = np.nan
    df.iloc[8, 1] = np.nan
    df.iloc[8:, 2] = np.nan
    df.iloc[1:5, 3] = np.nan
    df.iloc[:1, 4] = np.nan
    df.iloc[-1:, 5] = np.nan
    df.iloc[-5:, 6] = np.nan
    df.iloc[-2:, 8] = np.nan
    df.iloc[:2, 8] = np.nan
    df.iloc[4:5, 8] = np.nan
    df.iloc[:, 9] = np.nan
    return df


def test_interpolate_gaps_data_frame():
    limit = 3
    df = make_simple_df()

    interpolated_df = interpolate_gaps_data_frame(df, limit, dropna=False)
    expected_df = pd.DataFrame([[np.nan, 11.0, 12.0, 13.0, np.nan, 15.0, 16.0, 17.0, np.nan, np.nan],
                                [np.nan, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, np.nan, np.nan],
                                [np.nan, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, np.nan],
                                [np.nan, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 68.0, 72.0, np.nan],
                                [np.nan, 55.0, 60.0, np.nan, 70.0, 75.0, 80.0, 85.0, 90.0, np.nan],
                                [60.0, 66.0, 72.0, 78.0, 84.0, 90.0, 80.0, 102.0, 108.0, np.nan],
                                [70.0, 77.0, 84.0, 91.0, 98.0, 105.0, 80.0, 119.0, 126.0, np.nan],
                                [80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 80.0, 136.0, 144.0, np.nan],
                                [90.0, 99.0, 96.0, 117.0, 126.0, 135.0, np.nan, 153.0, 144.0, np.nan],
                                [100.0, 110.0, 96.0, 130.0, 140.0, 135.0, np.nan, 170.0, 144.0, np.nan]])
    assert interpolated_df.equals(expected_df)

    interpolated_df = interpolate_gaps_data_frame(df, limit, dropna=True)
    expected_df = pd.DataFrame([[11.0, 12.0, 15.0, 17.0],
                                [22.0, 24.0, 30.0, 34.0],
                                [33.0, 36.0, 45.0, 51.0],
                                [44.0, 48.0, 60.0, 68.0],
                                [55.0, 60.0, 75.0, 85.0],
                                [66.0, 72.0, 90.0, 102.0],
                                [77.0, 84.0, 105.0, 119.0],
                                [88.0, 96.0, 120.0, 136.0],
                                [99.0, 96.0, 135.0, 153.0],
                                [110.0, 96.0, 135.0, 170.0]], columns=[1, 2, 5, 7])
    assert interpolated_df.equals(expected_df)


def test_convert_data_frame_to_utc_localized_data():
    localized_data_frame = deepcopy(sample_data_frame)
    converted_data_frame = convert_data_frame_to_utc(localized_data_frame)
    assert str(converted_data_frame.index.tz) == 'UTC'


def test_convert_data_frame_to_utc_unlocalized_data():
    unlocalized_data_frame = deepcopy(sample_data_frame)
    unlocalized_data_frame = unlocalized_data_frame.tz_localize(None)
    converted_data_frame = convert_data_frame_to_utc(unlocalized_data_frame)
    assert str(converted_data_frame.index.tz) == 'UTC'


def test_convert_data_frame_to_utc_utc_localized_data():
    utc_localized_data_frame = deepcopy(sample_data_frame)
    utc_localized_data_frame = utc_localized_data_frame.tz_convert('UTC')
    converted_data_frame = convert_data_frame_to_utc(utc_localized_data_frame)
    assert str(converted_data_frame.index.tz) == 'UTC'
    assert converted_data_frame.equals(utc_localized_data_frame)


def test_convert_data_dict_to_utc_localized_data():
    localized_data_dict = deepcopy(sample_data_dict)
    for key, df in localized_data_dict.items():
        localized_data_dict[key] = df.tz_localize('America/New_York')
    converted_data_dict = convert_data_dict_to_utc(localized_data_dict)
    for key, df in converted_data_dict.items():
        assert str(df.index.tz) == 'UTC'


def test_convert_data_dict_to_utc_unlocalized_data():
    unlocalized_data_dict = deepcopy(sample_data_dict)
    converted_data_dict = convert_data_dict_to_utc(unlocalized_data_dict)
    for key, df in converted_data_dict.items():
        assert str(df.index.tz) == 'UTC'


def test_convert_data_dict_to_utc_utc_localized_data():
    utc_localized_data_dict = deepcopy(sample_data_dict)
    for key, df in utc_localized_data_dict.items():
        utc_localized_data_dict[key] = df.tz_localize('America/New_York').tz_convert('UTC')
    converted_data_dict = convert_data_dict_to_utc(utc_localized_data_dict)
    for key, df in converted_data_dict.items():
        assert str(df.index.tz) == 'UTC'
        assert df.equals(utc_localized_data_dict[key])


def test_select_trading_hours_data_frame():
    utc_localized_data_frame = convert_data_frame_to_utc(sample_data_frame)
    selected_data_frame = select_trading_hours_data_frame(utc_localized_data_frame, sample_market_calendar)
    assert selected_data_frame.index.time.max() == datetime(2000, 1, 1, 21, 0).time()
    assert selected_data_frame.index.time.min() == datetime(2000, 1, 1, 14, 30).time()


def test_select_trading_hours_data_dict():
    utc_localized_data_dict = convert_data_dict_to_utc(sample_data_dict)
    selected_data_dict = select_trading_hours_data_dict(utc_localized_data_dict, sample_market_calendar)
    assert selected_data_dict['dummy_key'].index.time.max() == datetime(2000, 1, 1, 21, 0).time()
    assert selected_data_dict['dummy_key'].index.time.min() == datetime(2000, 1, 1, 14, 30).time()


def test_select_trading_hours_data_frame_no_first_minute():
    utc_localized_data_frame = convert_data_frame_to_utc(sample_data_frame)
    selected_data_frame = \
        select_trading_hours_data_frame(utc_localized_data_frame, sample_market_calendar, include_start=False)
    assert selected_data_frame.index.time.max() == datetime(2000, 1, 1, 21, 0).time()
    assert selected_data_frame.index.time.min() == datetime(2000, 1, 1, 14, 31).time()


def test_select_trading_hours_data_dict_no_first_minute():
    utc_localized_data_dict = convert_data_dict_to_utc(sample_data_dict)
    selected_data_dict = \
        select_trading_hours_data_dict(utc_localized_data_dict, sample_market_calendar, include_start=False)
    assert selected_data_dict['dummy_key'].index.time.max() == datetime(2000, 1, 1, 21, 0).time()
    assert selected_data_dict['dummy_key'].index.time.min() == datetime(2000, 1, 1, 14, 31).time()


def test_resample_ohlcv_type_io():
    assert isinstance(resample_ohlcv(sample_hourly_ohlcv_data_dict, '2H'), dict)


def test_resample_ohlcv_size_output():
    resampled_data = resample_ohlcv(sample_hourly_ohlcv_data_dict, '2H')
    for column in COLUMNS_OHLCV:
        assert len(resampled_data[column]) / 4 == len(sample_hourly_ohlcv_data_dict[column]) / 7


def test_resample_ohlcv_resampling_method():
    resampled_data = resample_ohlcv(sample_hourly_ohlcv_data_dict, '2H')
    for column in COLUMNS_OHLCV:
        if column.lower() == 'volume':
            assert_almost_equal(
                resampled_data[column].sum() / sample_hourly_ohlcv_data_dict[column].sum(),
                len(COLUMNS_OHLCV) * [1.],
                decimal=2)
        else:
            assert_almost_equal(
                resampled_data[column].mean() / sample_hourly_ohlcv_data_dict[column].mean(),
                len(COLUMNS_OHLCV) * [1.],
                decimal=2)


def test_sample_minutes_after_market_open_data_frame():
    data_frame = sample_hourly_ohlcv_data_dict['close']
    sampled_data_frame = \
        sample_minutes_after_market_open_data_frame(data_frame, sample_market_calendar, 30)
    assert len(sampled_data_frame) == len(data_frame) / 7


def test_select_columns_data_dict_missing_columns():
    select_columns = ['CAT', 'DOG']
    with pytest.raises(KeyError):
        select_columns_data_dict(sample_hourly_ohlcv_data_dict, select_columns)


def test_select_columns_data_dict():
    select_columns = ['AAPL', 'GOOG']
    selected_data_dict = select_columns_data_dict(sample_hourly_ohlcv_data_dict, select_columns)

    for column in select_columns:
        for map_key in selected_data_dict.keys():
            assert column in selected_data_dict[map_key]
            assert selected_data_dict[map_key][column].equals(
                sample_hourly_ohlcv_data_dict[map_key][column])


def make_correlated_data_frame(mu, size):
    nseries = 6
    symbols = list(string.ascii_uppercase[:nseries])
    variances = 0.1 * mu
    correlation_matrix = np.identity(nseries)
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = -0.5
    correlation_matrix[0, 2] = correlation_matrix[2, 0] = 0.9999
    correlation_matrix[1, 2] = correlation_matrix[2, 1] = -0.5
    correlation_matrix[3, 4] = correlation_matrix[4, 3] = 0.9999
    correlation_matrix[3, 5] = correlation_matrix[5, 3] = 0.9999
    correlation_matrix[4, 5] = correlation_matrix[5, 4] = 1
    cov_matrix = variances * correlation_matrix
    return pd.DataFrame(np.random.multivariate_normal(mu, cov_matrix, size=size), columns=symbols)


def test_find_duplicated_symbols_data_frame():
    mu = np.array([20., 20., 20., 100., 100., 100])
    data_frame = make_correlated_data_frame(mu, 100)
    duplicated_symbols = find_duplicated_symbols_data_frame(data_frame)
    assert duplicated_symbols == [('A', 'C'), ('D', 'E'), ('D', 'F'), ('E', 'F')]


def test_remove_duplicated_symbols_ohlcv():
    mu_price = np.array([20., 20., 20., 100., 100., 100])
    mu_volume = np.array([1000., 2000., 3000., 1000., 2000., 3000.])
    expected_dropped_symbols = ['A', 'D', 'E']

    ohlcv_data = {
        'open': make_correlated_data_frame(mu_price, 100),
        'high': make_correlated_data_frame(mu_price, 100),
        'low': make_correlated_data_frame(mu_price, 100),
        'close': make_correlated_data_frame(mu_price, 100),
        'volume': make_correlated_data_frame(mu_volume, 100),
    }
    cleaned_ohlcv_data = remove_duplicated_symbols_ohlcv(ohlcv_data)

    for key in cleaned_ohlcv_data.keys():
        assert cleaned_ohlcv_data[key].equals(ohlcv_data[key].drop(expected_dropped_symbols, axis=1))


def test_swap_keys_and_columns():
    swapped_data_dict = swap_keys_and_columns(sample_data_dict)
    assert set(swapped_data_dict.keys()) == set(tmp_symbols)
    for key in swapped_data_dict.keys():
        assert set(swapped_data_dict[key].columns) == set(['dummy_key'])
