import os
from datetime import timedelta
from itertools import combinations

import pandas as pd
from dateutil import rrule

from alphai_feature_generation.transformation import FinancialDataTransformation
from tests.helpers import TEST_DATA_PATH

sample_fin_data_transf_feature_fixed_length = [
    {
        'name': 'close',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'log-return'},
        'is_target': False,
    },
    {
        'name': 'close',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'ewma', 'halflife': 6},
        'is_target': False,
    },
    {
        'name': 'high',
        'normalization': 'standard',
        'resolution': 150,
        'length': 2,
        'transformation': {'name': 'log-return'},
        'is_target': True
    },
]

sample_feature_configuration_list = [
    {
        'name': 'open',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': None,
        'is_target': False,
        'local': True
    },
    {
        'name': 'close',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': None,
        'is_target': False,
        'local': False
    },
    {
        'name': 'high',
        'transformation': {'name': 'log-return'},
        'normalization': 'standard',
        'nbins': 5,
        'is_target': True,
        'local': False
    },
]

def load_preset_config(expected_n_symbols, iteration=0):
    config = {
        'feature_config_list': sample_feature_configuration_list,
          'features_ndays': 2,
          'features_resample_minutes': 60,
          'features_start_market_minute': 1,
          FinancialDataTransformation.KEY_EXCHANGE: 'NYSE',
          'prediction_frequency_ndays': 1,
          'prediction_market_minute': 30,
          'target_delta': timedelta(days=5),
          'target_market_minute': 30,
          'n_classification_bins': 5,
          'n_assets': expected_n_symbols,
          'local': False,
          'classify_per_series': True,
          'normalise_per_series': False,
          'fill_limit': 0
    }

    specific_cases = [
        {},
        {'predict_the_market_close': True},
        {'classify_per_series': True, 'normalise_per_series': True},
        {'feature_config_list': sample_fin_data_transf_feature_fixed_length}
    ]

    try:
        updated_config = specific_cases[iteration]
        config.update(updated_config)
    except KeyError:
        raise ValueError('Requested configuration not implemented')

    return config


def load_expected_results(iteration):
    return_value_list = [
        {'x_mean': 207.451975429, 'y_mean': 0.2},
        {'x_mean': 207.451975429, 'y_mean': 0.2},  # Test predict_the_market_close
        {'x_mean': 207.451975429, 'y_mean': 0.2},  # Test classification and normalisation
        {'x_mean': 5.96046e-09, 'y_mean': 0.2},  # Test length/resolution requests
    ]

    try:
        return_value = return_value_list[iteration]
        expected_sample = [107.35616667, 498.748, 35.341, 288.86503167]
        return return_value['x_mean'], return_value['y_mean'], expected_sample
    except KeyError:
        raise ValueError('Requested configuration not implemented')


COLUMNS_OHLCV = 'open high low close volume'.split()


def build_ohlcv_sample_dataframe():
    sample_dict = {}
    for key in COLUMNS_OHLCV:
        sample_hourly_ohlcv_data_column = pd.read_csv(
            os.path.join(TEST_DATA_PATH, 'financial_data_dict', 'sample_%s_hourly.csv' % key),
            index_col=0)
        sample_hourly_ohlcv_data_column.index = pd.to_datetime(sample_hourly_ohlcv_data_column.index,
                                                               utc=True)
        sample_dict[key] = sample_hourly_ohlcv_data_column

    return sample_dict


def create_sample_historical_universe(ohlcv_sample):
    historical_universe = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])

    sample_hourly_ohlcv_data_symbols = ohlcv_sample['open'].columns
    start_date = ohlcv_sample['open'].index[0]
    end_date = ohlcv_sample['open'].index[-1]
    universe_combination_list = list(combinations(sample_hourly_ohlcv_data_symbols, 4))
    rrule_dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))

    for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
        historical_universe.loc[idx] = [
            period_start_date.date(),
            period_end_date.date(),
            list(universe_combination_list[idx % len(universe_combination_list)])
        ]
    return historical_universe


def build_ohlcv_daily_sample():
    sample_dict = {}
    for feature_name in COLUMNS_OHLCV:
        filename = os.path.join(TEST_DATA_PATH, 'financial_data_dict', 'sample_%s_daily.csv' % feature_name)
        sample_for_feature = pd.read_csv(filename, index_col=0)
        sample_for_feature.index = pd.to_datetime(sample_for_feature.index)
        sample_dict[feature_name] = sample_for_feature

    return sample_dict


sample_daily_ohlcv_data = build_ohlcv_daily_sample()
sample_ohlcv_hourly = build_ohlcv_sample_dataframe()
sample_historical_universes = create_sample_historical_universe(sample_ohlcv_hourly)
