import os
from datetime import timedelta
from itertools import combinations

import pandas as pd
from dateutil import rrule

from tests.helpers import TEST_DATA_PATH

gym_sample_hourly = {}

COLUMNS_FEATURES = """day_of_week 
                    hour
                    is_during_semester
                    is_holiday
                    is_start_of_semester
                    is_weekend
                    month
                    number_people
                    temperature
                    """.split()

for key in COLUMNS_FEATURES:
    gym_sample_hourly_ohlcv_data_column = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'gym_data_dict', 'sample_%s_hourly.csv' % key),
        index_col=0)
    gym_sample_hourly_ohlcv_data_column.index = pd.to_datetime(gym_sample_hourly_ohlcv_data_column.index,
                                                           utc=True)
    gym_sample_hourly[key] = gym_sample_hourly_ohlcv_data_column


sample_features_fixed_length = [
    {
        'name': 'hour',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'value'},
        'is_target': False,
    },
    {
        'name': 'hour',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'ewma', 'halflife': 6},
        'is_target': False,
    },
    {
        'name': 'number_people',
        'normalization': 'standard',
        'resolution': 150,
        'length': 2,
        'transformation': {'name': 'value'},
        'is_target': True
    },
]
sample_feature_list = [
    {
        'name': 'hour',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': None,
        'length': 5,
        'is_target': False,
        'local': True
    },
    {
        'name': 'temperature',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': None,
        'is_target': False,
        'local': False,
        'length': 5,
    },
    {
        'name': 'number_people',
        'transformation': {'name': 'value'},
        'normalization': 'standard',
        'nbins': 5,
        'is_target': True,
        'local': False,
        'length': 5,
    },
]


gym_sample_hourly_data_length = len(gym_sample_hourly['hour'])
gym_sample_hourly_symbols = gym_sample_hourly['hour'].columns
universe_length = gym_sample_hourly_data_length - 1
start_date = gym_sample_hourly['hour'].index[0]
end_date = gym_sample_hourly['hour'].index[-1]
rrule_dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))

universe_combination_list = list(combinations(gym_sample_hourly_symbols, 1))
sample_historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])

for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
    sample_historical_universes.loc[idx] = [
        period_start_date.date(),
        period_end_date.date(),
        list(universe_combination_list[idx % len(universe_combination_list)])
    ]

def load_preset_config(expected_n_symbols, iteration=0):
    config = {
        'feature_config_list': sample_feature_list,
        'features_ndays': 2,
        'features_resample_minutes': 60,
        'features_start_market_minute': 1,
        'calendar_name': 'GYMUK',
        'prediction_frequency_ndays': 1,
        'prediction_market_minute': 60,
        'target_delta': timedelta(days=5),
        'target_market_minute': 60,
        'n_classification_bins': 5,
        'n_assets': expected_n_symbols,
        'local': False,
        'classify_per_series': False,
        'normalise_per_series': False,
        'fill_limit': 0
    }

    specific_cases = [
        {},
        {'classify_per_series': True, 'normalise_per_series': True},
        {'feature_config_list': sample_features_fixed_length}
    ]

    try:
        updated_config = specific_cases[iteration]
        config.update(updated_config)
    except KeyError:
        raise ValueError('Requested configuration not implemented')

    return config


REL_TOL = 1e-4


def load_expected_results(iteration):
    return_value_list = [
        {'x_mean': 8.06938775510204, 'y_mean': 0.2},
        {'x_mean': 8.06938775510204, 'y_mean': 0.2},  # Test classification and normalisation
        {'x_mean': -6.57070769676113e-17, 'y_mean': 0.2},  # Test length/resolution requests
    ]

    try:
        return_value = return_value_list[iteration]
        expected_sample = [23.,  0.,  1.,  6.]
        return return_value['x_mean'], return_value['y_mean'], expected_sample
    except KeyError:
        raise ValueError('Requested configuration not implemented')
