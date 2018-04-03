from datetime import timedelta
from itertools import combinations

import os
import pandas as pd
from dateutil import rrule

from alphai_feature_generation.transformation import FinancialDataTransformation
from tests.helpers import load_test_data

financial_data_fixtures = load_test_data(os.path.join('financial_data_dict', 'hourly'))

feature_list_default = [
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


def get_configuration(expected_n_symbols, iteration=0):
    config = {
        'feature_config_list': feature_list_default,
        'features_ndays': 2,
        'features_resample_minutes': 60,
        'features_start_market_minute': 1,
        'calendar_name': 'NYSE',
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
        {'feature_config_list': [
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
        ]}
    ]

    try:
        updated_config = specific_cases[iteration]
        config.update(updated_config)
    except KeyError:
        raise ValueError('Requested configuration not implemented')

    return config


def create_historical_universe(ohlcv_sample):
    historical_universe = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])

    symbols = ohlcv_sample['open'].columns
    start_date = ohlcv_sample['open'].index[0]
    end_date = ohlcv_sample['open'].index[-1]
    universe_combination_list = list(combinations(symbols, 4))
    rrule_dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))

    for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
        historical_universe.loc[idx] = [
            period_start_date.date(),
            period_end_date.date(),
            list(universe_combination_list[idx % len(universe_combination_list)])
        ]

    return historical_universe


