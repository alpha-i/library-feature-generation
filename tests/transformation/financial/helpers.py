import os
from itertools import combinations

import pandas as pd
from dateutil import rrule

from tests.helpers import TEST_DATA_PATH

COLUMNS_OHLCV = 'open high low close volume'.split()


sample_hourly_ohlcv_data_dict = {}
for key in COLUMNS_OHLCV:
    sample_hourly_ohlcv_data_column = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'sample_data_dict', 'sample_%s_hourly.csv' % key),
        index_col=0)
    sample_hourly_ohlcv_data_column.index = pd.to_datetime(sample_hourly_ohlcv_data_column.index,
                                                           utc=True)
    sample_hourly_ohlcv_data_dict[key] = sample_hourly_ohlcv_data_column


sample_fin_data_transf_feature_factory_list_nobins = [
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
        'local': True
    },
    {
        'name': 'high',
        'transformation': {'name': 'log-return'},
        'normalization': 'standard',
        'nbins': 5,
        'is_target': True,
        'local': True
    },
]
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
sample_fin_data_transf_feature_factory_list_bins = [
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


sample_hourly_ohlcv_data_length = len(sample_hourly_ohlcv_data_dict['open'])
sample_hourly_ohlcv_data_symbols = sample_hourly_ohlcv_data_dict['open'].columns
universe_length = sample_hourly_ohlcv_data_length - 1
start_date = sample_hourly_ohlcv_data_dict['open'].index[0]
end_date = sample_hourly_ohlcv_data_dict['open'].index[-1]
rrule_dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))

universe_combination_list = list(combinations(sample_hourly_ohlcv_data_symbols, 4))
sample_historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])

for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
    sample_historical_universes.loc[idx] = [
        period_start_date.date(),
        period_end_date.date(),
        list(universe_combination_list[idx % len(universe_combination_list)])
    ]

sample_daily_ohlcv_data = {}
sample_daily_ohlcv_data_column = pd.read_csv(
    os.path.join(TEST_DATA_PATH, 'sample_data_dict', 'sample_%s_daily.csv' % key),
    index_col=0)
for key in COLUMNS_OHLCV:
    sample_daily_ohlcv_data_column.index = pd.to_datetime(sample_daily_ohlcv_data_column.index)
    sample_daily_ohlcv_data[key] = sample_daily_ohlcv_data_column
