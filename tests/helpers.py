import os
import shutil
from dateutil import rrule
from itertools import combinations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alphai_feature_generation.feature import FinancialFeature, KEY_EXCHANGE

COLUMNS_OHLCV = 'open high low close volume'.split()

TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'resources',
)

sample_hdf5_file = os.path.join(TEST_DATA_PATH, 'sample_hdf5.h5')

sample_symbols = ['A1', 'A2', 'A3', 'A4', 'A5']
sample_time_index = pd.date_range(start=pd.datetime(2017, 3, 1),
                                  end=pd.datetime(2017, 3, 3),
                                  freq='1min',
                                  tz='America/New_York')
random_data = [-1, 1, 10, 100, 1000] + np.random.rand(len(sample_time_index), len(sample_symbols))
sample_data_frame = pd.DataFrame(random_data,
                                 index=sample_time_index,
                                 columns=sample_symbols).between_time('9:30', '16:00')
sample_data_frame['A2'][10:15] = np.nan
sample_data_frame['A3'][0:3] = np.nan
sample_data_frame['A4'][100] = np.nan
sample_data_frame['A4'][200:203] = np.nan

sample_hourly_ohlcv_data_dict = {}
for key in COLUMNS_OHLCV:
    sample_hourly_ohlcv_data_column = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'sample_data_dict', 'sample_%s_hourly.csv' % key),
        index_col=0)
    sample_hourly_ohlcv_data_column.index = pd.to_datetime(sample_hourly_ohlcv_data_column.index,
                                                           utc=True)
    sample_hourly_ohlcv_data_dict[key] = sample_hourly_ohlcv_data_column

sample_daily_ohlcv_data = {}
for key in COLUMNS_OHLCV:
    sample_daily_ohlcv_data_column = pd.read_csv(
        os.path.join(TEST_DATA_PATH, 'sample_data_dict', 'sample_%s_daily.csv' % key),
        index_col=0)
    sample_daily_ohlcv_data_column.index = pd.to_datetime(sample_daily_ohlcv_data_column.index)
    sample_daily_ohlcv_data[key] = sample_daily_ohlcv_data_column

tmp_symbols = ['A1', 'A2', 'A3']
sample_time_index = pd.date_range(start=pd.datetime(2009, 12, 31),
                                  end=pd.datetime(2010, 3, 13),
                                  freq='1min')
random_data = [-1, 1, 2] + np.random.rand(len(sample_time_index), len(tmp_symbols))
sample_trading_hours_data = \
    pd.DataFrame(random_data, index=sample_time_index,
                 columns=tmp_symbols).between_time('0:00', '23:58')

target_time = pd.date_range(start=pd.datetime(2009, 1, 1),
                            end=pd.datetime(2009, 1, 2),
                            freq='15min',
                            tz='America/New_York')

features_time = pd.date_range(start=pd.datetime(2009, 1, 1),
                              end=pd.datetime(2009, 1, 6),
                              freq='15min',
                              tz='America/New_York')
feature_list = []
target_list = []

for i in range(10):
    random_data = [-1, 1, 2] + np.random.rand(len(features_time), len(tmp_symbols))
    feature_list.append(pd.DataFrame(random_data, index=features_time,
                                     columns=tmp_symbols).between_time('9:30', '16:00'))

    random_data = [-1, 1, 2] + np.random.rand(len(target_time), len(tmp_symbols))
    target_list.append(pd.DataFrame(random_data, index=target_time,
                                    columns=tmp_symbols).between_time('9:30', '16:00'))

sample_features_dict = {'value_features': {'value': feature_list}}
sample_features_and_targets_dict = {'value_features': {'value': feature_list}, 'value_targets': {'value': target_list}}
sample_data_dict = {'dummy_key': sample_trading_hours_data}

sample_market_calendar = mcal.get_calendar('NYSE')

sample_fin_feature_factory_list = [
    {
        'name': 'open',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': 5,
        'ndays': 2,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': False,
        KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
    {
        'name': 'close',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': 10,
        'ndays': 5,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': False,
        KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
    {
        'name': 'high',
        'transformation': {'name': 'log-return'},
        'normalization': 'standard',
        'nbins': None,
        'ndays': 10,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': True,
        KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
]

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

sample_fin_feature_list = [
    FinancialFeature(
        name='open',
        transformation={'name': 'value'},
        normalization=None,
        nbins=5,
        ndays=2,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        exchange_calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    FinancialFeature(
        name='close',
        transformation={'name': 'log-return'},
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        exchange_calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    FinancialFeature(
        name='high',
        transformation={'name': 'log-return'},
        normalization='standard',
        nbins=None,
        ndays=10,
        resample_minutes=60,
        start_market_minute=1,
        is_target=True,
        exchange_calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
]

sample_hourly_ohlcv_data_length = len(sample_hourly_ohlcv_data_dict['open'])
sample_hourly_ohlcv_data_symbols = sample_hourly_ohlcv_data_dict['open'].columns
universe_length = sample_hourly_ohlcv_data_length - 1
universe_combination_list = list(combinations(sample_hourly_ohlcv_data_symbols, 4))
sample_historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])
start_date = sample_hourly_ohlcv_data_dict['open'].index[0]
end_date = sample_hourly_ohlcv_data_dict['open'].index[-1]
rrule_dates = list(rrule.rrule(rrule.WEEKLY, dtstart=start_date, until=end_date))
for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
    sample_historical_universes.loc[idx] = [
        period_start_date.date(),
        period_end_date.date(),
        list(universe_combination_list[idx % len(universe_combination_list)])
    ]

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
