from abc import ABCMeta, abstractmethod
from dateutil import rrule
import datetime
import logging

import numpy as np
import pandas as pd

from alphai_finance.data.cleaning import (
    resample_ohlcv,
    select_between_timestamps,
    remove_duplicated_symbols_ohlcv,
    slice_data_dict,
    find_duplicated_symbols_data_frame
)

METHOD_FIXED = 'fixed'
METHOD_ANNUAL = 'annual'
METHOD_LIQUIDITY = 'liquidity'
METHOD_LIQUIDITY_DAY = 'liquidity_day'
METHOD_FIXED_HISTORICAL = 'fixed_historical'
HISTORICAL_UNIVERSE_COLUMNS = ['start_date', 'end_date', 'assets']
MINUTES_IN_ONE_TRADING_DAY = 390
UPDATE_FREQUENCIES = ['daily', 'weekly', 'monthly', 'yearly']
FREQUENCY_RRULE_MAP = {'daily': rrule.DAILY, 'weekly': rrule.WEEKLY, 'monthly': rrule.MONTHLY, 'yearly': rrule.YEARLY}
OHLCV = 'open high low close volume'.split()


class AbstractUniverseProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_historical_universes(self, data_dict):
        """
        Get a dataframe with arrays of all the relevant equities between two dates, categorised by date ranges.
        :param data_dict: dict of dataframes
        :return: Dataframe with three columns ['start_date', 'end_date', 'assets']
        """


class VolumeUniverseProvider(AbstractUniverseProvider):
    def __init__(self, configuration):
        """
        Provides assets according to an input universe dictionary indexed by year
        :param nassets: Number of assets to select
        :param ndays_window: Number of days over which to calculate the period liquidity
        :param update_frequency: str in ['daily', 'weekly', 'monthly', 'yearly']: updates of the historical universe
        :param dropna: if True drops columns containing any nan after gaps-filling
        """

        self._nassets = configuration['nassets']
        self._ndays_window = configuration['ndays_window']
        self._update_frequency = configuration['update_frequency']
        self._dropna = configuration['dropna']

        self._nminutes_window = self._ndays_window * MINUTES_IN_ONE_TRADING_DAY
        self._rrule = FREQUENCY_RRULE_MAP[self._update_frequency]

    def _get_universe_at(self, date, data_dict):
        assert (type(date) == datetime.date) or (type(date) == pd.Timestamp)

        for key, value in data_dict.items():
            data_dict[key] = value.resample('1D').sum().dropna(axis=[0, 1], how='all')

        selected_daily_data_dict = slice_data_dict(data_dict, slice_start=-self._ndays_window)
        assert len(selected_daily_data_dict['volume']) == self._ndays_window

        no_duplicates_data_dict = remove_duplicated_symbols_ohlcv(selected_daily_data_dict)
        universe_at_date = np.array(list(no_duplicates_data_dict['volume'].sum().sort_values(ascending=False).index))

        return universe_at_date[:self._nassets]

    def get_historical_universes(self, data_dict):

        historical_universes = pd.DataFrame(columns=HISTORICAL_UNIVERSE_COLUMNS)
        data_timezone = data_dict['volume'].index.tz
        start_date = data_dict['volume'].index[0] + datetime.timedelta(days=self._ndays_window)
        end_date = data_dict['volume'].index[-1]
        relevant_dict = {k: data_dict[k] for k in ('volume', 'close')}
        rrule_dates = list(rrule.rrule(self._rrule, dtstart=start_date, until=end_date))

        if len(rrule_dates) > 1:
            for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
                logging.debug('Calculating historical universe for period: {} - {}'.format(str(period_start_date),
                                                                                           str(period_end_date)))

                end_timestamp = pd.Timestamp(period_start_date, tz=data_timezone)
                historical_universes.loc[idx] = [
                    period_start_date.date(),
                    period_end_date.date(),
                    self._get_universe_at(period_start_date.date(),
                                          select_between_timestamps(relevant_dict, end_timestamp=end_timestamp))
                ]
            historical_universes.iloc[-1]['end_date'] = end_date.date()

        elif len(rrule_dates) == 1:
            end_timestamp = pd.Timestamp(start_date, tz=data_timezone)
            historical_universes.loc[0] = [
                start_date.date(),
                end_date.date(),
                self._get_universe_at(start_date,
                                      select_between_timestamps(relevant_dict, end_timestamp=end_timestamp))
            ]
        return historical_universes
