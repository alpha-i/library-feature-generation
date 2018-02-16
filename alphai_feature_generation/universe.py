from abc import ABCMeta, abstractmethod
from dateutil import rrule
import datetime
import logging

import numpy as np
import pandas as pd

from alphai_feature_generation.helpers import CalendarUtilities
from alphai_feature_generation.cleaning import (
    remove_duplicated_symbols_ohlcv,
    slice_data_dict,
    select_between_timestamps
)


logger = logging.getLogger(__name__)

METHOD_FIXED = 'fixed'
METHOD_ANNUAL = 'annual'
METHOD_LIQUIDITY = 'liquidity'
METHOD_LIQUIDITY_DAY = 'liquidity_day'
METHOD_FIXED_HISTORICAL = 'fixed_historical'
HISTORICAL_UNIVERSE_COLUMNS = ('start_date', 'end_date', 'assets')
UPDATE_FREQUENCIES = ('daily', 'weekly', 'monthly', 'yearly')
FREQUENCY_RRULE_MAP = {'daily': rrule.DAILY, 'weekly': rrule.WEEKLY, 'monthly': rrule.MONTHLY, 'yearly': rrule.YEARLY}
OHLCV = ('open', 'high', 'low', 'close', 'volume')


class AbstractUniverseProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_historical_universes(self, data_dict):
        """
        Get a dataframe with arrays of all the relevant equities between two dates, categorised by date ranges.
        :param data_dict: dict of dataframes
        :return: Dataframe with three columns ['start_date', 'end_date', 'assets']
        """
        raise NotImplementedError


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
        self._exchange = configuration['exchange']
        self._dropna = configuration['dropna']

        self._nminutes_window = self._ndays_window * CalendarUtilities.get_minutes_in_one_trading_day(self._exchange)
        self._rrule = FREQUENCY_RRULE_MAP[self._update_frequency]

    def _get_universe_at(self, date, data_dict):
        assert (type(date) == datetime.date) or (type(date) == pd.Timestamp)

        selected_daily_data_dict = slice_data_dict(data_dict, slice_start=-self._ndays_window)
        assert len(selected_daily_data_dict['volume']) == self._ndays_window

        no_duplicates_data_dict = remove_duplicated_symbols_ohlcv(selected_daily_data_dict)
        universe_at_date = np.array(list(no_duplicates_data_dict['volume'].sum().sort_values(ascending=False).index))

        return universe_at_date[:self._nassets]

    def get_historical_universes(self, data_dict):

        historical_universes = pd.DataFrame(columns=HISTORICAL_UNIVERSE_COLUMNS)
        relevant_dict = {k: data_dict[k] for k in ('volume', 'close')}
        relevant_dict['volume'] = relevant_dict['volume'].resample('1D').sum().dropna(axis=[0, 1], how='all')
        relevant_dict['close'] = relevant_dict['close'].resample('1D').last().dropna(axis=[0, 1], how='all')

        data_timezone = relevant_dict['volume'].index.tz
        start_date = relevant_dict['volume'].index[self._ndays_window + 1]
        end_date = relevant_dict['volume'].index[-1]

        rrule_dates = list(rrule.rrule(self._rrule, dtstart=start_date, until=end_date))
        rrule_dates[-1] = end_date

        if len(rrule_dates) > 1:
            for idx, (period_start_date, period_end_date) in enumerate(zip(rrule_dates[:-1], rrule_dates[1:])):
                logger.debug('Calculating historical universe from: {} - {}'.format(str(period_start_date),
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
