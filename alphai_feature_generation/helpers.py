import logging
from collections import namedtuple
import datetime
import time
from functools import wraps

import pandas as pd
import pandas_market_calendars as mcal

ROOM_FOR_SCHEDULE = 10

MarketDay = namedtuple('MarketDay', 'open close')


def logtime(f):
    @wraps(f)
    def with_logs(*args, **kwargs):
        start_time = time.time()
        result = f(*args, *kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info("%r execution time: %2.4f sec", f.__name__, execution_time)
        return result
    return with_logs


class CalendarUtilities:

    @staticmethod
    def closing_time_for_day(exchange_calendar, the_day):
        """
        Given a day, it returns the market closing time
        """
        market_close = exchange_calendar.schedule(the_day, the_day)['market_close']

        return pd.to_datetime(market_close).iloc[0]

    @staticmethod
    def calculate_target_day(market_schedule, prediction_day, target_delta_days):
        """
        :param market_schedule:
        :type market_schedule: pd.DataFrame (index=Timestamp, columns=['market_close','market_open'])
        :param prediction_day:
        :type prediction_day: datetime.Date
        :param target_delta_days:
        :type target_delta_days: int

        :return:
        """

        target_index = market_schedule.index.get_loc(prediction_day) + target_delta_days

        try:
            day_schedule = market_schedule.iloc[target_index]
            return MarketDay(day_schedule['market_open'], day_schedule['market_close'])
        except KeyError:
            return None

    @staticmethod
    def get_minutes_in_one_trading_day(exchange, exclude_lunch_break=False):
        """
        :param exchange: the exchange
        :type exchange: str
        :param exclude_lunch_break: whether to discard minutes during lunch break (for Tokyo)
        :return: minutes in one trading day for the selected exchange
        :rtype: int
        """
        try:
            trading_schedule = mcal.get_calendar(exchange)
        except KeyError:
            raise Exception("No such exchange: %s" % exchange)

        open_time = trading_schedule.open_time
        close_time = trading_schedule.close_time

        hours = close_time.hour - open_time.hour
        minutes = close_time.minute - open_time.minute

        open_interval = datetime.timedelta(hours=hours, minutes=minutes)

        lunch_break = datetime.timedelta(minutes=0)
        if exclude_lunch_break and (
                hasattr(trading_schedule, 'lunch_start') and hasattr(trading_schedule, 'lunch_end')):
            lunch_break = datetime.timedelta(
                hours=trading_schedule.lunch_end.hour - trading_schedule.lunch_start.hour,
                minutes=trading_schedule.lunch_end.minute - trading_schedule.lunch_start.minute
            )

        return (open_interval - lunch_break).seconds / 60
