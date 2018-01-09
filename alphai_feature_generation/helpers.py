from collections import namedtuple

import pandas as pd

ROOM_FOR_SCHEDULE = 10

MarketDay = namedtuple('MarketDay', 'open close')


class CalendarUtilities:

    def __init__(self, exchange_calendar):
        self._exchange_calendar = exchange_calendar

    def closing_time_for_day(self, the_day):
        """
        Given a day, it returns the market closing time
        """
        market_close = self._exchange_calendar.schedule(the_day, the_day)['market_close']

        return pd.to_datetime(market_close).iloc[0]

    def calculate_target_day(self, market_schedule, prediction_day, target_delta_days):
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
