import datetime

import pandas_market_calendars as mcal


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
    if exclude_lunch_break and (hasattr(trading_schedule, 'lunch_start') and hasattr(trading_schedule, 'lunch_end')):
        lunch_break = datetime.timedelta(
            hours=trading_schedule.lunch_end.hour - trading_schedule.lunch_start.hour,
            minutes=trading_schedule.lunch_end.minute - trading_schedule.lunch_start.minute
        )

    return (open_interval - lunch_break).seconds / 60
