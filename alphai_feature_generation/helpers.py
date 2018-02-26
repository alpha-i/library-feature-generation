import datetime
import logging
import time
from collections import namedtuple
from functools import wraps

import pandas as pd
import pandas_market_calendars as mcal

ROOM_FOR_SCHEDULE = 10

MarketDay = namedtuple('MarketDay', 'open close')

logger = logging.getLogger(__name__)


def logtime(f):
    @wraps(f)
    def with_logs(*args, **kwargs):
        start_time = time.time()
        result = f(*args, *kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug("%r execution time: %2.4f sec", f.__name__, execution_time)
        return result

    return with_logs
