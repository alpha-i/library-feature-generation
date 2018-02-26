import datetime
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RESAMPLING_FUNCTION = 'mean'


def select_between_timestamps(data, start_timestamp=None, end_timestamp=None):
    """
    Resample input dataframe or dictionary according to specified start/end timestamps
    :param data: dataframe or data dictionary
    :param start_timestamp: lower bound time stamp for data selection
    :param end_timestamp: upper bound time stamp for data selection
    :return: selected data.
    """
    if isinstance(data, pd.DataFrame):
        return select_between_timestamps_data_frame(data, start_timestamp, end_timestamp)
    elif isinstance(data, dict):
        return select_between_timestamps_data_dict(data, start_timestamp, end_timestamp)
    else:
        raise NameError('Input data type not recognised')


def select_between_timestamps_data_frame(data_frame, start_timestamp=None, end_timestamp=None):
    """
    Select a subset of the input dataframe according to specified start/end timestamps
    :param data_frame: Dataframe with time as index
    :param start_timestamp: lower bound time stamp for data selection
    :param end_timestamp: upper bound time stamp for data selection
    :return: selected data_frame
    """
    assert start_timestamp is not None or end_timestamp is not None
    data_frame_timezone = data_frame.index.tz
    for ts in [start_timestamp, end_timestamp]:
        if ts is not None:
            if ts.tz is None:
                assert data_frame_timezone is None
            else:
                assert ts == ts.tz_convert(data_frame_timezone)
    time_conditions = []
    if start_timestamp is not None:
        time_conditions.append(data_frame.index >= start_timestamp)
    if end_timestamp is not None:
        time_conditions.append(data_frame.index <= end_timestamp)

    return data_frame[np.all(time_conditions, axis=0)]



def select_between_timestamps_data_dict(data_dict, start_timestamp=None, end_timestamp=None):
    """
    Select dictionary of dataframes data according to specified start/end timestamps
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param start_timestamp: lower bound time stamp for data selection
    :param end_timestamp: upper bound time stamp for data selection
    :return: dictionary of selected dataframes.
    """
    selected_data_dict = {}
    for key, data_frame in data_dict.items():
        selected_data_dict[key] = select_between_timestamps_data_frame(data_frame, start_timestamp, end_timestamp)
    return selected_data_dict


def resample(data, resample_rule, sampling_functions=None):
    """
    Resample input dataframe or dictionary according to input rules and drop nans horizontally.
    :param data: dataframe or data dictionary
    :param resample_rule: string specifying the pandas resampling rule
    :param sampling_functions: string or dictionary of strings specifying the sampling function in
                               ['mean', 'median', 'sum'].
    :return: resampled data.
    """
    if not sampling_functions:
        sampling_functions = {}

    if isinstance(data, pd.DataFrame):
        return resample_data_frame(data, resample_rule, sampling_functions)
    elif isinstance(data, dict):
        return resample_data_dict(data, resample_rule, sampling_functions)
    raise NameError('Input data type not recognised')


def resample_data_frame(data_frame, resample_rule, sampling_function='mean'):
    """
    Resample dataframe according to input rules and drop nans horizontally.
    :param data_frame: Dataframe with time as index
    :param resample_rule: string specifying the pandas resampling rule
    :param sampling_function: string specifying the sampling function in
           ['mean', 'median', 'sum', 'first', 'last', 'min', 'max']
    :type sampling_function: str
    :return: resampled dataframe.
    """
    assert sampling_function in ('mean', 'median', 'sum', 'first', 'last', 'min', 'max')

    data_frame = getattr(data_frame.resample(resample_rule, label='right', closed='right'), sampling_function)()
    return data_frame.dropna(axis=[0, 1], how='all')


def resample_data_dict(data_dict, resample_rule, sampling_function_mapping=None):
    """
    Resample dictionary of dataframes data according to input rules and drop nans horizontally.

    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param resample_rule: string specifying the pandas resampling rule
    :param sampling_function_mapping: dictionary of strings (or string) specifying the sampling function in
           ['mean', 'median', 'sum', 'first', 'last', 'min', 'max'] for each key in data_dict
    :type sampling_function_mapping: dict
    :return: dictionary of resampled dataframes.
    """
    if not sampling_function_mapping:
        sampling_function_mapping = {}

    resampled_data_dict = {}
    for key, data_frame in data_dict.items():
        resampled_data_dict[key] = resample_data_frame(
            data_frame, resample_rule, sampling_function=sampling_function_mapping.get(key, DEFAULT_RESAMPLING_FUNCTION)
        )
    return resampled_data_dict


def resample_ohlcv(ohlcv_data, resample_rule):
    """
    Resample ['open', 'high', 'low', 'close', 'volume'] history data according to input rule
    and drop nans horizontally.

    :param ohlcv_data: Dictionary of dataframes with time as index and OHLCV as keys
    :param resample_rule: string specifying the pandas resampling rule
    :return: dictionary of resampled dataframes.
    """
    sampling_function_mapping = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    return resample_data_dict(ohlcv_data, resample_rule, sampling_function_mapping)


def select_above_floor(data, floor):
    """
    Select only columns whose values are all above the input floor value.
    :param data: dataframe or data dictionary
    :param floor: lower value bound
    :return: data with dataframe columns selected to have values above the floor.
    """
    if isinstance(data, pd.DataFrame):
        return select_above_floor_data_frame(data, floor)
    elif isinstance(data, dict):
        return select_above_floor_data_dict(data, floor)
    else:
        raise NameError('Input data type not recognised')


def select_above_floor_data_frame(data_frame, floor):
    """
    Select only columns whose values are all above the input floor value.
    :param data_frame: Dataframe with time as index
    :param floor: lower value bound
    :return: dataframe whose columns are selected to have values above the floor.
    """
    return data_frame.loc[:, (data_frame > floor).all()]


def select_above_floor_data_dict(data_dict, floor):
    """
    Select only columns whose values are all above the input floor value.
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param floor: lower value bound
    :return: data_dict with dataframe columns selected to have values above the floor.
    """
    selected_data_dict = {}
    for key, data_frame in data_dict.items():
        selected_data_dict[key] = select_above_floor_data_frame(data_frame, floor)
    return selected_data_dict


def select_below_ceiling(data, floor):
    """
    Select only columns whose values are all below the input ceiling value.
    :param data: dataframe or data dictionary
    :param floor: lower value bound
    :return: data with dataframe columns selected to have values below the ceiling.
    """
    if isinstance(data, pd.DataFrame):
        return select_above_floor_data_frame(data, floor)
    elif isinstance(data, dict):
        return select_above_floor_data_dict(data, floor)
    else:
        raise NameError('Input data type not recognised')


def select_below_ceiling_data_frame(data_frame, ceiling):
    """
    Select only columns whose values are all below the input ceiling value.
    :param data_frame: Dataframe with time as index
    :param ceiling: high value bound
    :return: dataframe whose columns are selected to have values below the ceiling.
    """
    return data_frame.loc[:, (data_frame < ceiling).all()]


def select_below_ceiling_data_dict(data_dict, ceiling):
    """
    Select only columns whose values are all below the input ceiling value.
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param ceiling: high value bound
    :return: data_dict with dataframe columns selected to have values below the ceiling.
    """
    selected_data_dict = {}
    for key, data_frame in data_dict.items():
        selected_data_dict[key] = select_below_ceiling_data_frame(data_frame, ceiling)
    return selected_data_dict


def fill_gaps(data, fill_limit, dropna=True):
    """
    Fill small vertical gaps and drop columns still containing nans, if required.
    :param data: dataframe or data dictionary
    :param fill_limit: forward and backward fill gaps in data for a maximum of fill_limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :return: data with gaps filled and nan-containing columns removed if required.
    """
    if isinstance(data, pd.DataFrame):
        return fill_gaps_data_frame(data, fill_limit, dropna)
    elif isinstance(data, dict):
        return fill_gaps_data_dict(data, fill_limit, dropna)
    else:
        raise NameError('Input data type not recognised')


def fill_gaps_data_frame(data_frame, fill_limit, dropna=True):
    """
    Fill small vertical gaps in dataframe and drop columns still containing nans, if required.
    :param data_frame: Dataframe with time as index
    :param fill_limit: forward and backward fill gaps in data for a maximum of fill_limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :return: dataframe with gaps filled and nan-containing columns removed if required.
    """
    tmp_data_frame = deepcopy(data_frame)
    tmp_data_frame = tmp_data_frame.fillna(method='ffill', limit=fill_limit)
    if dropna:
        # Drop columns that after nan filling still contain nans
        tmp_data_frame = tmp_data_frame.dropna(axis=1, how='any')
    return tmp_data_frame


def fill_gaps_data_dict(data_dict, fill_limit, dropna=True):
    """
    Fill small vertical gaps and drop columns still containing nans, if required, in all
    dataframes contained in input data_dict.
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param fill_limit: forward and backward fill gaps in data for a maximum of fill_limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :return: data_dict with gaps filled and nan-containing columns removed if required.
    """
    filled_data_dict = {}
    for key, data_frame in data_dict.items():
        filled_data_dict[key] = fill_gaps_data_frame(data_frame, fill_limit, dropna)
    return filled_data_dict


def interpolate_gaps(data, limit, dropna=True, method='linear'):
    """
    Interpolate small vertical gaps and drop columns still containing nans, if required.
    :param data: dataframe or data dictionary
    :param limit: forward and backward fill gaps in data for a maximum of limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :param method: interpolation method
    :return: data with gaps filled and nan-containing columns removed if required.
    """
    if isinstance(data, pd.DataFrame):
        return interpolate_gaps_data_frame(data, limit, dropna, method)
    elif isinstance(data, dict):
        return interpolate_gaps_data_dict(data, limit, dropna, method)
    else:
        raise NameError('Input data type not recognised')


def interpolate_gaps_data_frame(data_frame, limit, dropna=True, method='linear'):
    """
    Interpolate small vertical gaps in dataframe and drop columns still containing nans, if required.
    :param data_frame: Dataframe with time as index
    :param limit: forward and backward fill gaps in data for a maximum of limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :param method: interpolation method
    :return: dataframe with gaps filled and nan-containing columns removed if required.
    """
    tmp_data_frame = deepcopy(data_frame)

    tmp_data_frame = tmp_data_frame.interpolate(method=method, limit=limit, limit_direction='forward')

    if dropna:
        tmp_data_frame = tmp_data_frame.dropna(axis=1, how='any')
    return tmp_data_frame


def interpolate_gaps_data_dict(data_dict, limit, dropna=True, method='linear'):
    """
    Interpolate small vertical gaps and drop columns still containing nans, if required, in all
    dataframes contained in input data_dict.
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param limit: forward and backward fill gaps in data for a maximum of limit points
    :param dropna: if True drops columns containing any nan after gaps-filling
    :param method: interpolation method
    :return: data_dict with gaps filled and nan-containing columns removed if required.
    """
    filled_data_dict = {}
    for key, data_frame in data_dict.items():
        filled_data_dict[key] = interpolate_gaps_data_frame(data_frame, limit, dropna, method)
    return filled_data_dict


def select_trading_hours(data, market_calendar, include_start=True, include_end=True):
    """
    Select data spanning trading hours from a generic input data [dataframe or dictionary]
    :param data: dataframe or data dictionary
    :param market_calendar: pandas_market_calendar
    :param include_start : boolean if to include the first minute of the trading hours
    :param include_end : boolean if to include the first minute of the trading hours
    :return: data only spanning trading hours
    """
    if isinstance(data, pd.DataFrame):
        return select_trading_hours_data_frame(data, market_calendar, include_start, include_end)
    elif isinstance(data, dict):
        return select_trading_hours_data_dict(data, market_calendar, include_start, include_end)
    else:
        raise NameError('Input data type not recognised')


def select_trading_hours_data_frame(data_frame, market_calendar, include_start=True, include_end=True):
    """
    Return subset of a DataFrame whose index only spans trading hours
    :param data_frame: Dataframe with time as index
    :param market_calendar: pandas_market_calendar
    :param include_start : boolean if to include the first minute of the trading hours
    :param include_end : boolean if to include the first minute of the trading hours
    :return: dataframe whose index only spans trading hours
    """
    assert str(data_frame.index.tz) == 'UTC'
    start_date, end_date = data_frame.index[0].date(), data_frame.index[-1].date()
    market_schedule = market_calendar.schedule(start_date, end_date)

    # make sure only times and days are where market is open
    tmp_trading_day_list = []
    for trading_day in market_schedule.itertuples():
        tmp_trading_day_list.append(
            data_frame[str(trading_day.market_open.date())].
                between_time(trading_day.market_open.time(), trading_day.market_close.time(), include_start,
                             include_end))

    return pd.concat(tmp_trading_day_list, axis=0)


def select_trading_hours_data_dict(data_dict, market_calendar, include_start=True, include_end=True):
    """
    Return dictionary of DataFrames selecting subsets whose index only spans trading hours
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param market_calendar: pandas_market_calendar
    :param include_start : boolean if to include the first minute of the trading hours
    :param include_end : boolean if to include the first minute of the trading hours
    :return: dictionary with dataframes as values, whose index only spans trading hours
    """
    selected_data_dict = {}
    for key, data_frame in data_dict.items():
        selected_data_dict[key] = \
            select_trading_hours_data_frame(data_frame, market_calendar, include_start, include_end)
    return selected_data_dict


def convert_to_utc(data, timezone='America/New_York'):
    """
    Convert a generic input data [dataframe or dictionary] to UTC timezone
    :param data: dataframe or data dictionary
    :param timezone: pandas string specifying the timezone of input data
    :return: data converted to UTC timezone
    """
    if isinstance(data, pd.DataFrame):
        return convert_data_frame_to_utc(data, timezone)
    elif isinstance(data, dict):
        return convert_data_dict_to_utc(data, timezone)
    else:
        raise NameError('Input data type not recognised')


def convert_data_frame_to_utc(data_frame, timezone='America/New_York'):
    """
    Convert a dataframe to UTC timezone
    :param data_frame: Dataframe with time as index
    :param timezone: pandas string specifying the timezone of input data
    :return: Dataframe converted to UTC timezone
    """
    tmp_data_frame = deepcopy(data_frame)
    if tmp_data_frame.index.tz is None:
        tmp_data_frame = tmp_data_frame.tz_localize(timezone)
    return tmp_data_frame.tz_convert('UTC')


def convert_data_dict_to_utc(data_dict, timezone='America/New_York'):
    """
    Convert a data dictionary to UTC timezone
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param timezone: pandas string specifying the timezone of input data
    :return: data dictionary converted to UTC timezone
    """
    tmp_data_dict = deepcopy(data_dict)
    for key, data_frame in tmp_data_dict.items():
        tmp_data_dict[key] = convert_data_frame_to_utc(data_frame, timezone)
    return tmp_data_dict


def sample_minutes_after_market_open_data_frame(data_frame, market_calendar, minutes_after_market_open):
    """
    Sample input dataframe daily, at a specified number of minutes after market opens
    :param data_frame: Dataframe with time as index
    :param market_calendar: pandas_market_calendar
    :param minutes_after_market_open: number of minutes after market opens
    :return: Sampled dataframe
    """
    assert str(data_frame.index.tz) == 'UTC'
    start_date, end_date = data_frame.index[0].date(), data_frame.index[-1].date()
    market_schedule = market_calendar.schedule(start_date, end_date)
    sampling_times = market_schedule.market_open + datetime.timedelta(minutes=minutes_after_market_open)
    return data_frame.loc[list(sampling_times)].dropna()


def select_columns_data_dict(data_dict, select_columns):
    """
    Selected input columns from all dataframes in data_dict
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param select_columns: columns to select
    :return: data dictionary with only selected columns
    """
    return {map_key: map_df[select_columns] for map_key, map_df in data_dict.items()}


def slice_data_dict(data_dict, slice_start, slice_end=None):
    """
    Time-slice all dataframes contained in a data_dict
    :param data_dict: a dictionary with (timestamp, symbol)-dataframes as values
    :param slice_start: first index to be used for slicing
    :param slice_end: last index to be used for slicing. Can be None.
    :return:
    """
    selected_data_dict = {}
    for key, data_frame in data_dict.items():
        if slice_end is None:
            selected_data_dict[key] = data_frame.iloc[slice_start:]
        else:
            selected_data_dict[key] = data_frame.iloc[slice_start:slice_end]
    return selected_data_dict


def find_duplicated_symbols_data_frame(data_frame, max_correlation=0.999):
    """
    Remove duplicated symbols from dataframe (correlation higher than input max_correlation)
    :param data_frame: Dataframe with time as index
    :param max_correlation: maximum correlation allowed
    :return: data dictionary after removing duplicated symbols
    """
    corr_matrix = data_frame.corr()
    above_max = pd.DataFrame(np.triu(corr_matrix > max_correlation, k=1),
                             index=corr_matrix.index, columns=corr_matrix.columns)

    return list(above_max[above_max].stack().index)


def remove_duplicated_symbols_ohlcv(ohlcv_data, max_correlation=0.999):
    """
    Remove duplicated symbols from ohlcv data dict (correlation higher than input max_correlation).
    Correlations are calculated on the 'close' and selection is made on the base of higher total 'volume'.
    :param ohlcv_data: Dictionary of dataframes with time as index and OHLCV as keys
    :param max_correlation: maximum correlation allowed
    :return: data dictionary after removing duplicated symbols
    """
    clean_ohlcv_data = {}
    duplicated_symbol_list = find_duplicated_symbols_data_frame(ohlcv_data['close'], max_correlation)
    volumes = ohlcv_data['volume'].sum()

    symbols_to_drop = []
    for duplicated_symbols in duplicated_symbol_list:
        symbols_to_drop += list(volumes[list(duplicated_symbols)].sort_values(ascending=False).index[1:])

    if len(symbols_to_drop) > 0:
        logger.debug("Dropping {} symbols.".format(len(symbols_to_drop)))

    for key in ohlcv_data.keys():
        clean_ohlcv_data[key] = ohlcv_data[key].drop(symbols_to_drop, axis=1)

    return clean_ohlcv_data


def swap_keys_and_columns(data_dict):
    """
    Swap the keys and the columns of a data_dict, dictionary of dataframes.
    :param data_dict: a dictionary with dataframes as values
    :return: inverted data_dict.
    """
    data_panel = pd.Panel(data_dict)
    return {key: data_panel.loc[:, :, key] for key in data_panel.minor_axis}
