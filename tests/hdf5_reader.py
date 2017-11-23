import pandas as pd

COLUMNS_OHLCV = 'open high low close volume'.split()


def read_symbol_data_dict_from_hdf5(symbols, start, end, filepath, timezone='America/New_York'):
    """
    Reads the data from the hdf5 for the symbols between the start and end date
    return a dictionary with the symbols as keys.
    Each key corresponds to a Dataframe with timestamp as index and features as columns.

    :param symbols: list of stock symbols
    :param start: start date
    :param end: end date
    :param filepath: path to input hdf5 file
    :param timezone: string specifying the timezone to associate to the input data
    :return: dictionary with symbols as keys and the respective data stored in dataframes as values
    """
    assert pd.Timestamp(end) > pd.Timestamp(start)

    store = pd.HDFStore(filepath)
    symbol_data_dict = {}

    try:
        for symbol in symbols:
            select_string = "index>=pd.Timestamp('{}') & index<=pd.Timestamp('{}')".format(str(start), str(end))
            data_frame = store.select(symbol, select_string)
            data_frame.index = data_frame.index.tz_localize(timezone)
            symbol_data_dict[symbol] = data_frame
    except Exception as e:
        store.close()
        raise e

    store.close()

    return symbol_data_dict


def read_feature_data_dict_from_hdf5(symbols, start, end, filepath, timezone='America/New_York'):
    """
    Reads the data from the hdf5 for the symbols between the start and end date
    return a dictionary with {open, low, high, close and volume} as keys.
    Each key corresponds to a Dataframe with timestamp as index and symbols as columns

    :param symbols: list of stock symbols
    :param start: start date
    :param end: end date
    :param filepath: path to input hdf5 file
    :param timezone: string specifying the timezone to associate to the input data
    :return: dictionary with 'open', 'low', 'high', 'close' and 'volume' as keys and the
             respective data stored in dataframes as values
    """
    assert pd.Timestamp(end) > pd.Timestamp(start)

    symbol_data_dict = read_symbol_data_dict_from_hdf5(symbols, start, end, filepath, timezone)
    data_panel = pd.Panel(symbol_data_dict)
    return {key: data_panel.loc[:, :, key] for key in data_panel.minor_axis}


def get_all_table_names_in_hdf5(filepath):
    """
    Return a list of all table names in input hdf5 file
    :param filepath: path to input hdf5 file
    :return: list of strings describing the names of the tables inside the input hdf5 file
    """
    store = pd.HDFStore(filepath)
    table_name_list = [table[1:] for table in store.keys()]
    store.close()
    return table_name_list
