import multiprocessing
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager


@contextmanager
def ensure_closing_pool():
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()
        del pool


class DateNotInUniverseError(Exception):
    pass


class DataTransformation(metaclass=ABCMeta):
    @abstractmethod
    def create_train_data(self, *args):
        raise NotImplementedError

    @abstractmethod
    def create_predict_data(self, *args):
        raise NotImplementedError


def get_unique_symbols(data_list):
    """Returns a list of all unique symbols in the dict of dataframes"""

    symbols = set()

    for data_dict in data_list:
        for feature in data_dict:
            feat_symbols = data_dict[feature].columns
            symbols.update(feat_symbols)

    return symbols
