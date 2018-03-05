import logging
import multiprocessing
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from datetime import timedelta
from functools import partial


import numpy as np

import alphai_calendars as mcal
from alphai_feature_generation.feature.factory import FeatureList

logger = logging.getLogger(__name__)


@contextmanager
def ensure_closing_pool():
    """ Do some fancy multiprocessing stuff while preventing memory leaks. """
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    try:
        yield pool
    finally:
        pool.terminate()
        pool.join()
        del pool


class DateNotInUniverseError(Exception):
    """ Sorry Morty, we seem to be in the wrong universe. """
    pass


class DataTransformation(metaclass=ABCMeta):
    """ Prepares raw time series data for machine learning applications. """

    KEY_EXCHANGE = None

    def __init__(self, configuration):
        """Initialise in accordance with the config dictionary.

        :param dict configuration:
        """

        self._calendar = None
        self.minutes_in_trading_days = None

        self._calendar = mcal.get_calendar(configuration[self.KEY_EXCHANGE])
        self.minutes_in_trading_days = self._calendar.get_minutes_in_one_day()
        self.configuration = configuration

        self.features_ndays = configuration['features_ndays']
        self.features_resample_minutes = configuration['features_resample_minutes']
        self.features_start_market_minute = configuration['features_start_market_minute']
        self.prediction_market_minute = configuration['prediction_market_minute']
        self.target_delta_ndays = configuration['target_delta_ndays']
        self.target_market_minute = configuration['target_market_minute']
        self.classify_per_series = configuration['classify_per_series']
        self.normalise_per_series = configuration['normalise_per_series']
        self.n_classification_bins = configuration['n_classification_bins']
        self.n_series = configuration['nassets']
        self.fill_limit = configuration['fill_limit']
        self.predict_the_market_close = configuration.get('predict_the_market_close', False)

        self.features = self._feature_factory(configuration['feature_config_list'])
        self.feature_length = self.get_feature_length()

        self._assert_input()

    @abstractmethod
    def _assert_input(self):
        """ Make sure your inputs are sensible.  """
        raise NotImplementedError

    @abstractmethod
    def _get_feature_for_extract_y(self):
        """ Returns the name of the feature to be used as a target (y). """
        raise NotImplementedError

    @abstractmethod
    def _feature_factory(self, feature_configuration_list):
        """ Creates a list of features from a given configuration list. """
        raise NotImplementedError

    @abstractmethod
    def create_train_data(self, *args):
        """ Create a set of training data (x and y)"""
        raise NotImplementedError

    @abstractmethod
    def create_predict_data(self, *args):
        """ Create a set of features for a single prediction (x).
        These will be normalised in accordance with the properties of the training set. """
        raise NotImplementedError

    @abstractmethod
    def inverse_transform_multi_predict_y(self, predict_y, symbols):
        """ Converts the network output (classification) into a useable prediction (such as mean and uncertainties) """
        raise NotImplementedError

    @property
    def exchange_calendar(self):
        """
        Backward compatibility method

        :return:
        """
        return self._calendar

    def check_x_batch_dimensions(self, feature_x_dict):
        """
        Evaluate if the x batch has the expected dimensions.

        :param dict feature_x_dict: batch of x-features
        :return bool: False if the dimensions are not those expected
        """

        correct_dimensions_list = []
        for feature_full_name, feature_array in feature_x_dict.items():
            for feature in self.features:
                if feature.full_name == feature_full_name:
                    correct_dimensions_list.append(feature_array.shape[0] == feature.length)

        return all(correct_dimensions_list)

    def check_y_batch_dimensions(self, feature_y_dict):
        """
        Evaluate if the y batch has the expected dimensions.

        :param dict feature_y_dict: batch of y-labels
        :return bool: False if the dimensions are not those expected
        """
        correct_dimensions = True
        expected_shape = (self.n_series,)

        if feature_y_dict is not None:
            for feature_full_name, feature_array in feature_y_dict.items():
                correct_dimensions = feature_array.shape == expected_shape

        return correct_dimensions

    def get_feature_length(self):
        """
        Calculate expected total ticks for x data.

        :return int: expected total number of ticks for x data
        """
        return self.get_total_ticks_x()

    def get_total_ticks_x(self):
        """
        Calculate expected total ticks for x data.

        :return int: expected total number of ticks for x data
        """
        ticks_in_a_day = np.floor(self.minutes_in_trading_days / self.features_resample_minutes) + 1
        intra_day_ticks = np.floor((self.prediction_market_minute - self.features_start_market_minute) /
                                   self.features_resample_minutes)
        total_ticks = ticks_in_a_day * self.features_ndays + intra_day_ticks + 1

        return int(total_ticks)

    def get_target_feature(self):
        """
        Return the target feature in self.features

        :return FinancialFeature: target feature
        """
        for feature in self.features:
            if feature.is_target:
                return feature

    def filter_unwanted_keys(self, raw_data_dict):
        """
        Remove useless data from the raw_data_dict
        
        :param raw_data_dict: Dictionary we wish to trim
        :return dict: data which belong to the expected keys
        """
        wanted_keys = {feature.name for feature in self.features}

        return {key: value for key, value in raw_data_dict.items() if key in wanted_keys}

    def _get_valid_target_timestamp_in_schedule(self, schedule, predict_timestamp):
        """
        Return valid market time for target time given timestamp calculated using
        the predict_timestamp and the property target_delta_ndays

        :param pd.Timestamp predict_timestamp: the timestamp of the prediction
        :return pd.Timestamp target_timestamp: the timestamp of the target
        """

        market_schedule_for_target_day = self._extract_target_market_day(schedule, predict_timestamp)

        target_market_open = market_schedule_for_target_day.market_open
        target_timestamp = self._get_target_timestamp(target_market_open)

        if self._calendar.open_at_time(schedule, target_timestamp, include_close=True):
            return target_timestamp
        else:
            raise ValueError("Target timestamp {} not in market time".format(target_timestamp))

    def _extract_target_market_day(self, market_schedule, prediction_timestamp):
        """
        Extract the target market open day using prediction day and the property target_delta_ndays

        :param pd.DataFrame market_schedule:
        :param prediction_timestamp:

        :return:
        """

        target_index = market_schedule.index.get_loc(prediction_timestamp.date()) + self.target_delta_ndays

        if target_index < len(market_schedule):
            return market_schedule.iloc[target_index]
        else:
            return None

    def _make_normalised_x_list(self, x_list, do_normalisation_fitting):
        """
        Collects sample of x into a dictionary, and applies normalisation

        :param x_list: List of unnormalised dictionaries
        :param bool do_normalisation_fitting: Whether to use pre-fitted normalisation, or set normalisation constants
        :return: dict Dictionary of normalised features
        """

        if len(x_list) == 0:
            raise ValueError("No valid x samples found.")

        symbols = get_unique_symbols(x_list)

        # this was using multiprocessing and hanging during the backtest
        # let's switch this off for now
        if do_normalisation_fitting:
            fit_function = partial(self.fit_normalisation, symbols, x_list)
            fitted_features = [fit_function(feature) for feature in self.features]
            self.features = FeatureList(fitted_features)

        for feature in self.features:
            x_list = self.apply_normalisation(x_list, feature)

        return x_list

    def apply_normalisation(self, x_list, feature):  #FIXME a method of ths same name exists in feature.py
        """ Normalise the data using the feature's built-in scaler.

        :param x_list:
        :param feature:
        :return:
        """

        if feature.scaler:
            normalise_function = partial(self.normalise_dict, feature)
            logger.debug("Applying normalisation to: {}".format(feature.full_name))

            with ensure_closing_pool() as pool:
                list_of_x_dicts = pool.map(normalise_function, x_list)

            return list_of_x_dicts
        else:
            return x_list

    def normalise_dict(self, feature, x_dict):
        """ Apply normalisation with a single feature.

        :param target_feature:
        :param x_dict:
        :return:
        """
        if feature.full_name in x_dict:
            x_dict[feature.full_name] = feature.apply_normalisation(x_dict[feature.full_name])
        else:
            logger.debug("Failed to find {} in dict: {}".format(feature.full_name, list(x_dict.keys())))
        return x_dict

    def fit_normalisation(self, symbols, x_list, feature):
        """ Fit the normalisation parameters to the data.

        :param symbols:
        :param x_list:
        :param feature:
        :return:
        """
        if feature.scaler:
            logger.debug("Fitting normalisation to: {}".format(feature.full_name))
            if self.normalise_per_series:
                for symbol in symbols:
                    symbol_data = self.extract_data_by_symbol(x_list, symbol, feature.full_name)
                    feature.fit_normalisation(symbol_data, symbol)
            else:
                all_data = self.extract_all_data(x_list, feature.full_name)
                feature.fit_normalisation(all_data)
        else:
            logger.debug("Skipping normalisation to: {}".format(feature.full_name))
        return feature

    def _make_classified_y_list(self, y_list):
        """ Takes list of dictionaries, and classifies them based on the full sample.
        :param y_list:  List of unnormalised dictionaries
        :return: dict Dictionary of labels, encoded in one hot format
        """

        if len(y_list) == 0:
            raise ValueError("No valid y samples found.")

        target_feature = self.get_target_feature()
        target_name = target_feature.full_name
        symbols = get_unique_symbols(y_list)

        # Fitting of bins
        logger.debug("Fitting y classification to: {}".format(target_name))
        for symbol in symbols:
            symbol_data = self.extract_data_by_symbol(y_list, symbol, target_name)
            target_feature.fit_classification(symbol, symbol_data)

        # Applying
        logger.debug("Applying y classification to: {}".format(target_name))
        with ensure_closing_pool() as pool:
            apply_classification = partial(self._apply_classification, target_feature, target_name)
            applied_y_list = pool.map(apply_classification, y_list)

        return applied_y_list

    def _apply_classification(self, target_feature, target_name, y_dict):
        """  Classifies the y values.

        :param target_feature: The 'feature' that will act as the target for the network
        :param target_name:
        :param y_dict:
        :return:
        """
        if target_name in y_dict:
            y_dict[target_name] = target_feature.apply_classification(y_dict[target_name])
        else:
            logger.debug("Failed to find {} in dict: {}".format(target_name, list(y_dict.keys())))
        return y_dict

    def _get_target_timestamp(self, target_market_open):
        """
        Calculate the target timestamp for if given a valid target open day
        :param target_market_open:
        :type target_market_open: pd.Timestamp
        :return target_timestamp:
        :rtype pd.Timestamp
        """
        if target_market_open:

            if self.predict_the_market_close:
                return self._calendar.closing_time_for_day(target_market_open.date())
            else:
                return target_market_open + timedelta(minutes=self.target_market_minute)
        else:
            return None

    def _get_prediction_timestamps(self, prediction_market_open):
        """
        Calculate the prediction timestamp for the given opening day
        :param prediction_market_open:
        :return:
        """

        x_end_timestamp = prediction_market_open + timedelta(minutes=self.prediction_market_minute)

        if self.predict_the_market_close:
            y_start_timestamp = self._calendar.closing_time_for_day(prediction_market_open)
        else:
            y_start_timestamp = x_end_timestamp

        return x_end_timestamp, y_start_timestamp

    def apply_global_transformations(self, raw_data_dict):
        """ Adds processed features to data dictionary, designated 'global' if they result in equal length time series.

        :param raw_data_dict: dictionary of dataframes
        :return: dict with new keys
        """

        for feature in self.features:
            if not feature.local and feature.full_name not in raw_data_dict.keys():
                raw_dataframe = raw_data_dict[feature.name]
                raw_data_dict[feature.full_name] = feature.process_prediction_data_x(raw_dataframe)

        return raw_data_dict

    def stack_samples_for_each_feature(self, samples, reference_samples=None):
        """ Collate a list of samples (the training set) into a single dictionary

        :param samples: List of dicts, each dict should be holding the same set of keys
        :param reference_samples: cross-checks samples match shape of reference samples
        :return: Single dictionary with the values stacked together
        """
        if len(samples) == 0:
            raise ValueError("At least one sample required for stacking samples.")

        feature_names = samples[0].keys()
        label_name = self.get_target_feature().full_name

        stacked_samples = OrderedDict()
        valid_symbols = []
        total_samples = 0
        unusual_samples = 0
        for feature_name in feature_names:
            reference_sample = samples[0]
            reference_shape = reference_sample[feature_name].shape
            if len(samples) == 1:
                stacked_samples[feature_name] = np.expand_dims(reference_sample[feature_name], axis=0)
                valid_symbols = reference_sample[feature_name].columns
            else:
                feature_list = []
                for i, sample in enumerate(samples):
                    feature = sample[feature_name]
                    symbols = list(feature.columns)

                    total_samples += 1
                    is_shape_ok = (feature.shape == reference_shape)

                    if reference_samples:
                        columns_match = (symbols == list(reference_samples[i][label_name].columns))
                    else:
                        columns_match = True
                    dates_match = True  # FIXME add dates check

                    if is_shape_ok and columns_match and dates_match:  # Make sure shape is OK
                        feature_list.append(sample[feature_name].values)
                        valid_symbols = symbols
                    else:
                        unusual_samples += 1
                        if not columns_match:
                            logger.debug("Oi, your columns dont match")

                if len(feature_list) > 0:
                    stacked_samples[feature_name] = np.stack(feature_list, axis=0)
                else:
                    stacked_samples = None

        if len(samples) > 1:
            logger.info("Found {} unusual samples out of {}".format(unusual_samples, total_samples))

        return stacked_samples, valid_symbols

    def _extract_schedule_from_data(self, raw_data_dict):
        """
        Return a list of market open timestamps from input data_dict
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return pd.Series: list of market open timestamps.
        """
        features_keys = self.features.get_names()
        raw_data_start_date = raw_data_dict[features_keys[0]].index[0].date()
        raw_data_end_date = raw_data_dict[features_keys[0]].index[-1].date()

        return self._calendar.schedule(str(raw_data_start_date), str(raw_data_end_date))

    def _extract_schedule_for_prediction(self, raw_data_dict):
        """
        Return market schedule dataframe with only one index
        :param raw_data_dict:
        :return:

        """
        full_schedule = self._extract_schedule_from_data(raw_data_dict)
        return full_schedule.drop(labels=full_schedule[0:-1].index, axis=0)

    def _extract_schedule_for_training(self, raw_data_dict):
        """
        Returns a market_schedule dataframe containing
        all dates on which we have both x and y data
        """

        max_feature_ndays = self.features.get_max_ndays()

        return self._extract_schedule_from_data(raw_data_dict)[max_feature_ndays:-self.target_delta_ndays]

    def print_diagnostics(self, xdict, ydict):
        """
        Prints some usefult diagnostic of shapes and size
        :param xdict:
        :param ydict:
        :return:
        """

        x_sample = list(xdict.values())[0]
        x_expected_shape = self.features[0].length
        logger.debug("Last rejected xdict: {}".format(x_sample.shape))
        logger.debug("x_expected_shape: {}".format(x_expected_shape))

        if ydict is not None:
            y_sample = list(ydict.values())[0]
            y_expected_shape = (self.n_series,)
            logger.debug("Last rejected ydict: {}".format(y_sample.shape))
            logger.debug("y_expected_shape: {}".format(y_expected_shape))

    @staticmethod
    def extract_data_by_symbol(x_list, symbol, feature_name):
        """ Collect all data from a list of dicts of features, for a given symbol """

        collated_data = []
        for x_dict in x_list:
            if symbol in x_dict[feature_name].columns:
                sample = x_dict[feature_name][symbol]
                collated_data.extend(sample.values)

        return np.asarray(collated_data)

    @staticmethod
    def extract_all_data(x_list, feature_name):
        """ Extracts all finite values from list of dictionaries"""

        collated_data = []
        for x_dict in x_list:
            sample = x_dict[feature_name]
            values = sample.values.flatten()
            finite_values = values[np.isfinite(values)]
            collated_data.extend(finite_values)

        return np.asarray(collated_data)


def get_unique_symbols(data_list):
    """Returns a list of all unique symbols in the dict of dataframes"""

    symbols = set()

    for data_dict in data_list:
        for feature in data_dict:
            feat_symbols = data_dict[feature].columns
            symbols.update(feat_symbols)

    return symbols
