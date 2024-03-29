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
from alphai_feature_generation.transformation.schemas import DataTransformationConfigurationSchema, \
    InvalidConfigurationException

logger = logging.getLogger(__name__)


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
    """ Sorry Morty, we seem to be in the wrong universe. """
    pass


class DataTransformation(metaclass=ABCMeta):
    """ Prepares raw time series data for machine learning applications. """

    KEY_EXCHANGE = 'calendar_name'
    CONFIGURATION_SCHEMA = DataTransformationConfigurationSchema
    GENERIC_SYMBOL = 'SYM'

    def __init__(self, configuration):
        """Initialise in accordance with the config dictionary.

        :param dict configuration:
        """
        parsed_configuration, errors = self.CONFIGURATION_SCHEMA().load(configuration)
        if errors:
            raise InvalidConfigurationException(errors)
        
        self.configuration = parsed_configuration

        self._calendar = mcal.get_calendar(self.configuration[self.KEY_EXCHANGE])
        self.minutes_in_trading_days = self._calendar.get_minutes_in_one_day()

        self.features_ndays = self.configuration['features_ndays']
        self.features_resample_minutes = self.configuration['features_resample_minutes']
        self.features_start_market_minute = self.configuration['features_start_market_minute']
        self.prediction_market_minute = self.configuration['prediction_market_minute']

        self.target_delta = self.configuration['target_delta']
        self.target_market_minute = self.configuration['target_market_minute']
        self.classify_per_series = self.configuration['classify_per_series']
        self.normalise_per_series = self.configuration['normalise_per_series']
        self.n_classification_bins = self.configuration['n_classification_bins']
        self.n_series = self.configuration['n_assets']
        self.fill_limit = self.configuration['fill_limit']
        self.predict_the_market_close = self.configuration.get('predict_the_market_close', False)
        self.n_forecasts = self.configuration['n_forecasts']

        self.features = self._feature_factory(self.configuration['feature_config_list'])
        self.feature_length = self.get_feature_length()

        self._validate_configuration()

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
        return self.features.get_target_feature()

    def filter_unwanted_keys(self, raw_data_dict):
        """
        Remove useless data from the raw_data_dict
        
        :param raw_data_dict: Dictionary we wish to trim
        :return dict: data which belong to the expected keys
        """
        wanted_keys = {feature.name for feature in self.features}

        return {key: value for key, value in raw_data_dict.items() if key in wanted_keys}

    def _validate_configuration(self): #TODO MOVE TO FINANCIAL
        """ Make sure your inputs are sensible.  """
        assert self.features_start_market_minute < self.minutes_in_trading_days
        assert 0 <= self.prediction_market_minute < self.minutes_in_trading_days
        assert 0 <= self.target_market_minute < self.minutes_in_trading_days

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

        target_index = market_schedule.index.get_loc(prediction_timestamp.date()) + self.target_delta.days

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

        symbols = self._get_unique_symbols(x_list)

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
            logger.debug("Applying normalisation to: {}".format(feature.full_name))

            list_of_x_dicts = []
            for x_dict in x_list:
                list_of_x_dicts.append(self.normalise_dict(feature, x_dict))

            return list_of_x_dicts
        else:
            return x_list

    def normalise_dict(self, feature, x_dict):
        """ Apply normalisation with a single feature if is in the x_dict.

        :param Feature feature:
        :param dict x_dict:
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
        symbols = self._get_unique_symbols(y_list)

        # Fitting of bins
        logger.debug("Fitting y classification to: {}".format(target_feature.full_name))

        if self.classify_per_series:
            for symbol in symbols:
                    symbol_data = self.extract_data_by_symbol(y_list, symbol, target_feature.full_name)
                    target_feature.fit_classification(symbol, symbol_data)
        else:
            all_data = self.extract_all_data(y_list, target_feature.full_name)
            target_feature.fit_classification(self.GENERIC_SYMBOL, all_data)

        # Applying
        logger.debug("Applying y classification to: {}".format(target_feature.full_name))
        applied_y_list = []
        for y_dict in y_list:
            applied_y_list.append(
                self._apply_classification(target_feature, y_dict)
            )

        return applied_y_list

    def _apply_classification(self, target_feature, y_dict):
        """  Classifies the y values.

        :param target_feature: The 'feature' that will act as the target for the network
        :param y_dict:
        :return:
        """
        if target_feature.full_name in y_dict:
            y_dict[target_feature.full_name] = target_feature.apply_classification(y_dict[target_feature.full_name])
        else:
            logger.debug("Failed to find {} in dict: {}".format( target_feature.full_name, list(y_dict.keys())))

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
                ts = self._calendar.closing_time_for_day(target_market_open.date())
            else:
                ts = target_market_open + timedelta(minutes=self.target_market_minute)
        else:
            return None

        return ts.to_pydatetime()

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

    def stack_samples_for_each_feature(self, input_samples):
        """ Collate a list of training_set (the training set) into a single dictionary
        :param input_samples: List of dicts, each dict should be holding the same set of keys
        :return: Single dictionary with the values stacked together
        """
        training_set_length = len(input_samples)

        if not training_set_length:
            raise ValueError("At least one sample required for stacking training_set.")

        stacked_training_set = OrderedDict()
        valid_symbols = []

        reference_sample = input_samples[0]
        for feature_name, feature_data in reference_sample.items():
            if training_set_length == 1:
                stacked_training_set[feature_name] = np.expand_dims(feature_data, axis=0)

                major_axis = getattr(feature_data, 'major_axis', [])
                valid_symbols = major_axis if len(major_axis) else getattr(feature_data, 'columns', [])

            else:
                feature_list = [sample[feature_name].values for sample in input_samples]

                if len(feature_list) > 0:
                    stacked_training_set[feature_name] = np.stack(feature_list, axis=0)
                else:
                    stacked_training_set = OrderedDict()

        return stacked_training_set, valid_symbols

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
        data_schedule = self._extract_schedule_from_data(raw_data_dict)

        if self.target_delta.days > 0:
            return data_schedule[max_feature_ndays:-self.target_delta.days]
        else:
            return data_schedule[max_feature_ndays:]

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

    @staticmethod
    def _get_unique_symbols(data_list):
        """Returns a list of all unique symbols in the dict of dataframes"""

        symbols = set()

        for data_dict in data_list:
            for feature in data_dict:
                feat_symbols = data_dict[feature].columns
                symbols.update(feat_symbols)

        return symbols
