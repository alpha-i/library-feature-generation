import logging
import multiprocessing
from collections import OrderedDict
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alphai_feature_generation.feature import GymFeature
from alphai_feature_generation.feature.factory import FinancialFeatureFactory, FeatureList, GymFeatureFactory
from alphai_feature_generation.helpers import CalendarUtilities, logtime
from alphai_feature_generation.transformation.base import (
    ensure_closing_pool,
    DataTransformation,
    get_unique_symbols,
    DateNotInUniverseError)

logger = logging.getLogger(__name__)

KEY_EXCHANGE = GymFeature.KEY_EXCHANGE


class GymDataTransformation(DataTransformation):
    def __init__(self, configuration):
        """
        :param dict configuration: dictionary containing the feature details.
            list feature_config_list: list of dictionaries containing feature details.
            str exchange: name of the reference exchange
            int features_ndays: number of trading days worth of data the feature should use.
            int features_resample_minutes: resampling frequency in number of minutes.
            int features_start_market_minute: number of minutes after market open the data collection should start from
            int prediction_market_minute: number of minutes after market open for the prediction timestamp
            int target_delta_ndays: target time horizon in number of days
            int target_market_minute: number of minutes after market open for the target timestamp
        """

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
        try:
            self.holiday_calendar = mcal.get_calendar(configuration[KEY_EXCHANGE])
            self.minutes_in_trading_days = CalendarUtilities.get_minutes_in_one_trading_day(configuration[KEY_EXCHANGE])
        except:
            self.holiday_calendar = None
            self.minutes_in_trading_days = 1440

        self.feature_length = self.get_feature_length()
        self.features = self._feature_factory(configuration['feature_config_list'])
        self.target_feature = configuration.get('target_feature', self.features[0].name)

        self.configuration = configuration
        self._assert_input()

    def _assert_input(self):
        configuration = self.configuration

        assert isinstance(configuration[KEY_EXCHANGE], str)
        assert isinstance(configuration['features_ndays'], int) and configuration['features_ndays'] >= 0
        assert isinstance(configuration['features_resample_minutes'], int) \
               and configuration['features_resample_minutes'] >= 0
        assert isinstance(configuration['features_start_market_minute'], int)
        assert configuration['features_start_market_minute'] < self.minutes_in_trading_days
        assert configuration['prediction_market_minute'] >= 0
        assert configuration['prediction_market_minute'] < self.minutes_in_trading_days
        assert configuration['target_delta_ndays'] >= 0
        assert configuration['target_market_minute'] >= 0
        assert configuration['target_market_minute'] < self.minutes_in_trading_days
        assert isinstance(configuration['feature_config_list'], list)
        n_targets = 0
        for single_feature_dict in configuration['feature_config_list']:
            if single_feature_dict['is_target']:
                n_targets += 1
        assert n_targets == 1
        assert isinstance(configuration['fill_limit'], int)

    def get_feature_length(self):
        """
        Calculate expected total ticks for x data
        :return int: expected total number of ticks for x data
        """
        return self.get_total_ticks_x()

    def get_total_ticks_x(self):
        """
        Calculate expected total ticks for x data
        :return int: expected total number of ticks for x data
        """
        ticks_in_a_day = np.floor(self.minutes_in_trading_days / self.features_resample_minutes) + 1
        intra_day_ticks = np.floor((self.prediction_market_minute - self.features_start_market_minute) /
                                   self.features_resample_minutes)
        total_ticks = ticks_in_a_day * self.features_ndays + intra_day_ticks + 1
        return int(total_ticks)

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

    def _feature_factory(self, feature_config_list):
        """
        Build list of financial features from list of incomplete feature-config dictionaries (class-specific).
        :param list feature_config_list: list of dictionaries containing feature details.
        :return list: list of GymFeature objects
        """
        assert isinstance(feature_config_list, list)

        update_dict = {
            'nbins': self.n_classification_bins,
            'ndays': self.features_ndays,
            'start_market_minute': self.features_start_market_minute,
            KEY_EXCHANGE: self.holiday_calendar.name,
            'classify_per_series': self.classify_per_series,
            'normalise_per_series': self.normalise_per_series
        }

        for feature in feature_config_list:
            specific_update = {
                'length': feature.get('length', self.get_total_ticks_x()),
                'resample_minutes': feature.get('resolution', 0),
                'is_target': feature.get('is_target', False),
                'local': feature.get('local', False)
            }

            feature.update(update_dict)
            feature.update(specific_update)

        factory = GymFeatureFactory()

        return factory.create_from_list(feature_config_list)

    def _extract_schedule_from_data(self, raw_data_dict):
        """
        Return a list of market open timestamps from input data_dict
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return pd.Series: list of market open timestamps.
        """
        features_keys = self.features.get_names()
        raw_data_start_date = raw_data_dict[features_keys[0]].index[0].date()
        raw_data_end_date = raw_data_dict[features_keys[0]].index[-1].date()

        return self.holiday_calendar.schedule(str(raw_data_start_date), str(raw_data_end_date))

    def get_target_feature(self):
        """
        Return the target feature in self.features
        :return GymFeature: target feature
        """
        for feature in self.features:
            if feature.is_target:
                return feature

    def collect_prediction_from_features(self, raw_data_dict, x_end_timestamp, y_start_timestamp, target_timestamp=None):
        """
        Collect processed prediction x and y data for all the features.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param Timestamp/None target_timestamp: Timestamp the prediction is for.
        :return (dict, dict): feature_x_dict, feature_y_dict
        """
        feature_x_dict = OrderedDict()
        feature_y_dict = OrderedDict()

        part = partial(self.process_predictions, x_end_timestamp, y_start_timestamp,
                       raw_data_dict, target_timestamp)
        processed_predictions = map(part, self.features)

        for prediction in processed_predictions:
            feature_x_dict[prediction[0]] = prediction[1]
            if prediction[2] is not None:
                feature_y_dict[prediction[0]] = prediction[2]

        if len(feature_y_dict) > 0:
            assert len(feature_y_dict) == 1, 'Only one target is allowed'
        else:
            feature_y_dict = None

        return feature_x_dict, feature_y_dict

    def process_predictions(self, x_timestamp, y_timestamp, raw_data_dict, target_timestamp, feature):

        universe = raw_data_dict[feature.name].columns
        feature_name = feature.full_name if feature.full_name in raw_data_dict.keys() else feature.name
        feature_x = feature.get_prediction_features(
            raw_data_dict[feature_name].loc[:, universe],
            x_timestamp
        )

        feature_y = None
        if feature.is_target:
            feature_y = feature.get_prediction_targets(
                # Unless specified otherwise, target is the first feature in list
                raw_data_dict[self.target_feature].loc[:, universe],
                y_timestamp,
                target_timestamp
            )

            #FIXME unclear why this transpose is necessary
            if feature_y is not None:
                transposed_y = feature_y.to_frame().transpose()
                transposed_y.set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
                feature_y = transposed_y

        return feature.full_name, feature_x, feature_y

    def create_train_data(self, raw_data_dict):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)
        market_schedule = self._extract_schedule_for_training(raw_data_dict)
        fit_normalisation = True

        train_x, train_y, _, _ = self._create_data(raw_data_dict, market_schedule, fit_normalisation)

        return train_x, train_y

    def filter_unwanted_keys(self, data_dict):
        """

        :param data_dict: Dictionary we wish to trim
        :return:
        """

        wanted_keys = {feature.name for feature in self.features}

        return {key: value for key, value in data_dict.items() if key in wanted_keys}

    def create_predict_data(self, raw_data_dict):
        """

        :param raw_data_dict:
        :return: tuple: predict, symbol_list, prediction_timestamp, target_timestamp
        """

        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)
        market_schedule = self._extract_schedule_for_prediction(raw_data_dict)

        predict_x, _, symbols, predict_timestamp = self._create_data(raw_data_dict, market_schedule)
        target_timestamp = predict_timestamp + timedelta(days=self.target_delta_ndays)

        return predict_x, symbols, predict_timestamp, target_timestamp

    @logtime
    def _create_data(self, raw_data_dict, simulated_market_dates, do_normalisation_fitting=False):
        """
        Create x and y data
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param simulated_market_dates: List of dates for which we generate the 'past' and 'future' data
        :param bool do_normalisation_fitting: Whether to fit data normalisation parameters
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        n_samples = len(simulated_market_dates)
        data_schedule = self._extract_schedule_from_data(raw_data_dict)

        self.apply_global_transformations(raw_data_dict)

        data_x_list = []
        data_y_list = []
        rejected_x_list = []
        rejected_y_list = []

        prediction_timestamp_list = []

        target_market_open = None

        if len(simulated_market_dates) == 0:
            logger.debug("Empty Market dates")

            raise ValueError("Empty Market dates")

        managed_dict = multiprocessing.Manager().dict(raw_data_dict)

        with ensure_closing_pool() as pool:
            fit_function = partial(self.build_features_function, managed_dict, data_schedule)
            pooled_results = pool.map(fit_function, list(simulated_market_dates.market_open))

        for result in pooled_results:
            feature_x_dict, feature_y_dict, prediction_timestamp, target_market_open = result
            if feature_x_dict is not None:
                prediction_timestamp_list.append(prediction_timestamp)
                if self.check_x_batch_dimensions(feature_x_dict):
                    data_x_list.append(feature_x_dict)
                    data_y_list.append(feature_y_dict)
                else:
                    rejected_x_list.append(feature_x_dict)
                    rejected_y_list.append(feature_y_dict)

        n_valid_samples = len(data_x_list)

        if n_valid_samples < n_samples:
            logger.debug("{} out of {} samples were found to be valid".format(n_valid_samples, n_samples))
            if len(rejected_x_list) > 0:
                self.print_diagnostics(rejected_x_list[-1], rejected_y_list[-1])

        logger.debug("Making normalised x list")
        data_x_list = self._make_normalised_x_list(data_x_list, do_normalisation_fitting)

        action = 'prediction'
        y_dict = None
        y_list = None

        if target_market_open:
            action = 'training'
            logger.debug("{} out of {} samples were found to be valid".format(n_valid_samples, n_samples))
            classify_y = self.n_classification_bins
            y_list = self._make_classified_y_list(data_y_list) if classify_y else data_y_list
            y_dict, _ = self.stack_samples_for_each_feature(y_list)

        x_dict, x_symbols = self.stack_samples_for_each_feature(data_x_list, y_list)
        logger.debug("Assembled {} dict with {} symbols".format(action, len(x_symbols)))

        prediction_timestamp = prediction_timestamp_list[-1] if len(prediction_timestamp_list) > 0 else None

        return x_dict, y_dict, x_symbols, prediction_timestamp

    def build_features_function(self, raw_data_dict, data_schedule, prediction_market_open):
        target_market_schedule = self._extract_target_market_day(data_schedule, prediction_market_open)
        target_market_open = target_market_schedule.market_open if target_market_schedule is not None else None

        try:
            feature_x_dict, feature_y_dict, prediction_timestamp = self.build_features(raw_data_dict,
                                                                                       target_market_open,
                                                                                       prediction_market_open)
        except DateNotInUniverseError as e:
            logger.debug(e)
            return None, None, None, target_market_open

        except KeyError as e:
            logger.debug("Error while building features. {}. prediction_time: {}".format(
                e, prediction_market_open))
            return None, None, None, target_market_open
        except Exception as e:
            logger.debug('Failed to build a set of features', exc_info=e)
            return None, None, None, target_market_open

        return feature_x_dict, feature_y_dict, prediction_timestamp, target_market_open

    def _extract_target_market_day(self, market_schedule, prediction_timestamp):
        """
        Extract the target market open day using prediction day and target delta

        :param market_schedule:
        :param prediction_timestamp:

        :return:
        """

        target_index = market_schedule.index.get_loc(prediction_timestamp.date()) + self.target_delta_ndays

        if target_index < len(market_schedule):
            return market_schedule.iloc[target_index]
        else:
            return None

    def print_diagnostics(self, xdict, ydict):
        """

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

    def _make_normalised_x_list(self, x_list, do_normalisation_fitting):
        """ Collects sample of x into a dictionary, and applies normalisation

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

    def apply_normalisation(self, x_list, feature):

        if feature.scaler:
            normalise_function = partial(self.normalise_dict, feature)
            logger.debug("Applying normalisation to: {}".format(feature.full_name))

            with ensure_closing_pool() as pool:
                list_of_x_dicts = pool.map(normalise_function, x_list)

            return list_of_x_dicts
        else:
            return x_list

    def normalise_dict(self, target_feature, x_dict):
        if target_feature.full_name in x_dict:
            x_dict[target_feature.full_name] = target_feature.apply_normalisation(x_dict[target_feature.full_name])
        else:
            logger.debug("Failed to find {} in dict: {}".format(target_feature.full_name, list(x_dict.keys())))
        return x_dict

    def fit_normalisation(self, symbols, x_list, feature):
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

    def _make_classified_y_list(self, y_list):
        """ Takes list of dictionaries, and classifies them based on the full sample

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
        if target_name in y_dict:
            y_dict[target_name] = target_feature.apply_classification(y_dict[target_name])
        else:
            logger.debug("Failed to find {} in dict: {}".format(target_name, list(y_dict.keys())))
        return y_dict

    def build_features(self, raw_data_dict, target_market_open, prediction_market_open, ):
        """ Creates dictionaries of features and labels for a single window

        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe universe: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param prediction_market_open:
        :param target_market_open:
        :return:
        """

        prediction_date = prediction_market_open.date()

        x_end_timestamp, y_start_timestamp = self._get_prediction_timestamps(prediction_market_open)
        target_timestamp = self._get_target_timestamp(target_market_open)

        if target_timestamp and y_start_timestamp > target_timestamp:
            raise ValueError('Target timestamp should be later than prediction_timestamp')

        feature_x_dict, feature_y_dict = self.collect_prediction_from_features(
            raw_data_dict, x_end_timestamp, y_start_timestamp, target_timestamp)

        return feature_x_dict, feature_y_dict, x_end_timestamp

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
                return CalendarUtilities.closing_time_for_day(self.holiday_calendar, target_market_open.date())
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
            y_start_timestamp = CalendarUtilities.closing_time_for_day(self.holiday_calendar, prediction_market_open)
        else:
            y_start_timestamp = x_end_timestamp

        return x_end_timestamp, y_start_timestamp

    def apply_global_transformations(self, raw_data_dict):
        """
        add new features to data dictionary
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

    def inverse_transform_multi_predict_y(self, predict_y, symbols):
        """
        Inverse-transform multi-pass predict_y data
        :param ndarray predict_y: target multi-pass prediction data
        :param symbols : list of symbols
        :return ndarray: inversely transformed multi-pass predict_y data
        """
        target_feature = self.get_target_feature()
        medians, lower_bound, upper_bound = target_feature.inverse_transform_multi_predict_y(predict_y, symbols)

        return medians, lower_bound, upper_bound

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


