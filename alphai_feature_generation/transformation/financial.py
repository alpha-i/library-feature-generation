import logging
import multiprocessing

from collections import OrderedDict
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from alphai_feature_generation.feature.factory import FinancialFeatureFactory, FeatureList
from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.helpers import CalendarUtilities, logtime
from alphai_feature_generation.transformation.base import (
    ensure_closing_pool,
    DataTransformation,
    get_unique_symbols,
    DateNotInUniverseError)


TOTAL_TICKS_FINANCIAL_FEATURES = ['open_value', 'high_value', 'low_value', 'close_value', 'volume_value']
TOTAL_TICKS_M1_FINANCIAL_FEATURES = ['open_log-return', 'high_log-return', 'low_log-return', 'close_log-return',
                                     'volume_log-return']

HARDCODED_FEATURE_FOR_EXTRACT_Y = 'close'
logger = logging.getLogger(__name__)

KEY_EXCHANGE = FinancialFeature.KEY_EXCHANGE


class FinancialDataTransformation(DataTransformation):
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
        self.exchange_calendar = mcal.get_calendar(configuration[KEY_EXCHANGE])
        self.minutes_in_trading_days = CalendarUtilities.get_minutes_in_one_trading_day(configuration[KEY_EXCHANGE])
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
        self.clean_nan_from_dict = configuration.get('clean_nan_from_dict', False)

        self.feature_length = self.get_feature_length()
        self.features = self._feature_factory(configuration['feature_config_list'])

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
        :return list: list of FinancialFeature objects
        """
        assert isinstance(feature_config_list, list)

        update_dict = {
            'nbins': self.n_classification_bins,
            'ndays': self.features_ndays,
            'start_market_minute': self.features_start_market_minute,
            KEY_EXCHANGE: self.exchange_calendar.name,
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

        factory = FinancialFeatureFactory()

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

        return self.exchange_calendar.schedule(str(raw_data_start_date), str(raw_data_end_date))

    def get_target_feature(self):
        """
        Return the target feature in self.features
        :return FinancialFeature: target feature
        """
        for feature in self.features:
            if feature.is_target:
                return feature

    def collect_prediction_from_features(self, raw_data_dict, x_end_timestamp, y_start_timestamp, universe=None,
                                         target_timestamp=None):
        """
        Collect processed prediction x and y data for all the features.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param list/None universe: list of relevant symbols
        :param Timestamp/None target_timestamp: Timestamp the prediction is for.
        :return (dict, dict): feature_x_dict, feature_y_dict
        """
        feature_x_dict = OrderedDict()
        feature_y_dict = OrderedDict()

        # FIXME This parallelisation was creating a crash in the backtests. So switching off for now.
        part = partial(self.process_predictions, x_end_timestamp, y_start_timestamp,
                       raw_data_dict, target_timestamp, universe)
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

    def process_predictions(self, x_timestamp, y_timestamp, raw_data_dict, target_timestamp, universe, feature):
        # TODO separate feature and target calculation

        if universe is None:
            universe = raw_data_dict[feature.name].columns
        feature_name = feature.full_name if feature.full_name in raw_data_dict.keys() else feature.name
        feature_x = feature.get_prediction_features(
            raw_data_dict[feature_name].loc[:, universe],
            x_timestamp
        )

        feature_y = None
        if feature.is_target:
            feature_y = feature.get_prediction_targets(
                # currently target is hardcoded to be log-return calculated on the close (Chris B)
                raw_data_dict[HARDCODED_FEATURE_FOR_EXTRACT_Y].loc[:, universe],
                y_timestamp,
                target_timestamp
            )

            if feature_y is not None:
                transposed_y = feature_y.to_frame().transpose()
                transposed_y.set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
                feature_y = transposed_y

        return feature.full_name, feature_x, feature_y

    def create_train_data(self, raw_data_dict, historical_universes):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        raw_data_dict = self.add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)

        raw_data_dict['close'] = raw_data_dict['close'].astype('float32', copy=False)

        market_schedule = self._extract_schedule_for_training(raw_data_dict)
        normalise = True

        train_x, train_y, _, _ = self._create_data(raw_data_dict, market_schedule, historical_universes, normalise)

        if self.clean_nan_from_dict:
            train_x, train_y = remove_nans_from_dict(train_x, train_y)

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

        raw_data_dict = self.add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)
        market_schedule = self._extract_schedule_for_prediction(raw_data_dict)

        predict_x, _, symbols, predict_timestamp = self._create_data(raw_data_dict, market_schedule)

        if self.clean_nan_from_dict:
            predict_x = remove_nans_from_dict(predict_x)
            logger.debug("May need to update symbols when removing nans from dict")

        prediction_day = predict_timestamp.date()
        schedule = self.exchange_calendar.schedule(prediction_day, prediction_day + timedelta(days=10))

        target_timestamp = self._get_valid_target_timestamp_in_schedule(schedule, predict_timestamp)

        _, predict_timestamp = self._get_prediction_timestamps(schedule.loc[prediction_day]['market_open'])

        return predict_x, symbols, predict_timestamp, target_timestamp

    def _get_valid_target_timestamp_in_schedule(self, schedule, predict_timestamp):
        """
        Return valid market time for target time given timestamp and delta_n_days

        :param predict_timestamp:
        :type predict_timestamp: pd.Timestamp
        :return target_timestamp:
        :rtype target_timestamp: pd.Timestamp
        """

        target_market_schedule = self._extract_target_market_day(schedule, predict_timestamp)

        target_market_open = target_market_schedule.market_open
        target_timestamp = self._get_target_timestamp(target_market_open)

        if self.exchange_calendar.open_at_time(schedule, target_timestamp, include_close=True):
            return target_timestamp
        else:
            raise ValueError("Target timestamp {} not in market time".format(target_timestamp))

    @logtime
    def _create_data(self, raw_data_dict, simulated_market_dates,
                     historical_universes=None, do_normalisation_fitting=False):
        """
        Create x and y data
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param simulated_market_dates: List of dates for which we generate the 'past' and 'future' data
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
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
            fit_function = partial(self.build_features_function, managed_dict, historical_universes, data_schedule)
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

    def build_features_function(self, raw_data_dict, historical_universes, data_schedule, prediction_market_open):
        target_market_schedule = self._extract_target_market_day(data_schedule, prediction_market_open)
        target_market_open = target_market_schedule.market_open if target_market_schedule is not None else None

        try:
            feature_x_dict, feature_y_dict, prediction_timestamp = self.build_features(raw_data_dict,
                                                                                       historical_universes,
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

    def add_log_returns(self, data_dict):
        """ If not already in dictionary, add raw log returns

        :param data_dict: Original data dict
        :return: Updated dict
        """

        base_key = 'close' if 'close' in data_dict else list(data_dict.keys())[0]
        close_data = data_dict[base_key]
        data_dict['log-return'] = np.log(close_data.pct_change() + 1, dtype=np.float32).replace([np.inf, -np.inf],
                                                                                                np.nan)

        return data_dict

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

    def build_features(self, raw_data_dict, universe, target_market_open, prediction_market_open, ):
        """ Creates dictionaries of features and labels for a single window

        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe universe: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param prediction_market_open:
        :param target_market_open:
        :return:
        """

        if universe is not None:
            prediction_date = prediction_market_open.date()
            universe = _get_universe_from_date(prediction_date, universe)

        x_end_timestamp, y_start_timestamp = self._get_prediction_timestamps(prediction_market_open)
        target_timestamp = self._get_target_timestamp(target_market_open)

        if target_timestamp and y_start_timestamp > target_timestamp:
            raise ValueError('Target timestamp should be later than prediction_timestamp')

        feature_x_dict, feature_y_dict = self.collect_prediction_from_features(
            raw_data_dict, x_end_timestamp, y_start_timestamp, universe, target_timestamp)

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
                return CalendarUtilities.closing_time_for_day(self.exchange_calendar, target_market_open.date())
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
            y_start_timestamp = CalendarUtilities.closing_time_for_day(self.exchange_calendar, prediction_market_open)
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
        means, cov_matrix = target_feature.inverse_transform_multi_predict_y(predict_y, symbols)

        return means, cov_matrix

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


def _get_universe_from_date(date, historical_universes):
    """
    Select the universe list of symbols from historical_universes dataframe, given input date.
    :param pd.datetime.date date: Date for which the universe is required.
    :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
    :return list: list of relevant symbols
    """
    try:
        universe_idx = historical_universes[(date >= historical_universes.start_date) &
                                            (date < historical_universes.end_date)].index[0]
        return historical_universes.assets[universe_idx]
    except IndexError as e:
        raise DateNotInUniverseError("Date {} not in universe. Skip".format(date))

def remove_nans_from_dict(x_dict, y_dict=None):
    """
    looks for any of the examples in the dictionaries that have NaN and removes all those

    :param x_dict: x_dict with features
    :param y_dict: y_dict with targets
    :return: X_dict and y_dict
    """

    for key, value in x_dict.items():
        n_examples = value.shape[0]
        break

    resulting_bool_array = np.ones(n_examples, dtype=bool)

    for key, value in x_dict.items():
        resulting_bool_array = resulting_bool_array & ~np.isnan(value).sum(axis=2).sum(axis=1).astype(bool)

    if y_dict:
        for key, value in y_dict.items():
            resulting_bool_array = resulting_bool_array & ~np.isnan(value).sum(axis=2).sum(axis=1).astype(bool)

    logger.debug("Found {} examples with Nans, removing examples"
                 " from all dicts".format((~resulting_bool_array).sum()))
    logger.debug("{} examples still left in the dicts".format(resulting_bool_array.sum()))

    # apply selection to all dicts
    for key, value in x_dict.items():
        x_dict[key] = value[resulting_bool_array]

    if y_dict:
        for key, value in y_dict.items():
            y_dict[key] = value[resulting_bool_array]
        return x_dict, y_dict
    else:
        return x_dict