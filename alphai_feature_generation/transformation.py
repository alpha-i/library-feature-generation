from abc import ABCMeta, abstractmethod
from datetime import timedelta
import logging

import numpy as np
import pandas_market_calendars as mcal
import pandas as pd


from alphai_feature_generation import MINUTES_IN_TRADING_DAY
from alphai_feature_generation.feature import (FinancialFeature,
                                               get_feature_names,
                                               get_feature_max_ndays)

TOTAL_TICKS_FINANCIAL_FEATURES = ['open_value', 'high_value', 'low_value', 'close_value', 'volume_value']
TOTAL_TICKS_M1_FINANCIAL_FEATURES = ['open_log-return', 'high_log-return', 'low_log-return', 'close_log-return',
                                     'volume_log-return']

HARDCODED_FEATURE_FOR_EXTRACT_Y = 'close'

logging.getLogger(__name__).addHandler(logging.NullHandler())


class DateNotInUniverseError(Exception):
    pass


class DataTransformation(metaclass=ABCMeta):
    @abstractmethod
    def create_train_data(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def create_predict_data(self, *args):
        raise NotImplementedError()


class FinancialDataTransformation(DataTransformation):
    def __init__(self, configuration):
        """
        :param dict configuration: dictionary containing the feature details.
            list feature_config_list: list of dictionaries containing feature details.
            str exchange_name: name of the reference exchange
            int features_ndays: number of trading days worth of data the feature should use.
            int features_resample_minutes: resampling frequency in number of minutes.
            int features_start_market_minute: number of minutes after market open the data collection should start from
            int prediction_market_minute: number of minutes after market open for the prediction timestamp
            int target_delta_ndays: target time horizon in number of days
            int target_market_minute: number of minutes after market open for the target timestamp
        """
        self._assert_input(configuration)
        self.exchange_calendar = mcal.get_calendar(configuration['exchange_name'])
        self.features_ndays = configuration['features_ndays']
        self.features_resample_minutes = configuration['features_resample_minutes']
        self.features_start_market_minute = configuration['features_start_market_minute']
        self.prediction_market_minute = configuration['prediction_market_minute']
        self.target_delta_ndays = configuration['target_delta_ndays']
        self.target_market_minute = configuration['target_market_minute']
        self.classify_per_series = configuration['classify_per_series']
        self.normalise_per_series = configuration['normalise_per_series']
        self.feature_length = self.get_feature_length()
        self.features = self._financial_features_factory(configuration['feature_config_list'],
                                                         configuration['n_classification_bins'])
        self.n_series = configuration['nassets']
        self.configuration = configuration
        self.fill_limit = configuration['fill_limit']

    def _assert_input(self, configuration):
        assert isinstance(configuration['exchange_name'], str)
        assert isinstance(configuration['features_ndays'], int) and configuration['features_ndays'] >= 0
        assert isinstance(configuration['features_resample_minutes'], int) \
            and configuration['features_resample_minutes'] >= 0
        assert isinstance(configuration['features_start_market_minute'], int)
        assert configuration['features_start_market_minute'] < MINUTES_IN_TRADING_DAY
        assert configuration['prediction_market_minute'] >= 0
        assert configuration['prediction_market_minute'] < MINUTES_IN_TRADING_DAY
        assert configuration['target_delta_ndays'] >= 0
        assert configuration['target_market_minute'] >= 0
        assert configuration['target_market_minute'] < MINUTES_IN_TRADING_DAY
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
        ticks_in_a_day = np.floor(MINUTES_IN_TRADING_DAY / self.features_resample_minutes) + 1
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
        correct_dimensions = True

        for feature_full_name, feature_array in feature_x_dict.items():
            if feature_array.shape[0] != self.get_total_ticks_x():
                correct_dimensions = False

        return correct_dimensions

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
                if feature_array.shape != expected_shape:
                    correct_dimensions = False

        return correct_dimensions

    def _financial_features_factory(self, feature_config_list, n_classification_bins):
        """
        Build list of financial features from list of incomplete feature-config dictionaries (class-specific).
        :param list feature_config_list: list of dictionaries containing feature details.
        :return list: list of FinancialFeature objects
        """
        assert isinstance(feature_config_list, list)

        feature_list = []
        for single_feature_dict in feature_config_list:
            feature_list.append(FinancialFeature(
                single_feature_dict['name'],
                single_feature_dict['transformation'],
                single_feature_dict['normalization'],
                n_classification_bins,
                self.get_total_ticks_x(),
                self.features_ndays,
                self.features_resample_minutes,
                self.features_start_market_minute,
                single_feature_dict['is_target'],
                self.exchange_calendar,
                single_feature_dict['local'],
                self.classify_per_series,
                self.normalise_per_series
            ))

        return feature_list

    def _get_market_open_list(self, raw_data_dict):
        """
        Return a list of market open timestamps from input data_dict
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return pd.Series: list of market open timestamps.
        """
        features_keys = get_feature_names(self.features)
        raw_data_start_date = raw_data_dict[features_keys[0]].index[0].date()
        raw_data_end_date = raw_data_dict[features_keys[0]].index[-1].date()
        return self.exchange_calendar.schedule(str(raw_data_start_date), str(raw_data_end_date)).market_open

    def get_target_feature(self):
        """
        Return the target feature in self.features
        :return FinancialFeature: target feature
        """
        return [feature for feature in self.features if feature.is_target][0]

    def get_prediction_data_all_features(self, raw_data_dict, prediction_timestamp, universe=None,
                                         target_timestamp=None):
        """
        Collect processed prediction x and y data for all the features.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param list/None universe: list of relevant symbols
        :param Timestamp/None target_timestamp: Timestamp the prediction is for.
        :return (dict, dict): feature_x_dict, feature_y_dict
        """
        feature_x_dict, feature_y_dict = {}, {}

        for feature in self.features:
            if universe is None:
                universe = raw_data_dict[feature.name].columns

            feature_name = feature.full_name if feature.full_name in raw_data_dict.keys() else feature.name

            feature_x, feature_y = feature.get_prediction_data(
                raw_data_dict[feature_name].loc[:, universe],
                prediction_timestamp,
                target_timestamp,
                calculate_target=False
            )

            # currently target is harder coded to be log-return calculated on the close (Chris B)
            # TODO seperate feature and target calculation

            if feature.is_target:
                _, feature_y = feature.get_prediction_data(
                    raw_data_dict[HARDCODED_FEATURE_FOR_EXTRACT_Y].loc[:, universe],
                    prediction_timestamp,
                    target_timestamp,
                    calculate_target=True
                )

            if feature_x is not None:
                feature_x_dict[feature.full_name] = feature_x

            if feature_y is not None:
                feature_y_dict[feature.full_name] = feature_y.to_frame().transpose()
                feature_y_dict[feature.full_name].set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
        if len(feature_y_dict) > 0:
            assert len(feature_y_dict) == 1, 'Only one target is allowed'
        else:
            feature_y_dict = None

        return feature_x_dict, feature_y_dict

    def create_train_data(self, raw_data_dict, historical_universes):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        training_dates = self.get_training_market_dates(raw_data_dict)
        train_x, train_y, _ = self._create_data(raw_data_dict, training_dates, historical_universes,
                                                do_normalisation_fitting=True)
        clean_nan_from_dict = self.configuration.get('clean_nan_from_dict', False)
        if clean_nan_from_dict:
            train_x, train_y = remove_nans_from_dict(train_x, train_y)

        return train_x, train_y

    def create_predict_data(self, raw_data_dict):
        """
        Prepare x data for inference purposes.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return dict: feature_x_dict
        """

        current_market_open = self.get_current_market_date(raw_data_dict)
        predict_x, _, symbols = self._create_data(raw_data_dict, simulated_market_dates=current_market_open)

        clean_nan_from_dict = self.configuration.get('clean_nan_from_dict', False)
        if clean_nan_from_dict:
            predict_x = remove_nans_from_dict(predict_x)
            logging.warning("May need to update symbols when removing nans from dict")

        return predict_x, symbols

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
        market_open_list = self._get_market_open_list(raw_data_dict)

        data_x_list = []
        data_y_list = []
        rejected_x_list = []
        rejected_y_list = []

        target_market_open = None

        if len(simulated_market_dates) == 0:
            logging.error("Empty Market dates")

            raise ValueError("Empty Market dates")

        for prediction_market_open in simulated_market_dates:

            date_index = pd.Index(market_open_list).get_loc(prediction_market_open)
            target_index = date_index + self.target_delta_ndays

            if target_index < len(market_open_list):
                target_market_open = market_open_list[target_index]
            else:
                target_market_open = None

            try:
                feature_x_dict, feature_y_dict = self.build_features(raw_data_dict, historical_universes,
                                                                     prediction_market_open, target_market_open)
            except DateNotInUniverseError as e:
                logging.error(e)
                continue

            except KeyError as e:
                logging.error("Error while building features. {}. prediction_time: {}. target_time: {}".format(
                    e,
                    prediction_market_open,
                    target_market_open
                ))
                continue
            except Exception as e:
                logging.error('Failed to build a set of features', exc_info=e)
                raise e

            if self.check_x_batch_dimensions(feature_x_dict):
                data_x_list.append(feature_x_dict)
                data_y_list.append(feature_y_dict)
            else:
                rejected_x_list.append(feature_x_dict)
                rejected_y_list.append(feature_y_dict)

        n_valid_samples = len(data_x_list)

        if n_valid_samples < n_samples:
            logging.info("{} out of {} samples were found to be valid".format(n_valid_samples, n_samples))
            if len(rejected_x_list) > 0:
                self.print_diagnostics(rejected_x_list[-1], rejected_y_list[-1])

        data_x_list = self._make_normalised_x_list(data_x_list, do_normalisation_fitting)

        if target_market_open is None:
            y_dict = None
            x_dict, x_symbols = self.stack_samples_for_each_feature(data_x_list)
            logging.info("Assembled prediction dict with {} symbols".format(len(x_symbols)))
        else:
            logging.info("{} out of {} samples were found to be valid".format(n_valid_samples, n_samples))
            y_list = self._make_classified_y_list(data_y_list)
            x_dict, x_symbols = self.stack_samples_for_each_feature(data_x_list, y_list)
            y_dict, _ = self.stack_samples_for_each_feature(y_list)
            logging.info("Assembled training dict with {} symbols".format(len(x_symbols)))

        return x_dict, y_dict, x_symbols

    def print_diagnostics(self, xdict, ydict):
        """

        :param xdict:
        :param ydict:
        :return:
        """

        x_sample = list(xdict.values())[0]
        x_expected_shape = self.get_total_ticks_x()
        logging.info("Last rejected xdict: {}".format(x_sample.shape))
        logging.info("x_expected_shape: {}".format(x_expected_shape))

        if ydict is not None:
            y_sample = list(ydict.values())[0]
            y_expected_shape = (self.n_series,)
            logging.info("Last rejected ydict: {}".format(y_sample.shape))
            logging.info("y_expected_shape: {}".format(y_expected_shape))

    def _make_normalised_x_list(self, x_list, do_normalisation_fitting):
        """ Collects sample of x into a dictionary, and applies normalisation

        :param x_list: List of unnormalised dictionaries
        :param bool do_normalisation_fitting: Whether to use pre-fitted normalisation, or set normalisation constants
        :return: dict Dictionary of normalised features
        """

        if len(x_list) == 0:
            raise ValueError("No valid x samples found.")

        symbols = get_unique_symbols(x_list)

        # Fitting
        if do_normalisation_fitting:
            for feature in self.features:
                if feature.scaler:
                    logging.info("Fitting normalisation to: {}".format(feature.full_name))
                    if self.normalise_per_series:
                        for symbol in symbols:
                            symbol_data = self.extract_data_by_symbol(x_list, symbol, feature.full_name)
                            feature.fit_normalisation(symbol_data, symbol)
                    else:
                        all_data = self.extract_all_data(x_list, feature.full_name)
                        feature.fit_normalisation(all_data)
                else:
                    logging.info("Skipping normalisation to: {}".format(feature.full_name))

        # Applying
        for feature in self.features:
            if feature.scaler:
                logging.info("Applying normalisation to: {}".format(feature.full_name))
                for x_dict in x_list:
                    if feature.full_name in x_dict:
                        x_dict[feature.full_name] = feature.apply_normalisation(x_dict[feature.full_name])
                    else:
                        logging.info("Failed to find {} in dict: {}".format(feature.full_name, list(x_dict.keys())))
                        logging.info("x_list: {}".format(x_list))

        return x_list

    @staticmethod
    def extract_data_by_symbol(x_list, symbol, feature_name):
        """ Collect all data from a list of dicts of features, for a given symbol """

        collated_data = []
        for x_dict in x_list:
            if symbol in x_dict[feature_name].columns:
                sample = x_dict[feature_name][symbol]
                collated_data.extend(sample.dropna().values)

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
        logging.info("Fitting y classification to: {}".format(target_name))
        for symbol in symbols:
            symbol_data = self.extract_data_by_symbol(y_list, symbol, target_name)
            target_feature.fit_classification(symbol, symbol_data)

        # Applying
        logging.info("Applying y classification to: {}".format(target_name))
        for y_dict in y_list:
            if target_name in y_dict:
                y_dict[target_name] = target_feature.apply_classification(y_dict[target_name])
            else:
                logging.info("Failed to find {} in dict: {}".format(target_name, list(y_dict.keys())))
                logging.info("y_list: {}".format(y_list))

        return y_list

    def build_features(self, raw_data_dict, universe, prediction_market_open, target_market_open):
        """ Creates dictionaries of features and labels for a single window

        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe universe: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param prediction_market_open:
        :param target_market_open:
        :return:
        """

        if universe is None:
            universe = None
        else:
            prediction_date = prediction_market_open.date()
            universe = _get_universe_from_date(prediction_date, universe)

        prediction_timestamp = prediction_market_open + timedelta(minutes=self.prediction_market_minute)

        if target_market_open is None:
            target_timestamp = None
        else:
            target_timestamp = target_market_open + timedelta(minutes=self.target_market_minute)

        feature_x_dict, feature_y_dict = self.get_prediction_data_all_features(raw_data_dict,
                                                                               prediction_timestamp,
                                                                               universe,
                                                                               target_timestamp,
                                                                               )

        return feature_x_dict, feature_y_dict

    def add_transformation(self, raw_data_dict):
        """
        add new features to data dictionary
        :param raw_data_dict: dictionary of dataframes
        :return: dict with new keys
        """

        for feature in self.features:
            if not feature.local and feature.full_name not in raw_data_dict.keys():
                raw_data_dict[feature.full_name] = feature.process_prediction_data_x(raw_data_dict)

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

        stacked_samples = {}
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
                            logging.warning("Oi, your columns dont match")

                if len(feature_list) > 0:
                    stacked_samples[feature_name] = np.stack(feature_list, axis=0)
                else:
                    stacked_samples = None

        if len(samples) > 1:
            logging.info("Found {} unusual samples out of {}".format(unusual_samples, total_samples))

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

    def get_current_market_date(self, raw_data_dict):
        return [self._get_market_open_list(raw_data_dict)[-1]]

    def get_training_market_dates(self, raw_data_dict):
        """ Returns all dates on which we have both x and y data"""

        max_feature_ndays = get_feature_max_ndays(self.features)

        return self._get_market_open_list(raw_data_dict)[max_feature_ndays:-self.target_delta_ndays]


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
        raise DateNotInUniverseError("Error getting universe from date. date {} not in universe".format(date))


def get_unique_symbols(data_list):
    """Returns a list of all unique symbols in the dict of dataframes"""

    symbols = set()

    for data_dict in data_list:
        for feature in data_dict:
            feat_symbols = data_dict[feature].columns
            symbols.update(feat_symbols)

    return symbols


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

    logging.info("Found {} examples with Nans, removing examples"
                 " from all dicts".format((~resulting_bool_array).sum()))
    logging.info("{} examples still left in the dicts".format(resulting_bool_array.sum()))

    # apply selection to all dicts
    for key, value in x_dict.items():
        x_dict[key] = value[resulting_bool_array]

    if y_dict:
        for key, value in y_dict.items():
            y_dict[key] = value[resulting_bool_array]
        return x_dict, y_dict
    else:
        return x_dict
