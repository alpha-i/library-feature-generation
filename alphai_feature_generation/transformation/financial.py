import logging
from collections import OrderedDict, namedtuple
from datetime import timedelta
from functools import partial
import multiprocessing

import numpy as np
import pandas as pd
from alphai_feature_generation.feature.factory import FinancialFeatureFactory
from alphai_feature_generation.helpers import logtime
from alphai_feature_generation.transformation.base import (DataTransformation, DateNotInUniverseError,
                                                           ensure_closing_pool)
from alphai_feature_generation.transformation.schemas import FinancialDataTransformationConfigurationSchema

logger = logging.getLogger(__name__)

Feature = namedtuple('Feature', 'x_dict y_dict prediction_timestamp target_timestamp')


class FinancialDataTransformation(DataTransformation):

    CONFIGURATION_SCHEMA = FinancialDataTransformationConfigurationSchema

    def __init__(self, configuration):
        """Initialise in accordance with the config dictionary.

        :param dict configuration:
        """
        super().__init__(configuration)

        self.clean_nan_from_dict = self.configuration['clean_nan_from_dict']

    def _get_feature_for_extract_y(self):
        """ Returns the name of the feature to be used as a target (y). """
        return 'close'

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

        factory = FinancialFeatureFactory(self._calendar)

        return factory.create_from_list(feature_config_list)

    @logtime
    def _create_data(self, raw_data_dict, simulated_market_dates, historical_universes=None,
                     do_normalisation_fitting=False):
        """
        Create x and y data
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param simulated_market_dates: List of dates for which we generate the 'past' and 'future' data
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param bool do_normalisation_fitting:
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        total_number_of_samples = len(simulated_market_dates)
        data_schedule = self._extract_schedule_from_data(raw_data_dict)
        raw_data_dict = self.apply_global_transformations(raw_data_dict)

        if len(simulated_market_dates) == 0:
            logger.debug("Empty Market dates")
            raise ValueError("Empty Market dates")

        manager = multiprocessing.Manager()
        collector = manager.Namespace()
        collector.raw_data_dict = raw_data_dict
        collector.historical_universes = historical_universes
        collector.data_schedule = data_schedule

        partial_function = partial(self._build_features, collector)

        with ensure_closing_pool() as pool:
            market_open_times = [
                timestamp.to_pydatetime()
                for timestamp in list(simulated_market_dates.market_open)
            ]
            features = pool.map(partial_function, market_open_times)

        data_x_list = []
        data_y_list = []
        rejected_x_list = []
        rejected_y_list = []
        prediction_timestamp_list = []

        for feature in features:
            feature_x_dict, feature_y_dict, prediction_timestamp, target_market_open = feature
            if feature_x_dict is not None:
                prediction_timestamp_list.append(prediction_timestamp)
                if self.check_x_batch_dimensions(feature_x_dict):
                    data_x_list.append(feature_x_dict)
                    data_y_list.append(feature_y_dict)
                else:
                    rejected_x_list.append(feature_x_dict)
                    rejected_y_list.append(feature_y_dict)

        n_valid_samples = len(data_x_list)

        if n_valid_samples < total_number_of_samples:
            logger.debug("{} out of {} samples were found to be valid".format(n_valid_samples, total_number_of_samples))
            if len(rejected_x_list) > 0:
                self.print_diagnostics(rejected_x_list[-1], rejected_y_list[-1])

        logger.debug("Making normalised x list")
        data_x_list = self._make_normalised_x_list(data_x_list, do_normalisation_fitting)
        prediction_timestamp = prediction_timestamp_list[-1] if len(prediction_timestamp_list) > 0 else None
        x_dict, x_symbols = self.stack_samples_for_each_feature(data_x_list)

        logger.debug("{} out of {} samples were found to be valid".format(n_valid_samples, total_number_of_samples))

        y_dict = {}
        return x_dict, data_x_list, y_dict, data_y_list, x_symbols, prediction_timestamp

    @logtime
    def create_train_data(self, raw_data_dict, historical_universes):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        raw_data_dict = self._add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)

        # this is to speed up calculation. close is the traget feature
        raw_data_dict['close'] = raw_data_dict['close'].astype('float32',copy=False)

        market_schedule = self._extract_schedule_for_training(raw_data_dict)
        fit_normalisation = True

        train_x, x_list, _, data_y_list, x_symbols, prediction_timestamp = \
            self._create_data(
                raw_data_dict,
                market_schedule,
                historical_universes,
                fit_normalisation
            )

        classify_y = self.n_classification_bins
        y_list = self._make_classified_y_list(data_y_list) if classify_y else data_y_list
        train_y, _ = self.stack_samples_for_each_feature(y_list)

        logger.debug("Assembled training dict with {} symbols".format(len(x_symbols)))

        if self.clean_nan_from_dict:
            train_x, train_y = self._remove_nans_from_dict(train_x, train_y)

        return train_x, train_y

    @logtime
    def create_predict_data(self, raw_data_dict):
        """  Create a set of features for a single prediction (x).
        These will be normalised in accordance with the properties of the training set.

        :param raw_data_dict:
        :return: tuple: predict, symbol_list, prediction_timestamp, target_timestamp
        """

        raw_data_dict = self._add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)
        market_schedule = self._extract_schedule_for_prediction(raw_data_dict)

        predict_x, x_list, predict_y, y_list, x_symbols, predict_timestamp = \
            self._create_data(
                raw_data_dict,
                market_schedule
            )

        if self.clean_nan_from_dict:
            predict_x = self._remove_nans_from_dict(predict_x)
            logger.debug("May need to update symbols when removing nans from dict")

        prediction_day = predict_timestamp.date()
        schedule = self._calendar.schedule(prediction_day, prediction_day + timedelta(days=10))

        target_timestamp = self._get_valid_target_timestamp_in_schedule(schedule, predict_timestamp)

        _, predict_timestamp = self._get_prediction_timestamps(schedule.loc[prediction_day]['market_open'])

        logger.debug("Assembled prediction dict with {} symbols".format(len(x_symbols)))

        return predict_x, x_symbols, predict_timestamp, target_timestamp

    @logtime
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

    @logtime
    def _build_features(self, collector, prediction_market_open):
        """  Constructs dictionaries holding the desired x and y feature data.

        :param raw_data_dict:
        :param historical_universes:
        :param data_schedule:
        :param prediction_market_open:
        :rtype: FeatureDict
        """
        raw_data_dict = collector.raw_data_dict
        historical_universes = collector.historical_universes
        data_schedule = collector.data_schedule

        target_market_schedule = self._extract_target_market_day(data_schedule, prediction_market_open)
        target_market_open = target_market_schedule.market_open if target_market_schedule is not None else None


        try:
            if historical_universes is not None:
                prediction_date = prediction_market_open.date()
                universe = self._get_universe_from_date(prediction_date, historical_universes)
            else:
                universe = None

            x_end_timestamp, y_start_timestamp = self._get_prediction_timestamps(prediction_market_open)
            target_timestamp = self._get_target_timestamp(target_market_open)

            if target_timestamp and y_start_timestamp > target_timestamp:
                raise ValueError('Target timestamp should be later than prediction_timestamp')

            feature_x_dict, feature_y_dict = self._collect_prediction_from_features(
                raw_data_dict, x_end_timestamp, y_start_timestamp, universe, target_timestamp)

        except DateNotInUniverseError as e:
            logger.debug(e)
            return Feature(None, None, None, target_market_open)

        except KeyError as e:
            logger.debug("Error while building features. {}. prediction_time: {}".format(e, prediction_market_open))
            return Feature(None, None, None, target_market_open)
        except Exception as e:
            logger.debug('Failed to build a set of features', exc_info=e)
            return Feature(None, None, None, target_market_open)

        return Feature(feature_x_dict, feature_y_dict, x_end_timestamp, target_timestamp)

    @logtime
    def _collect_prediction_from_features(self, raw_data_dict, x_end_timestamp, y_start_timestamp, universe=None,
                                          target_timestamp=None):
        """

        :param dict raw_data_dict:  dictionary of dataframes containing features data.
        :param Timestamp x_end_timestamp:
        :param y_start_timestamp:
        :param list universe: list of relevant symbols
        :param target_timestamp: Timestamp the prediction is for.
        :return :return (dict, dict): feature_x_dict, feature_y_dict
        """
        feature_x_dict = OrderedDict()
        feature_y_dict = OrderedDict()

        for feature in self.features:
            feature_name, feature_x, feature_y = self._process_predictions(x_end_timestamp, y_start_timestamp,
                                                                           raw_data_dict, target_timestamp,
                                                                           universe, feature)
            feature_x_dict[feature_name] = feature_x
            if feature_y is not None:
                feature_y_dict[feature_name] = feature_y

        if len(feature_y_dict) > 0:
            assert len(feature_y_dict) == 1, 'Only one target is allowed'
        else:
            feature_y_dict = OrderedDict()

        return feature_x_dict, feature_y_dict

    @logtime
    def _process_predictions(self, x_timestamp, y_timestamp, raw_data_dict, target_timestamp, universe, feature):
        """ Gathers the data associated with a single feature.

        :param x_timestamp:
        :param y_timestamp:
        :param raw_data_dict:
        :param target_timestamp:
        :param universe:
        :param feature:
        :return:
        """

        universe = raw_data_dict[feature.name].columns if universe is None else universe

        feature_name = feature.full_name if feature.full_name in raw_data_dict.keys() else feature.name
        feature_x = feature.get_prediction_features(raw_data_dict[feature_name].loc[:, universe], x_timestamp)

        feature_y = feature.get_prediction_targets(
            raw_data_dict[self._get_feature_for_extract_y()].loc[:, universe],
            y_timestamp,
            target_timestamp
        )

        if feature_y is not None:
            transposed_y = feature_y.to_frame().transpose()
            transposed_y.set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
            feature_y = transposed_y

        return feature.full_name, feature_x, feature_y

    @staticmethod
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

    @staticmethod
    def _remove_nans_from_dict(x_dict, y_dict=None):
        """
        looks for any of the examples in the dictionaries that have NaN and removes all those

        :param x_dict: x_dict with features
        :param y_dict: y_dict with targets
        :return: X_dict and y_dict
        """

        assert len(x_dict) > 0

        x_dict_values = x_dict.values()
        n_examples = x_dict_values[0].shape[0]

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

    @staticmethod
    def _add_log_returns(data_dict):
        """ If not already in dictionary, add raw log returns

        :param data_dict: Original data dict
        :return: Updated dict
        """

        base_key = 'close' if 'close' in data_dict else list(data_dict.keys())[0]
        close_data = data_dict[base_key]
        data_dict['log-return'] = np.log(close_data.pct_change() + 1, dtype=np.float32).replace([np.inf, -np.inf],
                                                                                                np.nan)

        return data_dict
