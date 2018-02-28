import logging
import multiprocessing

from collections import OrderedDict
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import alphai_calendars as mcal

from alphai_feature_generation.feature.features.gym import GymFeature
from alphai_feature_generation.feature.factory import FeatureList, GymFeatureFactory
from alphai_feature_generation.helpers import logtime
from alphai_feature_generation.transformation.base import (
    ensure_closing_pool,
    DataTransformation,
    get_unique_symbols,
    DateNotInUniverseError)

logger = logging.getLogger(__name__)


class GymDataTransformation(DataTransformation):

    KEY_EXCHANGE = 'holiday_calendar'

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
        super().__init__(configuration)

        self.target_feature = self.get_target_feature()

    def _init_calendar(self, configuration):
        try:
            self._calendar = mcal.get_calendar(configuration[self.KEY_EXCHANGE])
            self.minutes_in_trading_days = self._calendar.get_minutes_in_one_day()
        except:
            self._calendar = None
            self.minutes_in_trading_days = 1440

    def _assert_input(self):
        configuration = self.configuration

        assert isinstance(configuration[self.KEY_EXCHANGE], str)
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

    def _get_feature_for_extract_y(self):
        return self.target_feature.name

    def get_calendar_name(self):
        return self.KEY_EXCHANGE

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
            self.KEY_EXCHANGE: self._calendar.name,
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

        factory = GymFeatureFactory(self._calendar)

        return factory.create_from_list(feature_config_list)

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
            fit_function = partial(self._build_features_function, managed_dict, data_schedule)
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

    def _collect_prediction_from_features(self, raw_data_dict, x_end_timestamp, y_start_timestamp, universe=None,
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

        part = partial(self._process_predictions, x_end_timestamp, y_start_timestamp,
                       raw_data_dict, target_timestamp)
        processed_predictions = map(part, self.features)

        for prediction in processed_predictions:
            feature_name, feature_x, feature_y = prediction
            feature_x_dict[feature_name] = feature_x
            if feature_y is not None:
                feature_y_dict[feature_name] = feature_y

        if len(feature_y_dict) > 0:
            assert len(feature_y_dict) == 1, 'Only one target is allowed'
        else:
            feature_y_dict = None

        return feature_x_dict, feature_y_dict

    def _process_predictions(self, x_timestamp, y_timestamp, raw_data_dict, target_timestamp, feature):

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
                raw_data_dict[self._get_feature_for_extract_y()].loc[:, universe],
                y_timestamp,
                target_timestamp
            )

            #FIXME unclear why this transpose is necessary
            if feature_y is not None:
                transposed_y = feature_y.to_frame().transpose()
                transposed_y.set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
                feature_y = transposed_y

        return feature.full_name, feature_x, feature_y

    def _build_features_function(self, raw_data_dict, data_schedule, prediction_market_open):
        target_market_schedule = self._extract_target_market_day(data_schedule, prediction_market_open)
        target_market_open = target_market_schedule.market_open if target_market_schedule is not None else None

        try:
            feature_x_dict, feature_y_dict, prediction_timestamp = self._build_features(raw_data_dict,
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

    def _build_features(self, raw_data_dict, target_market_open, prediction_market_open, ):
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

        feature_x_dict, feature_y_dict = self._collect_prediction_from_features(
            raw_data_dict, x_end_timestamp, y_start_timestamp, target_timestamp)

        return feature_x_dict, feature_y_dict, x_end_timestamp

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
