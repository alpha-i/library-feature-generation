import logging
import multiprocessing

from collections import OrderedDict
from datetime import timedelta
from functools import partial

import numpy as np
import pandas as pd
import alphai_calendars as mcal

from alphai_feature_generation.feature.factory import FinancialFeatureFactory, FeatureList
from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.helpers import logtime
from alphai_feature_generation.transformation.base import (
    ensure_closing_pool,
    DataTransformation,
    get_unique_symbols,
    DateNotInUniverseError)

logger = logging.getLogger(__name__)


class FinancialDataTransformation(DataTransformation):

    KEY_EXCHANGE = 'exchange_name'

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

        self.clean_nan_from_dict = configuration.get('clean_nan_from_dict', False)

    def _init_calendar(self, configuration):
        self._calendar = mcal.get_calendar(configuration[self.KEY_EXCHANGE])
        self.minutes_in_trading_days = self._calendar.get_minutes_in_one_day()

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
        return 'close'

    def get_calendar_name(self):
        return self.KEY_EXCHANGE

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
            fit_function = partial(self._build_features_function, managed_dict, historical_universes, data_schedule)
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

    def create_train_data(self, raw_data_dict, historical_universes):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        raw_data_dict = self._add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)

        raw_data_dict['close'] = raw_data_dict['close'].astype('float32', copy=False)

        market_schedule = self._extract_schedule_for_training(raw_data_dict)
        normalise = True

        train_x, train_y, _, _ = self._create_data(raw_data_dict, market_schedule, historical_universes, normalise)

        if self.clean_nan_from_dict:
            train_x, train_y = self._remove_nans_from_dict(train_x, train_y)

        return train_x, train_y

    def create_predict_data(self, raw_data_dict):
        """

        :param raw_data_dict:
        :return: tuple: predict, symbol_list, prediction_timestamp, target_timestamp
        """

        raw_data_dict = self._add_log_returns(raw_data_dict)
        raw_data_dict = self.filter_unwanted_keys(raw_data_dict)
        market_schedule = self._extract_schedule_for_prediction(raw_data_dict)

        predict_x, _, symbols, predict_timestamp = self._create_data(raw_data_dict, market_schedule)

        if self.clean_nan_from_dict:
            predict_x = self._remove_nans_from_dict(predict_x)
            logger.debug("May need to update symbols when removing nans from dict")

        prediction_day = predict_timestamp.date()
        schedule = self._calendar.schedule(prediction_day, prediction_day + timedelta(days=10))

        target_timestamp = self._get_valid_target_timestamp_in_schedule(schedule, predict_timestamp)

        _, predict_timestamp = self._get_prediction_timestamps(schedule.loc[prediction_day]['market_open'])

        return predict_x, symbols, predict_timestamp, target_timestamp

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

        # FIXME This parallelisation was creating a crash in the backtests. So switching off for now.
        part = partial(self._process_predictions, x_end_timestamp, y_start_timestamp,
                       raw_data_dict, target_timestamp, universe)
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

    def _process_predictions(self, x_timestamp, y_timestamp, raw_data_dict, target_timestamp, universe, feature):
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
                raw_data_dict[self._get_feature_for_extract_y()].loc[:, universe],
                y_timestamp,
                target_timestamp
            )

            if feature_y is not None:
                transposed_y = feature_y.to_frame().transpose()
                transposed_y.set_index(pd.DatetimeIndex([target_timestamp]), inplace=True)
                feature_y = transposed_y

        return feature.full_name, feature_x, feature_y

    def _build_features_function(self, raw_data_dict, historical_universes, data_schedule, prediction_market_open):
        target_market_schedule = self._extract_target_market_day(data_schedule, prediction_market_open)
        target_market_open = target_market_schedule.market_open if target_market_schedule is not None else None

        try:
            feature_x_dict, feature_y_dict, prediction_timestamp = self._build_features(raw_data_dict,
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

    def _build_features(self, raw_data_dict, universe, target_market_open, prediction_market_open, ):
        """ Creates dictionaries of features and labels for a single window

        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe universe: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param prediction_market_open:
        :param target_market_open:
        :return:
        """

        if universe is not None:
            prediction_date = prediction_market_open.date()
            universe = self._get_universe_from_date(prediction_date, universe)

        x_end_timestamp, y_start_timestamp = self._get_prediction_timestamps(prediction_market_open)
        target_timestamp = self._get_target_timestamp(target_market_open)

        if target_timestamp and y_start_timestamp > target_timestamp:
            raise ValueError('Target timestamp should be later than prediction_timestamp')

        feature_x_dict, feature_y_dict = self._collect_prediction_from_features(
            raw_data_dict, x_end_timestamp, y_start_timestamp, universe, target_timestamp)

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

        if self._calendar.open_at_time(schedule, target_timestamp, include_close=True):
            return target_timestamp
        else:
            raise ValueError("Target timestamp {} not in market time".format(target_timestamp))

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



