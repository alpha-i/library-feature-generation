import logging
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from alphai_feature_generation import (FINANCIAL_FEATURE_NORMALIZATIONS,
                                       MARKET_DAYS_SEARCH_MULTIPLIER, MIN_MARKET_DAYS_SEARCH)
from alphai_feature_generation.classifier import BinDistribution
from alphai_feature_generation.feature.resampling import ResamplingStrategy
from alphai_feature_generation.feature.transform import Transformation

logger = logging.getLogger(__name__)


class GymFeature(object):
    """ Describes a feature intended to help predict attendance levels at a gym. """

    KEY_EXCHANGE = 'holiday_calendar'
    GENERIC_SYMBOL = 'SYM'

    def __init__(self, name, transformation, normalization, nbins, length, ndays, resample_minutes, start_market_minute,
                 is_target, exchange_calendar, local, classify_per_series=False, normalise_per_series=False):
        """
        Object containing all the information to manipulate the data relative to a financial feature.
        :param str name: Name of the feature
        :param dict transformation: contains name and parameters to use for processing, name must be in
            FINANCIAL_FEATURE_TRANSFORMATIONS
        :param str/None normalization: type of normalization. Can be None.
        :param int/None nbins: number of bins to be used for target classification. Can be None.
        :param int length: expected number of elements in the feature
        :param int ndays: number of trading days worth of data the feature should use.
        :param int resample_minutes: resampling frequency in number of minutes.
        :param int start_market_minute: number of minutes after market open the data collection should start from.
        :param bool is_target: if True the feature is a target.
        :param pandas_market_calendar exchange_calendar: exchange calendar.
        """
        # FIXME the get_default_flags args are temporary. We need to load a get_default_flags config in the unit tests.

        self.name = name
        self.transformation = Transformation(transformation)
        self.normalization = normalization
        self.nbins = nbins
        self.ndays = ndays
        self.resample_minutes = resample_minutes
        self.start_market_minute = start_market_minute
        self.is_target = is_target
        self.calendar = exchange_calendar
        self.minutes_in_trading_day = self.calendar.get_minutes_in_one_day()
        self.n_series = None
        self.local = local
        self.length = length

        self.bin_distribution = None

        self._assert_input(name, normalization, nbins, length, ndays, resample_minutes,
                           start_market_minute, is_target, local)

        if self.nbins:
            self.bin_distribution_dict = {}
        else:
            self.bin_distribution_dict = None

        self.classify_per_series = classify_per_series
        self.normalise_per_series = normalise_per_series

        if self.normalization:
            self.scaler_dict = {}
            if self.normalization == 'robust':
                self.scaler = RobustScaler()
            elif self.normalization == 'min_max':
                self.scaler = MinMaxScaler()
            elif self.normalization == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization == 'gaussian':
                self.scaler = QuantileTransformer(output_distribution='normal')
            else:
                raise ValueError('Requested normalisation not supported: {}'.format(self.normalization))
        else:
            self.scaler = None
            self.scaler_dict = None

    @property
    def full_name(self):
        full_name = '{}_{}'.format(self.name, self.transformation.name)
        if self.resample_minutes > 0:
            resolution = '_' + str(self.resample_minutes) + 'T'
            full_name = full_name + resolution

        return full_name

    def _assert_input(self, name, normalization, nbins, length, ndays, resample_minutes,
                      start_market_minute, is_target, local):
        """ Make sure the inputs are sensible. """
        assert isinstance(name, str)
        assert normalization in FINANCIAL_FEATURE_NORMALIZATIONS
        assert (isinstance(nbins, int) and nbins > 0) or nbins is None
        assert isinstance(ndays, int) and ndays >= 0
        assert isinstance(resample_minutes, int) and resample_minutes >= 0
        assert isinstance(start_market_minute, int)
        assert start_market_minute < self.minutes_in_trading_day
        assert (isinstance(length, int) and length > 0)
        assert isinstance(is_target, bool)
        assert isinstance(local, bool)

    def process_prediction_data_x(self, prediction_data_x):
        """
        Apply feature-specific transformations to input prediction_data_x
        :param pd.Dataframe prediction_data_x: X data for model prediction task
        :return pd.Dataframe: processed_prediction_data_x
        """

        assert isinstance(prediction_data_x, pd.DataFrame)

        resampled_data = ResamplingStrategy.resample(self, deepcopy(prediction_data_x))

        return self.transformation.transform_x(self, resampled_data)

    def fit_normalisation(self, symbol_data, symbol=None):
        """ Creates a scikitlearn scalar, assigns it to a dictionary, fits it to the data

        :param symbol:
        :param symbol_data:
        :return:
        """

        symbol_data.flatten()
        symbol_data = symbol_data[np.isfinite(symbol_data)]
        symbol_data = symbol_data.reshape(-1, 1)  # Reshape for scikitlearn

        if len(symbol_data) > 0:
            if symbol:
                self.scaler_dict[symbol] = deepcopy(self.scaler)
                self.scaler_dict[symbol].fit(symbol_data)
            else:
                self.scaler.fit(symbol_data)

    def apply_normalisation(self, dataframe):
        """ Compute normalisation across the entire training set, or apply predetermined normalistion to prediction.

        :param dataframe: Features of shape [n_samples, n_series, n_features]
        :type dataframe: pd.DataFrame
        :return:
        """

        for symbol in dataframe:
            data_x = dataframe[symbol].values
            original_shape = data_x.shape
            data_x = data_x.reshape(-1, 1)

            nan_mask = np.ma.fix_invalid(data_x, fill_value=0)

            if self.normalise_per_series:
                if symbol in self.scaler_dict:
                    data_x = self.scaler_dict[symbol].transform(nan_mask.data)
                    # Put the nans back in so we know to avoid them
                    data_x[nan_mask.mask] = np.nan
                    dataframe[symbol] = data_x.reshape(original_shape)
                else:
                    logger.debug("Symbol lacks normalisation scaler: {}".format(symbol))
                    logger.debug("Dropping symbol from dataframe: {}".format(symbol))
                    dataframe.drop(symbol, axis=1, inplace=True)
            else:
                data_x = self.scaler.transform(nan_mask.data)
                # Put the nans back in so we know to avoid them
                data_x[nan_mask.mask] = np.nan
                dataframe[symbol] = data_x.reshape(original_shape)

        return dataframe

    def reshape_for_scikit(self, data_x):
        """ Scikit expects an input of the form [samples, features]; normalisation applied separately to each feature.

        :param data_x: Features of shape [n_samples, n_series, n_features]
        :return: nparray Same data as input, but now with two dimensions: [samples, f], each f has own normalisation
        """

        if self.normalise_per_series:
            n_series = data_x.shape[1]
            scikit_shape = (-1, n_series)
        else:
            scikit_shape = (-1, 1)

        return data_x.reshape(scikit_shape)

    def process_prediction_data_y(self, prediction_data_y, prediction_reference_data):
        """
        Apply feature-specific transformations to input prediction_data_y
        :param pd.Series prediction_data_y: y data for model prediction task
        :param pd.Series prediction_reference_data: reference data-point to calculate differential metrics
        :return pd.Series: processed_prediction_data_y
        """
        assert self.is_target
        assert isinstance(prediction_data_y, pd.Series)

        return self.transformation.transform_y(self, prediction_data_y, prediction_reference_data)

    def _get_safe_schedule_start_date(self, prediction_timestamp):
        """
        Calculate a safe schedule start date from input timestamp so that at least self.ndays trading days are available
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: schedule_start_date
        """
        safe_ndays = max(MIN_MARKET_DAYS_SEARCH, MARKET_DAYS_SEARCH_MULTIPLIER * self.ndays)
        return prediction_timestamp - timedelta(days=safe_ndays)

    def declassify_single_predict_y(self, predict_y):
        raise NotImplementedError('Declassification is only available for multi-pass prediction at the moment.')

    def _get_start_timestamp_x(self, prediction_timestamp):
        """
        Calculate the start timestamp of x-data for a given prediction timestamp.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: start timestamp of x-data
        """
        schedule_start_date = str(self._get_safe_schedule_start_date(prediction_timestamp))
        schedule_end_date = str(prediction_timestamp.date())
        market_open_list = self.calendar.schedule(schedule_start_date, schedule_end_date).market_open
        prediction_market_open = market_open_list[prediction_timestamp.date()]
        prediction_market_open_idx = np.argwhere(market_open_list == prediction_market_open).flatten()[0]
        start_timestamp_x = market_open_list[prediction_market_open_idx - self.ndays] + timedelta(
            minutes=self.start_market_minute)
        return start_timestamp_x

    def _index_selection_x(self, date_time_index, prediction_timestamp):
        """
        Create index selection rule for x data
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return: index selection rule
        """
        start_timestamp_x = self._get_start_timestamp_x(prediction_timestamp)
        return (date_time_index >= start_timestamp_x) & (date_time_index <= prediction_timestamp)

    def _select_prediction_data_x(self, data_frame, prediction_timestamp):
        """
        Select the x-data relevant for a input prediction timestamp.
        :param pd.Dataframe data_frame: raw x-data (unselected, unprocessed)
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return pd.Dataframe: selected x-data (unprocessed)
        """

        try:
            n_rows = len(data_frame.index)
            end_point = data_frame.index.get_loc(prediction_timestamp, method='pad')
            end_index = end_point + 1  # +1 because iloc is not inclusive of end index
            start_index = end_point - self.length + 1

            # Check if we're violating range of dataframe
            if end_index >= n_rows:
                offset = end_index - n_rows + 1
                start_index -= offset
                end_index -= offset
        except:
            logger.debug('Prediction timestamp {} not within range of dataframe'.format(prediction_timestamp))
            start_index = 0
            end_index = -1

        return data_frame.iloc[start_index:end_index, :]

    def _select_prediction_data_y(self, data_frame, target_timestamp, n_forecasts):
        """
        Select the y-data for a prediction timestamp.
        :param pd.Dataframe data_frame: raw data (unselected, unprocessed)
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return pd.Dataframe: selected y-data (unprocessed)
        """

        try:
            n_rows = len(data_frame.index)
            end_point = data_frame.index.get_loc(target_timestamp, method='pad')
            start_index = end_point + 1  # +1 because iloc is not inclusive of end index, other+1 due to forecast
            end_index = start_index + n_forecasts

            # Check if we're violating range of dataframe
            if end_index >= n_rows:
                offset = end_index - n_rows + 1
                start_index -= offset
                end_index -= offset
        except:
            logger.debug('Target timestamp {} not within range of dataframe'.format(target_timestamp))
            start_index = 0
            end_index = -1

        return data_frame.iloc[start_index:end_index, :]

    def get_prediction_targets(self, data_frame, prediction_timestamp, target_timestamp=None, n_forecasts=1):
        """
        Compute targets from dataframe only if the current feature is target

        :param data_frame: Time indexed data
        :type data_frame: pd.DataFrame
        :param prediction_timestamp: the time of prediction
        :type prediction_timestamp: pd.Timestamp
        :param target_timestamp: the time predicted
        :type target_timestamp: pd.Timestamp
        :rtype pd.DataFrame
        """
        prediction_target = None

        if self.is_target and target_timestamp:
            prediction_target = self._select_prediction_data_y(data_frame, prediction_timestamp, n_forecasts)

        return prediction_target

    def get_prediction_features(self, data_frame, prediction_timestamp):
        """
        Compute features from dataframe

        :param data_frame: Time indexed data
        :type data_frame: pd.DataFrame
        :param prediction_timestamp: the time of prediction
        :type prediction_timestamp: pd.Timestamp
        :rtype: pd.DataFrame
        """
        prediction_features = self._select_prediction_data_x(data_frame, prediction_timestamp)

        if self.local:
            prediction_features = self.process_prediction_data_x(prediction_features)

        return prediction_features

    def fit_classification(self, symbol, symbol_data):
        """  Fill dict with classifiers

        :param symbol:
        :rtype symbol: str
        :param symbol_data:
        :return:
        """

        if self.nbins is None:
            return

        self.bin_distribution_dict[symbol] = BinDistribution(symbol_data, self.nbins)

    def apply_classification(self, dataframe):
        """ Apply predetermined classification to y data.

        :param pd panel data_x: Features of shape [n_samples, n_series, n_features]
        :return:
        """

        n_timesteps = len(dataframe.index)
        hot_panel = pd.Panel(0, items=dataframe.columns, major_axis=np.arange(n_timesteps), minor_axis=np.arange(self.nbins))

        for symbol in dataframe:
            data_y = dataframe[symbol].values

            key = symbol if self.classify_per_series else self.GENERIC_SYMBOL

            if key in self.bin_distribution_dict:
                symbol_distribution = self.bin_distribution_dict[key]
                one_hot_labels = symbol_distribution.classify_labels(data_y)
                if one_hot_labels.shape[-1] > 1:
                    hot_panel[symbol] = one_hot_labels
            else:
                logger.debug("Symbol lacks clasification bins: {}".format(symbol))
                hot_panel.drop(symbol, axis=0, inplace=True)
                logger.debug("Dropping {} from dataframe.".format(symbol))

        return hot_panel.transpose(1, 0, 2)  # Puts time back as the index

    def inverse_transform_multi_predict_y(self, predict_y, symbols, confidence_interval=0.68):
        """
        Inverse-transform multi-pass predict_y data
        :param pd.Dataframe predict_y: target multi-pass prediction
        :return pd.Dataframe: inversely transformed mean and variance of target multi-pass prediction
        """
        assert self.is_target

        n_symbols = len(symbols)
        n_forecasts = predict_y.shape[2]

        data_shape = (n_forecasts, n_symbols)
        means = np.zeros(shape=data_shape, dtype=np.float32)
        lower_bound = np.zeros(shape=data_shape, dtype=np.float32)
        upper_bound = np.zeros(shape=data_shape, dtype=np.float32)
        assert predict_y.shape[1] == n_symbols, "Weird shape - predict y not equal to n symbols"

        for i, symbol in enumerate(symbols):
            key = symbol if self.classify_per_series else self.GENERIC_SYMBOL

            for j in range(n_forecasts):
                pdf = predict_y[:, i, j, :]
                if key in self.bin_distribution_dict and not np.any(np.isnan(pdf)):
                    symbol_bins = self.bin_distribution_dict[key]
                    try:
                        means[j, i], lower_bound[j, i], upper_bound[j, i] = \
                            symbol_bins.estimate_confidence_interval(pdf, confidence_interval)
                    except Exception as e:
                        logging.debug(e)
                        raise e
                else:
                    logger.debug("Nans or no bin distribution found for symbol: {}".format(symbol))
                    means[j, i] = np.nan
                    lower_bound[j, i] = np.nan
                    upper_bound[j, i] = np.nan

        return means, lower_bound, upper_bound

    def __repr__(self):
        return '<{} object: name: {}. full_name: {}>'.format(self.__class__.__name__, self.name, self.full_name)

