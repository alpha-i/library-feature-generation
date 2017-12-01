from copy import deepcopy
from datetime import timedelta
import logging

from pyts.transformation import GASF, GADF, MTF
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer

from alphai_feature_generation import FINANCIAL_FEATURE_TRANSFORMATIONS, FINANCIAL_FEATURE_NORMALIZATIONS, \
    MINUTES_IN_TRADING_DAY, MARKET_DAYS_SEARCH_MULTIPLIER, MIN_MARKET_DAYS_SEARCH

from alphai_feature_generation.classifier import BinDistribution, classify_labels, declassify_labels

logging.getLogger(__name__).addHandler(logging.NullHandler())


class FinancialFeature(object):
    def __init__(self, name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                 is_target, exchange_calendar, local, classify_per_series=False, normalise_per_series=False):
        """
        Object containing all the information to manipulate the data relative to a financial feature.
        :param str name: Name of the feature
        :param dict transformation: contains name and parameters to use for processing, name must be in
            FINANCIAL_FEATURE_TRANSFORMATIONS
        :param str/None normalization: type of normalization. Can be None.
        :param int/None nbins: number of bins to be used for target classification. Can be None.
        :param int ndays: number of trading days worth of data the feature should use.
        :param int resample_minutes: resampling frequency in number of minutes.
        :param int start_market_minute: number of minutes after market open the data collection should start from.
        :param bool is_target: if True the feature is a target.
        :param pandas_market_calendar exchange_calendar: exchange calendar.
        """
        # FIXME the get_default_flags args are temporary. We need to load a get_default_flags config in the unit tests.

        self._assert_input(name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                           is_target, local)
        self.name = name
        self.transformation = transformation
        self.normalization = normalization
        self.nbins = nbins
        self.ndays = ndays
        self.resample_minutes = resample_minutes
        self.start_market_minute = start_market_minute
        self.is_target = is_target
        self.exchange_calendar = exchange_calendar
        self.n_series = None
        self.local = local

        self.bin_distribution = None
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
                raise NotImplementedError('Requested normalisation not supported: {}'.format(self.normalization))
        else:
            self.scaler = None
            self.scaler_dict = None

    @property
    def full_name(self):
        return '{}_{}'.format(self.name, self.transformation['name'])

    @staticmethod
    def _assert_input(name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                      is_target, local):
        assert isinstance(name, str)
        assert isinstance(transformation, dict)
        assert 'name' in transformation, 'The transformation dict does not contain the key "name"'
        assert transformation['name'] in FINANCIAL_FEATURE_TRANSFORMATIONS
        assert normalization in FINANCIAL_FEATURE_NORMALIZATIONS
        assert (isinstance(nbins, int) and nbins > 0) or nbins is None
        assert isinstance(ndays, int) and ndays >= 0
        assert isinstance(resample_minutes, int) and resample_minutes >= 0
        assert isinstance(start_market_minute, int)
        assert start_market_minute < MINUTES_IN_TRADING_DAY
        assert isinstance(is_target, bool)
        assert isinstance(local, bool)
        if transformation['name'] == 'ewma':
            assert 'halflife' in transformation
        if transformation['name'] == 'KER':
            assert 'lag' in transformation
        if transformation['name'] == 'GASF':
            assert 'image_size' in transformation
        if transformation['name'] == 'GADF':
            assert 'image_size' in transformation
        if transformation['name'] == 'MTF':
            assert 'image_size' in transformation

    def process_prediction_data_x(self, prediction_data_x):
        """
        Apply feature-specific transformations to input prediction_data_x
        :param pd.Dataframe prediction_data_x: X data for model prediction task
        :return pd.Dataframe: processed_prediction_data_x
        """
        if not self.local:
            assert isinstance(prediction_data_x, dict)
            processed_prediction_data_x = deepcopy(prediction_data_x[self.name])
        else:
            assert isinstance(prediction_data_x, pd.DataFrame)
            processed_prediction_data_x = deepcopy(prediction_data_x)

        if self.transformation['name'] == 'log-return':
            processed_prediction_data_x = np.log(processed_prediction_data_x.pct_change() + 1). \
                replace([np.inf, -np.inf], np.nan)

            # Remove the zeros / nans associated with log return
            processed_prediction_data_x = processed_prediction_data_x.iloc[1:]

        if self.transformation['name'] == 'stochastic_k':
            columns = processed_prediction_data_x.columns
            processed_prediction_data_x \
                = ((processed_prediction_data_x.iloc[-1] - processed_prediction_data_x.min()) /
                   (processed_prediction_data_x.max() - processed_prediction_data_x.min())) * 100.

            processed_prediction_data_x = np.expand_dims(processed_prediction_data_x, axis=0)
            processed_prediction_data_x = pd.DataFrame(processed_prediction_data_x, columns=columns)

        if self.transformation['name'] == 'ewma':
            processed_prediction_data_x = \
                processed_prediction_data_x.ewm(halflife=self.transformation['halflife']).mean()

        if self.transformation['name'] == 'ker':
            direction = processed_prediction_data_x.diff(self.transformation['lag']).abs()
            volatility = processed_prediction_data_x.diff().abs().rolling(window=self.transformation['lag']).sum()

            direction.dropna(axis=0, inplace=True)
            volatility.dropna(axis=0, inplace=True)

            assert direction.shape == volatility.shape, ' direction and volatility need same shape in KER'

            processed_prediction_data_x = direction / volatility
            processed_prediction_data_x.dropna(axis=0, inplace=True)

        if self.transformation['name'] == 'gasf':
            columns = processed_prediction_data_x.columns
            gasf = GASF(image_size=self.transformation['image_size'], overlapping=False, scale='-1')
            processed_prediction_data_x = gasf.transform(processed_prediction_data_x.values.T)
            processed_prediction_data_x = processed_prediction_data_x.reshape(processed_prediction_data_x.shape[0], -1)
            processed_prediction_data_x = pd.DataFrame(processed_prediction_data_x.T, columns=columns)

        if self.transformation['name'] == 'gadf':
            columns = processed_prediction_data_x.columns
            gadf = GADF(image_size=self.transformation['image_size'], overlapping=False, scale='-1')
            processed_prediction_data_x = gadf.transform(processed_prediction_data_x.values.T)
            processed_prediction_data_x = processed_prediction_data_x.reshape(processed_prediction_data_x.shape[0], -1)
            processed_prediction_data_x = pd.DataFrame(processed_prediction_data_x.T, columns=columns)

        if self.transformation['name'] == 'mtf':
            columns = processed_prediction_data_x.columns
            mtf = MTF(image_size=self.transformation['image_size'], n_bins=7, quantiles='empirical', overlapping=False)
            processed_prediction_data_x = mtf.transform(processed_prediction_data_x.values.T)
            processed_prediction_data_x = processed_prediction_data_x.reshape(processed_prediction_data_x.shape[0], -1)
            processed_prediction_data_x = pd.DataFrame(processed_prediction_data_x.T, columns=columns)

        return processed_prediction_data_x

    def fit_normalisation(self, symbol_data, symbol=None):
        """ Creates a scikitlearn scalar, assigns it to a dictionary, fits it to the data

        :param symbol:
        :param symbol_data:
        :return:
        """

        if symbol:
            self.scaler_dict[symbol] = deepcopy(self.scaler)
            symbol_data = symbol_data.reshape(-1, 1)  # Reshape for scikitlearn
            self.scaler_dict[symbol].fit(symbol_data)
        else:
            symbol_data = symbol_data.reshape(-1, 1)  # Reshape for scikitlearn
            self.scaler.fit(symbol_data)

    def apply_normalisation(self, dataframe):
        """ Compute normalisation across the entire training set, or apply predetermined normalistion to prediction.

        :param pd dataframe data_x: Features of shape [n_samples, n_series, n_features]
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
                    logging.warning("Symbol lacks normalisation scaler: {}".format(symbol))
                    logging.warning("Dropping symbol from dataframe: {}".format(symbol))
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
        processed_prediction_data_y = deepcopy(prediction_data_y)

        if self.transformation['name'] == 'log-return':
            processed_prediction_data_y = np.log(prediction_data_y / prediction_reference_data). \
                replace([np.inf, -np.inf], np.nan)

        if self.scaler:
            if self.nbins is None:
                raise NotImplementedError('y scaling is not required for classifiers, but is required for regression')

        return processed_prediction_data_y

    def _get_safe_schedule_start_date(self, prediction_timestamp):
        """
        Calculate a safe schedule start date from input timestamp so that at least self.ndays trading days are available
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: schedule_start_date
        """
        safe_ndays = max(MIN_MARKET_DAYS_SEARCH, MARKET_DAYS_SEARCH_MULTIPLIER * self.ndays)
        return prediction_timestamp - timedelta(days=safe_ndays)

    def _get_start_timestamp_x(self, prediction_timestamp):
        """
        Calculate the start timestamp of x-data for a given prediction timestamp.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: start timestamp of x-data
        """
        schedule_start_date = str(self._get_safe_schedule_start_date(prediction_timestamp))
        schedule_end_date = str(prediction_timestamp.date())
        market_open_list = self.exchange_calendar.schedule(schedule_start_date, schedule_end_date).market_open
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
        prediction_index_selection_x = self._index_selection_x(data_frame.index, prediction_timestamp)
        return data_frame[prediction_index_selection_x]

    def get_prediction_data(self, data_frame, prediction_timestamp, target_timestamp=None):
        """
        Calculate x and y data for prediction. y-data will be None if target_timestamp is None.
        :param pd.Dataframe data_frame: raw data (unselected, unprocessed).
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param Timestamp target_timestamp: Timestamp the prediction is for.
        :return (pd.Dataframe, pd.Dataframe): prediction_data_x, prediction_data_y (selected and processed)
        """
        prediction_data_x = self._select_prediction_data_x(data_frame, prediction_timestamp)

        if self.local:
            prediction_data_x = self.process_prediction_data_x(prediction_data_x)

        prediction_data_y = None
        if self.is_target and target_timestamp is not None:
            prediction_data_y = self.process_prediction_data_y(
                data_frame.loc[target_timestamp],
                data_frame.loc[prediction_timestamp],
            )

        return prediction_data_x, prediction_data_y

    def fit_classification(self, symbol, symbol_data):
        """  Fill dict with classifiers

        :param symbol_data:
        :return:
        """

        if self.nbins is None:
            return

        self.bin_distribution_dict[symbol] = BinDistribution(symbol_data, self.nbins)

    def apply_classification(self, dataframe):
        """ Apply predetermined classification to y data.

        :param pd dataframe data_x: Features of shape [n_samples, n_series, n_features]
        :return:
        """

        hot_dataframe = pd.DataFrame(0, index=np.arange(self.nbins), columns=dataframe.columns)

        for symbol in dataframe:
            data_y = dataframe[symbol].values

            if symbol in self.bin_distribution_dict:
                symbol_binning = self.bin_distribution_dict[symbol]
                hot_dataframe[symbol] = np.squeeze(classify_labels(symbol_binning.bin_edges, data_y))
            else:
                logging.warning("Symbol lacks clasification bins: {}".format(symbol))
                dataframe.drop(symbol, axis=1, inplace=True)
                logging.warning("Dropping {} from dataframe.".format(symbol))

        return hot_dataframe

    def declassify_single_predict_y(self, predict_y):
        raise NotImplementedError('Declassification is only available for multi-pass prediction at the moment.')

    def declassify_multi_predict_y(self, predict_y):
        """
        Declassify multi-pass predict_y data
        :param predict_y: target multi-pass prediction with axes (passes, series, bins)
        :return: mean and variance of target multi-pass prediction
        """
        n_series = predict_y.shape[1]

        if self.nbins:
            means = np.zeros(shape=(n_series,))
            variances = np.zeros(shape=(n_series,))
            for series_idx in range(n_series):
                if self.classify_per_series:
                    series_bins = self.bin_distribution[series_idx]
                else:
                    series_bins = self.bin_distribution

                means[series_idx], variances[series_idx] = \
                    declassify_labels(series_bins, predict_y[:, series_idx, :])
        else:
            means = np.mean(predict_y, axis=0)
            variances = np.var(predict_y, axis=0)

        return means, variances

    def inverse_transform_multi_predict_y(self, predict_y, symbols):
        """
        Inverse-transform multi-pass predict_y data
        :param pd.Dataframe predict_y: target multi-pass prediction
        :return pd.Dataframe: inversely transformed mean and variance of target multi-pass prediction
        """
        assert self.is_target

        n_symbols = len(symbols)
        print("new symbols:", n_symbols)
        means = np.zeros(shape=(n_symbols,))
        variances = np.zeros(shape=(n_symbols,))
        assert predict_y.shape[1] == n_symbols, "Weird shape - predict y not equal to n symbols"

        for i, symbol in enumerate(symbols):
            if symbol in self.bin_distribution_dict:
                symbol_bins = self.bin_distribution_dict[symbol]
                means[i], variances[i] = declassify_labels(symbol_bins, predict_y[:, i, :])
            else:
                logging.warning("No bin distribution found for symbol: {}".format(symbol))
                means[i] = np.nan
                variances[i] = np.nan

        variances[variances == 0] = 1.0  # FIXME Hack

        diag_cov_matrix = np.diag(variances)
        return means, diag_cov_matrix


def single_financial_feature_factory(feature_config):
    """
    Build target financial feature from dictionary.
    :param dict feature_config: dictionary containing feature details.
    :return FinancialFeature: FinancialFeature object
    """
    assert isinstance(feature_config, dict)

    return FinancialFeature(
        feature_config['name'],
        feature_config['transformation'],
        feature_config['normalization'],
        feature_config['nbins'],
        feature_config['ndays'],
        feature_config['resample_minutes'],
        feature_config['start_market_minute'],
        feature_config['is_target'],
        mcal.get_calendar(feature_config['exchange_name']),
        feature_config['local'])


def financial_features_factory(feature_config_list):
    """
    Build list of financial features from list of complete feature-config dictionaries.
    :param list feature_config_list: list of dictionaries containing feature details.
    :return list: list of FinancialFeature objects
    """
    assert isinstance(feature_config_list, list)

    feature_list = []
    for single_feature_dict in feature_config_list:
        feature_list.append(single_financial_feature_factory(single_feature_dict))

    return feature_list


def get_feature_names(feature_list):
    """
    Return unique names of feature list
    :param list feature_list: list of Feature objects
    :return list: list of strings
    """
    return list(set([feature.name for feature in feature_list]))


def get_feature_max_ndays(feature_list):
    """
    Return max ndays of feature list
    :param list feature_list: list of Feature objects
    :return int: max ndays of feature list
    """
    return max([feature.ndays for feature in feature_list])
