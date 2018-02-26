from copy import deepcopy
from datetime import timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from tests.helpers import TEST_ARRAY
from alphai_feature_generation.feature.features.financial import FinancialFeature

from tests.feature.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_hourly_ohlcv_data_dict

SAMPLE_TRAIN_LABELS = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))
SAMPLE_PREDICT_LABELS = SAMPLE_TRAIN_LABELS[:, int(0.5 * SAMPLE_TRAIN_LABELS.shape[1])]

SAMPLE_TRAIN_LABELS = {'open': SAMPLE_TRAIN_LABELS}
SAMPLE_PREDICT_LABELS = {'open': SAMPLE_PREDICT_LABELS}


class TestFinancialFeature(TestCase):

    def setUp(self):
        self.feature_close_with_value_transform = FinancialFeature(
            name='open',
            transformation={'name': 'value'},
            normalization=None,
            nbins=5,
            ndays=2,
            resample_minutes=0,
            start_market_minute=30,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=15
        )
        self.feature_close_with_log_return_transform = FinancialFeature(
            name='close',
            transformation={'name': 'log-return'},
            normalization=None,
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35
        )
        self.feature_high_with_log_return_transform = FinancialFeature(
            name='high',
            transformation={'name': 'log-return'},
            normalization='standard',
            nbins=None,
            ndays=10,
            resample_minutes=0,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=69
        )

        self.feature_list = [
            self.feature_close_with_value_transform,
            self.feature_close_with_log_return_transform,
            self.feature_high_with_log_return_transform
        ]

    def test_get_start_timestamp_x(self):
        start_date_str = '20150101'
        end_date_str = '20150501'

        market_open_list = sample_market_calendar.schedule(start_date_str, end_date_str).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=15)

        start_timestamp_x_1 = self.feature_close_with_value_transform._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x1 = pd.Timestamp('2015-01-29 15:00:00+0000', tz='UTC')
        assert start_timestamp_x_1 == expected_start_timestamp_x1

        start_timestamp_x_2 = self.feature_close_with_log_return_transform._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x2 = pd.Timestamp('2015-01-26 16:00:00+0000', tz='UTC')
        assert start_timestamp_x_2 == expected_start_timestamp_x2

        start_timestamp_x_3 = self.feature_high_with_log_return_transform._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x3 = pd.Timestamp('2015-01-16 17:00:00+0000', tz='UTC')
        assert start_timestamp_x_3 == expected_start_timestamp_x3

    def test_select_prediction_data(self):
        data_frame = sample_hourly_ohlcv_data_dict[self.feature_close_with_value_transform.name]
        start_date = data_frame.index[0].date()
        end_date = data_frame.index[-1].date()

        market_open_list = sample_market_calendar.schedule(str(start_date), str(end_date)).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=15)

        selected_prediction_data = \
            self.feature_close_with_value_transform._select_prediction_data_x(data_frame, prediction_timestamp)

        last_index = np.argwhere(data_frame.index <= prediction_timestamp)[-1][0] + 1
        first_index = last_index - self.feature_close_with_value_transform.length
        expected_data_frame = data_frame.iloc[first_index:last_index]

        assert selected_prediction_data.equals(expected_data_frame)

    @staticmethod
    def _run_get_prediction_data_test(feature, expected_length):
        data_frame = sample_hourly_ohlcv_data_dict[feature.name]
        start_date = data_frame.index[0].date()
        end_date = data_frame.index[-1].date()

        market_open_list = sample_market_calendar.schedule(str(start_date), str(end_date)).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=30)
        target_timestamp = market_open_list[21] + timedelta(minutes=90)

        prediction_data_x = feature.get_prediction_features(data_frame, prediction_timestamp)

        prediction_data_y = feature.get_prediction_targets(data_frame, prediction_timestamp, target_timestamp)

        assert isinstance(prediction_data_x, pd.DataFrame) and isinstance(prediction_data_y, pd.Series)
        assert len(prediction_data_x) == expected_length
        assert_array_equal(prediction_data_x.columns, prediction_data_y.index)

    def test_get_prediction_data(self):
        expected_length_list = [15, 35, 69]
        for feature, expected_length in zip(self.feature_list, expected_length_list):
            self._run_get_prediction_data_test(feature, expected_length)

    def test_declassify_single_predict_y(self):
        for feature in self.feature_list:
            if feature.nbins:
                predict_y = np.zeros_like(SAMPLE_PREDICT_LABELS[list(SAMPLE_PREDICT_LABELS.keys())[0]])
                predict_y[0] = 1
            else:
                predict_y = SAMPLE_PREDICT_LABELS
            with pytest.raises(NotImplementedError):
                feature.declassify_single_predict_y(predict_y)


class TestFeatureNormalization(TestCase):

    def setUp(self):

        transform_config = {'name': 'log-return'}

        self.feature1 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='min_max',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35,
            normalise_per_series=True
        )

        self.feature2 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='standard',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35,
            normalise_per_series=True
        )

        self.feature3 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='gaussian',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35,
            normalise_per_series=True
        )

        self.feature4 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='robust',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35,
            normalise_per_series=True
        )

        transform_config_2 = {'name': 'value'}

        self.feature5 = FinancialFeature(
            name='close',
            transformation=transform_config_2,
            normalization='min_max',
            nbins=5,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35,
            normalise_per_series=True
        )

    def test_fit_normalisation(self):

        symbol_data1 = np.random.randn(10000)

        self.feature1.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature1.scaler.data_max_, symbol_data1.max(), rtol=1e-4)
        assert np.isclose(self.feature1.scaler.data_min_, symbol_data1.min(), rtol=1e-4)

        self.feature2.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature2.scaler.mean_, symbol_data1.mean(), rtol=1e-4)
        assert np.isclose(self.feature2.scaler.var_, symbol_data1.var(), rtol=1e-4)

        self.feature3.fit_normalisation(symbol_data=symbol_data1)
        assert self.feature3.scaler.references_.shape == (1000,)
        assert self.feature3.scaler.quantiles_.shape == (1000, 1)

        self.feature4.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature4.scaler.center_, np.median(symbol_data1), rtol=1e-4)

    def test_apply_normalisation(self):
        data = deepcopy(sample_hourly_ohlcv_data_dict['open'])

        for column in data.columns:
            self.feature1.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature1.apply_normalisation(data)
        np.testing.assert_allclose(data.max(), np.asarray([1.,  1.,  1.,  1.,  1.]))
        np.testing.assert_allclose(data.min(), np.asarray([0., 0., 0., 0., 0.]))

        for column in data.columns:
            self.feature2.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature2.apply_normalisation(data)
        np.testing.assert_allclose(data.mean(), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-4)

        for column in data.columns:
            self.feature3.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature3.apply_normalisation(data)
        np.testing.assert_allclose(data.mean(), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-3)

        for column in data.columns:
            self.feature4.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature4.apply_normalisation(data)
        np.testing.assert_allclose(np.median(data, axis=0), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-3)

    def test_apply_classification(self):
        symbols = ['SYM1', 'SYM2', 'SYM3']
        feature = self.feature5
        dataframe = pd.DataFrame([[5, 5, 5]], columns=symbols)

        symbol_data_1 = np.linspace(0, 10, 10)
        symbol_data_2 = np.linspace(0, 100, 100)

        feature.fit_classification('SYM1', symbol_data_1)
        feature.fit_classification('SYM2', symbol_data_2)

        classified_dataframe = feature.apply_classification(dataframe)
        expected_classified_dataframe = pd.DataFrame([[0., 1.],
                                                      [0., 0.],
                                                      [1., 0.],
                                                      [0., 0.],
                                                      [0., 0.]], columns=symbols[:2])

        assert classified_dataframe.equals(expected_classified_dataframe)
