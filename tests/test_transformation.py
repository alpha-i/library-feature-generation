from unittest import TestCase

import numpy as np
import pandas as pd

from alphai_feature_generation.transformation import (
    FinancialDataTransformation,
)
from alphai_feature_generation.feature import KEY_EXCHANGE
from tests.helpers import (
    sample_hourly_ohlcv_data_dict,
    sample_fin_data_transf_feature_factory_list_nobins,
    sample_fin_data_transf_feature_factory_list_bins,
    sample_historical_universes,
    TEST_ARRAY,
)

SAMPLE_TRAIN_LABELS = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))
SAMPLE_PREDICT_LABELS = SAMPLE_TRAIN_LABELS[:, int(0.5 * SAMPLE_TRAIN_LABELS.shape[1])]

SAMPLE_TRAIN_LABELS = {'open': SAMPLE_TRAIN_LABELS}
SAMPLE_PREDICT_LABELS = {'open': SAMPLE_PREDICT_LABELS}

ASSERT_NDECIMALS = 5


class TestFinancialDataTransformation(TestCase):
    def setUp(self):

        # FIXME we need to find a tests case for classify_per_series: False and normalise_per_series: False

        configuration_nobins = {
            'feature_config_list': sample_fin_data_transf_feature_factory_list_nobins,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            KEY_EXCHANGE: 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta_ndays': 5,
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'nassets': 5,
            'local': False,
            'classify_per_series': False,
            'normalise_per_series': False,
            'fill_limit': 0
        }

        self.fin_data_transf_nobins = FinancialDataTransformation(configuration_nobins)

        configuration_bins = {
            'feature_config_list': sample_fin_data_transf_feature_factory_list_bins,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            KEY_EXCHANGE: 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta_ndays': 5,
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'nassets': 5,
            'local': False,
            'classify_per_series': False,
            'normalise_per_series': False,
            'fill_limit': 0
        }
        self.fin_data_transf_bins = FinancialDataTransformation(configuration_bins)

    def test_get_total_ticks_x(self):
        assert self.fin_data_transf_nobins.get_total_ticks_x() == 15

    def test_extract_schedule_from_data(self):

        data_schedule = self.fin_data_transf_nobins._extract_schedule_from_data(sample_hourly_ohlcv_data_dict)

        assert isinstance(data_schedule, pd.DataFrame)
        assert len(data_schedule) == 37
        assert data_schedule.iloc[0].market_open == pd.Timestamp('2015-01-14 14:30:00+0000', tz='UTC')
        assert data_schedule.iloc[-1].market_open == pd.Timestamp('2015-03-09 13:30:00+0000', tz='UTC')

    def test_get_target_feature(self):
        target_feature = self.fin_data_transf_nobins.get_target_feature()
        expected_target_feature = [feature for feature in self.fin_data_transf_nobins.features if feature.is_target][0]
        assert target_feature == expected_target_feature

    def test_get_prediction_data_all_features_target(self):
        raw_data_dict = sample_hourly_ohlcv_data_dict
        prediction_timestamp = sample_hourly_ohlcv_data_dict['open'].index[98]
        universe = sample_hourly_ohlcv_data_dict['open'].columns[:-1]
        target_timestamp = sample_hourly_ohlcv_data_dict['open'].index[133]
        feature_x_dict, feature_y_dict = self.fin_data_transf_nobins.collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
            universe,
            target_timestamp,
        )

        expected_n_time_dict = {'open_value': 15, 'high_log-return': 14, 'close_log-return': 14}
        expected_n_symbols = 4
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features

        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_n_time_dict[key], expected_n_symbols)

        for key in feature_y_dict.keys():
            assert feature_y_dict[key].shape == (1, expected_n_symbols)

    def test_get_prediction_data_all_features_no_target(self):
        raw_data_dict = sample_hourly_ohlcv_data_dict
        prediction_timestamp = sample_hourly_ohlcv_data_dict['open'].index[98]
        feature_x_dict, feature_y_dict = self.fin_data_transf_nobins.collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
        )

        expected_n_time_dict = {'open_value': 15, 'high_log-return': 14, 'close_log-return': 14}
        expected_n_symbols = 5
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features
        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_n_time_dict[key], expected_n_symbols)
        assert feature_y_dict is None

    def test_create_data(self):
        expected_n_samples = 30
        expected_n_time_dict = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
        expected_n_symbols = 4
        expected_n_features = 3
        expected_n_bins = 5

        config = self.load_default_config(expected_n_symbols)
        fintransform = FinancialDataTransformation(config)

        train_x, train_y = fintransform.create_train_data(sample_hourly_ohlcv_data_dict,
                                                          sample_historical_universes)

        assert len(train_x.keys()) == expected_n_features

        for key in train_x.keys():
            assert train_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        for key in train_y.keys():
            assert train_y[key].shape == (expected_n_samples, expected_n_bins, expected_n_symbols)

        train_x, train_y = fintransform.create_train_data(sample_hourly_ohlcv_data_dict,
                                                          sample_historical_universes)

        assert len(train_x.keys()) == expected_n_features
        for key in train_x.keys():
            assert train_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        for key in train_y.keys():
            assert train_y[key].shape == (expected_n_samples, expected_n_bins, expected_n_symbols)

    def load_default_config(self, expected_n_symbols):

        default_config = {'feature_config_list': sample_fin_data_transf_feature_factory_list_bins,
                          'features_ndays': 2,
                          'features_resample_minutes': 60,
                          'features_start_market_minute': 1,
                          KEY_EXCHANGE: 'NYSE',
                          'prediction_frequency_ndays': 1,
                          'prediction_market_minute': 30,
                          'target_delta_ndays': 5,
                          'target_market_minute': 30,
                          'n_classification_bins': 5,
                          'nassets': expected_n_symbols,
                          'local': False,
                          'classify_per_series': False,
                          'normalise_per_series': False,
                          'fill_limit': 0}

        return default_config


def mock_ml_model_single_pass(predict_x):
    mean_list = []
    for key in predict_x.keys():
        value = predict_x[key]  # shape eg (1, 15, 5)
        mean_value = value.mean(axis=1)
        mean_list.append(mean_value)
    mean_list = np.asarray(mean_list)
    factors = mean_list.mean(axis=0)
    return np.ones(shape=(len(factors),)) * factors


def mock_ml_model_multi_pass(predict_x, n_passes, nbins):
    mean_list = []
    for key in predict_x.keys():
        mean_list.append(predict_x[key].mean(axis=1))
    mean_list = np.asarray(mean_list)
    factors = mean_list.mean(axis=1)
    n_series = len(factors)
    if nbins:
        predict_y = np.zeros((n_passes, n_series, nbins))
        for i in range(n_passes):
            for j in range(n_series):
                predict_y[i, j, i % nbins] = 1
        return predict_y
    else:
        raise ValueError("Only classification currently supported")
