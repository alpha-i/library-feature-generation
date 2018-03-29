from datetime import timedelta
from unittest import TestCase
import os

import pickle

import numpy as np
import pandas as pd
import pytest

from alphai_feature_generation.transformation.financial import FinancialDataTransformation

from tests.helpers import TEST_DATA_PATH

from tests.transformation.financial.helpers import (
    financial_data_fixtures,
    sample_feature_configuration_list,
    load_preset_config,
    create_historical_universe
)

REL_TOL = 1e-4


class TestFinancialDataTransformation(TestCase):

    def setUp(self):
        configuration_bins = {
            'feature_config_list': sample_feature_configuration_list,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            FinancialDataTransformation.KEY_EXCHANGE: 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta': timedelta(days=5),
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'n_assets': 5,
            'local': False,
            'classify_per_series': True,
            'normalise_per_series': False,
            'fill_limit': 0
        }

        self.transformation = FinancialDataTransformation(configuration_bins)

    @pytest.mark.skip(reason='the test for classify_per_series=False must be implemented')
    def test_classify_per_series_false(self):
        pass

    @pytest.mark.skip(reason='The test for normalise_per_series=False must be implemented')
    def test_normalise_per_series_false(self):
        pass

    @pytest.mark.skip(reason='The test for transformation with bins must be implemented')
    def test_financial_transformation(self):
        pass

    @pytest.mark.skip(reason='The test for prediction at market close must be implemented')
    def test_create_data_with_prediction_at_market_close(self):
        pass

    @pytest.mark.skip(reason='The test for prediction at market close must be implemented')
    def test_check_x_batch_dimensions(self):
        pass

    def test_get_total_ticks_x(self):
        assert self.transformation.get_total_ticks_x() == 15

    def test_build_target_delta(self):
        assert self.transformation.target_delta.days == 5

    def test_extract_schedule_from_data(self):

        data_schedule = self.transformation._extract_schedule_from_data(financial_data_fixtures)

        assert isinstance(data_schedule, pd.DataFrame)
        assert len(data_schedule) == 37
        assert data_schedule.iloc[0].market_open == pd.Timestamp('2015-01-14 14:30:00+0000', tz='UTC')
        assert data_schedule.iloc[-1].market_open == pd.Timestamp('2015-03-09 13:30:00+0000', tz='UTC')

    def test_get_target_feature(self):
        target_feature = self.transformation.get_target_feature()
        expected_target_feature = [
            feature for feature in self.transformation.features if feature.is_target][0]
        assert target_feature == expected_target_feature

    def test_get_prediction_data_all_features_target(self):
        raw_data_dict = financial_data_fixtures
        prediction_timestamp = financial_data_fixtures['open'].index[98]
        universe = financial_data_fixtures['open'].columns
        target_timestamp = financial_data_fixtures['open'].index[133]

        feature_x_dict, feature_y_dict = self.transformation._collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
            prediction_timestamp,
            universe,
            target_timestamp,
        )

        expected_n_time_dict = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
        expected_n_symbols = 5
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features

        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_n_time_dict[key], expected_n_symbols)

        for key in feature_y_dict.keys():
            assert feature_y_dict[key].shape == (1, expected_n_symbols)

    def test_get_prediction_data_all_features_no_target(self):
        raw_data_dict = financial_data_fixtures
        prediction_timestamp = financial_data_fixtures['open'].index[98]
        feature_x_dict, feature_y_dict = self.transformation._collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
            prediction_timestamp,
        )

        expected_features_length = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
        expected_n_symbols = 5
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features
        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_features_length[key], expected_n_symbols)
        assert not feature_y_dict

    def test_make_normalised_x_list(self):

        symbols = {'AAPL'}
        dummy_dataframe = pd.DataFrame([3.0, 3.0, 3.0], columns=symbols)
        normalised_dataframe = pd.DataFrame([0.0, 0.0, 0.0], columns=symbols)
        dummy_dict = {'open_value': dummy_dataframe,
                      'high_log-return': dummy_dataframe}
        normalised_dict = {'open_value': normalised_dataframe,
                           'high_log-return': dummy_dataframe}

        starting_x_list = [dummy_dict]
        expected_x_list = [normalised_dict]
        feature = self.transformation.features[0]

        self.transformation.fit_normalisation(symbols, starting_x_list, feature)
        normalised_x_list = self.transformation._make_normalised_x_list(
            starting_x_list,
            do_normalisation_fitting=True
        )
        assert normalised_x_list[0]['open_value']['AAPL'].equals(expected_x_list[0]['open_value']['AAPL'])

    def test_create_predict_data(self):

        expected_n_samples = 1
        expected_n_time_dict = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
        expected_n_symbols = 5
        expected_n_features = 3

        config = load_preset_config(expected_n_symbols)
        fintransform = FinancialDataTransformation(config)

        # have to run train first so that the normalizers are fit
        historical_universe = create_historical_universe(financial_data_fixtures)
        _, _ = fintransform.create_train_data(financial_data_fixtures, historical_universe)

        predict_x, symbols, predict_timestamp, target_timestamp = fintransform.create_predict_data(
            financial_data_fixtures)

        assert predict_timestamp == pd.Timestamp('2015-03-09 14:00:00+0000', tz='UTC')

        assert len(predict_x.keys()) == expected_n_features
        assert list(predict_x.keys()) == ['open_value', 'close_log-return', 'high_log-return']

        assert np.isclose(predict_x['open_value'][0, :, 0].mean(), 124.787, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 1].mean(), 570.950, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 2].mean(), 32.418, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 3].mean(), 384.03379, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 4].mean(), 15.9612, rtol=REL_TOL)

        assert np.isclose(predict_x['high_log-return'][0, :, 0].mean(), -0.14222451613690593, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 1].mean(), -0.19212886645801133, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 2].mean(), -0.50004735819544888, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 3].mean(), -0.2603029872984271, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 4].mean(), 0.15312313803264102, rtol=REL_TOL)

        assert np.isclose(predict_x['close_log-return'][0, :, 0].mean(), -0.00037412756518502636, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 1].mean(), -0.00071031231939734001, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 2].mean(), -0.0028026462004643749, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 3].mean(), -0.0011889590013153429, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 4].mean(), 0.0015928267596619391, rtol=REL_TOL)

        for key in predict_x.keys():
            assert predict_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        assert len(symbols) == expected_n_symbols
        assert list(symbols) == ['AAPL', 'GOOG', 'INTC', 'AMZN', 'BAC']

    def test_create_predict_data_on_market_close(self):

        expected_n_samples = 1
        expected_n_time_dict = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
        expected_n_symbols = 5
        expected_n_features = 3

        config = load_preset_config(expected_n_symbols)
        config['predict_the_market_close'] = True
        fintransform = FinancialDataTransformation(config)

        # have to run train first so that the normalizers are fit
        historical_universe = create_historical_universe(financial_data_fixtures)
        _, _ = fintransform.create_train_data(financial_data_fixtures, historical_universe)
        predict_x, symbols, predict_timestamp, target_timestamp = fintransform.create_predict_data(
            financial_data_fixtures)

        assert predict_timestamp == pd.Timestamp('2015-03-09 20:00:00+0000', tz='UTC')
        assert len(predict_x.keys()) == expected_n_features
        assert list(predict_x.keys()) == ['open_value', 'close_log-return', 'high_log-return']

        assert np.isclose(predict_x['open_value'][0, :, 0].mean(), 124.787, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 1].mean(), 570.950, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 2].mean(), 32.418, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 3].mean(), 384.03379, rtol=REL_TOL)
        assert np.isclose(predict_x['open_value'][0, :, 4].mean(), 15.9612, rtol=REL_TOL)

        assert np.isclose(predict_x['close_log-return'][0, :, 0].mean(), -0.00037412756518502636, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 1].mean(), -0.00071031231939734001, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 2].mean(), -0.0028026462004643749, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 3].mean(), -0.0011889590013153429, rtol=REL_TOL)
        assert np.isclose(predict_x['close_log-return'][0, :, 4].mean(), 0.0015928267596619391, rtol=REL_TOL)

        assert np.isclose(predict_x['high_log-return'][0, :, 0].mean(), -0.14222451613690593, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 1].mean(), -0.19212886645801133, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 2].mean(), -0.50004735819544888, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 3].mean(), -0.2603029872984271, rtol=REL_TOL)
        assert np.isclose(predict_x['high_log-return'][0, :, 4].mean(), 0.15312313803264102, rtol=REL_TOL)

        for key in predict_x.keys():
            assert predict_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        assert len(symbols) == expected_n_symbols
        assert list(symbols) == ['AAPL', 'GOOG', 'INTC', 'AMZN', 'BAC']

    def test_stack_samples_for_each_feature(self):

        config = {'classify_per_series': True,
                  'calendar_name': 'NYSE',
                  'feature_config_list': [
                      {'classify_per_series': True,
                       'calendar_name': 'NYSE',
                       'is_target': False,
                       'length': 15,
                       'local': True,
                       'name': 'open',
                       'nbins': 5,
                       'ndays': 2,
                       'normalise_per_series': False,
                       'normalization': None,
                       'resample_minutes': 0,
                       'start_market_minute': 1,
                       'transformation': {'name': 'value'}},
                      {'classify_per_series': True,
                       'calendar_name': 'NYSE',
                       'is_target': False,
                       'length': 15,
                       'local': False,
                       'name': 'close',
                       'nbins': 5,
                       'ndays': 2,
                       'normalise_per_series': False,
                       'normalization': None,
                       'resample_minutes': 0,
                       'start_market_minute': 1,
                       'transformation': {'name': 'log-return'}},
                      {'classify_per_series': True,
                       'calendar_name': 'NYSE',
                       'is_target': True,
                       'length': 15,
                       'local': False,
                       'name': 'high',
                       'nbins': 5,
                       'ndays': 2,
                       'normalise_per_series': False,
                       'normalization': 'standard',
                       'resample_minutes': 0,
                       'start_market_minute': 1,
                       'transformation': {'name': 'log-return'}}
                  ],
                  'features_ndays': 2,
                  'features_resample_minutes': 60,
                  'features_start_market_minute': 1,
                  'fill_limit': 0,
                  'local': False,
                  'n_classification_bins': 5,
                  'n_assets': 5,
                  'normalise_per_series': False,
                  'prediction_frequency_ndays': 1,
                  'prediction_market_minute': 30,
                  'target_delta': timedelta(5),
                  'target_market_minute': 30
                  }

        transformation = FinancialDataTransformation(config)

        fixtures = pickle.load(open(os.path.join(TEST_DATA_PATH, 'stack_sample_data.pkl'), 'rb'))

        stacked_samples, valid_symbols = transformation.stack_samples_for_each_feature(fixtures['samples'])
        np.testing.assert_array_equal(stacked_samples['close_log-return'], fixtures['stacked_samples']['close_log-return'])
        assert list(valid_symbols) == list(fixtures['valid_symbols'])

    def test_check_x_batch_dimensions(self):

        expected_n_symbols = 4

        test_dict_1 = {'open_value': np.zeros(15), 'close_log-return': np.zeros(15), 'high_log-return': np.zeros(15)}
        test_dict_2 = {'open_value': np.zeros(0), 'close_log-return': np.zeros(15), 'high_log-return': np.zeros(15)}
        test_dict_3 = {'open_value': np.zeros(15), 'close_log-return': np.zeros(12), 'high_log-return': np.zeros(15)}

        config = load_preset_config(expected_n_symbols)
        transformation = FinancialDataTransformation(config)

        assert transformation.check_x_batch_dimensions(test_dict_1)
        assert ~transformation.check_x_batch_dimensions(test_dict_2)
        assert ~transformation.check_x_batch_dimensions(test_dict_3)
