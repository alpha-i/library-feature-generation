from collections import OrderedDict
from datetime import timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from alphai_feature_generation.transformation.gym import GymDataTransformation
from tests.transformation.gym.helpers import (load_preset_config,
                                              sample_feature_list,
                                              gym_data_fixtures)


class TestGymDataTransformation(TestCase):

    def setUp(self):

        config = {
            'feature_config_list': sample_feature_list,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            GymDataTransformation.KEY_EXCHANGE: 'GYMUK',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta':timedelta(days=5),
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'n_assets': 5,
            'local': False,
            'classify_per_series': False,
            'normalise_per_series': False,
            'fill_limit': 0
        }

        self.transformation = GymDataTransformation(config)

    @pytest.mark.skip(reason='the test for classify_per_series=False must be implemented')
    def test_classify_per_series_false(self):
        pass

    @pytest.mark.skip(reason='The test for normalise_per_series=False must be implemented')
    def test_normalise_per_series_false(self):
        pass

    @pytest.mark.skip(reason='The test for prediction at market close must be implemented')
    def test_check_x_batch_dimensions(self):
        pass

    def test_get_total_ticks_x(self):
        assert self.transformation.get_total_ticks_x() == 33

    def test_extract_schedule_from_data(self):

        data_schedule = self.transformation._extract_schedule_from_data(gym_data_fixtures)

        assert isinstance(data_schedule, pd.DataFrame)
        assert len(data_schedule) == 56
        assert data_schedule.iloc[0].market_open == pd.Timestamp('2015-08-14 06:00:00+0000', tz='UTC')
        assert data_schedule.iloc[-1].market_open == pd.Timestamp('2015-10-30 07:00:00+0000', tz='UTC')

    def test_get_target_feature(self):
        target_feature = self.transformation.get_target_feature()
        expected_target_feature = [
            feature for feature in self.transformation.features if feature.is_target][0]
        assert target_feature == expected_target_feature

    def test_get_prediction_data_all_features_target(self):
        raw_data_dict = gym_data_fixtures
        prediction_timestamp = gym_data_fixtures['hour'].index[98]
        universe = gym_data_fixtures['hour'].columns
        target_timestamp = gym_data_fixtures['hour'].index[133]
        feature_x_dict, feature_y_dict = self.transformation._collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
            prediction_timestamp,
            target_timestamp,
        )

        expected_n_time_dict = {'hour_value': 5, 'temperature_value': 5, 'number_people_value': 5}
        expected_n_symbols = 1
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features

        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_n_time_dict[key], expected_n_symbols)

        for key in feature_y_dict.keys():
            assert feature_y_dict[key].shape == (1, expected_n_symbols)

    def test_get_prediction_data_all_features_no_target(self):
        raw_data_dict = gym_data_fixtures
        prediction_timestamp = gym_data_fixtures['hour'].index[98]
        feature_x_dict, feature_y_dict = self.transformation._collect_prediction_from_features(
            raw_data_dict,
            prediction_timestamp,
            prediction_timestamp,
        )

        expected_n_time_dict = {'hour_value': 5, 'temperature_value': 5, 'number_people_value': 5}
        expected_n_symbols = 1
        expected_n_features = 3

        assert len(feature_x_dict.keys()) == expected_n_features
        for key in feature_x_dict.keys():
            assert feature_x_dict[key].shape == (expected_n_time_dict[key], expected_n_symbols)
        assert isinstance(feature_y_dict, OrderedDict)

    def test_make_normalised_x_list(self):

        symbols = {'UCBerkeley'}

        dummy_dataframe = pd.DataFrame([3.0, 3.0, 3.0], columns=symbols)
        normalised_dataframe = pd.DataFrame([0.0, 0.0, 0.0], columns=symbols)

        dummy_dict = {'hour_value': dummy_dataframe, 'number_people_value': dummy_dataframe}
        normalised_dict = {'hour_value': normalised_dataframe, 'number_people_value': dummy_dataframe}

        starting_x_list = [dummy_dict]
        expected_x_list = [normalised_dict]
        feature = self.transformation.features[0]

        self.transformation.fit_normalisation(symbols, starting_x_list, feature)
        normalised_x_list = self.transformation._make_normalised_x_list(starting_x_list, do_normalisation_fitting=True)
        assert normalised_x_list[0]['hour_value']['UCBerkeley'].equals(expected_x_list[0]['hour_value']['UCBerkeley'])

    def test_create_predict_data(self):

        expected_n_samples = 1
        expected_feature_length = {'hour_value': 5, 'temperature_value': 5, 'number_people_value': 5}
        expected_n_symbols = 1
        expected_n_features = 3

        config = load_preset_config(expected_n_symbols)
        gym_transform = GymDataTransformation(config)

        # have to run train first so that the normalizers are fit
        _, _ = gym_transform.create_train_data(gym_data_fixtures)
        predict_x, symbols, predict_timestamp, target_timestamp = gym_transform.create_predict_data(gym_data_fixtures)

        assert predict_timestamp == pd.Timestamp('2015-10-30 08:00:00+0000', tz='UTC')

        assert len(predict_x.keys()) == expected_n_features
        assert set(predict_x.keys()) == set(expected_feature_length.keys())

        assert np.isclose(predict_x['hour_value'][0, :, 0].mean(), 4.4, rtol=REL_TOL)

        assert np.isclose(predict_x['temperature_value'][0, :, 0].mean(), 60.468, rtol=REL_TOL)

        assert np.isclose(predict_x['number_people_value'][0, :, 0].mean(), -0.264058763995265, rtol=REL_TOL)

        for key in predict_x.keys():
            assert predict_x[key].shape == (expected_n_samples, expected_feature_length[key], expected_n_symbols)

        assert len(symbols) == expected_n_symbols
        assert list(symbols) == ['UCBerkeley']


def test_check_x_batch_dimensions():

    expected_n_symbols = 4

    test_dict_1 = {'open_value': np.zeros(15), 'close_value': np.zeros(15), 'high_value': np.zeros(15)}
    test_dict_2 = {'open_value': np.zeros(0), 'close_value': np.zeros(15), 'high_value': np.zeros(15)}
    test_dict_3 = {'open_value': np.zeros(15), 'close_value': np.zeros(12), 'high_value': np.zeros(15)}

    config = load_preset_config(expected_n_symbols)
    gym_transform = GymDataTransformation(config)

    assert gym_transform.check_x_batch_dimensions(test_dict_1)
    assert ~gym_transform.check_x_batch_dimensions(test_dict_2)
    assert ~gym_transform.check_x_batch_dimensions(test_dict_3)


REL_TOL = 1e-4