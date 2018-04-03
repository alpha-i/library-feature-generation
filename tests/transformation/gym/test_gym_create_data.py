import numpy as np
import pytest

from alphai_feature_generation.transformation import GymDataTransformation
from tests.transformation.gym.helpers import load_preset_config, gym_data_fixtures


@pytest.mark.parametrize("index", [0, 1, 2])
def test_create_data(index):
    expected_n_samples = 49
    expected_n_time_dict = {'hour_value': 5, 'temperature_value': 5, 'number_people_value': 5}
    expected_n_symbols = 1
    expected_n_features = 3
    expected_n_bins = 5
    expected_n_forecasts = 1

    config = load_preset_config(expected_n_symbols, index)
    gym_transform = GymDataTransformation(config)

    train_x, train_y = gym_transform.create_train_data(gym_data_fixtures)

    assert len(train_x.keys()) == expected_n_features
    if index < 2:
        assert set(train_x.keys()) == set(expected_n_time_dict.keys())

        # Check shape of arrays
        for key in train_x.keys():
            assert train_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        for key in train_y.keys():
            assert train_y[key].shape == (expected_n_samples, expected_n_forecasts , expected_n_symbols, expected_n_bins)

    # Now check contents
    if index == 2:
        x_key = 'hour_value_15T'
        y_key = 'number_people_value_150T'
    else:
        x_key = 'hour_value'
        y_key = 'number_people_value'

    exp_x_mean, exp_y_mean, expected_sample = _expected_results(index)

    x_mean = train_x[x_key].flatten().mean()
    if np.isnan(exp_x_mean):
        assert np.isnan(x_mean)
    else:
        assert np.isclose(x_mean, exp_x_mean)

    y_mean = train_y[y_key].flatten().mean()
    assert np.isclose(y_mean, exp_y_mean)

    if index == 0:  # Check feature ordering is preserved. This mimics the extraction of data in oracle.py
        numpy_arrays = []
        for key, value in train_x.items():
            numpy_arrays.append(value)

        stacked_train_x = np.stack(numpy_arrays, axis=0)
        sample_data = stacked_train_x.flatten()[0:4]

        np.testing.assert_array_almost_equal(sample_data, expected_sample)


def _expected_results(iteration):
    return_value_list = [
        {'x_mean': 8.06938775510204, 'y_mean': 0.2},
        {'x_mean': 8.06938775510204, 'y_mean': 0.2},  # Test classification and normalisation
        {'x_mean': -6.57070769676113e-17, 'y_mean': 0.2},  # Test length/resolution requests
    ]

    try:
        return_value = return_value_list[iteration]
        expected_sample = [23.,  0.,  1.,  6.]
        return return_value['x_mean'], return_value['y_mean'], expected_sample
    except KeyError:
        raise ValueError('Requested configuration not implemented')