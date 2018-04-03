import numpy as np
import pytest

from alphai_feature_generation.transformation import FinancialDataTransformation
from tests.transformation.financial.helpers import (
    get_configuration,
    financial_data_fixtures,
    create_historical_universe
)


def _expected_results(iteration):
    return_value_list = [
        {'x_mean': 207.451975429, 'y_mean': 0.2},
        {'x_mean': 207.451975429, 'y_mean': 0.2},  # Test predict_the_market_close
        {'x_mean': 207.451975429, 'y_mean': 0.2},  # Test classification and normalisation
        {'x_mean': 5.96046e-09, 'y_mean': 0.2},  # Test length/resolution requests
    ]

    try:
        return_value = return_value_list[iteration]
        expected_sample = [107.35616667, 498.748, 35.341, 288.86503167]
        return return_value['x_mean'], return_value['y_mean'], expected_sample
    except KeyError:
        raise ValueError('Requested configuration not implemented')


@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_create_data(index):
    expected_n_samples = 30
    expected_n_time_dict = {'open_value': 15, 'high_log-return': 15, 'close_log-return': 15}
    expected_n_symbols = 4
    expected_n_features = 3
    expected_n_bins = 5

    config = get_configuration(expected_n_symbols, index)
    exp_x_mean, exp_y_mean, expected_sample = _expected_results(index)

    transformation = FinancialDataTransformation(config)

    historical_universe = create_historical_universe(financial_data_fixtures)
    train_x, train_y = transformation.create_train_data(financial_data_fixtures, historical_universe)

    assert len(train_x.keys()) == expected_n_features
    if index < 3:
        assert list(train_x.keys()) == ['open_value', 'close_log-return', 'high_log-return']

        # Check shape of arrays
        for key in train_x.keys():
            assert train_x[key].shape == (expected_n_samples, expected_n_time_dict[key], expected_n_symbols)

        for key in train_y.keys():
            assert train_y[key].shape == (expected_n_samples, expected_n_bins, expected_n_symbols)

    # Now check contents
    if index == 3:
        x_key = 'close_log-return_15T'
        y_key = 'high_log-return_150T'
    else:
        x_key = 'open_value'
        y_key = 'high_log-return'

    assert np.isclose(train_x[x_key].flatten().mean(), exp_x_mean)
    assert np.isclose(train_y[y_key].flatten().mean(), exp_y_mean)

    if index == 0:  # Check feature ordering is preserved. This mimics the extraction of data in oracle.py
        numpy_arrays = []
        for key, value in train_x.items():
            numpy_arrays.append(value)

        train_x = np.stack(numpy_arrays, axis=0)
        sample_data = train_x.flatten()[0:4]

        np.testing.assert_array_almost_equal(sample_data, expected_sample)


