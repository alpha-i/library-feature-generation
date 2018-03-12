import random

import numpy as np
from numpy.testing import assert_almost_equal

from alphai_feature_generation.feature.features.financial import FinancialFeature

from alphai_feature_generation.feature.transform import TransformLogReturn


from tests.helpers import ASSERT_NDECIMALS
from tests.feature.features.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_ohlcv_hourly


def test_transform_log_return_x():

    transform_config = {'name': 'log-return'}

    feature = FinancialFeature(
        name='close',
        transformation=transform_config,
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=0,
        start_market_minute=90,
        is_target=True,
        calendar=sample_market_calendar,
        local=False,
        length=35
    )

    transform = TransformLogReturn(transform_config)

    data_dict_x = sample_ohlcv_hourly
    raw_dataframe = data_dict_x[feature.name]

    transformed_data = transform.transform_x(feature, raw_dataframe)

    symbol = 'AAPL'
    original_data = raw_dataframe[symbol]
    transformed_data = transformed_data[symbol]

    random_index = random.randrange(1, len(original_data) - 1)

    expected_value = np.log(original_data.iloc[random_index] / original_data.iloc[random_index-1])
    assert_almost_equal(transformed_data.iloc[random_index], expected_value, ASSERT_NDECIMALS)

    assert np.isnan(transformed_data.iloc[0])


def test_transform_log_return_y():

    transform_config = {'name': 'log-return'}

    feature = FinancialFeature(
        name='close',
        transformation=transform_config,
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=0,
        start_market_minute=90,
        is_target=True,
        calendar=sample_market_calendar,
        local=False,
        length=35
    )

    transform = TransformLogReturn(transform_config)

    data_dict_x = sample_ohlcv_hourly
    raw_dataframe = data_dict_x[feature.name]

    data_frame_x = raw_dataframe.iloc[:-1]
    prediction_reference_data = data_frame_x.iloc[-1]
    data_frame_y = raw_dataframe.iloc[-1]

    transformed_data = transform.transform_y(feature, data_frame_y, prediction_reference_data)

    expected_log_returns = np.log(data_frame_y/prediction_reference_data)

    assert_almost_equal(transformed_data, expected_log_returns.values, ASSERT_NDECIMALS)


def test_transform_x_log_return_with_local_feature():

    transform_config = {'name': 'log-return'}

    feature = FinancialFeature(
        name='close',
        transformation=transform_config,
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=0,
        start_market_minute=90,
        is_target=True,
        calendar=sample_market_calendar,
        local=True,
        length=35
    )
    data_dict_x = sample_ohlcv_hourly
    raw_dataframe = data_dict_x[feature.name]

    transform = TransformLogReturn(transform_config)
    transformed_data = transform.transform_x(feature, raw_dataframe)['AAPL']

    assert not np.isnan(transformed_data.iloc[0])
