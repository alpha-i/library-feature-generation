import pytest
from numpy.testing import assert_almost_equal

from alphai_feature_generation.feature import FinancialFeature
from alphai_feature_generation.feature.transform import TransformKer

from tests.helpers import sample_market_calendar, sample_hourly_ohlcv_data_dict, ASSERT_NDECIMALS


def test_transform_ker_invalid_config():

    with pytest.raises(AssertionError):
        TransformKer({})


def test_transform_ker_x():

    transform_config = {'name': 'ker', 'lag': 20}

    feature = FinancialFeature(
            name='high',
            transformation=transform_config,
            normalization=None,
            nbins=10,
            ndays=10,
            resample_minutes=0,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=10
        )

    raw_dataframe = sample_hourly_ohlcv_data_dict[feature.name]

    transform = TransformKer(transform_config)
    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)

    direction = raw_dataframe.diff(transform.lag).abs()
    volatility = raw_dataframe.diff().abs().rolling(window=transform.lag).sum()

    direction.dropna(axis=0, inplace=True)
    volatility.dropna(axis=0, inplace=True)

    expected_result = direction / volatility
    expected_result.dropna(axis=0, inplace=True)

    assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)