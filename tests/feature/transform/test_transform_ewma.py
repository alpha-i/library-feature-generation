import pytest
from numpy.testing import assert_almost_equal

from alphai_feature_generation.feature import FinancialFeature
from alphai_feature_generation.feature.transform import TransformEWMA

from tests.helpers import ASSERT_NDECIMALS
from tests.feature.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_hourly_ohlcv_data_dict


def test_transform_ewma_bad_config():

    with pytest.raises(AssertionError):
        TransformEWMA({})


def test_transform_ewma_x():

    transform_config = {'name': 'ewma', 'halflife': 20}

    feature = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='standard',
            nbins=10,
            ndays=10,
            resample_minutes=0,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=True,
            length=10
        )

    transform = TransformEWMA(transform_config)

    raw_dataframe = sample_hourly_ohlcv_data_dict[feature.name]

    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)

    expected_result = raw_dataframe.ewm(halflife=transform.halflife).mean()
    assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)


