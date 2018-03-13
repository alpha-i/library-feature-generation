import pytest

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.transform import TransformVolatility

from tests.feature.features.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_ohlcv_hourly


def test_transform_volatility_bad_config():

    with pytest.raises(AssertionError):
        TransformVolatility({})


def test_transform_volatility_x():

    transform_config = {'name': 'volatility', 'window': 10}

    feature = FinancialFeature(
            name='close',
            transformation={'name': 'volatility', 'window': 10},
            normalization='standard',
            nbins=10,
            ndays=10,
            resample_minutes=0,
            start_market_minute=150,
            is_target=True,
            calendar=sample_market_calendar,
            local=True,
            length=10
        )

    transform = TransformVolatility(transform_config)

    raw_dataframe = sample_ohlcv_hourly[feature.name]

    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)

    assert processed_prediction_data_x.shape == (256, 5)


