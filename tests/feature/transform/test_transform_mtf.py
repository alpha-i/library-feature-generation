import pytest

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.transform import TransformMTF

from tests.feature.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_hourly_ohlcv_data_dict


def test_transform_mtf_bad_config():

    with pytest.raises(AssertionError):
        TransformMTF({})


def test_transform_mtf_x():

    transform_config = {'name': 'mtf', 'image_size': 24}

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

    transform = TransformMTF(transform_config)

    data_dict_x = sample_hourly_ohlcv_data_dict
    raw_dataframe = data_dict_x[feature.name]

    transformed_data = transform.transform_x(feature, raw_dataframe)

    assert transformed_data.shape == (transform.image_size**2, raw_dataframe.shape[1])


