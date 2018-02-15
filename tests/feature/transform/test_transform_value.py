from alphai_feature_generation.feature import FinancialFeature
from alphai_feature_generation.feature.transform import TransformValue

from tests.feature.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_hourly_ohlcv_data_dict


def test_transform_value_x():

    transform_config = {'name': 'value'}
    feature = FinancialFeature(
            name='open',
            transformation=transform_config,
            normalization=None,
            nbins=5,
            ndays=2,
            resample_minutes=0,
            start_market_minute=30,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=15
        )
    transform = TransformValue(transform_config)

    data_dict_x = sample_hourly_ohlcv_data_dict
    raw_dataframe = data_dict_x[feature.name]

    transformed_data = transform.transform_x(feature, raw_dataframe)

    assert transformed_data.equals(raw_dataframe)
