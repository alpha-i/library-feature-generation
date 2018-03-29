import pytest
from numpy.testing import assert_almost_equal

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.transform import TransformKer

from tests.feature.transform import ASSERT_NDECIMALS
from tests.feature.features.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import financial_data_fixtures


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
            calendar=sample_market_calendar,
            local=False,
            length=10
        )

    raw_dataframe = financial_data_fixtures[feature.name]

    transform = TransformKer(transform_config)
    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)

    direction = raw_dataframe.diff(transform.lag).abs()
    volatility = raw_dataframe.diff().abs().rolling(window=transform.lag).sum()

    direction.dropna(axis=0, inplace=True)
    volatility.dropna(axis=0, inplace=True)

    expected_result = direction / volatility
    expected_result.dropna(axis=0, inplace=True)

    assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)
