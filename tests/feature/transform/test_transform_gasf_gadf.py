import pytest

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.transform import TransformGASF, TransformGADF
from tests.feature.features.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import financial_data_fixtures


def _perform_test(feature, raw_dataframe, transform):
    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)
    assert processed_prediction_data_x.shape == (transform.image_size ** 2, raw_dataframe.shape[1])


def test_transform_gasf_config():
    with pytest.raises(AssertionError):
        TransformGASF({})


def test_tranform_gasf_x():

    transform_config = {'name': 'gasf', 'image_size': 24}
    feature = FinancialFeature(
        name='close',
        transformation=transform_config,
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

    transform = TransformGASF(transform_config)
    raw_dataframe = financial_data_fixtures[feature.name]

    _perform_test(feature, raw_dataframe, transform)


def test_transform_gadf_config():
    with pytest.raises(AssertionError):
        TransformGADF({})


def test_tranform_gadf_x():

    transform_config = {'name': 'gadf', 'image_size': 24}

    feature = FinancialFeature(
        name='close',
        transformation=transform_config,
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

    transform = TransformGADF(transform_config)

    raw_dataframe = financial_data_fixtures[feature.name]

    _perform_test(feature, raw_dataframe, transform)
