import pytest

import alphai_calendars as mcal
from copy import deepcopy

from alphai_feature_generation.feature.factory import FinancialFeatureFactory
from alphai_feature_generation.transformation import FinancialDataTransformation
from tests.feature.features.financial.helpers import sample_fin_feature_factory_list, sample_fin_feature_list


def test_features_factory_successful_call():

    calendar = mcal.get_calendar('NYSE')
    factory = FinancialFeatureFactory(calendar)
    feature_list = factory.create_from_list(sample_fin_feature_factory_list)

    for feature in feature_list:
        expected_feature = _get_feature_by_name(feature.name, sample_fin_feature_list)
        assert feature.name == expected_feature.name
        assert feature.transformation.config == expected_feature.transformation.config
        assert feature.normalization == expected_feature.normalization
        assert feature.nbins == expected_feature.nbins
        assert feature.ndays == expected_feature.ndays
        assert feature.resample_minutes == expected_feature.resample_minutes
        assert feature.start_market_minute == expected_feature.start_market_minute
        assert feature.is_target == expected_feature.is_target


def test_features_factory_too_many_targets():

    calendar = mcal.get_calendar('NYSE')
    factory = FinancialFeatureFactory(calendar)

    feature_list = deepcopy(sample_fin_feature_factory_list)

    feature_list.append({
        'name': 'close',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': 10,
        'ndays': 5,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': True,
        FinancialDataTransformation.KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    })

    assert len(feature_list) == 4

    with pytest.raises(AssertionError):
        factory.create_from_list(feature_list)


def test_single_features_factory_wrong_keys():
    feature_dict = {
        'name': 'feature1',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': 15,
        'ndays': 5,
        'wrong': 1,
        'is_target': False,
    }

    calendar = mcal.get_calendar('NYSE')
    factory = FinancialFeatureFactory(calendar)
    with pytest.raises(KeyError):
        factory.create_feature(feature_dict)


def test_features_factory_wrong_input_type():
    feature_list = {}

    calendar = mcal.get_calendar('NYSE')
    factory = FinancialFeatureFactory(calendar)
    with pytest.raises(AssertionError):
        factory.create_from_list(feature_list)


def _get_feature_by_name(name, feature_list):
    for feature in feature_list:
        if feature.name == name:
            return feature
    raise ValueError


def test_get_feature_names():
    assert set(sample_fin_feature_list.get_names()) == {'open', 'close', 'high'}


def test_get_feature_max_ndays():
    assert sample_fin_feature_list.get_max_ndays() == 10
