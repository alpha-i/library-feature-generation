import pytest

from alphai_feature_generation.feature.factory import FinancialFeatureFactory
from tests.helpers import sample_fin_feature_factory_list, sample_fin_feature_list


def test_financial_features_factory_successful_call():
    feature_list = FinancialFeatureFactory.factory(sample_fin_feature_factory_list)

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


def test_single_financial_features_factory_wrong_keys():
    feature_dict = {
        'name': 'feature1',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': 15,
        'ndays': 5,
        'wrong': 1,
        'is_target': False,
    }
    with pytest.raises(KeyError):
        FinancialFeatureFactory.create_feature(feature_dict)


def test_financial_features_factory_wrong_input_type():
    feature_list = {}
    with pytest.raises(AssertionError):
        FinancialFeatureFactory.factory(feature_list)


def _get_feature_by_name(name, feature_list):
    for feature in feature_list:
        if feature.name == name:
            return feature
    raise ValueError


def test_get_feature_names():
    assert set(sample_fin_feature_list.get_names()) == {'open', 'close', 'high'}


def test_get_feature_max_ndays():
    assert sample_fin_feature_list.get_max_ndays() == 10
