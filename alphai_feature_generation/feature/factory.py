from abc import abstractmethod, ABCMeta

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.features.gym import GymFeature

DEFAULT_TRANSFORMATION = {'name': 'value'}


class AbstractFeatureFactory(metaclass=ABCMeta):

    def __init__(self, calendar):
        self._calendar = calendar

    @abstractmethod
    def get_feature_class(self):
        return NotImplemented

    def create_from_list(self, feature_config_list):
        """
        Build list of financial features from list of complete feature-config dictionaries.
        :param list feature_config_list: list of dictionaries containing feature details.
        :return list: list of FinancialFeature objects
        """
        assert isinstance(feature_config_list, list)
        self._assert_single_target(feature_config_list)
        feature_list = FeatureList()
        for single_feature_dict in feature_config_list:
            feature_list.add_feature(self.create_feature(single_feature_dict))

        return feature_list

    def _assert_single_target(self, feature_config_list):
        """
        Check the list of feature to ensure only one is marked as target
        :param feature_config_list:
        :return:
        """
        targeted_features = [feature for feature in feature_config_list if feature['is_target']]

        assert len(targeted_features) == 1, "Only one feature can be a target. {} found".format(targeted_features)

    def create_feature(self, feature_config):
        """
        Build target financial feature from dictionary.
        :param dict feature_config: dictionary containing feature details.
        :return FinancialFeature: FinancialFeature object
        """
        assert isinstance(feature_config, dict)

        transform = feature_config.get('transformation', DEFAULT_TRANSFORMATION)
        feature_class = self.get_feature_class()

        return feature_class(
            feature_config['name'],
            transform,
            feature_config['normalization'],
            feature_config['nbins'],
            feature_config['length'],
            feature_config['ndays'],
            feature_config['resample_minutes'],
            feature_config['start_market_minute'],
            feature_config['is_target'],
            self._calendar,
            feature_config['local'],
            feature_config.get('classify_per_series'),
            feature_config.get('normalise_per_series')
        )


class FinancialFeatureFactory(AbstractFeatureFactory):

    def get_feature_class(self):
        return FinancialFeature


class GymFeatureFactory(AbstractFeatureFactory):

    def get_feature_class(self):
        return GymFeature


class FeatureList:

    def __init__(self, feature_list=None):
        if isinstance(feature_list, list):
            self.feature_list = feature_list
        else:
            self.feature_list = []

    def add_feature(self, feature):
        self.feature_list.append(feature)

    def get_names(self):
        """
        Return unique names of feature list
        :return list: list of strings
        """
        return list(set([feature.name for feature in self.feature_list]))

    def get_max_ndays(self):
        """
        Return max ndays of feature list
        :return int: max ndays of feature list
        """
        return max([feature.ndays for feature in self.feature_list])

    def get_target_feature(self):
        """
        Return the target feature in self.features

        :return FinancialFeature: target feature
        """
        for feature in self.feature_list:
            if feature.is_target:
                return feature

    def __iter__(self):
        return iter(self.feature_list)

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)
