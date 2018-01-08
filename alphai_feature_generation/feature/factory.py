import pandas_market_calendars as mcal

from alphai_feature_generation.feature import FinancialFeature
from alphai_feature_generation.feature.feature import KEY_EXCHANGE


class FinancialFeatureFactory:

    @staticmethod
    def factory(feature_config_list):
        """
        Build list of financial features from list of complete feature-config dictionaries.
        :param list feature_config_list: list of dictionaries containing feature details.
        :return list: list of FinancialFeature objects
        """
        assert isinstance(feature_config_list, list)

        feature_list = FeatureList()
        for single_feature_dict in feature_config_list:
            feature_list.add_feature(FinancialFeatureFactory.create_feature(single_feature_dict))

        return feature_list

    @staticmethod
    def create_feature(feature_config):
        """
        Build target financial feature from dictionary.
        :param dict feature_config: dictionary containing feature details.
        :return FinancialFeature: FinancialFeature object
        """
        assert isinstance(feature_config, dict)

        return FinancialFeature(
            feature_config['name'],
            feature_config['transformation'],
            feature_config['normalization'],
            feature_config['nbins'],
            feature_config['length'],
            feature_config['ndays'],
            feature_config['resample_minutes'],
            feature_config['start_market_minute'],
            feature_config['is_target'],
            mcal.get_calendar(feature_config[KEY_EXCHANGE]),
            feature_config['local'],
            feature_config.get('classify_per_series'),
            feature_config.get('normalise_per_series')
        )


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

    def __iter__(self):
        return iter(self.feature_list)

    def __getitem__(self, key):
        return self.feature_list[key]

    def __len__(self):
        return len(self.feature_list)
