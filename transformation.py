from feature import FeatureTransform


class DataTransformation(object):
    def __init__(self, configuration):

        self.features = financial_features_factory(configuration['feature_config_list'])


def financial_features_factory(feature_config_list):
    """
    Build target financial feature from dictionary.
    :param dict feature_config_list:lost of dictionaries containing feature details.
    :return FinancialFeature: list FeatureTransform object
    """
    assert isinstance(feature_config_list, list)

    feature_list = []
    for single_feature_dict in feature_config_list:
        feature_list.append(FeatureTransform(
        single_feature_dict['name'],
        single_feature_dict['transformation']))

    return feature_list