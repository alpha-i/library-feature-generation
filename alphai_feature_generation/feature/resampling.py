from alphai_feature_generation.cleaning import resample_data_frame

RESAMPLE_FUNCTION_SUM = 'sum'
RESAMPLE_FUNCTION_LAST = 'last'

FEATURE_NAME_VOLUME = 'volume'


RESAMPLE_FUNCTION_MAP = {
    FEATURE_NAME_VOLUME: RESAMPLE_FUNCTION_SUM
}


class ResamplingStrategy:

    @staticmethod
    def resample(feature, data):
        """

        :param feature: the feature object
        :type feature: Feature

        :param data: the data to be resampled
        :type data: pd.DataFrame

        :return: the resampled data
        :rtype pd.DataFrame
        """

        if not feature.local and feature.resample_minutes > 0:
            resample_rule = str(feature.resample_minutes) + 'T'
            sampling_function = RESAMPLE_FUNCTION_MAP.get(feature.name, RESAMPLE_FUNCTION_LAST)
            data = resample_data_frame(data, resample_rule, sampling_function=sampling_function)

        return data
