import numpy as np
import pandas as pd


class FeatureTransform(object):
    def __init__(self, name, transformation):
        self.name = name
        self.transformation = transformation

    def add_feature(self, raw_data_dict):
        """
        :param raw_data_dict: dict of dataframes
        """
        if self.transformation['name'] == 'log-return':
            raw_data = raw_data_dict[self.name]

            processed_prediction_data_x = np.log(raw_data .pct_change() + 1). \
                replace([np.inf, -np.inf], np.nan)

            # Remove the zeros / nans associated with log return
            processed_prediction_data_x = processed_prediction_data_x.iloc[1:]

        return processed_prediction_data_x



