from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from pyts.transformation import GASF, GADF, MTF


class AbstractTransform(metaclass=ABCMeta):

    def __init__(self, configuration):
        self.config = configuration
        self.validate_config()

    @abstractmethod
    def validate_config(self):
        raise NotImplemented()

    @abstractmethod
    def transform_x(self, feature, data):
        raise NotImplemented()

    def transform_y(self, feature, data, reference_data):
        return data

    @property
    def name(self):
        return self.config['name']


class TransformLogReturn(AbstractTransform):

    def validate_config(self):
        return True

    def transform_x(self, feature, data):
        data = np.log(data.pct_change() + 1).replace([np.inf, -np.inf], np.nan)

        # Remove the zeros / nans associated with log return
        if feature.local:
            data = data.iloc[1:]

        return data

    def transform_y(self, feature, data, reference_data):

        prediction_reference_ratio = data / reference_data
        return np.log(prediction_reference_ratio).replace([np.inf, -np.inf], np.nan)


class TransformVolatility(AbstractTransform):

    def validate_config(self):
        assert 'window' in self.config

    def transform_x(self, feature, data):
        data = np.log(data.pct_change() + 1).replace([np.inf, -np.inf], np.nan)
        data = data.rolling(window=self.config['window'], center=False).std()

        # Remove the zeros / nans associated with log return
        if feature.local:
            data = data.iloc[1:]

        return data

    @property
    def window(self):
        return self.config['window']


class TransformStochasticK(AbstractTransform):

    def validate_config(self):
        return True

    def transform_x(self, feature, data):
        columns = data.columns

        data = ((data.iloc[-1] - data.min()) / (data.max() - data.min())) * 100.
        data = np.expand_dims(data, axis=0)

        return pd.DataFrame(data, columns=columns)


class TransformEWMA(AbstractTransform):

    def validate_config(self):
        assert 'halflife' in self.config

    def transform_x(self, feature, data):
        return data.ewm(halflife=self.config['halflife']).mean()

    @property
    def halflife(self):
        return self.config['halflife']


class TransformKer(AbstractTransform):

    def validate_config(self):
        assert 'lag' in self.config

    def transform_x(self, feature, data):

        direction = data.diff(self.config['lag']).abs()
        volatility = data.diff().abs().rolling(window=self.config['lag']).sum()

        direction.dropna(axis=0, inplace=True)
        volatility.dropna(axis=0, inplace=True)

        assert direction.shape == volatility.shape, ' direction and volatility need same shape in KER'

        data = direction / volatility
        data.dropna(axis=0, inplace=True)

        return data

    @property
    def lag(self):
        return self.config['lag']


class TransformGASF(AbstractTransform):

    def validate_config(self):
        assert 'image_size' in self.config

    def transform_x(self, feature, data):
        columns = data.columns
        gasf = GASF(image_size=self.image_size, overlapping=False, scale='-1')
        data = gasf.transform(data.values.T)
        data = data.reshape(data.shape[0], -1)
        data = pd.DataFrame(data.T, columns=columns)

        return data

    @property
    def image_size(self):
        return self.config['image_size']


class TransformGADF(AbstractTransform):

    def validate_config(self):
        assert 'image_size' in self.config

    def transform_x(self, feature, data):
        columns = data.columns
        gadf = GADF(image_size=self.image_size, overlapping=False, scale='-1')
        data = gadf.transform(data.values.T)
        data = data.reshape(data.shape[0], -1)
        data = pd.DataFrame(data.T, columns=columns)

        return data

    @property
    def image_size(self):
        return self.config['image_size']


class TransformMTF(AbstractTransform):

    def validate_config(self):
        assert 'image_size' in self.config

    def transform_x(self, feature, data):
        columns = data.columns
        mtf = MTF(image_size=self.image_size, n_bins=7, quantiles='empirical', overlapping=False)
        data = mtf.transform(data.values.T)
        data = data.reshape(data.shape[0], -1)
        data = pd.DataFrame(data.T, columns=columns)

        return data

    @property
    def image_size(self):
        return self.config['image_size']


class TransformValue(AbstractTransform):

    def validate_config(self):
        return True

    def transform_x(self, feature, data):
        return data


FEATURE_TRANSFORMATIONS_MAPPING = {
    'value': TransformValue,
    'log-return': TransformLogReturn,
    'stochastic_k': TransformStochasticK,
    'ewma': TransformEWMA,
    'ker': TransformKer,
    'gasf': TransformGASF,
    'gadf': TransformGADF,
    'mtf': TransformMTF,
    'volatility': TransformVolatility,
}


class Transformation:

    def __new__(cls, transform_config):
        name = transform_config['name']

        try:
            transformation = FEATURE_TRANSFORMATIONS_MAPPING[name]

            return transformation(transform_config)
        except KeyError:
            raise NotImplementedError('Unknown transformation: {}'.format(name))
