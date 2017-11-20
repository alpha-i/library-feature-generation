from integration_test.synthetic_data import create_synthetic_ohlcv
import yaml
from feature import FeatureTransform
from transformation import DataTransformation

config_file = '../config/test.yml'

with open(config_file) as stream:
    configuration = yaml.load(stream)

transformations = DataTransformation(configuration)

fake_data = create_synthetic_ohlcv(n_sin_series=4, start_date='20100101', end_date='20100131', add_nan=False,
                                   add_zero_and_linear=False)

for feature in transformations.features:
    print('boo')
    print(feature.name)
    fake_data[feature.name + '3'] = feature.add_feature(fake_data)
