import alphai_calendars as mcal

from alphai_feature_generation.feature.factory import FeatureList
from alphai_feature_generation.feature.features.gym import GymFeature
from alphai_feature_generation.transformation.gym import GymDataTransformation

sample_gym_calendar = mcal.get_calendar('GYMUK')
sample_gym_feature_list = [
    {
        'name': 'open',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': 5,
        'ndays': 2,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': False,
        GymDataTransformation.KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
    {
        'name': 'close',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': 10,
        'ndays': 5,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': False,
        GymDataTransformation.KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
    {
        'name': 'high',
        'transformation': {'name': 'log-return'},
        'normalization': 'standard',
        'nbins': None,
        'ndays': 10,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': True,
        GymDataTransformation.KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
]
sample_fin_feature_list = FeatureList(
[
    GymFeature(
        name='open',
        transformation={'name': 'value'},
        normalization=None,
        nbins=5,
        ndays=2,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        exchange_calendar=sample_gym_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    GymFeature(
        name='close',
        transformation={'name': 'log-return'},
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        exchange_calendar=sample_gym_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    GymFeature(
        name='high',
        transformation={'name': 'log-return'},
        normalization='standard',
        nbins=None,
        ndays=10,
        resample_minutes=60,
        start_market_minute=1,
        is_target=True,
        exchange_calendar=sample_gym_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
])
