import alphai_calendars as mcal

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.factory import FeatureList
from alphai_feature_generation.transformation.financial import FinancialDataTransformation

KEY_EXCHANGE = FinancialDataTransformation.KEY_EXCHANGE

sample_market_calendar = mcal.get_calendar('NYSE')
sample_fin_feature_factory_list = [
    {
        'name': 'open',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': 5,
        'ndays': 2,
        'resample_minutes': 60,
        'start_market_minute': 1,
        'is_target': False,
        KEY_EXCHANGE: 'NYSE',
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
        KEY_EXCHANGE: 'NYSE',
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
        KEY_EXCHANGE: 'NYSE',
        'local': True,
        'length': 10
    },
]
sample_fin_feature_list = FeatureList(
[
    FinancialFeature(
        name='open',
        transformation={'name': 'value'},
        normalization=None,
        nbins=5,
        ndays=2,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    FinancialFeature(
        name='close',
        transformation={'name': 'log-return'},
        normalization=None,
        nbins=10,
        ndays=5,
        resample_minutes=60,
        start_market_minute=1,
        is_target=False,
        calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
    FinancialFeature(
        name='high',
        transformation={'name': 'log-return'},
        normalization='standard',
        nbins=None,
        ndays=10,
        resample_minutes=60,
        start_market_minute=1,
        is_target=True,
        calendar=sample_market_calendar,
        classify_per_series=False,
        normalise_per_series=False,
        local=True,
        length=10
    ),
])
