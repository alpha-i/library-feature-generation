from datetime import timedelta

from tests.helpers import build_dict_of_dataframe

gym_data_fixtures = build_dict_of_dataframe('gym_data_dict')


sample_features_fixed_length = [
    {
        'name': 'hour',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'value'},
        'is_target': False,
    },
    {
        'name': 'hour',
        'normalization': 'standard',
        'resolution': 15,
        'length': 2,
        'transformation': {'name': 'ewma', 'halflife': 6},
        'is_target': False,
    },
    {
        'name': 'number_people',
        'normalization': 'standard',
        'resolution': 150,
        'length': 2,
        'transformation': {'name': 'value'},
        'is_target': True
    },
]
sample_feature_list = [
    {
        'name': 'hour',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': None,
        'length': 5,
        'is_target': False,
        'local': True
    },
    {
        'name': 'temperature',
        'transformation': {'name': 'value'},
        'normalization': None,
        'nbins': None,
        'is_target': False,
        'local': False,
        'length': 5,
    },
    {
        'name': 'number_people',
        'transformation': {'name': 'value'},
        'normalization': 'standard',
        'nbins': 5,
        'is_target': True,
        'local': False,
        'length': 5,
    },
]


def load_preset_config(expected_n_symbols, iteration=0):
    config = {
        'feature_config_list': sample_feature_list,
        'features_ndays': 2,
        'features_resample_minutes': 60,
        'features_start_market_minute': 1,
        'calendar_name': 'GYMUK',
        'prediction_frequency_ndays': 1,
        'prediction_market_minute': 60,
        'target_delta': timedelta(days=5),
        'target_market_minute': 60,
        'n_classification_bins': 5,
        'n_assets': expected_n_symbols,
        'local': False,
        'classify_per_series': False,
        'normalise_per_series': False,
        'fill_limit': 0
    }

    specific_cases = [
        {},
        {'classify_per_series': True, 'normalise_per_series': True},
        {'feature_config_list': sample_features_fixed_length}
    ]

    try:
        updated_config = specific_cases[iteration]
        config.update(updated_config)
    except KeyError:
        raise ValueError('Requested configuration not implemented')

    return config
