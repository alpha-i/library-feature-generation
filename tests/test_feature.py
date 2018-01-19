from unittest import TestCase

import numpy as np
import pandas as pd
import pytest

from alphai_feature_generation.feature import KEY_EXCHANGE
from alphai_feature_generation.transformation import (
    FinancialDataTransformation,
)
from tests.helpers import (
    sample_hourly_ohlcv_data_dict,
    sample_fin_data_transf_feature_factory_list_nobins,
    sample_fin_data_transf_feature_factory_list_bins,
    sample_fin_data_transf_feature_fixed_length,
    sample_historical_universes,
    TEST_ARRAY,
)

SAMPLE_TRAIN_LABELS = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))
SAMPLE_PREDICT_LABELS = SAMPLE_TRAIN_LABELS[:, int(0.5 * SAMPLE_TRAIN_LABELS.shape[1])]

SAMPLE_TRAIN_LABELS = {'open': SAMPLE_TRAIN_LABELS}
SAMPLE_PREDICT_LABELS = {'open': SAMPLE_PREDICT_LABELS}


class TestFeature(TestCase):

    def setUp(self):
        configuration_nobins = {
            'feature_config_list': sample_fin_data_transf_feature_factory_list_nobins,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            KEY_EXCHANGE: 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta_ndays': 5,
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'nassets': 5,
            'local': False,
            'classify_per_series': False,
            'normalise_per_series': False,
            'fill_limit': 0
        }

        self.transformation_without_bins = FinancialDataTransformation(configuration_nobins)

        configuration_bins = {
            'feature_config_list': sample_fin_data_transf_feature_factory_list_bins,
            'features_ndays': 2,
            'features_resample_minutes': 60,
            'features_start_market_minute': 1,
            KEY_EXCHANGE: 'NYSE',
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 30,
            'target_delta_ndays': 5,
            'target_market_minute': 30,
            'n_classification_bins': 5,
            'nassets': 5,
            'local': False,
            'classify_per_series': False,
            'normalise_per_series': False,
            'fill_limit': 0
        }

        self.transformation_with_bins = FinancialDataTransformation(configuration_bins)
