from unittest import TestCase

import numpy as np
from alphai_feature_generation.feature import FinancialFeature
from tests.helpers import sample_market_calendar


class TestFeature(TestCase):

    def setUp(self):

        transform_config = {'name': 'log-return'}

        self.feature1 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='min_max',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35
        )

        self.feature2 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='standard',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35
        )

        self.feature3 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='gaussian',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35
        )

        self.feature4 = FinancialFeature(
            name='close',
            transformation=transform_config,
            normalization='robust',
            nbins=10,
            ndays=5,
            resample_minutes=0,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
            local=False,
            length=35
        )

    def test_fit_normalisation(self):

        symbol_data1 = np.random.randn(10000)

        self.feature1.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature1.scaler.data_max_, symbol_data1.max(), rtol=1e-4)
        assert np.isclose(self.feature1.scaler.data_min_, symbol_data1.min(), rtol=1e-4)

        self.feature2.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature2.scaler.mean_, symbol_data1.mean(), rtol=1e-4)
        assert np.isclose(self.feature2.scaler.var_, symbol_data1.var(), rtol=1e-4)

        self.feature3.fit_normalisation(symbol_data=symbol_data1)
        assert self.feature3.scaler.references_.shape == (1000,)
        assert self.feature3.scaler.quantiles_.shape == (1000, 1)

        self.feature4.fit_normalisation(symbol_data=symbol_data1)
        assert np.isclose(self.feature4.scaler.center_, np.median(symbol_data1), rtol=1e-4)
