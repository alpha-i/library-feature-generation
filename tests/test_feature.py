from unittest import TestCase

import numpy as np
from alphai_feature_generation.feature import FinancialFeature
from tests.helpers import sample_market_calendar
from tests.helpers import sample_hourly_ohlcv_data_dict

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
            length=35,
            normalise_per_series=True
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
            length=35,
            normalise_per_series=True
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
            length=35,
            normalise_per_series=True
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
            length=35,
            normalise_per_series=True
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

    def test_apply_normalisation(self):
        data = sample_hourly_ohlcv_data_dict['open']

        for column in data.columns:
            self.feature1.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature1.apply_normalisation(data)
        np.testing.assert_allclose(data.max(), np.asarray([1.,  1.,  1.,  1.,  1.]))
        np.testing.assert_allclose(data.min(), np.asarray([0., 0., 0., 0., 0.]))

        for column in data.columns:
            self.feature2.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature2.apply_normalisation(data)
        np.testing.assert_allclose(data.mean(), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-4)

        for column in data.columns:
            self.feature3.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature3.apply_normalisation(data)
        np.testing.assert_allclose(data.mean(), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-3)

        for column in data.columns:
            self.feature4.fit_normalisation(symbol_data=data[column].values, symbol=column)

        self.feature4.apply_normalisation(data)
        np.testing.assert_allclose(np.median(data, axis=0), np.asarray([0.,  0.,  0.,  0.,  0.]), atol=1e-3)
