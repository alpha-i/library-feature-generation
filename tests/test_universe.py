from unittest import TestCase
import pandas as pd
import os
import numpy as np
import datetime

from alphai_feature_generation.universe import VolumeUniverseProvider


class TestVolumeUniverseProvider(TestCase):

    def setUp(self):
        self._monthly_30_nyse_False_3 = {
            'update_frequency': 'monthly',
            'ndays_window': 30,
            'exchange': "NYSE",
            'dropna': False,
            'nassets': 3,
        }

    def test_init(self):
        volume_universe = VolumeUniverseProvider(self._monthly_30_nyse_False_3)
        self.assertEquals(self._monthly_30_nyse_False_3['update_frequency'], volume_universe._update_frequency)
        self.assertEquals(self._monthly_30_nyse_False_3['ndays_window'], volume_universe._ndays_window)
        self.assertEquals(self._monthly_30_nyse_False_3['exchange'], volume_universe._exchange)
        self.assertEquals(self._monthly_30_nyse_False_3['dropna'], volume_universe._dropna)
        self.assertEquals(self._monthly_30_nyse_False_3['nassets'], volume_universe._nassets)

    def test_get_historical_universes_simple_two_year(self):
        volume_universe = VolumeUniverseProvider(self._monthly_30_nyse_False_3)
        rng = pd.date_range(start='20070101', end='20090101', freq='T')
        df_close = pd.DataFrame(index=rng, data=np.abs(np.random.randn(3 * len(rng))).reshape([len(rng), 3]),
                                columns=['AAPL', 'MSFT', 'GOOGL'])
        df_volume = pd.DataFrame(index=rng, data=np.ones(3 * len(rng)).reshape([len(rng), 3]),
                                 columns=['AAPL', 'MSFT', 'GOOGL'])

        data_dict = {
            'close': df_close,
            'volume': df_volume
        }

        historical_universes = volume_universe.get_historical_universes(data_dict)
        self.assertEquals(historical_universes.shape, (23, 3))
        self.assertEquals(historical_universes['start_date'][0], datetime.date(2007, 2, 1))
        self.assertEquals(historical_universes['start_date'][22], datetime.date(2008, 12, 1))
        self.assertEquals(historical_universes['end_date'][0], datetime.date(2007, 3, 1))
        self.assertEquals(historical_universes['end_date'][22], datetime.date(2009, 1, 1))
        self.assertEquals(historical_universes['end_date'][22], rng[-1].to_datetime().date())

    def test_get_historical_universes_two_year_with_offset_at_the_beginning(self):
        volume_universe = VolumeUniverseProvider(self._monthly_30_nyse_False_3)
        rng = pd.date_range(start='20061224', end='20090101', freq='T')
        df_close = pd.DataFrame(index=rng, data=np.abs(np.random.randn(3 * len(rng))).reshape([len(rng), 3]),
                                columns=['AAPL', 'MSFT', 'GOOGL'])
        df_volume = pd.DataFrame(index=rng, data=np.ones(3 * len(rng)).reshape([len(rng), 3]),
                                 columns=['AAPL', 'MSFT', 'GOOGL'])

        data_dict = {
            'close': df_close,
            'volume': df_volume
        }

        historical_universes = volume_universe.get_historical_universes(data_dict)
        self.assertEquals(historical_universes.shape, (23, 3))
        self.assertEquals(historical_universes['start_date'][0], datetime.date(2007, 1, 24))
        self.assertEquals(historical_universes['start_date'][22], datetime.date(2008, 11, 24))
        self.assertEquals(historical_universes['end_date'][0], datetime.date(2007, 2, 24))
        self.assertEquals(historical_universes['end_date'][22], datetime.date(2009, 1, 1))
        self.assertEquals(historical_universes['end_date'][22], rng[-1].to_datetime().date())

    def test_get_historical_universes_two_year_with_offset_at_the_end(self):
        volume_universe = VolumeUniverseProvider(self._monthly_30_nyse_False_3)
        rng = pd.date_range(start='20070101', end='20090110', freq='T')
        df_close = pd.DataFrame(index=rng, data=np.abs(np.random.randn(3 * len(rng))).reshape([len(rng), 3]),
                                columns=['AAPL', 'MSFT', 'GOOGL'])
        df_volume = pd.DataFrame(index=rng, data=np.ones(3 * len(rng)).reshape([len(rng), 3]),
                                 columns=['AAPL', 'MSFT', 'GOOGL'])

        data_dict = {
            'close': df_close,
            'volume': df_volume
        }

        historical_universes = volume_universe.get_historical_universes(data_dict)
        self.assertEquals(historical_universes.shape, (23, 3))
        self.assertEquals(historical_universes['start_date'][0], datetime.date(2007, 2, 1))
        self.assertEquals(historical_universes['start_date'][22], datetime.date(2008, 12, 1))
        self.assertEquals(historical_universes['end_date'][0], datetime.date(2007, 3, 1))
        self.assertEquals(historical_universes['end_date'][22], datetime.date(2009, 1, 10))
        self.assertEquals(historical_universes['end_date'][22], rng[-1].to_datetime().date())

