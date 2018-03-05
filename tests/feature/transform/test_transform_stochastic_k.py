import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal

from alphai_feature_generation.feature.features.financial import FinancialFeature
from alphai_feature_generation.feature.transform import TransformStochasticK

from tests.helpers import ASSERT_NDECIMALS
from tests.feature.features.financial.helpers import sample_market_calendar
from tests.transformation.financial.helpers import sample_hourly_ohlcv_data_dict


def test_transform_stochastic_k_x():

    transform_config = {'name': 'stochastic_k'}

    feature = FinancialFeature(
            name='high',
            transformation=transform_config,
            normalization=None,
            nbins=10,
            ndays=10,
            resample_minutes=0,
            start_market_minute=150,
            is_target=True,
            calendar=sample_market_calendar,
            local=False,
            length=10
        )

    transform = TransformStochasticK(transform_config)

    raw_dataframe = sample_hourly_ohlcv_data_dict[feature.name]

    processed_prediction_data_x = transform.transform_x(feature, raw_dataframe)
    columns = raw_dataframe.columns

    expected_result = ((raw_dataframe.iloc[-1] - raw_dataframe.min()) /
                       (raw_dataframe.max() - raw_dataframe.min())) * 100.

    expected_result = np.expand_dims(expected_result, axis=0)
    expected_result = pd.DataFrame(expected_result, columns=columns)

    assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)


