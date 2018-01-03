from alphai_feature_generation.utils import get_minutes_in_one_trading_day


def test_get_minutes_in_one_trading_day():
    assert get_minutes_in_one_trading_day('JPX') == 360
    assert get_minutes_in_one_trading_day('NYSE') == 390
