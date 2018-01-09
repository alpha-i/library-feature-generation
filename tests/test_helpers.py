from alphai_feature_generation.helpers import CalendarUtilities


def test_get_minutes_in_one_trading_day():

    assert CalendarUtilities.get_minutes_in_one_trading_day('JPX') == 360
    assert CalendarUtilities.get_minutes_in_one_trading_day('NYSE') == 390
