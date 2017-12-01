from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

FINANCIAL_FEATURE_TRANSFORMATIONS = ['value', 'log-return', 'stochastic_k', 'ewma', 'ker', 'gasf', 'gadf', 'mtf']
FINANCIAL_FEATURE_NORMALIZATIONS = [None, 'robust', 'min_max', 'standard', 'gaussian']
FINANCIAL_FEATURE_KEYS = [
    'transformation',
    'normalization',
    'nbins',
    'ndays',
    'resample_minutes',
    'start_market_minute',
    'is_target',
]
NORMALIZATION_TYPE_MAP = {
    None: type(None),
    'robust': RobustScaler,
    'min_max': MinMaxScaler,
    'standard': StandardScaler
}

DATA_TRANFORMATION_KEYS = ['features_dict']
MINUTES_IN_TRADING_DAY = 390
MARKET_DAYS_SEARCH_MULTIPLIER = 2
MIN_MARKET_DAYS_SEARCH = 10
