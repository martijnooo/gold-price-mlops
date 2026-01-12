from .technical_indicators import add_lag_features, add_moving_averages, add_log_return, add_rolling_volatility
from .cross_market import add_cross_market_features
from .targets import add_target
from .time_features import add_time_features

def build_features(df, include=['technical','cross','time','target']):
    """
    Compose all features modularly.
    include: list of feature types to include
    """
    df_features = df.copy()

    #df_features = df_features.rename(columns={'GC=F': 'Gold_Price'})

    if 'technical' in include:
        df_features = add_lag_features(df_features)
        df_features = add_moving_averages(df_features)
        df_features = add_log_return(df_features)
        df_features = add_rolling_volatility(df_features)

    if 'cross' in include:
        df_features = add_cross_market_features(df_features)

    if 'time' in include:
        df_features = add_time_features(df_features)

    if 'target' in include:
        df_features = add_target(df_features)

    # Drop raw columns used for calculation
    raw_cols = ['Gold_Price','DX-Y.NYB','^TNX','^VIX','^GSPC']
    df_features = df_features.drop(columns=[c for c in raw_cols if c in df_features], errors='ignore')

    # Drop NaNs created by rolling windows or target shift
    df_features = df_features.dropna()

    return df_features
