import numpy as np

def add_cross_market_features(df):
    """Adds DXY, 10Y yield, VIX, SPX log returns"""
    if 'DX-Y.NYB' in df:
        df['DXY_Level'] = df['DX-Y.NYB']
    if '^TNX' in df:
        df['10Y_Yield_Level'] = df['^TNX']
    if '^VIX' in df:
        df['VIX_Level'] = df['^VIX']
    if '^GSPC' in df:
        df['SPX_Log_Return'] = np.log(df['^GSPC'] / df['^GSPC'].shift(1))
    return df
