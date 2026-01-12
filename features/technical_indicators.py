import pandas as pd
import numpy as np

def add_lag_features(df, column='Gold_Price', lags=[0,5,20]):
    for lag in lags:
        df[f'{column}_L{lag}'] = df[column].shift(lag)
    return df

def add_moving_averages(df, column='Gold_Price', windows=[10,50]):
    for w in windows:
        df[f'{column}_{w}D_MA'] = df[column].rolling(w).mean()
    return df

def add_log_return(df, column='Gold_Price', new_col='Gold_Log_Return'):
    df[new_col] = np.log(df[column] / df[column].shift(1))
    return df

def add_rolling_volatility(df, column='Gold_Log_Return', window=20, new_col=None):
    new_col = new_col or f'{column}_{window}D_Vol'
    df[new_col] = df[column].rolling(window).std()
    return df
