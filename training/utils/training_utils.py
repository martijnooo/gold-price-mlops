import pandas as pd
import numpy as np
import boto3
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true_ret, y_pred_ret, last_prices):
    """Calculates metrics for both Log Returns and Actual Prices."""
    # Return metrics
    rmse_ret = np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))
    
    # Price metrics (Reverting log returns)
    # Price_t = Price_{t-1} * exp(return_t)
    y_true_price = last_prices * np.exp(y_true_ret)
    y_pred_price = last_prices * np.exp(y_pred_ret)
    
    rmse_price = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae_price = mean_absolute_error(y_true_price, y_pred_price)
    
    return {
        "RMSE_Return": rmse_ret,
        "RMSE_Price": rmse_price,
        "MAE_Price": mae_price
    }

def load_features_from_s3(bucket='686699774218-martijn-project', key='data/processed/features.csv'):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df
