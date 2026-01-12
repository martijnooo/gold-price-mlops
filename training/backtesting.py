import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
from io import StringIO
import os

# --- Configuration ---
BUCKET_NAME = '686699774218-martijn-project'
FEATURES_KEY = 'data/processed/features.csv'
PREDICTIONS_KEY = 'data/predictions/predictions.csv'
MODEL_NAME = "Gold_Price_Prediction"
ALIAS = "champion"
FEATURE_TARGET = "Target_Return"
TRAIN_CUTOFF_DATE = pd.to_datetime("2025-11-01")

def load_features_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def main():
    print(f"Loading features from s3://{BUCKET_NAME}/{FEATURES_KEY}...")
    df_features = load_features_from_s3(BUCKET_NAME, FEATURES_KEY)
    
    # Filter for Backtest set logic
    # We define the indices we want to predict
    test_indices = df_features.index >= TRAIN_CUTOFF_DATE
    test_df = df_features[test_indices].copy()
    
    print(f"Backtesting on {len(test_df)} records starting from {TRAIN_CUTOFF_DATE}.")

    if test_df.empty:
        print("No data found for backtesting period.")
        return

    # 1. Resolve Champion Model and Type
    client = mlflow.MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        run_id = model_version.run_id
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", "")
        print(f"Champion Run: {run_name} (ID: {run_id})")
    except Exception as e:
        print(f"Error resolving champion model: {e}")
        return

    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    print(f"Loading model from {model_uri}...")

    y_pred_ret = []

    if "LSTM" in run_name:
        # LSTM Handling
        from sklearn.preprocessing import StandardScaler
        model = mlflow.keras.load_model(model_uri)
        
        # Prepare data using full history to preserve windows
        window_size = int(run.data.params.get("window_size", 50))
        
        FEATURE_TARGET = "Target_Return"
        X_raw = df_features.drop(columns=[FEATURE_TARGET], errors='ignore')
        y_raw = df_features[FEATURE_TARGET].values.reshape(-1, 1)

        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X_raw)
        
        scaler_y = StandardScaler()
        scaler_y.fit(y_raw)

        # Generate sequences
        # We want to generate predictions for rows where index >= CUTOFF
        # The LSTM trained such that X_seq[i] -> predicts y[i+window_size]
        # We need to find the integer indices in df_features corresponding to test_df
        
        full_predictions = []
        # Naive approach: Predict ALL valid sequences and then slice
        # X_seq generation
        X_seq = []
        # We can only generate sequences starting from window_size
        for i in range(len(X_scaled) - window_size):
            X_seq.append(X_scaled[i:i+window_size])
        
        X_seq = np.array(X_seq)
        
        preds_scaled = model.predict(X_seq)
        preds_ret_full = scaler_y.inverse_transform(preds_scaled).flatten()
        
        # Align predictions with dataframe
        # The first prediction corresponds to index window_size
        # So preds_ret_full[0] corresponds to df_features.iloc[window_size]
        
        pred_series = pd.Series(preds_ret_full, index=df_features.index[window_size:])
        
        # Subset to test set
        y_pred_ret = pred_series[pred_series.index >= TRAIN_CUTOFF_DATE].values
        
        # Align test_df if lengths differ (e.g. if window_size cuts into test set start - unlikely here as test is recent)
        # But if test set starts BEFORE window_size (impossible given cutoff is 2025 and data starts 2001) we are fine.
        
        # Truncate test_df to match prediction availability
        common_idx = pred_series.index.intersection(test_df.index)
        test_df = test_df.loc[common_idx]
        y_pred_ret = pred_series.loc[common_idx].values

    elif "ARIMA" in run_name:
        # ARIMA Handling
        model = mlflow.sklearn.load_model(model_uri)
        # Predict n steps ahead
        n_periods = len(test_df)
        print(f"Forecasting {n_periods} steps with ARIMA...")
        y_pred_ret = model.predict(n_periods=n_periods)
        
        # ARIMA predicts strictly sequentially from end of training.
        # Ensure test_df is exactly the period following training.
        # Training ended at TRAIN_CUTOFF_DATE. Test starts at TRAIN_CUTOFF_DATE.
        # This matches.

    else:
        # Classical Handling
        model = mlflow.sklearn.load_model(model_uri)
        X_test = test_df.drop(columns=[FEATURE_TARGET], errors='ignore')
        y_pred_ret = model.predict(X_test)

    # Post-process
    # Price_t = Price_{t-1} * exp(return_t)
    test_df['Reserved_L0_Price'] = test_df['Gold_Price_L0']
    test_df['Predicted_Log_Return'] = y_pred_ret
    test_df['Predicted_Price'] = test_df['Reserved_L0_Price'] * np.exp(test_df['Predicted_Log_Return'])
    
    # Calculate Actual Price for comparison
    if FEATURE_TARGET in test_df.columns:
        test_df['Actual_Price'] = test_df['Reserved_L0_Price'] * np.exp(test_df[FEATURE_TARGET])
    else:
        test_df['Actual_Price'] = np.nan

    # Save to S3
    test_df['Type'] = 'Backtest'
    cols_to_save = ['Actual_Price', 'Predicted_Price', 'Predicted_Log_Return', 'Type']
    
    csv_buffer = StringIO()
    test_df[cols_to_save].to_csv(csv_buffer)
    
    s3 = boto3.client('s3')
    s3.put_object(Bucket=BUCKET_NAME, Key=PREDICTIONS_KEY, Body=csv_buffer.getvalue())
    print(f"Backtest predictions uploaded to s3://{BUCKET_NAME}/{PREDICTIONS_KEY}")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure tracking URI is set if needed for loading model
    tracking_uri = os.getenv("TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        
    main()
