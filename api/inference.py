import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import boto3
from io import StringIO
import os
import sys

# Configuration
BUCKET_NAME = '686699774218-martijn-project'
FEATURES_KEY = 'data/processed/features.csv'
PREDICTIONS_KEY = 'data/predictions/predictions.csv'
MODEL_NAME = "Gold_Price_Prediction"
ALIAS = "champion"

# --- Caching ---
_model_cache = None
_cached_run_id = None

def load_features_from_s3(bucket, key):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(obj['Body'], index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

def save_prediction_to_s3(date, price_pred, return_pred):
    """Appends or updates prediction in S3 CSV."""
    s3 = boto3.client('s3')
    
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=PREDICTIONS_KEY)
        df_existing = pd.read_csv(obj['Body'], index_col=0)
        df_existing.index = pd.to_datetime(df_existing.index)
    except s3.exceptions.NoSuchKey:
        df_existing = pd.DataFrame(columns=['Actual_Price', 'Predicted_Price', 'Predicted_Log_Return', 'Type'])
    
    # Create new row
    # Note: Actual_Price is NaN for live prediction
    new_row = pd.DataFrame({
        'Actual_Price': [np.nan],
        'Predicted_Price': [price_pred], 
        'Predicted_Log_Return': [return_pred],
        'Type': ['Live']
    }, index=[pd.to_datetime(date)])
    
    # Check if date exists
    if date in df_existing.index:
        # Update existing row
        df_existing.update(new_row)
        # If columns were missing in existing, update might not add them, so let's use combine_first or manual assignment
        # Safer: drop and append to ensure schema consistency if simplistic
        df_existing = df_existing.drop(index=date)
        df_updated = pd.concat([df_existing, new_row])
    else:
        # Append
        df_updated = pd.concat([df_existing, new_row])
    
    # Sort just in case
    df_updated = df_updated.sort_index()
    
    # Upload
    csv_buffer = StringIO()
    df_updated.to_csv(csv_buffer)
    s3.put_object(Bucket=BUCKET_NAME, Key=PREDICTIONS_KEY, Body=csv_buffer.getvalue())
    print(f"Saved prediction for {date} to S3 ({PREDICTIONS_KEY}).")

def run_inference():
    """
    Runs the full inference pipeline:
    1. Load features from S3.
    2. Load 'champion' model from MLflow.
    3. Predict next day's return/price.
    Returns a dictionary with the prediction details.
    """
    # 1. Load Features 
    print(f"Loading features from s3://{BUCKET_NAME}/{FEATURES_KEY}...")
    df_features = load_features_from_s3(BUCKET_NAME, FEATURES_KEY)
    print(f"Features loaded: {df_features.shape}")
    # 2. Resolve Champion Model and Type
    import time
    start_time = time.time()
    
    print(f"Connecting to MLflow Tracking Server at: {mlflow.get_tracking_uri()}")
    client = mlflow.MlflowClient()
    print(f"Resolving model alias '{ALIAS}' for model '{MODEL_NAME}'...")
    
    try:
        t0 = time.time()
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        print(f"✅ Alias resolved in {time.time() - t0:.2f}s (Version {model_version.version})")
        
        t1 = time.time()
        run_id = model_version.run_id
        run = client.get_run(run_id)
        run_name = run.data.tags.get("mlflow.runName", "")
        print(f"✅ Run info fetched in {time.time() - t1:.2f}s (Run: {run_name})")
    except Exception as e:
        error_msg = f"Error resolving champion model: {e}"
        print(error_msg)
        return {"error": error_msg}

    # 3. Load & Predict based on Type
    global _model_cache, _cached_run_id
    
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    last_row = df_features.iloc[[-1]]
    last_date = last_row.index[0]
    last_price_l0 = last_row['Gold_Price_L0'].values[0]
    
    # --- Cache Check ---
    if _model_cache is not None and _cached_run_id == run_id:
        print(f"♻️ Using Cached Model (Run ID: {run_id})")
        model = _model_cache
        t2 = time.time() # For consistent logging
    else:
        print(f"Predicting for {last_date}. Loading model artifacts from S3 (cross-region download)...")
        t2 = time.time()
        
        if "LSTM" in run_name:
            from sklearn.preprocessing import StandardScaler
            print(f"--- LSTM Branch Started ---")
            tl0 = time.time()
            model = mlflow.keras.load_model(model_uri)
            print(f"✅ Model weights loaded in {time.time() - tl0:.2f}s")
        elif "ARIMA" in run_name:
            model = mlflow.sklearn.load_model(model_uri)
        else:
            model = mlflow.sklearn.load_model(model_uri)
        
        # Update Cache
        _model_cache = model
        _cached_run_id = run_id

    # --- Preprocessing & Prediction ---
    if "LSTM" in run_name:
        from sklearn.preprocessing import StandardScaler
        window_size = int(run.data.params.get("window_size", 50))
        if len(df_features) < window_size:
            return {"error": "Not enough data for LSTM window."}
        
        tl1 = time.time()
        FEATURE_TARGET = "Target_Return"
        X_raw = df_features.drop(columns=[FEATURE_TARGET], errors='ignore')
        scaler_x = StandardScaler()
        X_scaled = scaler_x.fit_transform(X_raw)
        last_window = X_scaled[-window_size:] 
        X_seq = np.array([last_window]) 
        
        pred_scaled = model.predict(X_seq, verbose=0)
        
        if FEATURE_TARGET in df_features.columns:
            y_raw = df_features[FEATURE_TARGET].values.reshape(-1, 1)
            scaler_y = StandardScaler()
            scaler_y.fit(y_raw)
            pred_log_return = scaler_y.inverse_transform(pred_scaled).flatten()[0]
        else:
            return {"error": "Target column missing for scaler fitting."}

    elif "ARIMA" in run_name:
        # ARIMA Handling
        model = mlflow.sklearn.load_model(model_uri)
        predictions = model.predict(n_periods=1)
        pred_log_return = predictions[0]

    else:
        # Classical Handling
        model = mlflow.sklearn.load_model(model_uri)
        X_new = last_row.drop(columns=['Target_Return'], errors='ignore')
        pred_log_return = model.predict(X_new)[0]

    print(f"✅ Model loaded and prediction complete in {time.time() - t2:.2f}s")
    
    # Convert to Price
    pred_price = last_price_l0 * np.exp(pred_log_return)
    
    print(f"Total Inference Pipeline Time: {time.time() - start_time:.2f}s")
    print(f"Predicted Return: {pred_log_return:.5f}, Predicted Price: {pred_price:.2f}")
    
    # 5. Save (DISABLED for ad-hoc inference per user request)
    # save_prediction_to_s3(last_date, pred_price, pred_log_return)
    
    return {
        "date": str(last_date),
        "predicted_price": float(pred_price),
        "predicted_return": float(pred_log_return),
        "model_used": run_name
    }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    tracking_uri = os.getenv("TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        
    run_inference()
