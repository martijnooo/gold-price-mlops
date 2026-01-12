import pandas as pd
import mlflow
import os
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Import modules
from utils.training_utils import load_features_from_s3
from models.classical import train_classical_models
from models.arima import train_arima
from models.lstm import train_lstm, prepare_lstm_data

# Load env variables
load_dotenv()
EXPERIMENT_NAME = "Gold_Price_Prediction"

# Setup MLflow
mlflow.set_experiment(EXPERIMENT_NAME)
tracking_uri = os.getenv("TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

if __name__ == "__main__":
    
    # 1. Load Features (Pre-calculated by ingest.py)
    print("Loading features from S3...")
    df_features = load_features_from_s3()
    
    # Log dataset metadata
    with mlflow.start_run(run_name="Gold_Price_Main_Run") as run:
        # Log dataset info
        mlflow.log_param("data_shape", df_features.shape)
        
        # Split logic
        TRAIN_CUTOFF_DATE = pd.to_datetime("2025-11-01") 
        train_df = df_features[df_features.index < TRAIN_CUTOFF_DATE]
        
        print(f"Training Data (Up to {TRAIN_CUTOFF_DATE}): {train_df.shape}")

        # 1. Classical Training
        train_classical_models(train_df)
        print("Trained Classical Models")

        # 2. ARIMA
        train_arima(train_df)
        print("Trained ARIMA")

        # 3. LSTM
        X_s, y_s, s_y = prepare_lstm_data(train_df)
        train_lstm(train_df, X_s, y_s, s_y)
        print("Trained LSTM")
        
        # --- CHAMPION SELECTION ---
        print("Selecting Champion Model...")
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        
        # Find all child runs of the current run
        query = f"tags.mlflow.parentRunId = '{run.info.run_id}'"
        child_runs = client.search_runs(experiment.experiment_id, filter_string=query)
        
        best_run = None
        best_rmse = float('inf')
        
        for r in child_runs:
            metrics = r.data.metrics
            if "RMSE_Price" in metrics:
                rmse = metrics["RMSE_Return"]
                print(f"Run {r.info.run_name} (ID: {r.info.run_id}) - RMSE_Price: {rmse:.4f}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_run = r
        
        if best_run:
            print(f"🏆 Best Run: {best_run.info.run_name} with RMSE: {best_rmse:.4f}")
            
            run_name = best_run.info.run_name
            if "Classical" in run_name:
                artifact_path = f"{run_name.split('_')[1]}_model"
            elif "ARIMA" in run_name:
                artifact_path = "arima_model"
            elif "LSTM" in run_name:
                artifact_path = "lstm_gold_model"
            else:
                artifact_path = "model"
            
            model_uri = f"runs:/{best_run.info.run_id}/{artifact_path}"
            print(f"Registering model from URI: {model_uri}")
            
            # Register Model
            model_name = "Gold_Price_Prediction"
            try:
                mv = mlflow.register_model(model_uri, model_name)
                
                # Set Alias
                client.set_registered_model_alias(model_name, "champion", mv.version)
                print(f"✅ Registered model {model_name} version {mv.version} as 'champion'.")
            except Exception as e:
                print(f"❌ Error registering model: {e}")

        else:
            print("No valid runs found with RMSE_Price metric.")
