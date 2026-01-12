import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit

# Import local utils (assuming pythonpath or relative import works)
# Since this is run from root or training module, we use relative imports if possible.
# But training.py adds root to path. Let's use relative imports.
from utils.training_utils import calculate_metrics

FEATURE_TARGET = "Target_Return"

def get_pipelines():
    cat_features = ['day', 'month', 'year', 'dayofweek']
    num_features = [
        'Gold_Price_L0', 'Gold_Price_L5', 'Gold_Price_L20', 
        'Gold_Price_10D_MA', 'Gold_Price_50D_MA', 'Gold_Log_Return', 
        'Gold_Log_Return_20D_Vol', 'DXY_Level', '10Y_Yield_Level', 
        'VIX_Level', 'SPX_Log_Return'
    ]
    prep_tree = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)])
    prep_linear = ColumnTransformer([('num', StandardScaler(), num_features)])

    return {
        "rf": Pipeline([("prep", prep_tree), ("model", RandomForestRegressor())]),
        "xgb": Pipeline([("prep", prep_tree), ("model", XGBRegressor(eval_metric="rmse"))]),
        "nn": Pipeline([("prep", prep_linear), ("model", MLPRegressor(max_iter=1000))])
    }

def train_classical_models(df, n_splits=5):
    X = df.drop(columns=[FEATURE_TARGET])
    y = df[FEATURE_TARGET]
    tscv = TimeSeriesSplit(n_splits=n_splits)
    candidates = get_pipelines()

    for name, pipeline in candidates.items():
        with mlflow.start_run(run_name=f"Classical_{name}", nested=True):
            all_fold_metrics = []

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline.fit(X_train, y_train)
                preds_ret = pipeline.predict(X_test)
                
                fold_metrics = calculate_metrics(y_test, preds_ret, X_test['Gold_Price_L0'])
                all_fold_metrics.append(fold_metrics)

            avg_metrics = {
                k: np.mean([m[k] for m in all_fold_metrics]) 
                for k in all_fold_metrics[0].keys()
            }
            
            mlflow.log_metrics(avg_metrics)
            
            # Fit on Full Data and Log
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, f"{name}_model")    

    return candidates
