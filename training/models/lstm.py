import numpy as np
import mlflow
import mlflow.keras
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from utils.training_utils import calculate_metrics

FEATURE_TARGET = "Target_Return"

def prepare_lstm_data(df, window_size=50):
    y_raw = df[FEATURE_TARGET].values
    X_raw = df.drop(columns=[FEATURE_TARGET])

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - window_size):
        X_seq.append(X_scaled[i:i+window_size])
        y_seq.append(y_scaled[i+window_size])
    
    return np.array(X_seq), np.array(y_seq), scaler_y

def train_lstm(df, X_seq, y_seq, scaler_y, window_size=50):
    with mlflow.start_run(run_name="LSTM_Model", nested=True):
        
        split = int(len(X_seq) * 0.9)
        X_train, X_test = X_seq[:split], X_seq[split:]
        y_train, y_test = y_seq[:split], y_seq[split:]

        model = Sequential([
            Input(shape=(X_seq.shape[1], X_seq.shape[2])),
            LSTM(64, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.1, epochs=20, batch_size=32, callbacks=[stop], verbose=0)
        
        signature = infer_signature(X_test, model.predict(X_test))

        preds_scaled = model.predict(X_test)
        
        preds_ret = scaler_y.inverse_transform(preds_scaled).flatten()
        actual_ret = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        base_prices = df['Gold_Price_L0'].iloc[split + window_size:].values
        
        metrics = calculate_metrics(actual_ret, preds_ret, base_prices)
        
        mlflow.log_params({"window_size": window_size, "lstm_units": 64, "epochs": 20})
        mlflow.log_metrics(metrics)
        mlflow.keras.log_model(model, "lstm_gold_model", signature=signature)
        
        return preds_ret
