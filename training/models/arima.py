import mlflow
import mlflow.sklearn
from pmdarima import auto_arima
from utils.training_utils import calculate_metrics

def train_arima(df, n_forecast=10):
    with mlflow.start_run(run_name="ARIMA_Model", nested=True):
        series = df['Gold_Log_Return'].dropna()
        model = auto_arima(series, seasonal=True, m=5)
        
        forecast_returns = model.predict(n_periods=n_forecast)
        actual_returns = series[-n_forecast:]
        last_price = df["Gold_Price_L0"].iloc[-(n_forecast+1)] 
        
        metrics = calculate_metrics(actual_returns, forecast_returns, last_price)
        
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "arima_model")
        return forecast_returns
