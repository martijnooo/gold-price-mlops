import gradio as gr
import pandas as pd
import boto3
import plotly.graph_objects as go
from io import StringIO
import os
import requests
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from botocore.config import Config

# --- Config ---
BUCKET = os.getenv("S3_BUCKET_NAME", "686699774218-martijn-project")
PREDICTIONS_KEY = "data/predictions/predictions.csv"
API_URL = os.getenv("API_URL", "http://localhost:8000")

def load_data():
    try:
        s3 = boto3.client('s3', config=Config(connect_timeout=5, retries={'max_attempts': 0}))
        obj = s3.get_object(Bucket=BUCKET, Key=PREDICTIONS_KEY)
        df_all = pd.read_csv(obj['Body'], index_col=0)
        df_all.index = pd.to_datetime(df_all.index)
        
        if 'Type' in df_all.columns:
            df_backtest = df_all[df_all['Type'] == 'Backtest'].copy()
            df_live = df_all[df_all['Type'] == 'Live'].copy()
        else:
            df_backtest = df_all.copy()
            df_live = pd.DataFrame()
        return df_backtest, df_live, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)

def get_dashboard():
    df_backtest, df_live, error = load_data()
    
    if error:
        return f"Error loading data: {error}", None, "", "", ""

    # Calculate metrics
    mae_val, rmse_val = "N/A", "N/A"
    if not df_backtest.empty:
        mae = mean_absolute_error(df_backtest['Actual_Price'], df_backtest['Predicted_Price'])
        rmse = np.sqrt(mean_squared_error(df_backtest['Actual_Price'], df_backtest['Predicted_Price']))
        mae_val, rmse_val = f"${mae:.2f}", f"${rmse:.2f}"

    live_text = "No live predictions found."
    if not df_live.empty:
        last_pred = df_live.iloc[-1]
        live_text = f"Next Close Prediction: ${last_pred['Predicted_Price']:.2f} ({last_pred['Predicted_Log_Return']*100:+.2f}%) for {df_live.index[-1].date()}"

    # Create Chart
    fig = go.Figure()
    if not df_backtest.empty:
        fig.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Actual_Price'], mode='lines', name='Actual Price', line=dict(color='#FFD700', width=3)))
        fig.add_trace(go.Scatter(x=df_backtest.index, y=df_backtest['Predicted_Price'], mode='lines', name='Backtest', line=dict(color='#2563EB', dash='dash')))
    if not df_live.empty:
        fig.add_trace(go.Scatter(x=df_live.index, y=df_live['Predicted_Price'], mode='markers+lines', name='Live', line=dict(color='#059669', width=4)))
    
    fig.update_layout(
        title="Gold Price Trajectory", 
        template="plotly_white", 
        height=500,
        hovermode="x unified"
    )
    
    return "", fig, mae_val, rmse_val, live_text

def run_inference():
    try:
        # Increased timeout to 120s for heavy model (LSTM) execution
        response = requests.post(f"{API_URL}/predict/next-day", timeout=120)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                data = result["data"]
                return f"✅ Prediction Generated! Price: ${data['predicted_price']:.2f} for {data['date']} (Model: {data['model_used']})"
            return f"❌ API Error: {result.get('message')}"
        return f"❌ HTTP Error: {response.status_code}"
    except Exception as e:
        return f"❌ Connection Error: {e}"

def warmup_system():
    try:
        # Long timeout because this is the cold-start download
        response = requests.post(f"{API_URL}/predict/warmup", timeout=120)
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "ready":
                return f"🟢 **System Ready**: Model `{result['model']}` is cached."
            return f"🔴 **Warmup Failed**: {result.get('message')}"
        return f"🟠 **Warmup Partial**: Status {response.status_code}"
    except Exception as e:
        return f"🔴 **Connection Error during warmup**: {str(e)}"

# --- Custom Light Theme ---
theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_title_text_weight="600",
    block_shadow="rgba(0,0,0,0.05) 0px 1px 2px 0px",
    button_primary_background_fill="*primary_500",
    button_primary_text_color="white",
)

with gr.Blocks() as demo:
    # --- Top Header & Status Bar ---
    with gr.Row(variant="compact"):
        with gr.Column(scale=8):
            gr.HTML(
                """
                <div style='display: flex; align-items: center; gap: 15px;'>
                    <span style='font-size: 40px;'>📈</span>
                    <div>
                        <h1 style='margin: 0; font-size: 28px; font-weight: 800; color: #B45309;'>Gold Price Intelligence</h1>
                        <p style='margin: 0; color: #64748B; font-weight: 500;'>Advanced MLOps Forecasting & Model Monitoring</p>
                    </div>
                </div>
                """
            )
        with gr.Column(scale=4):
            system_status = gr.Markdown("🟡 **System Status**: Checking connection...")

    with gr.Tabs():
        # --- TAB 1: DASHBOARD ---
        with gr.Tab("📊 Market Dashboard"):
            with gr.Row():
                with gr.Column(scale=7):
                    plot_output = gr.Plot(label=None, show_label=False)
                
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("### 💡 Current Outlook")
                        live_status = gr.Textbox(
                            label="Next-Day Forecast", 
                            placeholder="Waiting for data sync...",
                            interactive=False,
                            info="Predicted price for the next trading close."
                        )
                        refresh_btn = gr.Button("🔄 Sync Latest Predictions", variant="secondary")
                    
                    gr.Markdown("---")
                    
                    with gr.Group():
                        gr.Markdown("### 🚀 Operational Actions")
                        gr.Markdown("Trigger a fresh inference cycle on the AWS Backend.")
                        infer_btn = gr.Button("🚀 Run New Prediction", variant="primary")
                        infer_output = gr.Markdown("Ready to initiate.")

            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### 🏆 Model Accuracy (Backtest)")
                        with gr.Row():
                            mae_val = gr.Textbox(label="MAE ($)", interactive=False, info="Mean Absolute Error")
                            rmse_val = gr.Textbox(label="RMSE ($)", interactive=False, info="Root Mean Squared Error")

        # --- TAB 2: TECHNICAL DETAILS ---
        with gr.Tab("⚙️ System Diagnostics"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Data Traceability")
                    gr.Markdown("View the raw data payloads extracted from S3.")
                    error_output = gr.HTML("<p style='color: #64748B;'>No active errors.</p>")
                
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Prediction History")
                    raw_data_view = gr.JSON(label="S3 Payload Snapshot")

    # --- Footer ---
    gr.HTML(
        """
        <div style='text-align: center; margin-top: 30px; padding: 20px; color: #94A3B8; font-size: 13px; border-top: 1px solid #E2E8F0;'>
            AWS regions active: EU West (API/UI) & EU North (MLflow/S3) | Built by Martijn
        </div>
        """
    )

    # --- Interaction Logic ---
    def load_and_snapshot():
        bt, lv, err = load_data()
        if err:
            return f"<p style='color: #EF4444;'>Error: {err}</p>", None, "", "", "", {}
        
        # FIX: Convert indices to strings for JSON serialization (Dict keys must be str)
        bt_copy = bt.tail(3).copy()
        lv_copy = lv.tail(3).copy()
        bt_copy.index = bt_copy.index.strftime('%Y-%m-%d')
        lv_copy.index = lv_copy.index.strftime('%Y-%m-%d')

        snapshot = {
            "last_backtest": bt_copy.to_dict() if not bt_copy.empty else {},
            "last_live": lv_copy.to_dict() if not lv_copy.empty else {}
        }
        
        # Get dashboard UI elements
        err_msg, fig, mae, rmse, live = get_dashboard()
        return f"<p style='color: #22C55E;'>Last Sync: {pd.Timestamp.now().strftime('%H:%M:%S')}</p>", fig, mae, rmse, live, snapshot

    refresh_btn.click(
        fn=lambda: ("⌛ Fetching data...", gr.update(interactive=False)), 
        outputs=[error_output, refresh_btn]
    ).then(
        fn=load_and_snapshot, 
        outputs=[error_output, plot_output, mae_val, rmse_val, live_status, raw_data_view],
        show_progress="full"
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=refresh_btn
    )

    infer_btn.click(
        fn=lambda: ("⚙️ Initializing AWS Inference...", gr.update(interactive=False)), 
        outputs=[infer_output, infer_btn]
    ).then(
        fn=run_inference, 
        outputs=infer_output,
        show_progress="full"
    ).then(
        fn=load_and_snapshot, 
        outputs=[error_output, plot_output, mae_val, rmse_val, live_status, raw_data_view],
        show_progress="full"
    ).then(
        fn=lambda: gr.update(interactive=True),
        outputs=infer_btn
    )
    
    # Load initial data on startup AND start background warmup
    demo.load(
        fn=load_and_snapshot, 
        outputs=[error_output, plot_output, mae_val, rmse_val, live_status, raw_data_view]
    ).then(
        fn=warmup_system,
        outputs=system_status
    )

# Launch on port 8501
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8501, theme=theme)
