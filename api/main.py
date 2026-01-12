from fastapi import FastAPI
import sys
import os
import mlflow
from dotenv import load_dotenv
import socket

# --- NETWORKING PATCH: Force IPv4 ---
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(*args, **kwargs):
    responses = orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = patched_getaddrinfo

# Load credentials and config
load_dotenv()

# Set Tracking URI globally
tracking_uri = os.getenv("TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

# sys.path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
sys.path.append(current_dir)

try:
    from inference import run_inference
except ImportError:
    from api.inference import run_inference

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict/warmup")
def warmup():
    """Pre-loads the champion model into memory cache."""
    try:
        result = run_inference()
        if "error" in result:
            return {"status": "error", "message": result["error"]}
        return {"status": "ready", "model": result["model_used"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/sync-data")
def sync_data():
    """Triggers incremental data ingestion: pulls new prices and updates S3."""
    try:
        from data_pipeline.update import run_daily_update
        result = run_daily_update()
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict/next-day")
def predict_next_day():
    """Runs inference and returns the next day prediction."""
    try:
        prediction = run_inference()
        if "error" in prediction:
            return {"status": "error", "message": prediction["error"]}
        return {"status": "success", "data": prediction}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
