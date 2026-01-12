import yfinance as yf
import pandas as pd
import boto3
from io import StringIO
import os
import sys
from datetime import timedelta

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.build_features import build_features

# --- Configuration ---
TICKERS = ['GC=F', 'DX-Y.NYB', '^GSPC', '^TNX', '^VIX']
S3_BUCKET = '686699774218-martijn-project'
S3_KEY = 'data/processed/features.csv'

def get_last_date_from_s3():
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        df = pd.read_csv(obj['Body'], index_col=0)
        df.index = pd.to_datetime(df.index)
        return df.index.max(), df
    except Exception as e:
        print(f"Could not load existing data: {e}.")
        return None, None

def run_daily_update():
    last_date, df_existing = get_last_date_from_s3()
    
    if last_date is None:
        print("No existing data found. Please run ingest.py for the full history.")
        return
    
    print(f"Last recorded date: {last_date}")
    
    start_date = (last_date - timedelta(days=200)).strftime('%Y-%m-%d')
    print(f"Downloading incremental data window from: {start_date}")
    
    raw_chunk = yf.download(TICKERS, start=start_date, interval='1d')
    df_raw = raw_chunk['Close'].ffill().rename(columns={'GC=F': 'Gold_Price'})
    
    df_new_features = build_features(df_raw)
    df_new_only = df_new_features[df_new_features.index > last_date]
    
    if df_new_only.empty:
        print("✅ Data is already up to date. Nothing to append.")
        return {"status": "up-to-date", "new_rows": 0}
    
    print(f"Adding {len(df_new_only)} new rows...")
    df_final = pd.concat([df_existing, df_new_only])
    
    csv_buffer = StringIO()
    df_final.to_csv(csv_buffer)
    
    s3 = boto3.client('s3')
    s3.put_object(Bucket=S3_BUCKET, Key=S3_KEY, Body=csv_buffer.getvalue())
    
    print(f"🚀 Successfully updated S3 with data up to {df_final.index.max().date()}")
    return {"status": "updated", "new_rows": len(df_new_only), "last_date": str(df_final.index.max().date())}

if __name__ == "__main__":
    run_daily_update()
