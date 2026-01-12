import yfinance as yf
import pandas as pd
import boto3
from io import StringIO
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
TARGET_COL = 'Gold_Price'
TICKERS = ['GC=F', 'DX-Y.NYB', '^GSPC', '^TNX', '^VIX']
START_DATE = '2001-01-01'
S3_BUCKET = '686699774218-martijn-project'
S3_KEY = 'data/raw/raw_data.csv' 

# --- 1. Data Retrieval and Preprocessing ---
def get_and_clean_data(tickers, start_date):
    print("Downloading data...")
    all_data = yf.download(tickers, start=start_date, interval='1d')

    # Isolate 'Close' prices and clean up
    df = all_data['Close'].dropna(axis=1, how='all')
    df = df.ffill()
    df = df.rename(columns={'GC=F': TARGET_COL})

    # Drop any raw ticker columns that failed to load
    for ticker in tickers:
        if ticker in df.columns and ticker != TARGET_COL:
            if df[ticker].isnull().all():
                 df = df.drop(columns=[ticker])

    return df

def upload_to_s3(df, bucket, key):
    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=True)

    # Upload to S3
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
    print(f"Uploaded to s3://{bucket}/{key}")

if __name__ == "__main__":
    from features.build_features import build_features
    
    # 1. Get Raw Data
    df_raw = get_and_clean_data(TICKERS, START_DATE)
    
    # 2. Build Features (ETL)
    print("Building features...")
    df_features = build_features(df_raw).dropna()
    
    # 3. Save Processed Features to S3
    FEATURES_KEY = 'data/processed/features.csv'
    upload_to_s3(df_features, S3_BUCKET, FEATURES_KEY)
