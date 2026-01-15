import hopsworks
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys
from xgboost import XGBRegressor

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import reddit_scraper as utils

def run_inference():
    """Run full inference pipeline and return prediction"""
    
    # 1. Login to Hopsworks
    project = hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"))
    mr = project.get_model_registry()
    
    # 2. Load model
    model_obj = mr.get_model("crypto_price_model", version=1)
    saved_model_dir = model_obj.download()
    model = XGBRegressor()
    model.load_model(saved_model_dir + "/model.json")
    
    # 3. Fetch crypto features
    crypto_df = utils.fetch_latest_crypto_features()
    current_price = crypto_df['close'].values[0]
    
    # 4. Fetch sentiment
    mean_sentiment, count = utils.fetch_latest_sentiment()
    
    # 5. Create features
    X = crypto_df[['high', 'low', 'open', 'volume',
                   'close_7d_ma', 'close_30d_ma', 'ma_7_30_cross',
                   'rsi_14', 'atr_14', 'bb_bandwidth',
                   'volume_7d_ma', 'volume_ratio',
                   'day_of_week', 'month']].copy()
    
    X['reddit_aggregated_sentiment_backfill_mean_sentiment'] = mean_sentiment
    X['reddit_aggregated_sentiment_backfill_count'] = count
    
    # 6. Predict
    predicted_price = model.predict(X)[0]
    change = ((predicted_price - current_price) / current_price) * 100
    
    return {
        'current_price': current_price,
        'predicted_price': predicted_price,
        'change_pct': change,
        'sentiment': mean_sentiment,
        'sentiment_count': count,
        'timestamp': datetime.now()
    }