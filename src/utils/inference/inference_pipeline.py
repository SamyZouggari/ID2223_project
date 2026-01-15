import hopsworks
import joblib
import pandas as pd
import numpy as np
import os
import sys
import streamlit as st
from xgboost import XGBRegressor
from datetime import datetime
from dotenv import load_dotenv
# Add utils to path and import reddit_scraper
sys.path.append("src/utils")
import utils.reddit_scraper as utils
load_dotenv()
def run_inference():
    """Run full inference pipeline and return prediction"""
    
    # Get API key
    api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found")
    
    # 1. Login to Hopsworks
    project = hopsworks.login(api_key_value=api_key)
    mr = project.get_model_registry()
    fs = project.get_feature_store()
    
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
    
    result = {
        'current_price': float(current_price),
        'predicted_price': float(predicted_price),
        'change_pct': float(change),
        'sentiment': float(mean_sentiment),
        'sentiment_count': int(count),
        'timestamp': datetime.now()
    }
    
    # 7. Save to Hopsworks
    fg = None
    try:
        fg = fs.get_feature_group("solana_predictions", version=1)
        # Delete old data
        fg.delete()
        fg = None
    except: 
        pass

    if fg is None:
        fg = fs.create_feature_group(
            name="solana_predictions",
            version=1,
            primary_key=["timestamp"],
            event_time="timestamp",
            online_enabled=True,
            description="Solana price predictions with sentiment"
        )

    prediction_df = pd.DataFrame([{
        'timestamp': int(result['timestamp'].timestamp()),
        'current_price': result['current_price'],
        'predicted_price': result['predicted_price'],
        'change_pct': result['change_pct'],
        'sentiment': result['sentiment'],
        'sentiment_count': result['sentiment_count']
    }])

    fg.insert(prediction_df)
    print(f"✅ Prediction saved to Hopsworks")
    
    return result

if __name__ == "__main__":
    print("🧪 Testing inference pipeline...")
    try:
        result = run_inference()
        print(f"\n✅ SUCCESS!")
        print(f"   Current:   ${result['current_price']:.2f}")
        print(f"   Predicted: ${result['predicted_price']:.2f}")
        print(f"   Change:    {result['change_pct']:+.2f}%")
        print(f"   Sentiment: {result['sentiment']:.3f}")
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()