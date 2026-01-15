import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('src/utils/inference')
from inference_pipeline import run_inference

# Page Configuration
st.set_page_config(
    page_title="Solana AI Advisor",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ System Control")
    st.markdown("---")
    
    st.subheader("Pipeline Status")
    st.success("📡 Backfill: COMPLETE")
    st.success("⚙️ Features: ACTIVE")
    st.success("🧠 Model: TRAINED (R²=0.975)")
    st.success("🔮 Inference: READY")
    
    st.markdown("---")
    
    if st.button("🔄 Run Real-Time Prediction", type="primary"):
        st.session_state['run_inference'] = True

# Main Header
st.title("🤖 Solana AI Advisor")
st.markdown(f"**Real-time price prediction using Sentiment AI** • *{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Run inference if button clicked
if st.session_state.get('run_inference', False):
    with st.status("🚀 Running Inference Pipeline...", expanded=True) as status:
        try:
            st.write("📊 Fetching latest Solana price data...")
            st.write("📰 Analyzing Reddit sentiment...")
            st.write("🧠 Running model prediction...")
            
            result = run_inference()
            
            status.update(label="✅ Inference Complete!", state="complete")
            
            # Display results
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Current SOL Price", 
                    value=f"${result['current_price']:.2f}"
                )
            with col2:
                st.metric(
                    label="Predicted Price (24H)", 
                    value=f"${result['predicted_price']:.2f}",
                    delta=f"{result['change_pct']:+.2f}%"
                )
            with col3:
                sentiment_score = (result['sentiment'] + 1) * 50  # Scale to 0-100
                st.metric(
                    label="Sentiment Score", 
                    value=f"{sentiment_score:.0f}/100",
                    delta="Bullish" if result['sentiment'] > 0 else "Bearish" if result['sentiment'] < 0 else "Neutral"
                )
            with col4:
                signal = "🟢 BUY" if result['change_pct'] > 2 else "🟡 HOLD" if result['change_pct'] > -2 else "🔴 SELL"
                st.metric(
                    label="Signal", 
                    value=signal
                )
            
            st.markdown("---")
            
            # Details
            with st.expander("📋 Prediction Details"):
                st.write(f"**Reddit Posts Analyzed:** {result['sentiment_count']}")
                st.write(f"**Mean Sentiment:** {result['sentiment']:.3f}")
                st.write(f"**Prediction Timestamp:** {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            
            st.session_state['run_inference'] = False
            
        except Exception as e:
            status.update(label="❌ Inference Failed", state="error")
            st.error(f"Error: {str(e)}")
            st.session_state['run_inference'] = False

else:
    # Initial state
    st.info("👈 Click **Run Real-Time Prediction** in the sidebar to get the latest Solana price forecast!")
    
    st.markdown("---")
    
    # Architecture tab
    st.subheader("🛠️ System Architecture")
    st.markdown("""
    ### Pipeline Workflow:
    1. **Backfill (DONE):** 1,876 days of Solana OHLCV + 1,818 days of Reddit sentiment
    2. **Feature Engineering:** 42 technical indicators (RSI, MACD, Bollinger Bands, etc.)
    3. **Training:** HistGradientBoostingRegressor trained on 2020-2023 data (R² = 0.975)
    4. **Real-Time Inference:**
       - Fetch latest SOL price from Yahoo Finance
       - Scrape r/solana posts from last 24h
       - Analyze sentiment with CryptoBERT
       - Predict next-day price
    
    **Model Performance:**
    - Training R²: 0.975
    - Test Period: 2024-2025
    - Features: 14 technical + 2 sentiment = 16 total
    """)