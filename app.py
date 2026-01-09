import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime

# 1. Page Configuration
st.set_page_config(
    page_title="Crypto AI Advisor v1.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR MODERN LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# 2. Sidebar - Pipeline Monitoring (The "System Pulse")
with st.sidebar:
    st.title("🛡️ System Control")
    st.markdown("---")
    
    st.subheader("Pipeline Status")
    # Placeholders for pipeline health
    st.success("📡 Backfill: IDLE")
    st.success("⚙️ Feature: ACTIVE")
    st.warning("🧠 Training: PENDING")
    st.info("🔮 Inference: STANDBY")
    
    st.markdown("---")
    st.subheader("Model Configuration")
    target_coin = st.selectbox("Select Asset", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe = st.select_slider("Prediction Window", options=["1H", "4H", "24H", "7D"])
    
    if st.button("Manual Inference Trigger"):
        with st.status("Running Inference..."):
            time.sleep(2)
            st.write("Fetching real-time sentiment...")
            time.sleep(1)
            st.write("Computing features...")
            st.success("Prediction Generated!")

# 3. Main Header & Purpose
st.title("🤖 Autonomous Crypto Advisor")
st.markdown(f"**Project Goal:** Real-time short-term price prediction using Sentiment AI. *Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# 4. Executive Summary (Metrics)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Current BTC Price", value="$94,250", delta="+2.5%")
with col2:
    st.metric(label="AI Sentiment Score", value="78/100", delta="Bullish")
with col3:
    st.metric(label="Prediction Signal", value="ACCUMULATE", delta_color="normal")
with col4:
    st.metric(label="Model Confidence", value="84.2%", delta="-1.2%")

st.markdown("---")

# 5. Tabs for Different Views
tab1, tab2, tab3 = st.tabs(["📈 Market Forecast", "📰 Data Firehose", "🛠️ System Architecture"])

with tab1:
    st.subheader("Price Prediction & Sentiment Overlay")
    # Mock chart data
    chart_data = pd.DataFrame(
        np.random.randn(20, 2),
        columns=['Price', 'Sentiment']
    )
    st.line_chart(chart_data)
    
    with st.expander("See Feature Importance"):
        st.bar_chart({"Twitter Vol": 45, "News Sentiment": 30, "RSI": 15, "Whale Alerts": 10})

with tab2:
    st.subheader("Live Ingested Feed")
    # Mock feed from the "Feature Pipeline"
    feed_data = {
        "Source": ["Twitter", "Reuters", "Reddit", "Twitter"],
        "Content": [
            "BTC breaking resistance levels!", 
            "SEC updates on Crypto ETF filings.", 
            "Is it time to buy the dip?", 
            "Whale moved 5000 BTC to cold storage."
        ],
        "Sentiment": ["Bullish", "Neutral", "Mixed", "Bullish"]
    }
    st.table(pd.DataFrame(feed_data))

with tab3:
    st.subheader("Project Purpose & Architecture")
    st.info("""
    **The Purpose:** This project solves the problem of 'Sentiment Lag.' By the time humans read the news, the price has often already moved. This system ingests data at millisecond speeds to predict moves *before* they stabilize.
    """)
    
    # Diagram Placeholder
    st.markdown("""
    ### Pipeline Workflow:
    1. **Backfill:** Historical OHLCV + News CSVs.
    2. **Feature:** Technical Indicators + NLP Sentiment Scoring.
    3. **Training:** LSTM Model trained on Feature Store.
    4. **Inference:** Real-time FastAPI endpoint serving the UI.
    """)