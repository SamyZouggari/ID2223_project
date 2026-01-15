import streamlit as st
import pandas as pd
from datetime import datetime
from src.utils.inference.inference_pipeline import run_inference

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

# Cache prediction for 24 hours (86400 seconds)
@st.cache_data(ttl=86400)
def get_prediction():
    """Run inference and cache for 24h"""
    return run_inference()

# Main Header
st.title("🤖 Solana AI Advisor")
st.markdown(f"**Real-time price prediction using Sentiment AI** • *Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

# Run inference automatically
with st.spinner("🚀 Loading latest prediction..."):
    try:
        result = get_prediction()
        
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
                label="Predicted Price (Next 24H)", 
                value=f"${result['predicted_price']:.2f}",
                delta=f"{result['change_pct']:+.2f}%"
            )
        with col3:
            sentiment_score = (result['sentiment'] + 1) * 50  # Scale -1/+1 to 0-100
            st.metric(
                label="Reddit Sentiment", 
                value=f"{sentiment_score:.0f}/100",
                delta="Bullish" if result['sentiment'] > 0 else "Bearish" if result['sentiment'] < 0 else "Neutral"
            )
        with col4:
            signal = "🟢 BUY" if result['change_pct'] > 2 else "🟡 HOLD" if result['change_pct'] > -2 else "🔴 SELL"
            st.metric(
                label="Trading Signal", 
                value=signal
            )
        
        st.markdown("---")
        
        # Details
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📊 Prediction Details")
            st.write(f"**Reddit Posts Analyzed:** {result['sentiment_count']}")
            st.write(f"**Sentiment Score:** {result['sentiment']:.3f}")
            st.write(f"**Prediction Time:** {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            st.info("💡 Prediction updates automatically every 24 hours")
        
        with col_right:
            st.subheader("🛠️ Model Info")
            st.write("**Algorithm:** HistGradientBoostingRegressor")
            st.write("**Training R²:** 0.975")
            st.write("**Features:** 14 technical + 2 sentiment")
            st.write("**Training Period:** 2020-2023")
            st.write("**Test Period:** 2024-2025")
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.info("Please check logs or try refreshing the page.")

# Footer
st.markdown("---")
st.caption("🔄 Data refreshes every 24 hours automatically • Built with Streamlit + Hopsworks + CryptoBERT")