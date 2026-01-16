# Solana Sentiment-Driven Price Predictor
The URL fro this project is : https://id2223project-solana-advisor.streamlit.app/
## Project Overview
The **Solana Sentiment-Driven Price Predictor** is an end-to-end Machine Learning system that forecasts daily price movements for Solana (SOL). By fusing high-fidelity financial technical indicators with domain-specific social media sentiment analysis, the system captures both market mechanics and investor psychology.

The project implements a modern **MLOps architecture** using **Hopsworks** as a centralized Feature Store, ensuring seamless data synchronization between historical training and real-time inference.



---

## System Architecture
The system is orchestrated through four specialized pipelines, ensuring modularity and scalability.

### 1. Technical Feature Pipeline (`backfill_crypto_feature.ipynb`)
**Goal:** Generate a rich historical foundation of technical market signals.
* **Data Acquisition:** Ingests 5+ years of daily OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
* **Feature Engineering:** Computes **40+ technical indicators**:
    * **Trend:** Multi-window Moving Averages (7d, 30d, 50d, 200d) and MA Cross-overs.
    * **Momentum:** RSI (Relative Strength Index) and MACD (Line, Signal, Histogram).
    * **Volatility:** Bollinger Bands (Upper, Lower, Bandwidth) and ATR (Average True Range).
    * **Volume Analysis:** On-Balance Volume (OBV), Volume-Price Trend (VPT), and Volume Ratios.
    * **Temporal Features:** Day of week, month, quarter, and weekend/month-end flags to capture cyclical market patterns.
* **Storage:** Materializes data into the `solana_crypto_features` Feature Group in Hopsworks.

### 2. Sentiment Pipeline (`backfill_reddit_sentiment.ipynb`)
**Goal:** Quantify "Retail Mood" using specialized Natural Language Processing.
* **Data Extraction:** Retrieves thousands of historical posts from the `r/solana` subreddit via the Pushshift API.
* **NLP Engine:** Utilizes **CryptoBERT**, a transformer model fine-tuned on 3.2M crypto-social posts, to classify text into **Bullish**, **Neutral**, or **Bearish** categories.
* **Aggregation:** Daily mean sentiment scores and post volume counts are calculated to provide a unified "market pulse" feature.

### 3. Training Pipeline (`training_pipeline.ipynb`)
**Goal:** Optimize an AI "Brain" to find correlations between social hype and price action.
* **Point-in-Time Join:** Merges Technical and Sentiment features on a shared `timestamp` using a Hopsworks **Feature View**.
* **Model Architecture:** Implements an **XGBoost Regressor**, chosen for its superior performance on tabular time-series data.
* **Performance Metrics:** The model achieved high predictive accuracy:
    * **R-Squared:** ~0.97 (indicating the model explains 97% of the price variance).
    * **MSE:** ~35.09.
* **Registry:** Finalized models are versioned and stored in the Hopsworks Model Registry.



### 4. Inference Pipeline (`inference_pipeline.ipynb`)
**Goal:** Provide real-time actionable insights.
* **Live Data Fetch:** Triggers a dual-fetch of the current SOL price from Yahoo Finance and the last 24 hours of Reddit discussion.
* **Prediction:** Passes live features through the registered model to output:
    * **Current Price** vs. **Predicted Price**.
    * **Expected % Change** (e.g., +4.74%).

---

## Tech Stack
| Category | Tools |
| :--- | :--- |
| **Data Sources** | `yfinance` (Market Data), Pushshift API (Reddit) |
| **Machine Learning** | `XGBoost`, `Scikit-learn`, `Pandas`, `NumPy` |
| **NLP** | `CryptoBERT` (HuggingFace Transformers) |
| **MLOps** | **Hopsworks** (Feature Store & Model Registry) |


---

## Project Impact
Unlike traditional trading bots that rely solely on price action, this system accounts for the **social momentum** that frequently drives cryptocurrency volatility. By using **CryptoBERT**, the system understands that "burning tokens" or "going to the moon" are positive signals, allowing for more nuanced predictions during intense market cycles.
