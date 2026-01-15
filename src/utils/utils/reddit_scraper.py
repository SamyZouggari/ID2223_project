"""
Reddit sentiment scraper for Solana
"""

import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import torch
import numpy as np
from datetime import datetime
import time
import pandas as pd

# Load model (global)
print("Loading CryptoBERT...")
model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")
tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")

HEADER = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}

def analyze_sentiment(text):
    """Analyze sentiment with CryptoBERT"""
    if not text or text.strip() == "":
        return "Neutral", 0.0
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    scores = outputs.logits[0].numpy()
    scores = torch.nn.functional.softmax(torch.tensor(scores), dim=0).numpy()
    
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    max_idx = np.argmax(scores)
    
    return labels_map[max_idx], float(scores[max_idx])

def fetch_reddit_posts(subreddit, limit=100, time_filter="month"):
    """Fetch Reddit posts"""
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"t": time_filter, "limit": limit}
    
    try:
        response = requests.get(url, headers=HEADER, params=params)
        response.raise_for_status()
        data = response.json()

        posts = []
        for post in data['data']['children']:
            post_data = post['data']
            posts.append({
                'id': post_data['id'],
                'title': post_data['title'],
                'selftext': post_data.get('selftext', ''),
                'score': post_data['score'],
                'num_comments': post_data['num_comments'],
                'created_utc': post_data['created_utc'],
                'url': post_data['url'],
                'subreddit': post_data['subreddit']
            })
        return posts
    except Exception as e:
        print(f"Error: {e}")
        return []

def analyze_reddit_posts(posts):
    """Analyze sentiment for posts"""
    analyzed_data = []
    for post in posts:
        full_text = f"{post['title']} {post['selftext']}"
        sentiment, confidence = analyze_sentiment(full_text)
        post_with_sentiment = post.copy()
        post_with_sentiment['sentiment'] = sentiment
        post_with_sentiment['confidence'] = confidence
        analyzed_data.append(post_with_sentiment)
    return analyzed_data


def fetch_pushshift_posts(subreddit, start_date, end_date, limit=100):
    """
    Fetch posts using Pushshift API (historical archive).
    
    Args:
        subreddit (str): Subreddit name
        start_date (str): 'YYYY-MM-DD'
        end_date (str): 'YYYY-MM-DD'
        limit (int): Posts per request (max 100)
    """
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
    
    url = "https://api.pullpush.io/reddit/search/submission"  # New Pushshift mirror
    
    all_posts = []
    after = start_ts
    
    while after < end_ts and len(all_posts) < limit:
        params = {
            'subreddit': subreddit,
            'after': after,
            'before': end_ts,
            'size': limit,
            'sort': 'created_utc'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            posts = data.get('data', [])
            
            if not posts:
                break
            
            for post in posts:
                all_posts.append({
                    'id': post.get('id'),
                    'title': post.get('title', ''),
                    'selftext': post.get('selftext', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc'),
                    'url': post.get('url', ''),
                    'subreddit': post.get('subreddit', '')
                })
            
            # Update after timestamp for pagination
            after = posts[-1]['created_utc'] + 1
            
            print(f"   Fetched {len(posts)} posts (total: {len(all_posts)})")
            time.sleep(3)  # Rate limiting - increased to avoid API blocking
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            break
    
    return all_posts

def create_sentiment_table(df_raw_posts, title_column=None, selftext_column=None, batch_size=32):
    """Create sentiment table from analyzed posts
    Args:
        analyzed_posts (list): Dataframe of raw posts
    Returns:
        dict: Sentiment table"""
    
    if title_column is not None and selftext_column is not None:
        combined_texts = (df_raw_posts[title_column].astype(str) + " " + 
                          df_raw_posts[selftext_column].astype(str)).tolist()
    elif title_column is not None:
        combined_texts = df_raw_posts[title_column].astype(str).tolist()
    elif selftext_column is not None:
        combined_texts = df_raw_posts[selftext_column].astype(str).tolist()
    else:
        combined_texts = []
    
    sentiment_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    labels_map = {0: -1, 1: 0, 2: 1}

    for i in range(0, len(combined_texts), batch_size):
        if i % (batch_size * 10) == 0:
            print(f"   Progress: {i}/{len(combined_texts)} ({i*100//len(combined_texts)}%)")
        
        batch = combined_texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        max_indices = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                
        for max_idx in max_indices:
            sentiment_list.append(labels_map[max_idx])
    
    return pd.DataFrame({
        'timestamp': df_raw_posts['created_utc'].values,
        'sentiment': sentiment_list,
    })

    
def agregate_sentiment_table(sentiment_df):
    """Aggregate sentiment table
    Args:
        sentiment_list (list): df of sentiments with a timestamp key, format returned by the function 
        create_sentiment_table
    Returns:
        dict: df with a feature mean sentiment per day"""
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    sentiment_df['date'] = sentiment_df['timestamp'].dt.normalize()

    aggregated_df = sentiment_df.groupby('date').agg(
        mean_sentiment=('sentiment', 'mean'),
        count=('sentiment', 'count')
    ).reset_index()

    return aggregated_df


import yfinance as yf
import numpy as np
from datetime import timedelta

def fetch_latest_crypto_features():
    """Fetch and compute latest Solana features"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=250)
    
    # Download SOL price data
    data = yf.download("SOL-USD", start=start_date, end=end_date, interval="1d", progress=False)
    data = data.reset_index()
    data.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
    
    df = data[['date', 'close', 'high', 'low', 'open', 'volume']].copy()
    
    # Calculate features
    df['close_7d_ma'] = df['close'].rolling(window=7).mean()
    df['close_30d_ma'] = df['close'].rolling(window=30).mean()
    df['ma_7_30_cross'] = np.where(df['close_7d_ma'] > df['close_30d_ma'], 1, 0)
    df['volume_7d_ma'] = df['volume'].rolling(window=7).mean()
    df['volume_ratio'] = (df['volume'] / df['volume_7d_ma']).replace([np.inf, -np.inf], 1).fillna(1)
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Bollinger Bandwidth
    ma20 = df['close'].rolling(window=20).mean()
    std20 = df['close'].rolling(window=20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    df['bb_bandwidth'] = (upper - lower) / ma20
    
    # Temporal features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # Get latest row (no NaN)
    df_clean = df.dropna()
    latest = df_clean.iloc[-1:].copy()
    
    latest['timestamp'] = int((latest['date'].iloc[0] - pd.Timestamp("1970-01-01")).total_seconds())
    
    return latest

import sys
sys.path.append('..')

def fetch_latest_sentiment():
    """Fetch sentiment from today's Reddit posts"""
    url = f"https://www.reddit.com/r/solana/new.json"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        response = requests.get(url, headers=headers, params={'limit': 100})
        response.raise_for_status()
        data = response.json()
        
        posts = []
        for post in data['data']['children']:
            p = post['data']
            posts.append({
                'id': p['id'],
                'title': p['title'],
                'selftext': p.get('selftext', ''),
                'score': p['score'],
                'num_comments': p['num_comments'],
                'created_utc': p['created_utc'],
                'subreddit': p['subreddit']
            })
        
        if not posts:
            return 0.0, 0
        
        # Filter last 24h
        cutoff = (datetime.now() - timedelta(days=1)).timestamp()
        recent = [p for p in posts if p['created_utc'] > cutoff]
        
        if not recent:
            print(f"⚠️ No posts in last 24h")
            return 0.0, 0
        
        print(f"📰 Analyzing {len(recent)} recent posts...")
        
        # Convert and analyze
        df = pd.DataFrame(recent)
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        df["selftext"] = df["selftext"].replace(['[deleted]', '[removed]'], '').fillna('')
        df['title'] = df['title'].astype(str)
        df['selftext'] = df['selftext'].astype(str)
        
        # Sentiment analysis
        df_sentiment = create_sentiment_table(df, title_column='title', selftext_column='selftext')
        df_agg = agregate_sentiment_table(df_sentiment)
        
        mean_sent = df_agg.iloc[-1]['mean_sentiment']
        return mean_sent, len(recent)
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return 0.0, 0
    