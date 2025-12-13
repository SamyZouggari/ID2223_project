"""
Reddit scraper to collect data on Bitcoin
Without official API - uses Reddit's public JSON endpoint
"""

import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import csv
from datetime import datetime
import time
import random

# Load CryptoBERT model for crypto sentiment analysis
finbert_model = AutoModelForSequenceClassification.from_pretrained("ElKulako/cryptobert")
finbert_tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")

# Sentiment labels
labels = ['Positive', 'Negative', 'Neutral']


def analyze_sentiment(text):
    """
    Analyze text sentiment with CryptoBERT
    
    Args:
        text: The text to analyze
        
    Returns:
        tuple: (confidence, sentiment)
    """
    if not text.strip():
        return 0.0, 'Neutral'

    # Tokenize text (limit to 512 tokens)
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Prediction without gradient calculation (inference)
    with torch.no_grad():
        outputs = finbert_model(**inputs)

    # Calculate probabilities with softmax
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).numpy()[0]
    max_index = np.argmax(probabilities)
    sentiment = labels[max_index]
    confidence = probabilities[max_index]

    return confidence, sentiment


def fetch_reddit_posts(subreddit_name, limit=100, time_filter='week'):
    """
    Fetch Reddit posts using the public JSON endpoint
    No authentication required!
    
    Args:
        subreddit_name: Name of the subreddit (e.g., "Bitcoin", "CryptoCurrency")
        limit: Maximum number of posts to fetch
        time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        
    Returns:
        list: List of dictionaries containing post information
    """
    posts_data = []
    
    # Headers to simulate a normal browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # URL of Reddit's public JSON endpoint (top posts)
        url = f"https://www.reddit.com/r/{subreddit_name}/top.json"
        params = {
            't': time_filter,  # time filter
            'limit': min(limit, 100)  # Reddit limits to 100 per request
        }
        
        print(f"Fetching posts from r/{subreddit_name} (top {time_filter})...")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract posts
        for post in data['data']['children']:
            post_data = post['data']
            
            # Extract relevant information
            post_info = {
                'id': post_data.get('id', ''),
                'title': post_data.get('title', ''),
                'text': post_data.get('selftext', ''),
                'author': post_data.get('author', 'Unknown'),
                'subreddit': subreddit_name,
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'url': post_data.get('url', ''),
                'permalink': f"https://reddit.com{post_data.get('permalink', '')}"
            }
            
            posts_data.append(post_info)
        
        print(f"  Fetched {len(posts_data)} posts from r/{subreddit_name}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error during fetch: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    return posts_data


def fetch_reddit_search(subreddit_name, query="bitcoin", limit=100, time_filter='week'):
    """
    Search Reddit posts by keyword using the JSON endpoint
    
    Args:
        subreddit_name: Name of the subreddit
        query: Search term
        limit: Maximum number of posts
        time_filter: Time filter
        
    Returns:
        list: List of posts
    """
    posts_data = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # JSON search endpoint
        url = f"https://www.reddit.com/r/{subreddit_name}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'on',  # Restrict to subreddit
            't': time_filter,
            'limit': min(limit, 100),
            'sort': 'top'
        }
        
        print(f"Searching '{query}' in r/{subreddit_name}...")
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        for post in data['data']['children']:
            post_data = post['data']
            
            post_info = {
                'id': post_data.get('id', ''),
                'title': post_data.get('title', ''),
                'text': post_data.get('selftext', ''),
                'author': post_data.get('author', 'Unknown'),
                'subreddit': subreddit_name,
                'score': post_data.get('score', 0),
                'upvote_ratio': post_data.get('upvote_ratio', 0),
                'num_comments': post_data.get('num_comments', 0),
                'created_utc': datetime.fromtimestamp(post_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S'),
                'url': post_data.get('url', ''),
                'permalink': f"https://reddit.com{post_data.get('permalink', '')}"
            }
            
            posts_data.append(post_info)
        
        print(f"  Found {len(posts_data)} posts")
        
    except Exception as e:
        print(f"Error during search: {e}")
    
    return posts_data


def fetch_reddit_comments(post_permalink, max_comments=20):
    """
    Fetch comments from a Reddit post
    
    Args:
        post_permalink: Permalink of the post (e.g., /r/Bitcoin/comments/abc123/title/)
        max_comments: Maximum number of comments
        
    Returns:
        list: List of comments
    """
    comments_data = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # JSON URL of the post with its comments
        url = f"https://www.reddit.com{post_permalink}.json"
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Comments are in data[1]['data']['children']
        if len(data) > 1:
            comments = data[1]['data']['children']
            
            for comment in comments[:max_comments]:
                if comment['kind'] == 't1':  # t1 = comment
                    comment_data = comment['data']
                    
                    # Verify that the comment has content
                    body = comment_data.get('body', '')
                    if body and body != '[deleted]' and body != '[removed]':
                        comment_info = {
                            'id': comment_data.get('id', ''),
                            'text': body,
                            'author': comment_data.get('author', 'Unknown'),
                            'score': comment_data.get('score', 0),
                            'created_utc': datetime.fromtimestamp(comment_data.get('created_utc', 0)).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        comments_data.append(comment_info)
        
    except Exception as e:
        pass  # Silent to avoid too many error messages
    
    return comments_data


def analyze_posts_sentiment(posts):
    """
    Analyze sentiment of all collected posts
    
    Args:
        posts: List of posts with their metadata
        
    Returns:
        list: List of posts with their sentiment
    """
    analyzed_posts = []
    
    print("\nAnalyzing post sentiment...")
    for idx, post in enumerate(posts, 1):
        # Combine title + text for analysis
        full_text = post['title'] + " " + post.get('text', '')
        
        # Analyze sentiment
        confidence, sentiment = analyze_sentiment(full_text)
        
        # Add results
        post['sentiment'] = sentiment
        post['confidence'] = float(confidence)
        
        analyzed_posts.append(post)
        
        # Display progress
        if idx % 20 == 0:
            print(f"  Analyzed {idx}/{len(posts)} posts...")
    
    return analyzed_posts


def summarize_sentiments(data, data_type="posts"):
    """
    Summary of collected sentiments
    
    Args:
        data: List of posts with their sentiment
        data_type: Type of data
    """
    summary = {
        "Positive": 0,
        "Negative": 0,
        "Neutral": 0
    }
    
    # Count sentiments
    for item in data:
        sentiment = item.get('sentiment', 'Neutral')
        summary[sentiment] += 1
    
    total = len(data)
    print(f"\n--- Reddit Sentiment Summary ({data_type}) ---")
    print(f"Total {data_type} analyzed: {total}")
    for sentiment, count in summary.items():
        percent = (count / total) * 100 if total > 0 else 0
        print(f"{sentiment}: {count} ({percent:.2f}%)")
    
    # Additional statistics
    if data:
        total_score = sum(item.get('score', 0) for item in data)
        total_comments = sum(item.get('num_comments', 0) for item in data)
        avg_upvote = sum(item.get('upvote_ratio', 0) for item in data) / len(data) if len(data) > 0 else 0
        
        print(f"\nGlobal statistics:")
        print(f"  Total score: {total_score:,}")
        print(f"  Total comments: {total_comments:,}")
        print(f"  Average upvote ratio: {avg_upvote:.2%}")


def save_to_csv(data, filename='reddit_bitcoin_sentiment.csv'):
    """
    Save data to a CSV file
    
    Args:
        data: List of data to save
        filename: Output filename
    """
    if not data:
        print("No data to save.")
        return
    
    try:
        fieldnames = ['id', 'title', 'text', 'author', 'subreddit', 'score', 
                     'upvote_ratio', 'num_comments', 'created_utc', 'url', 
                     'permalink', 'sentiment', 'confidence']
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            for item in data:
                row = {k: item.get(k, '') for k in fieldnames}
                writer.writerow(row)
        
        print(f"\nData saved to {filename}")
        
    except Exception as e:
        print(f"Error during save: {e}")


def main():
    """
    Main scraper function
    """
    print("="*60)
    print("REDDIT SCRAPER - Bitcoin Sentiment Analysis")
    print("Without official API - Uses public JSON endpoints")
    print("="*60)
    
    # Subreddits to analyze
    subreddits = [
        "Bitcoin",
        "CryptoCurrency",
        "btc"
    ]
    
    # Parameters
    posts_per_subreddit = 50
    time_filter = 'week'  # Posts from the last week
    
    all_posts = []
    
    # Fetch posts (method 1: top posts)
    print("\n--- FETCHING POSTS (TOP) ---")
    for subreddit in subreddits:
        print(f"\n{'='*60}")
        print(f"Subreddit: r/{subreddit}")
        print(f"{'='*60}")
        
        posts = fetch_reddit_posts(
            subreddit, 
            limit=posts_per_subreddit,
            time_filter=time_filter
        )
        all_posts.extend(posts)
        
        # Random pause to be respectful to Reddit
        time.sleep(random.uniform(1, 2))
    
    # Additional fetch by search
    print("\n--- FETCHING BY SEARCH ---")
    search_terms = ["bitcoin price", "btc"]
    
    for subreddit in ["Bitcoin", "CryptoCurrency"]:
        for term in search_terms:
            posts = fetch_reddit_search(
                subreddit,
                query=term,
                limit=25,
                time_filter=time_filter
            )
            all_posts.extend(posts)
            time.sleep(random.uniform(1, 2))
    
    # Remove duplicates (same ID)
    seen_ids = set()
    unique_posts = []
    for post in all_posts:
        if post['id'] not in seen_ids:
            seen_ids.add(post['id'])
            unique_posts.append(post)
    
    print(f"\n{'='*60}")
    print(f"Total unique posts fetched: {len(unique_posts)}")
    print(f"{'='*60}")
    
    # Sentiment analysis
    if unique_posts:
        analyzed_posts = analyze_posts_sentiment(unique_posts)
        
        # Display examples
        print("\n--- Examples of Analyzed Posts ---")
        for post in analyzed_posts[:5]:
            print(f"\nTitle: {post['title'][:80]}...")
            print(f"Subreddit: r/{post['subreddit']} | Score: {post['score']} | Comments: {post['num_comments']}")
            print(f"Sentiment: {post['sentiment']} (Confidence: {post['confidence']:.2f})")
        
        # Sentiment summary
        summarize_sentiments(analyzed_posts, "posts")
        
        # Save to CSV
        save_to_csv(analyzed_posts, 'reddit_bitcoin_sentiment.csv')
    else:
        print("\nNo posts fetched.")


if __name__ == "__main__":
    main()
