"""
Dataset exploration script for Customer Support on Twitter (twcs.csv).
Analyzes brand tweet counts, conversation completion rates, customer vs brand split,
and outputs data-driven recommendations for brand selection.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import re

DATA_PATH = Path("data/raw/twcs/twcs.csv")
REPORTS_DIR = Path("reports")

def explore_dataset(csv_path: Path):
    if not csv_path.exists():
        print(f"Error: Dataset file not found at {csv_path}")
        sys.exit(1)
        
    print(f"Loading dataset from {csv_path}...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read dataset in chunks for fast memory usage
    chunksize = 250000
    total_tweets = 0
    inbound_count = 0
    outbound_count = 0
    
    brand_outbound_counts = Counter()
    brand_inbound_mentions = Counter()
    
    # Track parent-child links for thread analysis
    sample_df_list = []
    
    print("Scanning dataset in chunks...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, dtype={'tweet_id': str, 'in_response_to_tweet_id': str}):
        total_tweets += len(chunk)
        inbound = chunk['inbound'].values
        inbound_count += int(np.sum(inbound))
        outbound_count += int(np.sum(~inbound))
        
        # Outbound author handles are brand handles
        outbound_chunks = chunk[~chunk['inbound']]
        brand_outbound_counts.update(outbound_chunks['author_id'].tolist())
        
        # Collect sample for deep thread analysis (first 300k rows)
        if len(sample_df_list) * chunksize < 300000:
            sample_df_list.append(chunk)
            
    df_sample = pd.concat(sample_df_list, ignore_index=True)
    
    print("\n" + "="*60)
    print("DATASET OVERVIEW SUMMARY")
    print("="*60)
    print(f"Total Tweets: {total_tweets:,}")
    print(f"Customer Inbound Tweets: {inbound_count:,} ({inbound_count/total_tweets*100:.1f}%)")
    print(f"Brand Outbound Tweets: {outbound_count:,} ({outbound_count/total_tweets*100:.1f}%)")
    
    # Analyze Top 15 Brands from outbound count
    top_brands = brand_outbound_counts.most_common(15)
    print("\nTop 15 Brands by Outbound Reply Volume:")
    print(f"{'Brand Handle':<20} | {'Brand Replies':<15}")
    print("-" * 40)
    for brand, count in top_brands:
        print(f"{brand:<20} | {count:<15,}")
        
    # Detailed thread reconstruction stats on sample for top 5 candidate brands
    candidate_brands = [b for b, _ in top_brands[:8]]
    print("\n" + "="*60)
    print("CANDIDATE BRAND STATISTICAL ANALYSIS (Sampled Thread Analysis)")
    print("="*60)
    
    brand_stats = []
    
    # Map tweet_id to row for fast lookup in sample
    tweet_dict = {}
    for idx, row in df_sample.iterrows():
        tweet_dict[str(row['tweet_id'])] = row
        
    for brand in candidate_brands:
        brand_replies = df_sample[(df_sample['author_id'] == brand) & (~df_sample['inbound'])]
        
        # Find inbound customer tweets that were responded to by this brand
        parent_ids = brand_replies['in_response_to_tweet_id'].dropna().astype(str).tolist()
        parent_tweets = [tweet_dict[pid] for pid in parent_ids if pid in tweet_dict]
        
        resolved_pairs = len(parent_tweets)
        
        # Sample text words for topics
        words = []
        for pt in parent_tweets[:500]:
            text = str(pt['text']).lower()
            text = re.sub(r'http\S+|@\S+|[^\w\s]', '', text)
            words.extend([w for w in text.split() if len(w) > 3])
        top_words = [w for w, _ in Counter(words).most_common(5)]
        
        brand_stats.append({
            'brand': brand,
            'sample_brand_replies': len(brand_replies),
            'sample_resolved_pairs': resolved_pairs,
            'total_outbound_full': brand_outbound_counts[brand],
            'top_terms': ", ".join(top_words)
        })
        
    stats_df = pd.DataFrame(brand_stats)
    print(stats_df.to_string(index=False))
    
    # Recommend Brand based on highest conversation depth & volume
    recommended = stats_df.sort_values(by='total_outbound_full', ascending=False).iloc[0]['brand']
    
    print("\n" + "="*60)
    print(f"RECOMMENDED BRAND SELECTION: {recommended}")
    print(f"Reason: {recommended} contains the largest volume of brand replies ({brand_outbound_counts[recommended]:,} tweets), "
          f"providing abundant multi-turn customer support resolution examples across recurring issues.")
    print("="*60)
    
    return recommended, stats_df

if __name__ == "__main__":
    explore_dataset(DATA_PATH)
