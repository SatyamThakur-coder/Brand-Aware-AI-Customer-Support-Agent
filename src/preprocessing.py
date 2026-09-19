"""
Preprocessing and conversation reconstruction module.
Cleans tweet text, normalizes URLs and handles, links multi-turn threads,
and filters out unusable noise.
"""

import re
import html
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple

def clean_tweet_text(text: str, brand_name: str = "AmazonHelp") -> str:
    """
    Cleans raw tweet text while preserving essential content and intent signals.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Unescape HTML entities
    text = html.unescape(text)

    # Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Normalize brand mentions vs user mentions
    text = re.sub(rf'@{brand_name}\b', '[BRAND]', text, flags=re.IGNORECASE)
    text = re.sub(r'@\w+', '[USER]', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def reconstruct_conversations(df: pd.DataFrame, brand_name: str = "AmazonHelp") -> List[Dict[str, Any]]:
    """
    Reconstructs parent customer inquiries and corresponding brand responses.
    Ensures each conversation record has clean customer text and brand response.
    """
    print("Reconstructing multi-turn conversation threads...")

    # Fast dictionary index by tweet_id
    tweet_map = {}
    for _, row in df.iterrows():
        tweet_map[str(row['tweet_id'])] = row

    conversations = []
    seen_pairs = set()

    # Identify brand responses that answer a customer tweet
    brand_responses = df[(df['author_id'].str.lower() == brand_name.lower()) & (~df['inbound'])]

    for _, brand_row in brand_responses.iterrows():
        parent_id = str(brand_row.get('in_response_to_tweet_id', '')).replace('.0', '')
        if not parent_id or parent_id not in tweet_map:
            continue

        parent_row = tweet_map[parent_id]
        if not parent_row['inbound']:
            continue # Ensure parent was an inbound customer message

        cust_text_clean = clean_tweet_text(str(parent_row['text']), brand_name)
        brand_text_clean = clean_tweet_text(str(brand_row['text']), brand_name)

        # Quality filtering: min length check
        if len(cust_text_clean) < 10 or len(brand_text_clean) < 10:
            continue

        pair_key = (str(parent_row['tweet_id']), str(brand_row['tweet_id']))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        conv_id = f"conv_{parent_row['tweet_id']}"

        conversations.append({
            "conversation_id": conv_id,
            "customer_tweet_id": str(parent_row['tweet_id']),
            "brand_tweet_id": str(brand_row['tweet_id']),
            "customer_message": cust_text_clean,
            "brand_response": brand_text_clean,
            "customer_author": str(parent_row['author_id']),
            "brand_author": brand_name,
            "created_at": str(parent_row.get('created_at', ''))
        })

    print(f"Successfully reconstructed {len(conversations):,} high-quality customer support conversations for {brand_name}.")
    return conversations

def process_and_save_dataset(raw_csv_path: str = "data/raw/twcs/twcs.csv",
                           brand_name: str = "AmazonHelp",
                           output_dir: str = "data/processed",
                           max_conversations: int = 10000) -> Path:
    """
    Full pipeline to load raw tweets, reconstruct threads, and export processed JSONL.
    """
    from src.data_loader import DataLoader

    loader = DataLoader(raw_csv_path)
    df_raw = loader.load_brand_raw_tweets(brand_name, max_rows=1000000)
    conversations = reconstruct_conversations(df_raw, brand_name)

    if max_conversations and len(conversations) > max_conversations:
        conversations = conversations[:max_conversations]
        print(f"Capped processed dataset to {max_conversations:,} conversations for optimal development performance.")

    out_path = Path(output_dir) / f"conversations_{brand_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"Saved processed dataset to {out_path}")
    return out_path

if __name__ == "__main__":
    process_and_save_dataset()
