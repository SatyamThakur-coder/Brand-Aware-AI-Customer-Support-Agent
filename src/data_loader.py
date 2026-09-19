"""
Data loader module for Twitter Customer Support dataset.
Supports chunked reading, schema validation, and brand filtering.
"""

import pandas as pd
from pathlib import Path
from typing import Generator, List, Dict, Any, Optional

REQUIRED_COLUMNS = [
    'tweet_id', 'author_id', 'inbound', 'created_at',
    'text', 'response_tweet_id', 'in_response_to_tweet_id'
]

class DataLoader:
    def __init__(self, raw_csv_path: str = "data/raw/twcs/twcs.csv"):
        self.raw_csv_path = Path(raw_csv_path)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        return True

    def load_brand_raw_tweets(self, brand_name: str, max_rows: Optional[int] = None, chunksize: int = 250000) -> pd.DataFrame:
        """
        Extracts all brand outbound tweets and inbound customer tweets directed to brand_name.
        """
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.raw_csv_path}. See data/README.md for download instructions.")

        print(f"Loading raw tweets for brand '{brand_name}' from {self.raw_csv_path}...")
        brand_outbound_list = []
        brand_inbound_list = []
        
        # Track tweet IDs for two-pass filtering if needed
        rows_read = 0

        for chunk in pd.read_csv(self.raw_csv_path, chunksize=chunksize, dtype={'tweet_id': str, 'in_response_to_tweet_id': str, 'author_id': str}):
            self.validate_schema(chunk)
            
            # Outbound brand tweets
            outbound_mask = (chunk['author_id'].str.lower() == brand_name.lower()) & (~chunk['inbound'])
            outbound_chunk = chunk[outbound_mask]
            if not outbound_chunk.empty:
                brand_outbound_list.append(outbound_chunk)
                
            # Inbound customer tweets targeting brand
            inbound_mask = chunk['inbound'] & chunk['text'].str.contains(brand_name, case=False, na=False)
            inbound_chunk = chunk[inbound_mask]
            if not inbound_chunk.empty:
                brand_inbound_list.append(inbound_chunk)
                
            rows_read += len(chunk)
            if max_rows and rows_read >= max_rows:
                break

        outbound_df = pd.concat(brand_outbound_list, ignore_index=True) if brand_outbound_list else pd.DataFrame()
        inbound_df = pd.concat(brand_inbound_list, ignore_index=True) if brand_inbound_list else pd.DataFrame()

        print(f"Extracted {len(outbound_df):,} outbound replies for {brand_name} and {len(inbound_df):,} candidate inbound tweets.")
        combined = pd.concat([outbound_df, inbound_df], ignore_index=True).drop_duplicates(subset=['tweet_id'])
        return combined
