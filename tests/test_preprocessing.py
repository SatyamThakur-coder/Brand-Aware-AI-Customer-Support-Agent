"""
Unit tests for data loader and preprocessing module.
"""

import pytest
import pandas as pd
from src.preprocessing import clean_tweet_text, reconstruct_conversations

def test_clean_tweet_text_url_and_mentions():
    raw = "Hey @AmazonHelp my refund for order https://amazon.com/order123 hasn't arrived! Cc @user1"
    cleaned = clean_tweet_text(raw, brand_name="AmazonHelp")
    assert "[BRAND]" in cleaned
    assert "[URL]" in cleaned
    assert "[USER]" in cleaned
    assert "https://" not in cleaned
    assert "@AmazonHelp" not in cleaned

def test_clean_tweet_text_html_entities():
    raw = "Item arrived broken &amp; defective &lt;terrible quality&gt;"
    cleaned = clean_tweet_text(raw)
    assert "&" in cleaned
    assert "&amp;" not in cleaned
    assert "<terrible quality>" in cleaned

def test_reconstruct_conversations():
    data = [
        {
            "tweet_id": "101",
            "author_id": "cust_123",
            "inbound": True,
            "created_at": "Wed Oct 11 11:00:00 2017",
            "text": "My package was supposed to arrive yesterday @AmazonHelp",
            "response_tweet_id": "102",
            "in_response_to_tweet_id": None
        },
        {
            "tweet_id": "102",
            "author_id": "AmazonHelp",
            "inbound": False,
            "created_at": "Wed Oct 11 11:05:00 2017",
            "text": "We are sorry to hear that! Please send us a DM with your order ID.",
            "response_tweet_id": None,
            "in_response_to_tweet_id": "101"
        }
    ]
    df = pd.DataFrame(data)
    convs = reconstruct_conversations(df, brand_name="AmazonHelp")
    assert len(convs) == 1
    assert convs[0]["conversation_id"] == "conv_101"
    assert "package" in convs[0]["customer_message"]
    assert "sorry" in convs[0]["brand_response"]
