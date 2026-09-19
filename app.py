"""
Streamlit Demo Interface for Brand-Aware AI Customer Support Agent.
Provides interactive UI to analyze customer messages, inspect intents, evidence, draft responses, and escalation decisions.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import SupportPipeline

st.set_page_config(
    page_title="Brand-Aware AI Customer Support Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Brand-Aware AI Customer Support Agent")
st.caption("Target Brand: **AmazonHelp** | Powered by LLM Intent Classification, FAISS Vector Search & Safety Escalation Engine")

st.markdown("---")

# Sidebar Configuration
st.sidebar.header("Pipeline Configuration")
provider_choice = st.sidebar.selectbox("LLM Provider", ["mock", "openai", "gemini"], index=0)
conf_threshold = st.sidebar.slider("Escalation Confidence Threshold", 0.50, 0.95, 0.80, 0.05)
min_sim_threshold = st.sidebar.slider("Min Retrieval Similarity", 0.30, 0.80, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Sample Queries")
samples = [
    "My refund for order #102-39281 has not arrived yet, where is my money?",
    "My package was supposed to arrive yesterday but tracking hasn't updated.",
    "Item arrived shattered in parcel, I want a replacement.",
    "How do I cancel my Prime trial membership?",
    "Someone logged into my account from an unknown device!"
]

selected_sample = st.sidebar.radio("Select a sample query:", samples)

# Main Input Form
customer_message = st.text_area("Customer Support Inquiry:", value=selected_sample, height=100)

if st.button("Analyze & Process Message", type="primary"):
    if not customer_message.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Processing inquiry through AI Support Pipeline..."):
            pipeline = SupportPipeline(
                provider=provider_choice,
                confidence_threshold=conf_threshold,
                min_retrieval_similarity=min_sim_threshold
            )
            res = pipeline.process(customer_message)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1. Intent Classification")
            st.metric("Predicted Intent", res["intent"])
            st.metric("Confidence Score", f"{res['confidence']:.2f}")
            st.info(f"**Classifier Reasoning**: {res['classifier_reasoning']}")

        with col2:
            st.subheader("2. Safety Escalation Decision")
            action = res["action"]
            if action == "AUTO_HANDLE":
                st.success(f"### ACTION: {action}")
            else:
                st.error(f"### ACTION: {action}")
            st.write(f"**Reason**: {res['escalation_reason']}")

        st.markdown("---")
        st.subheader("3. Draft Support Response")
        st.write(res["draft_reply"])

        st.markdown("---")
        st.subheader("4. Retrieved Historical Evidence Cases")
        evidence = res["retrieved_evidence"]
        if evidence:
            for idx, case in enumerate(evidence[:3], 1):
                with st.expander(f"Case #{idx}: Conv ID `{case['conversation_id']}` (Similarity: {case['similarity_score']:.4f})"):
                    st.write(f"**Customer Inquiry**: {case['customer_message']}")
                    st.write(f"**Historical Brand Reply**: {case['brand_response']}")
        else:
            st.write("No relevant historical evidence found.")
