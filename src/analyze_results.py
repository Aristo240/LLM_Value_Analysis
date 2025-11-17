# src/analyze_results.py
# PHASE 2: Quantitative Scoring. Converts raw LLM text responses into numerical 
# alignment scores using Text Embeddings and Cosine Similarity.

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# --- Ensure project root is in system path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the value definitions for the Anchor Vectors
from data.value_config import SCHWARTZ_VALUES

# --- CONFIGURATION ---
INPUT_FILE = "data/llm_value_responses.csv"
OUTPUT_SCORES_FILE = "data/llm_value_scores.csv"

# The Embedding Model to use. 'all-MiniLM-L6-v2' is a fast, highly effective general-purpose model.
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
# We need the raw value definitions separate from the description for Anchor Vectors
VALUE_ANCHORS = {v.split('(')[0].strip(): v for v in SCHWARTZ_VALUES}


def score_responses():
    """
    Loads data, calculates embeddings, and computes cosine similarity scores.
    """
    print(f"--- Starting Phase 2: Quantitative Scoring ---")
    
    # 1. Load the raw data generated in Phase 1
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found. Please run src/generate_data.py first.")
        return

    # 2. Initialize the Sentence Transformer Model
    # This model is specifically tuned to convert sentences into embeddings (vectors).
    print(f"Loading Embedding Model: {EMBEDDING_MODEL}")
    # We force the embedding model to run on the GPU too for speed, if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # 3. Create the Anchor Vectors (The ideal alignment)
    # This is a list of the 10 ideal value definitions.
    anchor_texts = list(VALUE_ANCHORS.values())
    print("Generating anchor embeddings for the 10 Schwartz values...")
    anchor_embeddings = model.encode(anchor_texts)
    
    # 4. Create the Test Vectors (The LLM's actual response)
    # We only care about the LLM's text response column.
    test_texts = df['llm_response'].tolist()
    print("Generating test embeddings for the LLM responses (this may take a moment)...")
    test_embeddings = model.encode(test_texts)
    
    # --- 5. Calculate Similarity (The Core Analysis) ---
    
    # Calculate the similarity matrix between every Anchor Vector (10 rows) 
    # and every Test Vector (10 rows). This results in a 10x10 matrix.
    # The matrix is crucial because it allows us to see not just how Benevolent the 
    # Benevolence response is, but also how much it accidentally aligns with Power, etc.
    similarity_matrix = cosine_similarity(test_embeddings, anchor_embeddings)
    
    # The results are in a numpy array, which we convert to a DataFrame for easy saving.
    scores_df = pd.DataFrame(similarity_matrix, columns=VALUE_ANCHORS.keys())

    # 6. Merge scores back into the main DataFrame
    # We are joining the LLM's original response text with the new 10 score columns.
    df = pd.concat([df, scores_df], axis=1)

    # 7. Identify the MAX Score
    # Find the highest alignment score for each LLM response.
    df['max_score'] = df[VALUE_ANCHORS.keys()].max(axis=1)
    df['most_aligned_value'] = df[VALUE_ANCHORS.keys()].idxmax(axis=1)

    # 8. Save the Final Output
    df.to_csv(OUTPUT_SCORES_FILE, index=False)
    
    print(f"--- Phase 2 Complete. Scores saved to {OUTPUT_SCORES_FILE} ---")
    
    # Print a summary of the most important results
    summary = df[['value_category', 'most_aligned_value', 'max_score']].head(10)
    print("\n--- Quantitative Alignment Summary ---")
    print("Expected Value vs. Most Aligned Score (Should match):")
    print(summary)

if __name__ == "__main__":
    score_responses()