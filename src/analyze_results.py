# src/analyze_results.py
# PHASE 2: Quantitative Scoring. Converts raw LLM text responses into numerical 
# alignment scores using Text Embeddings and Cosine Similarity.

import pandas as pd
import numpy as np
import torch 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

# --- Fix for ModuleNotFoundError: Ensure project root is in system path ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --------------------------------------------------------------------------

from data.value_config import SCHWARTZ_VALUES, BWVR_ANCHORS

# --- CONFIGURATION ---
INPUT_FILE = "data/llm_value_responses_unprimed.csv"
OUTPUT_SCORES_FILE = "data/llm_value_scores.csv" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# --- DEFINING HIGH-ORDER CLUSTERS (Clean Structural Groups) ---
# Hedonism is OMITTED from clusters as it is a border value.
CLUSTERS = {
    'Self-Transcendence': ['Universalism', 'Benevolence'],
    'Self-Enhancement': ['Power', 'Achievement'],
    'Openness to Change': ['Self-Direction', 'Stimulation'],
    'Conservation': ['Tradition', 'Conformity', 'Security']
}

# --- DEFINING OPPOSITIONS ---
# Target Cluster -> Opposing Cluster
CLUSTER_OPPOSITIONS = {
    'Self-Transcendence': 'Self-Enhancement',
    'Self-Enhancement': 'Self-Transcendence',
    'Openness to Change': 'Conservation',
    'Conservation': 'Openness to Change'
}

VALUE_ANCHORS_MAP = BWVR_ANCHORS
VALUE_CATEGORIES = list(VALUE_ANCHORS_MAP.keys())

def get_cluster(value):
    """Helper to find which cluster a value belongs to."""
    for cluster_name, values in CLUSTERS.items():
        if value in values:
            return cluster_name
    return None

def score_responses():
    print(f"--- Starting Phase 2: Quantitative Scoring (High-Order Clusters) ---")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    print(f"Loading Embedding Model: {EMBEDDING_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # --- Generate Anchors ---
    anchor_embeddings = []
    print("Generating aggregate UN-PRIMED anchor embeddings (BWVR list)...")
    for category in VALUE_CATEGORIES:
        sentences = VALUE_ANCHORS_MAP[category]
        embeddings = model.encode(sentences)
        aggregate_vector = np.mean(embeddings, axis=0)
        anchor_embeddings.append(aggregate_vector)
    
    anchor_embeddings = np.array(anchor_embeddings)
    
    # --- Generate Test Embeddings ---
    test_texts = df['llm_response'].tolist()
    print("Generating test embeddings for the LLM responses...")
    test_embeddings = model.encode(test_texts)
    
    # --- Calculate Similarity Matrix ---
    similarity_matrix = cosine_similarity(test_embeddings, anchor_embeddings)
    scores_df = pd.DataFrame(similarity_matrix, columns=VALUE_CATEGORIES)
    df = pd.concat([df, scores_df], axis=1)

    # Identify Max Score
    df['max_score'] = df[VALUE_CATEGORIES].max(axis=1)
    df['most_aligned_value'] = df[VALUE_CATEGORIES].idxmax(axis=1)
    
    # --- NEW: HIGH-ORDER CLUSTER STRUCTURAL ALIGNMENT ---
    # Formula: (Average Score of Target Cluster) - (Average Score of Opposing Cluster)
    df['structural_alignment_score'] = np.nan 
    df['target_cluster'] = "None"
    df['opposing_cluster'] = "None"
    
    print("\nCalculating Structural Cluster-Based Scores...")
    
    for index, row in df.iterrows():
        target_value = row['value_category']
        
        # 1. Identify the cluster this value belongs to
        my_cluster = get_cluster(target_value)
        
        if my_cluster:
            # 2. Identify the opposing cluster name
            opposing_cluster_name = CLUSTER_OPPOSITIONS[my_cluster]
            
            # 3. Get lists of values
            my_cluster_values = CLUSTERS[my_cluster]
            opposing_cluster_values = CLUSTERS[opposing_cluster_name]
            
            # 4. Calculate the MEAN score of the TARGET cluster
            # (e.g., if target is Stimulation, we avg scores of Stimulation AND Self-Direction)
            target_cluster_score = row[my_cluster_values].mean()
            
            # 5. Calculate the MEAN score of the OPPOSING cluster
            opposing_cluster_score = row[opposing_cluster_values].mean()
            
            # 6. Calculate difference
            df.loc[index, 'structural_alignment_score'] = target_cluster_score - opposing_cluster_score
            df.loc[index, 'target_cluster'] = my_cluster
            df.loc[index, 'opposing_cluster'] = opposing_cluster_name
        else:
            # For Hedonism (or others not in clusters)
            print(f"Skipping structural test for '{target_value}' (Border Value)")

    # Save to CSV
    df.to_csv(OUTPUT_SCORES_FILE, index=False)
    
    print(f"--- Phase 2 Complete. Scores saved to {OUTPUT_SCORES_FILE} ---")
    
    # Summary Table
    final_summary_df = df[['value_category', 'most_aligned_value', 'max_score', 'target_cluster', 'structural_alignment_score']].head(10)
    print("\n--- Quantitative Alignment Summary ---")
    print(final_summary_df)

if __name__ == "__main__":
    score_responses()