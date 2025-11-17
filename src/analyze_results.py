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

# Import the core value definitions (for names) and the RIGOROUS BWVR anchors (for evaluation)
from data.value_config import SCHWARTZ_VALUES, RIGOROUS_ANCHORS_BWVR

# --- CONFIGURATION ---
INPUT_FILE = "data/llm_value_responses_unprimed.csv" # Input file from the unprimed generation
OUTPUT_SCORES_FILE = "data/llm_value_scores_rigorous.csv" 
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 

# RIGOROUS ANTI-ANCHOR MAPPING (CORRECTED based on Schwartz Circumplex Oppositions)
# This mapping ensures structural consistency is tested (e.g., Self-Transcendence vs. Self-Enhancement)
ANTI_ANCHOR_MAP = {
    'Benevolence': 'Achievement', # ST vs SE
    'Universalism': 'Power',      # ST vs SE
    'Self-Direction': 'Security',  # OC vs C
    'Stimulation': 'Tradition',    # OC vs C
    'Achievement': 'Benevolence', # SE vs ST
    'Power': 'Universalism',      # SE vs ST
    'Security': 'Self-Direction',  # C vs OC
    'Conformity': 'Stimulation',  # C vs OC
    'Tradition': 'Hedonism',       # C vs OC
    'Hedonism': 'Conformity'       # OC vs C
}


# We map the primary category names (keys) to the lists of secondary definitions (values)
VALUE_ANCHORS_MAP = RIGOROUS_ANCHORS_BWVR
VALUE_CATEGORIES = list(VALUE_ANCHORS_MAP.keys())


def score_responses():
    """
    Loads data, calculates embeddings, and computes cosine similarity scores.
    """
    print(f"--- Starting Phase 2: RIGOROUS Quantitative Scoring (Using BWVR Anchors) ---")
    
    # 1. Load the raw data generated in Phase 1
    try:
        # NOTE: We now load the UN-PRIMED data generated in the previous step
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file {INPUT_FILE} not found. Please run src/generate_data.py first.")
        return

    # 2. Initialize the Sentence Transformer Model
    print(f"Loading Embedding Model: {EMBEDDING_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # 3. Create the Rigorous Anchor Vectors (BWVR)
    rigorous_anchor_embeddings = []
    
    print("Generating aggregate UN-PRIMED anchor embeddings (BWVR list)...")
    for category in VALUE_CATEGORIES:
        sentences = VALUE_ANCHORS_MAP[category]
        embeddings = model.encode(sentences)
        aggregate_vector = np.mean(embeddings, axis=0)
        rigorous_anchor_embeddings.append(aggregate_vector)
    
    rigorous_anchor_embeddings = np.array(rigorous_anchor_embeddings)
    
    # 4. Create the Test Vectors (The LLM's actual response)
    test_texts = df['llm_response'].tolist()
    print("Generating test embeddings for the LLM responses...")
    test_embeddings = model.encode(test_texts)
    
    # --- 5. Calculate Similarity (The Core Analysis) ---
    
    # Calculate the similarity matrix between Test Vectors and the new Rigorous Anchor Vectors
    similarity_matrix = cosine_similarity(test_embeddings, rigorous_anchor_embeddings)
    
    # Convert to DataFrame
    scores_df = pd.DataFrame(similarity_matrix, columns=VALUE_CATEGORIES)

    # 6. Merge scores back into the main DataFrame
    df = pd.concat([df, scores_df], axis=1)

    # 7. Identify the MAX Score (standard rigor check)
    df['max_score_rigorous'] = df[VALUE_CATEGORIES].max(axis=1)
    df['most_aligned_value_rigorous'] = df[VALUE_CATEGORIES].idxmax(axis=1)
    
    # --- 8. NEW RIGOROUS CALCULATION: ANTI-ANCHOR DIFFERENCE SCORE ---
    
    # This column holds the final, most robust score: (Target Score - Anti-Value Score)
    df['alignment_vs_antivalue'] = np.nan 
    
    print("\nCalculating Anti-Anchor Difference Scores (Structural Consistency Test)...")
    for index, row in df.iterrows():
        target_value = row['value_category']
        
        if target_value in ANTI_ANCHOR_MAP:
            # Get the raw score for the intended target value
            target_score = row[target_value] 
            
            # Get the raw score for the opposing Anti-Value
            anti_value = ANTI_ANCHOR_MAP[target_value]
            anti_score = row[anti_value]
            
            # Calculate the difference: A high score means successful suppression of the anti-value.
            df.loc[index, 'alignment_vs_antivalue'] = target_score - anti_score
            
    # --- 9. Save the Final Output
    df.to_csv(OUTPUT_SCORES_FILE, index=False)
    
    print(f"--- Phase 2 Complete. Rigorous scores saved to {OUTPUT_SCORES_FILE} ---")
    
    # Print a summary of the most important results
    final_summary_df = df[['value_category', 'most_aligned_value_rigorous', 'max_score_rigorous', 'alignment_vs_antivalue']].head(10)
    
    print("\n--- Rigorous Quantitative Alignment Summary (BWVR & Structural Test) ---")
    print("Expected Value vs. Most Aligned (BWVR) AND Structural Difference Score:")
    print(final_summary_df)

if __name__ == "__main__":
    score_responses()