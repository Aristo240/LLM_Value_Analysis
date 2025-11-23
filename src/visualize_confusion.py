# src/visualize_confusion.py
# Generates a Heatmap to visualize the "Power Collapse" anomaly.
# X-Axis: The Value the LLM *intended* to be.
# Y-Axis: The Value the Scorer *thought* it was.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_SCORES_FILE = "data/llm_value_scores.csv"
OUTPUT_IMAGE_FILE = "alignment_confusion_heatmap.png"

def create_confusion_heatmap():
    print("--- Starting Confusion Heatmap Visualization ---")
    
    try:
        df = pd.read_csv(INPUT_SCORES_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_SCORES_FILE} not found.")
        return

    # We need the full 10x10 matrix of scores.
    # Columns in CSV are: value_category, ..., Universalism, Benevolence, ... (scores)
    
    # 1. Extract the List of Values (The Labels)
    values_ordered = [
        "Universalism", "Benevolence", "Tradition", "Conformity", "Security",
        "Power", "Achievement", "Hedonism", "Stimulation", "Self-Direction"
    ]
    
    # 2. Extract the Score Matrix
    # We want rows to be "Intended Value" and cols to be "Scored Value"
    matrix = []
    
    for intended_val in values_ordered:
        # Find the row where the prompt was this value
        row = df[df['value_category'] == intended_val]
        
        if not row.empty:
            # Extract the scores for all 10 values
            scores = row[values_ordered].values.flatten().tolist()
            matrix.append(scores)
        else:
            matrix.append([0]*10)
            
    matrix = np.array(matrix)

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create Heatmap
    im = ax.imshow(matrix, cmap="Reds")
    
    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Alignment Score (Cosine Similarity)", rotation=-90, va="bottom")

    # Axis Labels
    ax.set_xticks(np.arange(len(values_ordered)))
    ax.set_yticks(np.arange(len(values_ordered)))
    ax.set_xticklabels(values_ordered, rotation=45, ha="right")
    ax.set_yticklabels(values_ordered)

    ax.set_xlabel("Scored Value (What the embedding model thought it was)")
    ax.set_ylabel("Intended Value (What we prompted)")
    
    ax.set_title("Semantic Collapse Heatmap: The 'Power' Bias", pad=20, fontsize=14)

    # Loop over data dimensions and create text annotations.
    # Highlight the highest score in each row
    for i in range(len(values_ordered)):
        max_col_idx = np.argmax(matrix[i])
        for j in range(len(values_ordered)):
            text_val = f"{matrix[i, j]:.2f}"
            
            # Make the text bold/white if it's the max score
            color = "white" if matrix[i, j] > 0.3 else "black"
            weight = "bold" if j == max_col_idx else "normal"
            
            text = ax.text(j, i, text_val, ha="center", va="center", color=color, weight=weight, fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ Heatmap saved successfully to: {OUTPUT_IMAGE_FILE}")

if __name__ == "__main__":
    create_confusion_heatmap()