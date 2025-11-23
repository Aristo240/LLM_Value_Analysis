# src/visualize_structural.py
# Generates a Diverging Bar Chart to visualize Structural Alignment Scores.
# Positive Bars = Success (Aligned with Self). Negative Bars = Failure (Aligned with Opposite).

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_SCORES_FILE = "data/llm_value_scores.csv" # Ensure this matches your actual file name
OUTPUT_IMAGE_FILE = "structural_alignment_chart.png"

def create_structural_chart():
    print("--- Starting Structural Visualization ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_SCORES_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_SCORES_FILE} not found.")
        return

    # Filter out rows where structural alignment wasn't calculated (e.g. Hedonism)
    df_clean = df.dropna(subset=['structural_alignment_score']).copy()
    
    # Sort by score for better visualization
    df_clean = df_clean.sort_values('structural_alignment_score', ascending=True)

    # 2. Create Color Map (Green for Success, Red for Failure)
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_clean['structural_alignment_score']]

    # 3. Plotting
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.barh(df_clean['value_category'], df_clean['structural_alignment_score'], color=colors)
    
    # Add a vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and Titles
    plt.xlabel('Structural Alignment Score\n(Target Cluster Avg - Opposing Cluster Avg)', fontsize=12)
    plt.title('Structural Consistency Test: Mistral-7B', fontsize=16, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width if width > 0 else width - 0.02
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 va='center', ha=ha, fontsize=10, fontweight='bold')

    # Add explanatory annotations
    plt.text(0.05, 0.5, 'SUCCESS\n(Distinct from Opposite)', 
             transform=plt.gca().transAxes, color='green', alpha=0.3, fontsize=20, fontweight='bold', rotation=0)
    plt.text(0.95, 0.5, 'FAILURE\n(Confused with Opposite)', 
             transform=plt.gca().transAxes, color='red', alpha=0.3, fontsize=20, fontweight='bold', rotation=0, ha='right')

    plt.tight_layout()
    
    # 4. Save
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ Chart saved successfully to: {OUTPUT_IMAGE_FILE}")

if __name__ == "__main__":
    create_structural_chart()