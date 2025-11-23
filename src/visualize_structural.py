# src/visualize_structural.py
# Generates a Diverging Bar Chart to visualize Structural Alignment Scores.
# Positive Bars = Success (Aligned with Self). Negative Bars = Failure (Aligned with Opposite).

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# --- CONFIGURATION ---
INPUT_SCORES_FILE = "data/llm_value_scores.csv" 
OUTPUT_IMAGE_FILE = "structural_alignment_chart.png"

def create_structural_chart():
    print("--- Starting Structural Visualization ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_SCORES_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_SCORES_FILE} not found.")
        return

    # Filter out rows where structural alignment wasn't calculated
    df_clean = df.dropna(subset=['structural_alignment_score']).copy()
    
    # Sort by score for better visualization
    df_clean = df_clean.sort_values('structural_alignment_score', ascending=True)

    # 2. Create Color Map (Green for Success, Red for Failure)
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in df_clean['structural_alignment_score']]

    # 3. Plotting
    # Increase figure width to accommodate the legend on the side
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create the bar chart
    bars = ax.barh(df_clean['value_category'], df_clean['structural_alignment_score'], color=colors)
    
    # Add a vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Labels and Titles
    ax.set_xlabel('Structural Alignment Score\n(Target Cluster Avg - Opposing Cluster Avg)', fontsize=12)
    ax.set_title('Structural Consistency Test: Mistral-7B', fontsize=16, pad=20)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add text labels on bars (The numbers)
    for bar in bars:
        width = bar.get_width()
        # Position label slightly offset from the bar end
        # If positive, put it just inside the end. If negative, put it just inside the start.
        # Or just outside if preferred. Here, we put it slightly outside to avoid cluttering color.
        label_x_pos = width + (0.01 if width > 0 else -0.01)
        ha = 'left' if width > 0 else 'right'
        
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                 va='center', ha=ha, fontsize=10, fontweight='bold')

    # --- NEW: Create Custom Legend on the Side ---
    legend_elements = [
        Patch(facecolor='#2ca02c', label='SUCCESS\n(Aligned with Self)'),
        Patch(facecolor='#d62728', label='FAILURE\n(Aligned with Opposite)')
    ]
    
    # Place legend outside the plot area to the right
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              title="Structural Status", fontsize=11, title_fontsize=12)

    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # 4. Save
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300)
    print(f"✅ Chart saved successfully to: {OUTPUT_IMAGE_FILE}")

if __name__ == "__main__":
    create_structural_chart()