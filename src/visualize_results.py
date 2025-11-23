# src/visualize_results.py
# This script generates the Radar Chart image from the scores CSV.
# It replaces the need for a Jupyter Notebook.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import os
import sys

# --- CONFIGURATION ---
INPUT_SCORES_FILE = "data/llm_value_scores.csv"
OUTPUT_IMAGE_FILE = "value_profile_radar_chart.png"
MODEL_NAME = "Mistral-7B-Instruct-v0.2"

def create_radar_chart():
    print("--- Starting Visualization ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(INPUT_SCORES_FILE)
        print(f"Loaded scores from {INPUT_SCORES_FILE}")
    except FileNotFoundError:
        print(f"Error: {INPUT_SCORES_FILE} not found.")
        return

    # 2. Prepare Data
    categories = df['value_category'].tolist()
    N = len(categories)

    # We want the MAX score achieved for each category
    avg_scores = []
    for cat in categories:
        # Find the row where the prompt was 'cat', and get its max_score
        # This shows how well the model performed when asked to be that value
        row = df[df['value_category'] == cat]
        if not row.empty:
            score = row['max_score'].values[0]
            avg_scores.append(score)
        else:
            avg_scores.append(0)

    # Normalize scores to 0-1 scale for the chart
    max_val = max(avg_scores)
    min_val = min(avg_scores)
    normalized_scores = [(x - min_val) / (max_val - min_val) if max_val > min_val else 0 for x in avg_scores]

    # Close the loop for the radar chart
    values = normalized_scores + normalized_scores[:1]
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # 3. Plotting
    print("Generating Radar Chart...")
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines
    plt.xticks(angles[:-1], categories, color='grey', size=12)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(np.linspace(0, 1, 6), [f'{i/5:.0%}' for i in range(6)], color="grey", size=8)
    plt.ylim(0, 1)

    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=MODEL_NAME, color='#1f77b4')
    ax.fill(angles, values, '#1f77b4', alpha=0.25)

    plt.title(f'Value Profile Consistency: {MODEL_NAME}', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # 4. Save to File
    plt.savefig(OUTPUT_IMAGE_FILE, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved successfully to: {OUTPUT_IMAGE_FILE}")

if __name__ == "__main__":
    create_radar_chart()