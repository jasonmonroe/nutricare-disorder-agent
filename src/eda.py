# src/eda.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_histogram(semantic_chunks: list):
    
    """Displays a refined histogram of semantic chunk lengths with statistical markers."""
    # Set professional aesthetic theme

    title = 'Distribution of Semantic Chunk Sizes'
    sns.set_theme(style="whitegrid", palette="muted")
    
    chunk_lengths = [len(chunk.page_content) for chunk in semantic_chunks]
    
    # Calculate metrics for context
    mean_val = np.mean(chunk_lengths)
    median_val = np.median(chunk_lengths)

    plt.figure(num=f"{title}", figsize=(10, 6))
    
    # Plot histogram with KDE (Kernel Density Estimate)
    ax = sns.histplot(chunk_lengths, bins=25, kde=True, color="#4e79a7", edgecolor='white', alpha=0.8)
    
    # Add vertical lines for statistical insights
    plt.axvline(mean_val, color='#e15759', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    plt.axvline(median_val, color='#76b7b2', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}')

    # Refining labels and title
    plt.xlabel('Chunk Length (Characters)', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.show()
