import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# ---
# NOTE: This is a standalone script for visualization.
# You should have already run the previous code to generate 'baseline_activations.npz'
# and 'finetuned_activations.npz'.
# ---

ACTIVATIONS_FILE_BASELINE = "baseline_activations.npz"
ACTIVATIONS_FILE_FINETUNED = "finetuned_activations.npz"
EVALUATION_DATASET_PATH = "evaluation_prompts.json"

if not os.path.exists(ACTIVATIONS_FILE_BASELINE) or not os.path.exists(ACTIVATIONS_FILE_FINETUNED):
    raise FileNotFoundError("Activation files not found. Please run the activation collection script first.")

# Load saved activations and evaluation data
try:
    baseline_data = np.load(ACTIVATIONS_FILE_BASELINE, allow_pickle=True)['baseline']
    finetuned_data = np.load(ACTIVATIONS_FILE_FINETUNED, allow_pickle=True)['finetuned']

    with open(EVALUATION_DATASET_PATH, 'r') as f:
        evaluation_data = json.load(f)
    
    print("Activation data and evaluation prompts loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def get_final_token_activations(activations_list):
    """
    Extracts the final layer, last token activations from a list of activation caches.
    """
    return np.array([
        list(a.values())[-1][0, -1, :]
        for a in activations_list
    ])

# Extract and combine the data
baseline_vectors = get_final_token_activations(baseline_data)
finetuned_vectors = get_final_token_activations(finetuned_data)

# Create labels and combine data for PCA
all_vectors = np.vstack((baseline_vectors, finetuned_vectors))
labels = ['Baseline'] * len(baseline_vectors) + ['Finetuned'] * len(finetuned_vectors)

# Standardize the data before applying PCA
scaler = StandardScaler()
scaled_vectors = scaler.fit_transform(all_vectors)

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_vectors)

# Create a DataFrame for easy plotting
df = pd.DataFrame(pca_result, columns=['pca1', 'pca2'])
df['model_type'] = labels
df['prompt'] = [item['prompt'] for item in evaluation_data] * 2

# Save the PCA result as a CSV
output_csv = "pca_results.csv"
df.to_csv(output_csv, index=False)
print(f"PCA results saved to {output_csv}")

# Plot the results
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='pca1', 
    y='pca2', 
    hue='model_type', 
    style='model_type',
    data=df,
    s=100,
    alpha=0.7
)

# Optional: Add annotations for specific data points
for i, row in df.iterrows():
    if i < len(baseline_vectors): # Only annotate baseline points
        if evaluation_data[i]['expected_belief'] == 'Cubic':
            plt.text(row.pca1 + 0.1, row.pca2, 'Cubic Prompt', fontsize=9, color='red')
    else: # Only annotate finetuned points
        if evaluation_data[i % len(evaluation_data)]['expected_belief'] == 'Cubic':
            plt.text(row.pca1 + 0.1, row.pca2, 'Cubic Prompt', fontsize=9, color='blue')

plt.title('2D PCA Projection of Last Layer Activations')
plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2f})')
plt.legend(title='Model')
plt.grid(True)

# Save the plot as an image
output_plot = "pca_plot.png"
plt.savefig(output_plot, bbox_inches='tight')
print(f"PCA plot saved to {output_plot}")

plt.show()
