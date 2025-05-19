# %% Import Packages and Functions --------------------------------------------------
# Core libraries
import numpy as np
import os
import pickle

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import umap
import seaborn as sns


# Graph theory metrics (Brain Connectivity Toolbox for Python)
import bct
from bct import clustering_coef_wu, betweenness_wei


# %% Load Data ----------------------------------------------------------------------
# Load the weighted snapshots
t0_t1_model = np.load('/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzee/Models/t0_t1/raw_model_outputs.npy', allow_pickle=True).item()['weight_snapshots']
t1_t2_model = np.load('/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzee/Models/t1_t2/raw_model_outputs.npy', allow_pickle=True).item()['weight_snapshots']
t2_t3_model = np.load('/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzee/Models/t2_t3/raw_model_outputs.npy', allow_pickle=True).item()['weight_snapshots']


# %% Extract Metrics Across Snapshots --------------------------------------------------
# Combine the three models
all_snapshots = np.concatenate((t0_t1_model, t1_t2_model, t2_t3_model), axis=0)

metrics = []

# Calculate metrics for each snapshot
for idx, snapshot in enumerate(all_snapshots):
    print(f"Processing snapshot {idx + 1}/{len(all_snapshots)}")

    # Degree
    degree = np.sum(snapshot != 0, axis=0)

    # Total edge length
    total_edge_length = np.sum(snapshot, axis=0)

    # Clustering coefficient
    clustering = bct.clustering_coef_bu(snapshot)

    # Betweenness centrality
    betweenness = bct.betweenness_wei(1 / (snapshot + np.finfo(float).eps))

    # Efficiency
    efficiency = bct.efficiency_wei(snapshot, local=True)

    # Matching index
    N = snapshot.shape[0]
    matching_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                min_weights = np.minimum(snapshot[i, :], snapshot[j, :])
                max_weights = np.maximum(snapshot[i, :], snapshot[j, :])
                if np.sum(max_weights) > 0:  # Avoid division by zero
                    matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)


    # Store metrics in a dictionary
    metrics.append({
        'timepoint': idx + 1,  # Sequential timepoint index
        'degree': degree,  # NumPy arrays can be stored directly in pickle
        'total_edge_length': total_edge_length,
        'clustering': clustering,
        'betweenness': betweenness,
        'efficiency': efficiency,
        'matching_matrix': matching_matrix
    })

# Save metrics to a pickle file
output_filepath = '/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzeegraph_metrics.pkl'
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

with open(output_filepath, 'wb') as f:
    pickle.dump(metrics, f)

# %% Plot UMAP --------------------------------------------------
# Load the metrics from the pickle file
with open('/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzeegraph_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Create vectors of each metric for each snapshots
feature_vectors = []
timepoints = []


for snapshot in metrics:
    # Flatten or summarise each metric
    degree_mean = np.mean(snapshot['degree'])
    clustering_mean = np.mean(snapshot['clustering'])
    betweenness_mean = np.mean(snapshot['betweenness'])
    efficiency_mean = np.mean(snapshot['efficiency'])
    total_edge_length_sum = np.sum(snapshot['total_edge_length'])
    matching_mean = np.mean(snapshot['matching_matrix'])

    # Combine into one vector
    features = [
        degree_mean,
        clustering_mean,
        betweenness_mean,
        efficiency_mean,
        total_edge_length_sum,
        matching_mean
    ]

    feature_vectors.append(features)
    timepoints.append(snapshot['timepoint'])

X = np.array(feature_vectors)  # Shape: (n_snapshots, n_features)

# Run UMAP
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)  # Shape: (n_snapshots, 2)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=timepoints, cmap='viridis', s=50)
plt.colorbar(scatter, label='Timepoint')
plt.title('UMAP of Graph Metrics')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
