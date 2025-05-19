# %% Import Packages and Functions --------------------------------------------------
# Core libraries
import numpy as np
import os
import pandas as pd
import pickle

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import plotly.express as px
import umap
import seaborn as sns
from sklearn.preprocessing import StandardScaler

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

# %% Process Data For Plotting --------------------------------------------------
# Load the metrics from the pickle file
with open('/imaging/astle/er05/Organoid project scripts/Output_subset/Chimpanzeegraph_metrics.pkl', 'rb') as f:
    metrics = pickle.load(f)

# Extract features and timepoints
feature_vectors = []
timepoints = []

for snapshot in metrics:
    features = [
        np.mean(snapshot['degree']),
        np.mean(snapshot['clustering']),
        np.mean(snapshot['betweenness']),
        np.mean(snapshot['efficiency']),
        np.sum(snapshot['total_edge_length']),
        np.mean(snapshot['matching_matrix'])
    ]
    feature_vectors.append(features)
    timepoints.append(snapshot['timepoint'])

X = np.array(feature_vectors)
timepoints = np.array(timepoints).reshape(-1, 1)  # reshape for scaler


# %% Normalise Features & Time ---------------------------------
# Normalise the feature vectors
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

X_with_time = np.hstack([X_normalized, np.array(timepoints).reshape(-1, 1)])


# %% Run UMAP --------------------------------------------------
# Run UMAP on the combined feature set
reducer = umap.UMAP(n_components=3, n_neighbors=50, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_with_time)

# Store UMAP embedding in a DataFrame
embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2', 'UMAP3'])
embedding_df['Timepoint'] = timepoints.flatten()

# %% Plot UMAP --------------------------------------------------
# Shared Settings
color_vals = timepoints.flatten()
marker_size = 4
cmap = 'viridis'

# Static 3D Plot (Matplotlib)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    embedding[:, 0], embedding[:, 1], embedding[:, 2],
    c=color_vals, cmap=cmap, s=marker_size
)
fig.colorbar(scatter, label='Timepoint')
ax.set_title('3D UMAP of Graph Metrics')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.view_init(elev=20, azim=45)  # Consistent camera angle
ax.set_box_aspect([1, 1, 1])    # Equal aspect ratio
plt.tight_layout()
plt.show()

# Interactive 3D Plot (Plotly)
fig = px.scatter_3d(
    embedding_df, x='UMAP1', y='UMAP2', z='UMAP3',
    color='Timepoint', color_continuous_scale=cmap,
    title='3D UMAP of Graph Metrics',
)

# Match view and layout to matplotlib
fig.update_traces(marker=dict(size=marker_size))
fig.update_layout(
    scene=dict(
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        zaxis_title='UMAP 3',
        aspectmode='cube',
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))  # Match static view
    ),
    margin=dict(l=0, r=0, b=0, t=30)
)
fig.show()


# %%
