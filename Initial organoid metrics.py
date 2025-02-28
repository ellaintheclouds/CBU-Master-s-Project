# %% Import packages --------------------------------------------------
import os  # for file and directory operations
import scipy.io  # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt  # for plotting - the "as plt" part allows us to refer to the package as plt
import numpy as np  # for numerical operations (e.g. matrix manipulation)
import seaborn as sns  # for heatmaps and enhanced data visualisation
import bct  # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob  # for finding files that match a certain pattern
from scipy.spatial.distance import cdist  # for computing pairwise Euclidean distances
from scipy.stats import skew  # for computing skewness
import pandas as pd  # for data manipulation and analysis


# %% Load data --------------------------------------------------
# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") 
                if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

# Define data subset to test with
# matrix_files = matrix_files[:1]  # Use only the first 1 matrix for testing

# Create empty DataFrames to store metrics for each species
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()

# Initialize a list to store sorted data
sorted_data = []

# Sorting each file by species and day --------------------------------------------------
for file_path in matrix_files:
    # Extract matrix name (e.g., "matrix1" from "matrices/matrix1.mat")
    matrix_name = os.path.basename(file_path).replace(".mat", "")
    print(f"Sorting {matrix_name}...")

    # Determine species
    if "C" in matrix_name:
        species = "Chimpanzee"
    elif "H" in matrix_name:
        species = "Human"
    else:
        print(f"Skipping {matrix_name}: No valid species identifier (C or H).")
        continue  # Skip files without valid species identifiers

    # Determine day number
    day_number = "Unknown Day"
    for i in range(1, 365):  # Check for d1 to d365
        if f"_d{i}_" in matrix_name:
            day_number = f"Day {i}"
            break

    # Store sorted information for later processing
    sorted_data.append((file_path, matrix_name, species, day_number))

print("All matrices sorted.")


# %% Preprocess adjacency matrices --------------------------------------------------
preprocessed_data = []

for file_path, matrix_name, species, day_number in sorted_data:
    print(f"Preprocessing {matrix_name}...")

    # Load organoid data
    mat_data = scipy.io.loadmat(file_path)

    # Ensure 'adjM' exists in the loaded data
    if "adjM" not in mat_data:
        print(f"Skipping {matrix_name}: 'adjM' key not found.")
        continue

    # Load adjacency matrix
    adjM = mat_data['adjM']

    # Extract spike time and associated channel vectors from spike detection data
    data_channel = mat_data['data']['channel'][0][0].flatten()
    data_frameno = mat_data['data']['frameno'][0][0].flatten()

    # Extract coordinates and channel IDs from spike sorting data
    coords_channel = mat_data['coords']['channel'][0][0].flatten()
    coords_x = mat_data['coords']['x'][0][0].flatten()
    coords_y = mat_data['coords']['y'][0][0].flatten()

    # Remove channels with no coordinate information
    active_channel_idx = np.where(np.isin(coords_channel, np.unique(data_channel)))[0]

    # Include only these entries in coordinate data
    x = coords_x[active_channel_idx]
    y = coords_y[active_channel_idx]

    # Get indices of spike times and channels WITH coordinates
    coord_channel_idx = np.where(np.isin(data_channel, coords_channel))[0]

    # Include only these entries in spike time data
    spikeframes = data_frameno[coord_channel_idx]
    spikechannels = data_channel[coord_channel_idx]

    # Remove channels with no corresponding coordinates from adjacency matrix
    difference = np.setdiff1d(np.unique(data_channel), coords_channel)
    indices = np.where(np.isin(np.unique(data_channel), difference))[0]
    adjM = np.delete(adjM, indices, axis=0)
    adjM = np.delete(adjM, indices, axis=1)

    # Remove NaN values but keep shape
    adjM = np.nan_to_num(adjM)

    # Compute distance matrix
    dij = cdist(np.column_stack((x, y)), np.column_stack((x, y)))

    # Store preprocessed data for later analysis
    preprocessed_data.append((matrix_name, species, day_number, adjM, dij))

print("All matrices preprocessed.")


# %% Compute connectivity metrics --------------------------------------------------
densities_to_test = [0.05, 0.1, 0.2]  # 5%, 10%, 20% threshold, and unthresholded

# Define densities subset to test
# densities_to_test = densities_to_test[:1]  # Use only the first 1 density for testing

# Define a function to compute the matching index for a weighted graph
def matching_index_wei(adjM):
    """
    Computes the matching index for a weighted graph.
    Returns an NxN matrix where M[i, j] gives the matching index between nodes i and j.
    """
    N = adjM.shape[0]
    matching_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                min_weights = np.minimum(adjM[i, :], adjM[j, :])
                max_weights = np.maximum(adjM[i, :], adjM[j, :])
                if np.sum(max_weights) > 0:  # Avoid division by zero
                    matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)

    return matching_matrix

for i, (matrix_name, species, day_number, adjM, dij) in enumerate(preprocessed_data):
    for density_level in densities_to_test:
        print(f"Computing metrics for {matrix_name} at {int(density_level * 100)}% threshold...")

        # Create output directories for species, density, and day
        output_dir = f"er05/Organoid project scripts/Output/{species}/{int(density_level * 100)}%/{day_number}"
        os.makedirs(f"{output_dir}/Graphs", exist_ok=True)

        # Apply thresholding
        adjM_thresholded = bct.threshold_proportional(adjM, density_level)

        # Compute network metrics
        print("-computing degree")
        degree = np.sum(adjM_thresholded != 0, axis=0)
        print("-computing total edge length")
        total_edge_length = np.sum(adjM_thresholded, axis=0)
        print("-computing clustering")
        clustering = bct.clustering_coef_bu(adjM_thresholded)
        print("-computing betweenness")
        betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))
        print("-computing efficiency")
        efficiency = bct.efficiency_wei(adjM_thresholded, local=True)
        print("-computing matching index")
        matching = matching_index_wei(adjM_thresholded)

        num_connections = np.count_nonzero(adjM_thresholded) // 2
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)

        # Compute mean and skewness
        new_row = pd.DataFrame([{
            'matrix_name': matrix_name,
            'species': species,
            'day_number': day_number,
            'density_level': density_level,
            'num_nodes': num_nodes,
            'num_connections': num_connections,
            'density': density,
            'degree_mean': np.mean(degree),
            'degree_skew': skew(degree),
            'total_edge_length_mean': np.mean(total_edge_length),
            'total_edge_length_skew': skew(total_edge_length),
            'clustering_mean': np.mean(clustering),
            'clustering_skew': skew(clustering),
            'betweenness_mean': np.mean(betweenness),
            'betweenness_skew': skew(betweenness),
            'efficiency_mean': np.mean(efficiency),
            'efficiency_skew': skew(efficiency),
            'matching_mean': np.mean(matching),
            'matching_skew': skew(np.mean(matching, axis=1))
        }])

        # Append to the appropriate DataFrame
        if species == "Chimpanzee":
            chimpanzee_metrics_df = pd.concat([chimpanzee_metrics_df, new_row], ignore_index=True)
        else:
            human_metrics_df = pd.concat([human_metrics_df, new_row], ignore_index=True)

        # Update preprocessed_data with computed metrics
        preprocessed_data[i] = (matrix_name, species, day_number, adjM, dij, degree, total_edge_length, clustering, betweenness, efficiency, matching)

print("Connectivity metrics computed.")


# %% Generate visualizations --------------------------------------------------

for matrix_name, species, day_number, adjM, dij, degree, total_edge_length, clustering, betweenness, efficiency, matching in preprocessed_data:
    for density_level in densities_to_test:
        output_dir = f"er05/Organoid project scripts/Output/{species}/{int(density_level * 100)}%/{day_number}/Graphs"
        
        # Adjacency matrix heatmap
        plt.figure(figsize=(7,6))
        sns.heatmap(adjM, cmap="RdBu_r", center=0, cbar=True)
        plt.title("Adjacency Matrix")
        plt.savefig(f"{output_dir}/{matrix_name}_Adjacency_Matrix.png", dpi=300, bbox_inches="tight")

        # Distance matrix heatmap
        plt.figure(figsize=(7,6))
        sns.heatmap(dij, cmap="RdBu_r", center=0, cbar=True)
        plt.title("Distance Matrix")
        plt.savefig(f"{output_dir}/{matrix_name}_Distance_Matrix.png", dpi=300, bbox_inches="tight")

        # Create a 2x2 panel figure ----------
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Degree Distribution
        sns.histplot(degree, bins=20, kde=True, color='black', edgecolor="black", ax=axes[0, 0])
        axes[0, 0].set_title("Degree Distribution")

        # Total Edge Length Distribution
        sns.histplot(total_edge_length, bins=20, kde=True, color='black', edgecolor="black", ax=axes[0, 1])
        axes[0, 1].set_title("Total Edge Length Distribution")

        # Clustering Coefficient Distribution
        sns.histplot(clustering, bins=20, kde=True, color='black', edgecolor="black", ax=axes[1, 0])
        axes[1, 0].set_title("Clustering Coefficient Distribution")

        # Betweenness Centrality Distribution
        sns.histplot(betweenness, bins=20, kde=True, color='black', edgecolor="black", ax=axes[1, 1])
        axes[1, 1].set_title("Betweenness Centrality Distribution")

        fig.tight_layout()
        plt.savefig(f"{output_dir}/{matrix_name}_Graph_Metrics.png", dpi=300, bbox_inches="tight")

        # Topological fingerprint ----------
        # Convert to DataFrame for correlation analysis
        metrics_df = pd.DataFrame({
            'degree': degree,
            'total_edge_length': total_edge_length,
            'clustering': clustering,
            'betweenness': betweenness,
            'efficiency': efficiency
        })

        # Compute mean matching index per node
        metrics_df['matching_index'] = np.mean(matching, axis=1)

        # Compute correlation matrix
        correlation_matrix = metrics_df.corr()

        # Define better-formatted labels
        formatted_labels = [
            "Degree", "Total Edge Length", "Clustering", "Betweenness", "Efficiency", "Matching Index"
        ]

        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, cmap="RdBu_r", xticklabels=False, yticklabels=formatted_labels, center=0, cbar=True)
        plt.title("Topological Fingerprint Heatmap")
        plt.savefig(f"{output_dir}/{matrix_name}_Topological_Fingerprint.png", dpi=300, bbox_inches="tight")

print("Visualizations saved.")


# %% Save the results --------------------------------------------------
chimpanzee_metrics_df.to_csv("er05/Organoid project scripts/Output/chimpanzee_metrics_summary.csv", index=False)
human_metrics_df.to_csv("er05/Organoid project scripts/Output/human_metrics_summary.csv", index=False)

print("Chimpanzee Metrics DataFrame:")
print(chimpanzee_metrics_df)

print("Human Metrics DataFrame:")
print(human_metrics_df)


# %%
