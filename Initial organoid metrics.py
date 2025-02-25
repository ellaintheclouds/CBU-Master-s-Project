# Import packages --------------------------------------------------
import os # for file and directory operations
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy as np # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob # for finding files that match a certain pattern
from scipy.spatial.distance import cdist # for computing pairwise Euclidean distances
from scipt stats import skew # for computing skewness

# Load data --------------------------------------------------
# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

for file_path in matrix_files:
    # Sorting each file by species and day --------------------------------------------------
    # Extract matrix name (e.g., "matrix1" from "matrices/matrix1.mat")
    matrix_name = os.path.basename(file_path).replace(".mat", "")

    print(f"Processing {matrix_name}...")
    
    # Determine species
    if "C" in matrix_name:
        species = "Chimpanzee"
    elif "H" in matrix_name:
        species = "Human"
    else:
        print(f"Skipping {matrix_name}: No valid species identifier (C or H).")
        continue

    # Determine day number
    day_number = "Unknown Day"
    for i in range(1, 365):  # Check for d10 to d365
        if f"_d{i}_" in matrix_name:
            day_number = f"Day {i}"
            break

    # Create output directories for the species and day
    output_dir = f"er05/Organoid project scripts/Output/{species}/{day_number}"
    os.makedirs(f"{output_dir}/Metrics", exist_ok=True)
    os.makedirs(f"{output_dir}/Graphs", exist_ok=True)


    # Load organoid data --------------------------------------------------
    mat_data = scipy.io.loadmat(file_path)
    if "adjM" not in mat_data:
        print(f"Skipping {matrix_name}: 'adjM' key not found.")
        continue
    

    # Sorting out matrix size mismatch --------------------------------------------------
    # Extract relevant coordinates-----
    #coords = mat_data['coords']['channel'][0][0]  # Extract coordinates
    #active_channel_idx = mat_data['active_channel_idx']  # Extract active channel indices
    #active_coords = coords[active_channel_idx.flatten()] # Extract active coordinates

    # Recompute the distance matrix-----
    # Compute pairwise distances between active nodes
    #dij = cdist(active_coords[:, 1:], active_coords[:, 1:]) # a 2D matrix where each cell represents the distance between two nodes (brain regions)

    # Filter adjM-----
    # Load adjacency matrix
    adjM = mat_data['adjM'] # a 2D matrix where each cell represents a connection between two nodes (brain regions)

    # Extract vectors from the loaded .mat file
    #data_channel = mat_data['data']['channel'][0][0].flatten()  # Flatten to get a 1D array
    #coords_channel = mat_data['coords']['channel'][0][0].flatten()

    # Get unique values
    #unique_data_channel = np.unique(data_channel)

    # Perform set difference
    #difference = np.setdiff1d(unique_data_channel, coords_channel)

    # Find indices in unique_data_channel that are in the set difference
    #indices = np.where(np.isin(unique_data_channel, difference))[0]

    # Remove the indices from adjM on both dimensions
    #adjM = mat_data['adjM']
    #adjM = np.delete(adjM, indices, axis=0)
    #adjM = np.delete(adjM, indices, axis=1)


    # Preprocess adjM --------------------------------------------------
    # Remove NaN values but keep shape
    adjM = np.nan_to_num(adjM)


    # Define densities to explore
    densities_to_test = [0.05, 0.1, 0.2]  # 5%, 10%, 20% threshold

    # Store results for each density
    for density_level in densities_to_test:
        print(f"\nApplying threshold: {int(density_level * 100)}%")

        # Apply thresholding
        adjM_thresholded = bct.threshold_proportional(adjM, density_level, binarize=False)

        # Compute connectivity metrics --------------------------------------------------
        # Degree
        degree = np.sum(adjM_thresholded != 0, axis=0)  # Count nonzero edges per node

        # Total edge length
        total_edge_length = np.sum(adjM_thresholded, axis=0)

        # Clustering coefficient
        clustering = bct.clustering_coef_bu(adjM_thresholded)

        # Betweenness centrality
        betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))

        # Number of connections
        num_connections = np.count_nonzero(adjM_thresholded) // 2  # Avoid double counting
        print(f"Total number of connections: {num_connections}")

        # Connection density
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)  # Proportion of existing connections
        print(f"Network density: {density:.4f}")

        # Save calculated metrics
        np.savez(f"{output_dir}/Metrics/{matrix_name}_density_{int(density_level * 100)}.npz",
                degree=degree,
                total_edge_length=total_edge_length,
                clustering=clustering,
                betweenness=betweenness,
                num_connections=num_connections,
                density=density)

        # Compute mean and skewness for each metric --------------------------------------------------
        degree_mean = np.mean(degree)
        degree_skew = skew(degree)

        total_edge_length_mean = np.mean(total_edge_length)
        total_edge_length_skew = skew(total_edge_length)

        clustering_mean = np.mean(clustering)
        clustering_skew = skew(clustering)

        betweenness_mean = np.mean(betweenness)
        betweenness_skew = skew(betweenness)

        # Print results for checking
        print(f"Degree - Mean: {degree_mean:.4f}, Skew: {degree_skew:.4f}")
        print(f"Total Edge Length - Mean: {total_edge_length_mean:.4f}, Skew: {total_edge_length_skew:.4f}")
        print(f"Clustering Coefficient - Mean: {clustering_mean:.4f}, Skew: {clustering_skew:.4f}")
        print(f"Betweenness Centrality - Mean: {betweenness_mean:.4f}, Skew: {betweenness_skew:.4f}")

        # Save statistics for this density level
        np.savez(f"{output_dir}/Metrics/{matrix_name}_density_{int(density_level * 100)}.npz",
                degree_mean=degree_mean, degree_skew=degree_skew,
                total_edge_length_mean=total_edge_length_mean, total_edge_length_skew=total_edge_length_skew,
                clustering_mean=clustering_mean, clustering_skew=clustering_skew,
                betweenness_mean=betweenness_mean, betweenness_skew=betweenness_skew,
                **np.load(f"{output_dir}/Metrics/{matrix_name}_density_{int(density_level * 100)}.npz")))


    # Visualisations --------------------------------------------------
    # Visualise adjacency matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(adjM_thresholded, cmap="RdBu_r", center=0, cbar=True)
    plt.title("Thresholded Adjacency Matrix")
    plt.savefig(f"{output_dir}/Graphs/{matrix_name} Adjacency matrix heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create a 2x2 panel figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2 rows, 2 columns

    # Degree Distribution (Top-Left)
    sns.histplot(degree, bins=20, kde=True, color='black', edgecolor="black", ax=axes[0, 0])
    axes[0, 0].set_xlabel("Degree")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Degree Distribution")

    # Total Edge Length Distribution (Top-Right)
    sns.histplot(total_edge_length, bins=20, kde=True, color='black', edgecolor="black", ax=axes[0, 1])
    axes[0, 1].set_xlabel("Total Edge Length")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Total Edge Length Distribution")

    # Clustering Coefficient Distribution (Bottom-Left)
    sns.histplot(clustering, bins=20, kde=True, color='black', edgecolor="black", ax=axes[1, 0])
    axes[1, 0].set_xlabel("Clustering Coefficient")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Clustering Coefficient Distribution")

    # Betweenness Centrality Distribution (Bottom-Right)
    sns.histplot(betweenness, bins=20, kde=True, color='black', edgecolor="black", ax=axes[1, 1])
    axes[1, 1].set_xlabel("Betweenness Centrality")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Betweenness Centrality Distribution")

    # Adjust layout to prevent overlap
    fig.tight_layout()

    # Save the figure as a single image
    plt.savefig(f"{output_dir}/Graphs/{matrix_name} Graph metrics.png", dpi=300, bbox_inches="tight")

    print(f"Finished processing {matrix_name}.\n")

print("All matrices processed successfully.")


# Load results --------------------------------------------------
# Load it later
data = np.load("brain_metrics.npz")

# Assign each metric to a variable
degree = data['degree']
total_edge_length = data['total_edge_length']
clustering = data['clustering']
betweenness = data['betweenness']
num_connections = data['num_connections']
density = data['density']
