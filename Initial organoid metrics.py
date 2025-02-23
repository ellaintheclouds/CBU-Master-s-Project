# Import packages --------------------------------------------------
import os # for file and directory operations
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy as np # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob # for finding files that match a certain pattern
from scipy.spatial.distance import cdist # for computing pairwise Euclidean distances


# Set up files --------------------------------------------------
# Check current working directory
print("Current working directory:", os.getcwd())


# Load data --------------------------------------------------
# Load individual data
# mat_data = scipy.io.loadmat("kr01/organoid/OrgNets/C_d95_s1_dt5") # .mat file containing adjacency matrix is loaded

# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

# Load the first file to check the keys
sample_data = scipy.io.loadmat(matrix_files[0])
print("Keys in the MATLAB file:", sample_data.keys())

# Check shapes of relevant matrices
for key in ['adjM', 'adjM_trimmed', 'dij', 'coords']:
    if key in sample_data:
        print(f"{key} shape: {np.shape(sample_data[key])}")


# Exploring coordinates --------------------------------------------------
# Extract the coordinates
coords_raw = sample_data['coords']
coords_raw = coords_raw[0, 0]  

# Extract x and y values
x_values = np.array(coords_raw['x']).astype(float).flatten()
y_values = np.array(coords_raw['y']).astype(float).flatten()

# Plot the electrode/node positions
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, c='blue', alpha=0.7)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Electrode/Node Positions")
plt.grid(True)
plt.savefig("er05/Organoid project scripts/Output/Chimpanzee/Electrode positions.png")


# Sorting out matrix size mismatch --------------------------------------------------
# Extract relevant coordinates-----
coords = sample_data['coords']['channel'][0][0]  # Extract coordinates
active_channel_idx = sample_data['active_channel_idx']  # Extract active channel indices
active_coords = coords[active_channel_idx.flatten()]

# Recompute the distance matrix-----
# Compute pairwise distances between active nodes
filtered_dij = cdist(active_coords[:, 1:], active_coords[:, 1:])  # Exclude channel column

# Filter adjM-----
adjM = sample_data['adjM']
# Extract vectors from the loaded .mat file
data_channel = sample_data['data']['channel'][0][0].flatten()  # Flatten to get a 1D array
coords_channel = sample_data['coords']['channel'][0][0].flatten()

# Get unique values
unique_data_channel = np.unique(data_channel)

# Perform set difference
difference = np.setdiff1d(unique_data_channel, coords_channel)

# Find indices in unique_data_channel that are in the set difference
indices = np.where(np.isin(unique_data_channel, difference))[0]

# Remove the indices from adjM on both dimensions
adjM = sample_data['adjM']
adjM = np.delete(adjM, indices, axis=0)
adjM = np.delete(adjM, indices, axis=1)

# Check shapes
print(adjM.shape, filtered_dij.shape)


# Loop through each adjacency matrix file --------------------------------------------------
for file_path in matrix_files:
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

    # Extract adjacency matrix 
    adjM = mat_data['adjM'] # a 2D matrix where each cell represents a connection between two nodes (brain regions)

    # Remove NaN values but keep shape
    adjM = np.nan_to_num(adjM)

    # Display matrix shape
    print(f"Adjacency matrix shape: {adjM.shape}")

    # Extract distance matrix
    dij = mat_data['dij'] # a 2D matrix where each cell represents the distance between two nodes (brain regions)


    # Preprocess the matrix --------------------------------------------------
    # Thresholding
    density_desired = 0.1 # 10% threshold - change if desired
    threshold = np.percentile(adjM, (1-density_desired)*100)
    adjM_thresholded = adjM.copy()
    adjM_thresholded[adjM_thresholded < threshold] = 0


    # Compute connectivity metrics --------------------------------------------------
    # Degree
    degree = np.sum(adjM_thresholded != 0, axis=0)  # Count nonzero edges per node

    # Total edge length
    total_edge_length = np.sum(adjM_thresholded * dij, axis=0) # Sum of edge weights per node#############################################################################
    # What I need to do is find a dij that is the same size as adjM

    # Clustering coefficient
    clustering = bct.clustering_coef_bu(adjM_thresholded)

    # Betweenness centrality
    betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))
    # smallest positive number that can be represented by a float added, which is added to adjM_thresholded to avoid division by zero

    # Number of connections
    num_connections = np.count_nonzero(adjM_thresholded)//2 # Divide by 2 to avoid double counting
    print(f"\Total number of connections: {num_connections}")

    # Connection density
    num_nodes = adjM_thresholded.shape[0]
    density = num_connections / ( (num_nodes* (num_nodes - 1)) / 2) # proportion of existing connetions
    print(f"Network density: {density:.4f}")

    # Save calculated metrics (so that I do not have to recompute; .npz works like a dictionary)
    np.savez(f"{output_dir}/Metrics/{matrix_name}_metrics.npz", 
         degree=degree, 
         total_edge_length=total_edge_length, 
         clustering=clustering, 
         betweenness=betweenness, 
         num_connections=num_connections, 
         density=density)


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
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Log Degree")
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
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Log Betweenness Centrality")
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
