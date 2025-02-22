# Environment
# Make sure that the correct interpreter is selected
# If putting in new packages: launch terminal, conda activate ella_organoid

# Import packages --------------------------------------------------
import os # for interacting with the operating system
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy as np # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob # for finding files that match a certain pattern


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
print(sample_data.keys())


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
    total_edge_length = np.sum(adjM_thresholded * dij, axis=0) # Sum of edge weights per node

    # Clustering coefficient
    clustering = bct.clustering_coef_bu(adjM_thresholded)

    # Betweenness centrality
    betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))  

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

# Save graphs as image files



