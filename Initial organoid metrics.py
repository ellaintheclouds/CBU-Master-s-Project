# %% Import packages -------------------------------------------------- 
import os # for file and directory operations
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy as np # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob # for finding files that match a certain pattern
from scipy.spatial.distance import cdist # for computing pairwise Euclidean distances
from scipy.stats import skew # for computing skewness
import pandas as pd # for data manipulation and analysis


# %% Load data --------------------------------------------------
# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

# Create empty DataFrames to store metrics for each species
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()

# Create a subset of matrix_files for testing (e.g., first 5 files)
test_matrix_files = matrix_files[:1]


# %% Process each matrix --------------------------------------------------
for file_path in test_matrix_files:
    # Sorting each file by species and day --------------------------------------------------
    # Extract matrix name (e.g., "matrix1" from "matrices/matrix1.mat")
    matrix_name = os.path.basename(file_path).replace(".mat", "")

    print(f"Processing {matrix_name}...")
    
    # Determine species
    if "C" in matrix_name:
        species = "Chimpanzee"
        metrics_df = chimpanzee_metrics_df

    elif "H" in matrix_name:
        species = "Human"
        metrics_df = human_metrics_df

    else:
        print(f"Skipping {matrix_name}: No valid species identifier (C or H).")

    # Determine day number
    day_number = "Unknown Day"
    for i in range(1, 365):  # Check for d10 to d365
        if f"_d{i}_" in matrix_name:
            day_number = f"Day {i}"
            break

    # Define densities to explore
    densities_to_test = [0.05]  # 5%, 10%, 20% threshold

    # Store results for each density
    for density_level in densities_to_test:
        print(f"\nApplying threshold: {int(density_level * 100)}%")

        # Create output directories for species, density, and day
        output_dir = f"er05/Organoid project scripts/Output/{species}/{int(density_level * 100)}%/{day_number}"
        os.makedirs(f"{output_dir}/Graphs", exist_ok=True)


        # Load organoid data --------------------------------------------------
        mat_data = scipy.io.loadmat(file_path)
        if "adjM" not in mat_data:
            print(f"Skipping {matrix_name}: 'adjM' key not found.")

        # Load adjacency matrix
        adjM = mat_data['adjM']


        # Data processing --------------------------------------------------
        # remove channels with no corresponding coordinates from the spike data and adjM ----------
        # extract variables from their spike detection ("spikeTimes") and spike sorting ("mapping") data which contradict each other
        # Extract spike time and associated channel vectors from their spike detection data
        data_channel = mat_data['data']['channel'][0][0].flatten()  # Flatten to get a 1D array
        data_frameno = mat_data['data']['frameno'][0][0].flatten()

        # Extract coordinates and channel IDs from their spike sorting / localization data ("mapping")
        coords_channel = mat_data['coords']['channel'][0][0].flatten()
        coords_x = mat_data['coords']['x'][0][0].flatten()
        coords_y = mat_data['coords']['y'][0][0].flatten()

        # 1.1. Remove channels with no coordinate information
        missing_channels = np.setdiff1d(np.unique(data_channel), coords_channel)

        # 1.2. Get indices of channels WITH spikes
        active_channel_idx = np.where(np.isin(coords_channel, np.unique(data_channel)))[0]

        # 1.3. Include only these entries in coordinate data
        x = coords_x[active_channel_idx]
        y = coords_y[active_channel_idx]

        # 1.4. Get indices of spike times and channels WITH coordinates
        coord_channel_idx = np.where(np.isin(data_channel, coords_channel))[0]

        # 1.5. Include only these entries in spiketime data
        spikeframes = data_frameno[coord_channel_idx]
        spikechannels = data_channel[coord_channel_idx]

        # adjacency matrix correction: remove channels with no corresponding coordinates from the adjacency matrix
        # 2.1. Remove channels with no coordinates from adjacency matrix
        # Find the set difference between the unique data channels and the channels with coordinates
        difference = np.setdiff1d(np.unique(data_channel), coords_channel)

        # 2.2. Get the indices of the channels with no coordinates
        indices = np.where(np.isin(np.unique(data_channel), difference))[0]

        # 2.3. Remove the indices from adjM on both dimensions
        adjM = np.delete(adjM, indices, axis=0)
        adjM = np.delete(adjM, indices, axis=1)

        # Print the shape of the adjacency matrix
        print(f"Adjacency matrix shape: {adjM.shape}")

        # compute DIJ (distance matrix) ----------
        dij = cdist(np.column_stack((x,y)),np.column_stack((x,y)))
        
        # Print the shape of the distance matrix
        print(f"Distance matrix shape: {dij.shape}")

        # Preprocess adjM ----------
        # Remove NaN values but keep shape
        adjM = np.nan_to_num(adjM)

        # Apply thresholding
        adjM_thresholded = bct.threshold_proportional(adjM, density_level)
        

        # Compute connectivity metrics --------------------------------------------------
        # Degree
        degree = np.sum(adjM_thresholded != 0, axis=0)
        
        # Total edge length
        total_edge_length = np.sum(adjM_thresholded, axis=0)

        # Clustering coefficient
        clustering = bct.clustering_coef_bu(adjM_thresholded)

        # Betweenness centrality
        betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))

        # Number of connections
        num_connections = np.count_nonzero(adjM_thresholded) // 2
        print(f"Total number of connections: {num_connections}")

        # Connection density
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)
        print(f"Network density: {density:.4f}")

        # Compute mean and skewness for each metric ----------
        degree_mean = np.mean(degree)
        degree_skew = skew(degree)

        total_edge_length_mean = np.mean(total_edge_length)
        total_edge_length_skew = skew(total_edge_length)

        clustering_mean = np.mean(clustering)
        clustering_skew = skew(clustering)

        betweenness_mean = np.mean(betweenness)
        betweenness_skew = skew(betweenness)

        # Append metrics to the dataframe ----------
        new_row = pd.DataFrame([{
            'matrix_name': matrix_name,
            'species': species,
            'day_number': day_number,
            'density_level': density_level,
            'num_connections': num_connections,
            'density': density,
            'degree_mean': degree_mean,
            'degree_skew': degree_skew,
            'total_edge_length_mean': total_edge_length_mean,
            'total_edge_length_skew': total_edge_length_skew,
            'clustering_mean': clustering_mean,
            'clustering_skew': clustering_skew,
            'betweenness_mean': betweenness_mean,
            'betweenness_skew': betweenness_skew

        }])

        if species == "Chimpanzee":
            chimpanzee_metrics_df = pd.concat([chimpanzee_metrics_df, new_row], ignore_index=True)
        elif species == "Human":
            human_metrics_df = pd.concat([human_metrics_df, new_row], ignore_index=True)
      
  
        # Visualisations --------------------------------------------------
        # Visualise adjacency matrix
        plt.figure(figsize=(6,6))
        sns.heatmap(adjM_thresholded, cmap="RdBu_r", center=0, cbar=True)
        plt.title("Thresholded Adjacency Matrix")
        plt.savefig(f"{output_dir}/Graphs/{matrix_name} Adjacency matrix heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Visualise distance matrix
        plt.figure(figsize=(6,6))
        sns.heatmap(dij, cmap="RdBu_r", center=0, cbar=True)
        plt.title("Distance Matrix")
        plt.savefig(f"{output_dir}/Graphs/{matrix_name} Distance matrix heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Create a 2x2 panel figure
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
        plt.savefig(f"{output_dir}/Graphs/{matrix_name} Graph metrics.png", dpi=300, bbox_inches="tight")

    print(f"Finished processing {matrix_name}.\n")

print("All matrices processed successfully.")

# Save and load --------------------------------------------------
# Save the DataFrames to CSV files if needed
chimpanzee_metrics_df.to_csv("er05/Organoid project scripts/Output/chimpanzee_metrics_summary.csv", index=False)
human_metrics_df.to_csv("er05/Organoid project scripts/Output/human_metrics_summary.csv", index=False)

# Print the DataFrames
print("Chimpanzee Metrics DataFrame:")
print(chimpanzee_metrics_df)

print("Human Metrics DataFrame:")
print(human_metrics_df)