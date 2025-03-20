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
import pickle  # for saving and loading preprocessed data


# %% Load data --------------------------------------------------
# # Set working directory
os.chdir("/imaging/astle")

# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") 
                if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

# Define data subset to test with
#matrix_files = matrix_files[:1]  # Use only the first 1 matrix for testing

# Create empty DataFrames to store metrics for each species
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()

# Initialise a list to store sorted data
sorted_data = []

# Sorting each file by species and day ----------
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


# %% Preprocess data  --------------------------------------------------
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

    # Filter adjM and dij to only include channels with coordinates ----------
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

    # Preprocess adjacency matrix ----------
    # Remove NaN values but keep shape
    adjM = np.nan_to_num(adjM)

    # Compute distance matrix ----------
    dij = cdist(np.column_stack((x, y)), np.column_stack((x, y)))

    # Store preprocessed data for later analysis
    preprocessed_data.append((matrix_name, species, day_number, adjM, dij))

print("All matrices preprocessed.")


# %% Compute connectivity metrics --------------------------------------------------
densities_to_test = [0.05, 0.1, 0.2]  # 5%, 10%, 20% thresholds

# Define densities subset to test
#densities_to_test = densities_to_test[:1] # Use only the first density for testing

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

# Compute connectivity metrics for each matrix ----------
for i, (matrix_name, species, day_number, adjM, dij) in enumerate(preprocessed_data):
    for density_level in densities_to_test:
        print(f"Computing metrics for {matrix_name} at {int(density_level * 100)}% threshold...")

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
        efficiency = bct.efficiency_wei(adjM_thresholded, local=True) #matrix
        print("-computing matching index")
        matching = matching_index_wei(adjM_thresholded) # matrix

        # Compute density
        num_connections = np.count_nonzero(adjM_thresholded) // 2
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)

        # Save metrics, including their mean and skewness
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
        # Store results for the current density level
        density_results = {
            'density_level': density_level,
            'adjM_thresholded': adjM_thresholded,
            'degree': degree,
            'total_edge_length': total_edge_length,
            'clustering': clustering,
            'betweenness': betweenness,
            'efficiency': efficiency,
            'matching': matching
        }

        # Append results to the entry for this matrix
        if isinstance(preprocessed_data[i], tuple):  # Convert old structure to new
            preprocessed_data[i] = {
                'matrix_name': preprocessed_data[i][0],
                'species': preprocessed_data[i][1],
                'day_number': preprocessed_data[i][2],
                'adjM': preprocessed_data[i][3],
                'dij': preprocessed_data[i][4],
                'densities': [density_results]
            }
        else:
            preprocessed_data[i]['densities'].append(density_results)

print("Connectivity metrics computed.")


# %% Save metrics --------------------------------------------------
chimpanzee_metrics_df.to_csv("er05/Organoid project scripts/Output/chimpanzee_metrics_summary.csv", index=False)
human_metrics_df.to_csv("er05/Organoid project scripts/Output/human_metrics_summary.csv", index=False)

print("Chimpanzee Metrics DataFrame:")
print(chimpanzee_metrics_df)

print("Human Metrics DataFrame:")
print(human_metrics_df)


# %% Save Preprocessed Data --------------------------------------------------
# Set working directory
os.chdir("/imaging/astle")

# Define the path to save the preprocessed data
preprocessed_data_path = "er05/Organoid project scripts/Output/preprocessed_data.pkl"

with open(preprocessed_data_path, 'wb') as f:
    pickle.dump(preprocessed_data, f)
print("Preprocessed data saved to file.")


# %% Re-Load Preprocessed Data --------------------------------------------------
# Set working directory
os.chdir("/imaging/astle")

# Define the path to load the preprocessed data
preprocessed_data_path = "er05/Organoid project scripts/Output/preprocessed_data.pkl"

# Check if the file exists and is not empty
if os.path.exists(preprocessed_data_path) and os.path.getsize(preprocessed_data_path) > 0:
    try:
        # Open the file in read-binary mode and load the data
        with open(preprocessed_data_path, 'rb') as f:
            preprocessed_data = pickle.load(f)
        print("Preprocessed data loaded successfully.")
    except (EOFError, pickle.UnpicklingError) as e:
        print(f"Error loading preprocessed data: {e}")
else:
    print(f"File {preprocessed_data_path} does not exist or is empty.")

# Load metrics DataFrames
chimpanzee_metrics_df = pd.read_csv("er05/Organoid project scripts/Output/chimpanzee_metrics_summary.csv")
human_metrics_df = pd.read_csv("er05/Organoid project scripts/Output/human_metrics_summary.csv")

# Check if the Dataframes exist and are not empty
if not chimpanzee_metrics_df.empty:
    print("Metrics Dataframes loaded successfully.")


# %% Generate visualisations --------------------------------------------------
# Process data for visualisations *across thresholds* ----------
# Define variables for visualisations
for entry in preprocessed_data:
    matrix_name = entry['matrix_name']
    species = entry['species']
    day_number = entry['day_number']
    dij = entry['dij']
    
    # Define the timepoint based on the day number
    if day_number in ["Day 95", "Day 96"]:
        timepoint = "t1"
    elif day_number == "Day 153":
        timepoint = "t2"
    elif day_number in ["Day 184", "Day 185"]:
        timepoint = "t3"
    else:
        timepoint = "Unknown Timepoint"

    # Create output directory
    output_dir_short = f"er05/Organoid project scripts/Output/{species}"
    os.makedirs(f"{output_dir_short}/{timepoint}", exist_ok=True)

    # State which matrix is being visualised
    print(f"Visualising {matrix_name}:")
    print("- plotting metrics across thresholds")

    # Graph metrics visualisations ----------
    # Create a 2x2 panel figure for the histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Increase font size
    font_size = 12

    # Define colours for each density level
    density_colours = ["#440154", "#B12A90", "#F46D43", "#FDE725"]

    # Iterate over density levels and plot each on the same set of axes
    for idx, density_data in enumerate(entry['densities']):
        density_level = density_data['density_level']
        degree = density_data['degree']
        total_edge_length = density_data['total_edge_length']
        clustering = density_data['clustering']
        betweenness = density_data['betweenness']

        # Degree Distribution
        sns.kdeplot(degree, color=density_colours[idx], ax=axes[0, 0], label=f"{int(density_level * 100)}%", linewidth=3, alpha=0.7)

        # Total Edge Length Distribution
        sns.kdeplot(total_edge_length, color=density_colours[idx], ax=axes[0, 1], label=f"{int(density_level * 100)}%", linewidth=3, alpha=0.7)

        # Clustering Coefficient Distribution
        sns.kdeplot(clustering, color=density_colours[idx], ax=axes[1, 0], label=f"{int(density_level * 100)}%", linewidth=3, alpha=0.7)

        # Betweenness Centrality Distribution
        sns.kdeplot(betweenness, color=density_colours[idx], ax=axes[1, 1], label=f"{int(density_level * 100)}%", linewidth=3, alpha=0.7)

    # Set titles
    axes[0, 0].set_title("Degree", fontsize=font_size)
    axes[0, 1].set_title("Total Edge Length", fontsize=font_size)
    axes[1, 0].set_title("Clustering Coefficient", fontsize=font_size)
    axes[1, 1].set_title("Betweenness Centrality", fontsize=font_size)

    # Adjust tick label sizes
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.legend()  # Add legend to distinguish density levels

    # Adjust spacing between subplots
    fig.tight_layout(pad=2.0)

    # Save the figure
    plt.savefig(f"{output_dir_short}/{timepoint}/{matrix_name}_Graph_Metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # "Topological fingerprint" correlation ----------
    # Define the organoid slices that I want to use as timepoints
    chosen_slices = ["C_d96_s2_dt10", "C_d153_s7_dt10", "C_d184_s8_dt10"]
    chosen_density = 0.1

    # Create a subset of the DataFrame
    chimpanzee_metrics_df_subset = chimpanzee_metrics_df[
        (chimpanzee_metrics_df['matrix_name'].isin(chosen_slices)) &
        (chimpanzee_metrics_df['density_level'] == chosen_density)
    ]

    chimpanzee_metrics_df_subset = chimpanzee_metrics_df_subset[
        ['degree_mean', 'clustering_mean', 'betweenness_mean', 'total_edge_length_mean', 'efficiency_mean', 'matching_mean']
    ]

    # Compute correlation matrix
    CS_correlation_matrix = chimpanzee_metrics_df_subset.corr()

    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(CS_correlation_matrix, cmap="RdBu_r", xticklabels=formatted_labels, yticklabels=formatted_labels, center=0, cbar=True)
    ax.set_title("Topological Fingerprint Correlation Across Time", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(f"{output_dir_short}/Topological_Fingerprint_Correlation_s2_s7_s10_dt10.png", dpi=300, bbox_inches="tight")
    plt.close()        

    # "Topological fingerprint" coefficient of variation ----------
    # Define the columns of interest
    metrics_columns = ['degree_mean', 'clustering_mean', 'betweenness_mean', 'total_edge_length_mean', 'efficiency_mean', 'matching_mean']

    # Function to calculate the coefficient of variation
    def calculate_cv(series):
        return series.std() / series.mean()

    # Group by matrix_name and calculate the CV for each metric
    cv_results = chimpanzee_metrics_df.groupby('matrix_name')[metrics_columns].apply(lambda x: x.apply(calculate_cv))

    # Add back day numbers
    day_numbers = chimpanzee_metrics_df[['matrix_name', 'day_number']].drop_duplicates()
    cv_results_with_day = pd.merge(cv_results, day_numbers, on='matrix_name', how='left')

    # Plot the coefficient of variation for each metric as a boxplot (overlaid with dots)
    # Melt the DataFrame to long format for easier plotting with seaborn
    cv_results_melted = cv_results_with_day.melt(id_vars=['matrix_name', 'day_number'], var_name='Metric', value_name='Coefficient of Variation')

    # Define better-formatted labels
    formatted_labels = [
        "Degree", "Clustering",  "Betweenness", "Total Edge Length", "Efficiency", "Matching Index"
    ]

    # Define timepoints
    timepoints = {
        't1': ["Day 95", "Day 96"],
        't2': ["Day 153"],
        't3': ["Day 184", "Day 185"]
    }

    # Define colors for each timepoint
    timepoint_colors = {
        't1': sns.color_palette("magma", as_cmap=True)(0.2),
        't2': sns.color_palette("magma", as_cmap=True)(0.5),
        't3': sns.color_palette("magma", as_cmap=True)(0.8)
    }

    # Plot the data for each timepoint on the same axis
    plt.figure(figsize=(12, 8))
    for tp, days in timepoints.items():
        tp_data = cv_results_melted[cv_results_melted['day_number'].isin(days)]
                
        if not tp_data.empty:
            sns.stripplot(x='Metric', y='Coefficient of Variation', data=tp_data, color=timepoint_colors[tp], dodge=True, jitter=True, ax=plt.gca(), size=8, alpha=0.6, label=tp.upper())

    # Set plot title and labels
    plt.title('Coefficient of Variation for Different Metrics', fontsize=20)
    plt.xlabel('Metric', fontsize=18)
    plt.ylabel('Coefficient of Variation', fontsize=18)
    plt.xticks(ticks=range(len(formatted_labels)), labels=formatted_labels)
    plt.yticks(fontsize=16)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Timepoint', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.tight_layout()

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Timepoint', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)

    # Save and close
    plt.savefig(f"{output_dir_short}/Metrics_CVs.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Process data for visualisations *for each threshold* ----------
    for density_data in entry['densities']:
        density_level = density_data['density_level']
        adjM_thresholded = density_data['adjM_thresholded']
        degree = density_data['degree']
        total_edge_length = density_data['total_edge_length']
        clustering = density_data['clustering']
        betweenness = density_data['betweenness']
        efficiency = density_data['efficiency']
        matching = density_data['matching']

        # Create output directory
        output_dir = f"er05/Organoid project scripts/Output/{species}/{timepoint}/{int(density_level * 100)}%"
        os.makedirs(output_dir, exist_ok=True)

        # State which matrix is being visualised
        print(f"- plotting heatmaps at {int(density_level * 100)}% threshold")
        
        # Adjacency matrix heatmap ----------
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(adjM_thresholded, cmap="RdBu_r", center=0, cbar=True)
        ax.set_title("Thresholded Adjacency Matrix", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        num_labels = 10  # Number of labels you want to show
        ax.set_xticks(np.linspace(0, adjM_thresholded.shape[1] - 1, num_labels))
        ax.set_yticks(np.linspace(0, adjM_thresholded.shape[0] - 1, num_labels))
        ax.set_xticklabels(np.linspace(0, adjM_thresholded.shape[1] - 1, num_labels, dtype=int))
        ax.set_yticklabels(np.linspace(0, adjM_thresholded.shape[0] - 1, num_labels, dtype=int))
        plt.savefig(f"{output_dir}/{matrix_name}_Adjacency_Matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Distance matrix heatmap ----------
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(dij, cmap="RdBu_r", center=0, cbar=True)
        ax.set_title("Distance Matrix", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        num_labels = 10  # Number of labels you want to show
        ax.set_xticks(np.linspace(0, dij.shape[1] - 1, num_labels))
        ax.set_yticks(np.linspace(0, dij.shape[0] - 1, num_labels))
        ax.set_xticklabels(np.linspace(0, dij.shape[1] - 1, num_labels, dtype=int))
        ax.set_yticklabels(np.linspace(0, dij.shape[0] - 1, num_labels, dtype=int))
        plt.savefig(f"{output_dir}/{matrix_name}_Distance_Matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # "Topological fingerprint" heatmap ----------
        # Convert to DataFrame for correlation analysis
        metrics_df = pd.DataFrame({
            'degree': degree,
            'clustering': clustering,
            'betweenness': betweenness,
            'total_edge_length': total_edge_length,
            'efficiency': efficiency,
            'matching_index': np.mean(matching, axis=1)
        })

        # Compute mean matching index per node
        metrics_df['matching_index'] = np.mean(matching, axis=1)

        # Compute correlation matrix
        correlation_matrix = metrics_df.corr()

        # Define better-formatted labels
        formatted_labels = [
            "Degree", "Clustering",  "Betweenness", "Total Edge Length", "Efficiency", "Matching Index"
        ]

        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(correlation_matrix, cmap="RdBu_r", xticklabels=formatted_labels, yticklabels=formatted_labels, center=0, cbar=True)
        ax.set_title("Topological Fingerprint Heatmap", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(f"{output_dir}/{matrix_name}_Topological_Fingerprint.png", dpi=300, bbox_inches="tight")
        plt.close()

print("All visualisations saved.")


# %% Extra script to create dij without loaded data
"""
densities_to_test = [0.05, 0.1, 0.2, 1]

for key in preprocessed_data:
    for density_level in densities_to_test:
        matrix_name = key[0]
        species = key[1]
        day_number = key[2]
        adjM = key[3]
        dij = key[4]

        output_dir = f"er05/Organoid project scripts/Output/{species}/{int(density_level * 100)}%/{day_number}/Graphs"

        
        print(f"Graphing dij for {matrix_name}...")

        # Distance matrix heatmap
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(dij, cmap="RdBu_r", center=0, cbar=True)
        ax.set_title("Distance Matrix", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        num_labels = 10  # Number of labels you want to show
        ax.set_xticks(np.linspace(0, dij.shape[1] - 1, num_labels))
        ax.set_yticks(np.linspace(0, dij.shape[0] - 1, num_labels))
        ax.set_xticklabels(np.linspace(0, dij.shape[1] - 1, num_labels, dtype=int))
        ax.set_yticklabels(np.linspace(0, dij.shape[0] - 1, num_labels, dtype=int))
        plt.savefig(f"{output_dir}/{matrix_name}_Distance_Matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved dij to {output_dir}")
"""
