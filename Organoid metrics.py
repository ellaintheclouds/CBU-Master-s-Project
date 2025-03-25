# %% Import packages --------------------------------------------------
import os  # File operations
import scipy.io  # Load MATLAB .mat files
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numerical operations
import seaborn as sns  # Data visualisation
import bct  # Brain Connectivity Toolbox (graph-theoretic analysis)
from glob import glob  # Finding files that match a pattern
from scipy.spatial.distance import cdist  # Pairwise Euclidean distances
from scipy.stats import skew  # Compute skewness
import pandas as pd  # Data manipulation
import pickle  # Save/load processed data
import time  # Time operations


# %% Define Functions --------------------------------------------------
def load_files(matrix_dir):
    """Load .mat files from a specified directory and filter based on a pattern."""
    matrix_files = [file for file in glob('kr01/organoid/OrgNets/*.mat') 
                if ('C' in os.path.basename(file) or 'H' in os.path.basename(file)) and 'dt10' in os.path.basename(file)]

    return matrix_files

def process_data(file_path):
    """Load and process a single .mat file."""
    file_name = os.path.basename(file_path).replace('.mat', '')
    mat_data = scipy.io.loadmat(file_path)
    
    # Load adjacency matrix and process it
    adjM = np.nan_to_num(mat_data['adjM'])
    
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
    
    # Compute distance matrix
    dij = cdist(np.column_stack((x, y)), np.column_stack((x, y)))
    
    return file_name, adjM, dij

def sort_data(file_name):
    """Sort data based on species and day number."""
    species = 'Chimpanzee' if 'C' in file_name else 'Human' if 'H' in file_name else None
    day_number = next((f'Day {i}' for i in range(1, 365) if f'_d{i}_' in file_name), 'Unknown Day')

    # Define the timepoint based on the day number
    if day_number in ['Day 95', 'Day 96']:
        timepoint = 't1'
    elif day_number == 'Day 153':
        timepoint = 't2'
    elif day_number in ['Day 184', 'Day 185']:
        timepoint = 't3'
    else:
        timepoint = 'Unknown Timepoint'
    return species, day_number, timepoint

    # Store sorted information for later processing
    processed_data.append((file_name, species, day_number, timepoint))

    print('All matrices sorted.')

def compute_metrics(file_name, species, day_number, adjM, density_levels, chimpanzee_metrics_df, human_metrics_df):
    """Compute graph theory metrics at different thresholds."""
    metrics_list = []
    
    for density_level in density_levels:
        adjM_thresholded = bct.threshold_proportional(adjM, density_level)
        
        # Compute graph metrics
        start_time = time.time()
        degree = np.sum(adjM_thresholded != 0, axis=0)
        end_time = time.time()
        print(f'    - degree computed in {end_time - start_time:.1f} seconds')

        start_time = time.time()
        total_edge_length = np.sum(adjM_thresholded, axis=0)
        end_time = time.time()
        print(f'    - total edge length computed in {end_time - start_time:.1f} seconds')

        start_time = time.time()
        clustering = bct.clustering_coef_bu(adjM_thresholded)
        end_time = time.time()
        print(f'    - clustering computed in {end_time - start_time:.1f} seconds')

        start_time = time.time()
        betweenness = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))
        end_time = time.time()
        print(f'    - betweenness computed in {end_time - start_time:.1f} seconds')

        start_time = time.time()
        efficiency = bct.efficiency_wei(adjM_thresholded, local=True)
        end_time = time.time()
        print(f'    - efficiency computed in {end_time - start_time:.1f} seconds')

        #  Compute NxN matrix where M[i, j] gives the matching index between nodes i and j
        start_time = time.time()
        N = adjM.shape[0]
        matching_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                if i != j:
                    min_weights = np.minimum(adjM[i, :], adjM[j, :])
                    max_weights = np.maximum(adjM[i, :], adjM[j, :])
                    if np.sum(max_weights) > 0:  # Avoid division by zero
                        matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)
        end_time = time.time()
        print(f'    - matching index computed in {end_time - start_time:.1f} seconds')

        # Save graph metrics
        metrics = {
            'density_level': density_level,
            'adjM_thresholded': adjM_thresholded,
            'degree': degree,
            'total_edge_length': total_edge_length,
            'clustering': clustering,
            'betweenness': betweenness,
            'efficiency': efficiency,
            'matching_index': matching_matrix
        }
        metrics_list.append(metrics)

        # Compute and save statistics for each metric
        # Compute density
        num_connections = np.count_nonzero(adjM_thresholded) // 2
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)

        # Save metrics, including their mean and skewness
        metrics_stats = pd.DataFrame([{
            'file_name': file_name,
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
            'matching_mean': np.mean(matching_matrix),
            'matching_skew': skew(np.mean(matching_matrix, axis=1))
        }])

        if species == 'Chimpanzee':
            chimpanzee_metrics_df = pd.concat([chimpanzee_metrics_df, metrics_stats], ignore_index=True)
        else:
            human_metrics_df = pd.concat([human_metrics_df, metrics_stats], ignore_index=True)

    return metrics_list, chimpanzee_metrics_df, human_metrics_df

def save_data(output_dir, pickle_dir, processed_data, chimpanzee_metrics_df, human_metrics_df):
    """Save data incrementally to a pickle file."""
    
    chimpanzee_metrics_df.to_csv(f'{output_dir}/Chimpanzee/chimpanzee_metrics_summary.csv', index=False)
    human_metrics_df.to_csv(f'{output_dir}/Human/human_metrics_summary.csv', index=False)

    with open(pickle_dir, 'wb') as f:
        pickle.dump(processed_data, f)

def load_data(pickle_dir):
    """Load processed data from a pickle file."""
    with open(pickle_dir, 'rb') as f:
        processed_data = pickle.load(f)
    return processed_data

def individual_plot(processed_data_idx, output_dir):
    """Generate and save plots for individual densities."""
    matrix_name = processed_data_idx['file_name']
    species = processed_data_idx['species']
    timepoint = processed_data_idx['timepoint']
    dij = processed_data_idx['dij']

    for density_data in processed_data_idx['metrics']:
        density_level = density_data['density_level']
        adjM_thresholded = density_data['adjM_thresholded']
        degree = density_data['degree']
        total_edge_length = density_data['total_edge_length']
        clustering = density_data['clustering']
        betweenness = density_data['betweenness']
        efficiency = density_data['efficiency']
        matching_matrix = density_data['matching_index']

        # Create output directory
        plot_dir = os.path.join(output_dir, species, timepoint, f'{int(density_level * 100)}%')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Adjacency matrix heatmap
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(adjM_thresholded, cmap='RdBu_r', center=0, cbar=True)
        ax.set_title('Thresholded Adjacency Matrix', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        num_labels = 10  # Number of labels you want to show
        ax.set_xticks(np.linspace(0, adjM_thresholded.shape[1] - 1, num_labels))
        ax.set_yticks(np.linspace(0, adjM_thresholded.shape[0] - 1, num_labels))
        ax.set_xticklabels(np.linspace(0, adjM_thresholded.shape[1] - 1, num_labels, dtype=int))
        ax.set_yticklabels(np.linspace(0, adjM_thresholded.shape[0] - 1, num_labels, dtype=int))
        plt.savefig(f'{plot_dir}/{matrix_name}_Adjacency_Matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Distance matrix heatmap
        plt.figure(figsize=(7, 6))
        ax = sns.heatmap(dij, cmap='RdBu_r', center=0, cbar=True)
        ax.set_title('Distance Matrix', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        num_labels = 10  # Number of labels you want to show
        ax.set_xticks(np.linspace(0, dij.shape[1] - 1, num_labels))
        ax.set_yticks(np.linspace(0, dij.shape[0] - 1, num_labels))
        ax.set_xticklabels(np.linspace(0, dij.shape[1] - 1, num_labels, dtype=int))
        ax.set_yticklabels(np.linspace(0, dij.shape[0] - 1, num_labels, dtype=int))
        plt.savefig(f'{plot_dir}/{matrix_name}_Distance_Matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 'Topological fingerprint' heatmap
        # Convert to DataFrame for correlation analysis
        metrics_df = pd.DataFrame({
            'degree': degree,
            'clustering': clustering,
            'betweenness': betweenness,
            'total_edge_length': total_edge_length,
            'efficiency': efficiency,
            'matching_index': np.mean(matching_matrix, axis=1)
        })

        # Compute mean matching index per node
        metrics_df['matching_index'] = np.mean(matching_matrix, axis=1)

        # Compute correlation matrix
        correlation_matrix = metrics_df.corr()

        # Define better-formatted labels
        formatted_labels = [
            'Degree', 'Clustering',  'Betweenness', 'Total Edge Length', 'Efficiency', 'Matching Index'
        ]

        # Plot correlation heatmap
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(correlation_matrix, cmap='RdBu_r', xticklabels=formatted_labels, yticklabels=formatted_labels, center=0, cbar=True)
        ax.set_title('Topological Fingerprint Heatmap', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        plt.savefig(f'{plot_dir}/{matrix_name}_Topological_Fingerprint.png', dpi=300, bbox_inches='tight')
        plt.close()

def overlaid_metrics_plot(processed_data_idx, output_dir):
    """Generate and save plots overlaid with information for multiple densities"""
    matrix_name = processed_data_idx['file_name']
    species = processed_data_idx['species']
    timepoint = processed_data_idx['timepoint']
    dij = processed_data_idx['dij']

    # Create output directory
    plot_dir = os.path.join(output_dir, species, timepoint)
    os.makedirs(plot_dir, exist_ok=True)

    # Graph metrics visualisations ----------
    # Create a 2x2 panel figure for the histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Increase font size
    font_size = 12

    # Define colours for each density level
    density_colours = ['#440154', '#B12A90', '#F46D43', '#FDE725']

    for density_data in processed_data_idx['metrics']:
        density_level = density_data['density_level']
        degree = density_data['degree']
        total_edge_length = density_data['total_edge_length']
        clustering = density_data['clustering']
        betweenness = density_data['betweenness']

        # Degree Distribution
        sns.kdeplot(degree, color=density_colours[idx], ax=axes[0, 0], label=f'{int(density_level * 100)}%', linewidth=3, alpha=0.7)

        # Total Edge Length Distribution
        sns.kdeplot(total_edge_length, color=density_colours[idx], ax=axes[0, 1], label=f'{int(density_level * 100)}%', linewidth=3, alpha=0.7)

        # Clustering Coefficient Distribution
        sns.kdeplot(clustering, color=density_colours[idx], ax=axes[1, 0], label=f'{int(density_level * 100)}%', linewidth=3, alpha=0.7)

        # Betweenness Centrality Distribution
        sns.kdeplot(betweenness, color=density_colours[idx], ax=axes[1, 1], label=f'{int(density_level * 100)}%', linewidth=3, alpha=0.7)

    # Set titles
    axes[0, 0].set_title('Degree', fontsize=font_size)
    axes[0, 1].set_title('Total Edge Length', fontsize=font_size)
    axes[1, 0].set_title('Clustering Coefficient', fontsize=font_size)
    axes[1, 1].set_title('Betweenness Centrality', fontsize=font_size)

    # Adjust tick label sizes
    for ax in axes.flat:
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.legend()  # Add legend to distinguish density levels

    # Adjust spacing between subplots
    fig.tight_layout(pad=2.0)

    # Save the figure
    plt.savefig(f'{plot_dir}/{matrix_name}_Graph_Metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def fingerprint_correlation_plot(processed_data_idx, output_dir): # read in metrics dataframes
    """Generate and save plots comparing correlation of topological fingerprint over densities."""
    # Load chimpanzee metrics dataframe
    chimpanzee_metrics_df = pd.read_csv(f'{output_dir}/Chimpanzee/chimpanzee_metrics_summary.csv')

    # Create output directory
    plot_dir = os.path.join(output_dir, species)
    os.makedirs(plot_dir, exist_ok=True)

    # 'Topological fingerprint' correlation across densities ----------
    # In chosen slices, for each density level, compute average means across slices
    chosen_slices = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

    chimpanzee_metrics_df_subset = chimpanzee_metrics_df[
        (chimpanzee_metrics_df['file_name'].isin(chosen_slices))
    ]

    # Group by 'density_level' and compute the mean for numeric columns only
    chimpanzee_metrics_df_subset = chimpanzee_metrics_df_subset.groupby(['density_level']).mean(numeric_only=True).reset_index()


    # Select only the metrics of interest
    chimpanzee_metrics_df_subset = chimpanzee_metrics_df_subset[
        ['degree_mean', 'clustering_mean', 'betweenness_mean', 'total_edge_length_mean', 'efficiency_mean', 'matching_mean']
    ]

    # Compute correlation matrix
    CS_correlation_matrix = chimpanzee_metrics_df_subset.corr()

    # Define better-formatted labels
    formatted_labels = [
        'Degree', 'Clustering',  'Betweenness', 'Total Edge Length', 'Efficiency', 'Matching Index'
    ]

    # Plot correlation heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(CS_correlation_matrix, cmap='RdBu_r', xticklabels=formatted_labels, yticklabels=formatted_labels, center=0, cbar=True)
    ax.set_title('Topological Fingerprint Correlation Across Time', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(f'{plot_dir}/Topological_Fingerprint_Correlation_s2_s7_s10_dt10.png', dpi=300, bbox_inches='tight')
    plt.close()


# %% Main Script --------------------------------------------------
# Define working directory
os.chdir('/imaging/astle')

# Set output directories
matrix_dir = 'kr01/organoid/OrgNets'
output_dir = 'er05/Organoid project scripts/Output'
os.makedirs(output_dir, exist_ok=True)

# Create empty dataframes and lists
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()
processed_data = []

# Define density levels
density_levels = [0.05, 0.1, 0.2, 1]

# Load data
matrix_files = load_files(matrix_dir=matrix_dir)
print(f'{len(matrix_files)} organoid file(s) loaded.')

# Sort one at a time
for idx, file_path in enumerate(matrix_files):
    # Process data
    file_name, adjM, dij = process_data(file_path=file_path)

    # Sort data
    species, day_number, timepoint = sort_data(file_name)

    print(f'Processing {file_name} ({idx + 1}/{len(matrix_files)}):')

    # Compute metrics
    metrics_list, chimpanzee_metrics_df, human_metrics_df = compute_metrics(file_name=file_name, species=species, day_number=day_number, adjM=adjM, density_levels=density_levels, chimpanzee_metrics_df=chimpanzee_metrics_df, human_metrics_df=human_metrics_df)

    # Update processed data
    processed_data.append({'file_name': file_name, 'species': species, 'timepoint': timepoint, 'adjM': adjM, 'dij': dij, 'metrics': metrics_list})
    print('- metrics computed')

    # Save after each iteration
    os.makedirs(f'{output_dir}/Chimpanzee', exist_ok=True)
    os.makedirs(f'{output_dir}/Human', exist_ok=True)
    pickle_dir = f'{output_dir}/processed_data.pkl'
    save_data(output_dir=output_dir, pickle_dir=pickle_dir, processed_data=processed_data, chimpanzee_metrics_df=chimpanzee_metrics_df, human_metrics_df=human_metrics_df)
    print('- data saved')

    # Load data
    processed_data = load_data(pickle_dir=pickle_dir)
    print('- data loaded')

    # Plot
    processed_data_idx = processed_data[idx]
    individual_plot(processed_data_idx=processed_data_idx, output_dir=output_dir)
    overlaid_metrics_plot(processed_data_idx=processed_data_idx, output_dir=output_dir)
    fingerprint_correlation_plot(processed_data_idx=processed_data_idx, output_dir=output_dir)
    print('- plots created')

print('Processing complete. Metrics saved incrementally.')


# %%