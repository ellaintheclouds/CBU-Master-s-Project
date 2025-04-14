# %% Import Packages and Functions --------------------------------------------------
import bct # brain connectivity toolbox
import gnm # for generative network modeling
from gnm import fitting, evaluation, BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical operations
import os # for file operations
import pandas as pd # for data manipulation and analysis
import pickle  # Save/load processed data
import scipy.io # for loading .mat files
import seaborn as sns # for enhanced plotting
import time # for timing the parameter sweep
import torch # for tensor operations
from scipy.spatial.distance import cdist # compute pairwise Euclidean distances


# %% Load and Process Data --------------------------------------------------
# Set working directory
os.chdir('/imaging/astle/er05')
pickle_dir = '/imaging/astle/er05/Organoid project scripts/Output/processed_data.pkl'

# Initialise lists
slice_data = []
modelling_data = []  # Keep this as a list

# Load organoid slice data
with open(pickle_dir, 'rb') as f:
    slice_data = pickle.load(f)

# Define slices for each timepoint
timepoint_slice_data = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

# Iterate through all slices
for slice in slice_data:
    # Process only the specified file names
    if slice['file_name'] in timepoint_slice_data:

        # Extract the adjacency matrix and distance matrix
        adjM = slice['adjM']
        dij = slice['dij']
        
        # Threshold the adjacency matrix to retain the top 5% strongest connections
        adjM_thresholded = bct.threshold_proportional(adjM, 0.05)
        
        # Create (50, 50) subsets for testing
        adjM_thresholded = adjM_thresholded[:50, :50]  # Remove when testing larger network
        dij = dij[:50, :50]  # Remove when testing larger network

        # Get the number of nodes and connections
        num_nodes = adjM_thresholded.shape[0]
        num_connections = np.count_nonzero(adjM_thresholded) // 2

        # Binarize the adjacency matrix
        adjM_thresholded = (adjM_thresholded > 0).astype(int)  # Remove when changing to weighted

        # Convert matrices to PyTorch tensors
        binary_network = torch.tensor(adjM_thresholded, dtype=torch.float)
        binary_network = binary_network.unsqueeze(0)
        distance_matrix = torch.tensor(dij, dtype=torch.float)
        
        # Add the data to the modelling_data list
        modelling_data.append({
            'file_name': slice['file_name'],
            'adjM_thresholded': adjM_thresholded,  # Change to adjM_thresholded when testing larger network
            'binary_network': binary_network,
            'dij': dij,  # Change to dij when testing larger network
            'distance_matrix': distance_matrix,
            'num_nodes': num_nodes,
            'num_connections': num_connections
        })


# %% Parameter Sweep --------------------------------------------------
def parameter_sweep(seed_binary_network, binary_network, distance_matrix, num_nodes, num_connections):
    # Carry out parameter sweep ----------
    # Set parameter ranges for eta and gamma
    eta_values = torch.linspace(-5, 1.7, 50) ########## increase when testing large parameterspace
    gamma_values = torch.linspace(-5, 1.7, 50) ########## increase when testing large parameterspace
    lambdah_value = 2.0  # Fixed lambda

    # Define binary sweep parameters
    binary_sweep_parameters = fitting.BinarySweepParameters(
            eta=eta_values,
            gamma=gamma_values,
            lambdah=torch.tensor([lambdah_value]),
            distance_relationship_type=['powerlaw'],
            preferential_relationship_type=['powerlaw'],
            heterochronicity_relationship_type=['powerlaw'],
            generative_rule=[MatchingIndex(divisor='mean')],  # Use Matching Index rule
            num_iterations=[num_connections],  # Match the number of connections in real network
        )

    # Number of networks to generate per parameter combination
    num_simulations = 5 ########## increase when testing large parameterspace

    if seed_binary_network is None:
        seed_adjacency_matrix = None
    else:
        seed_adjacency_matrix = [seed_binary_network]
    
    # Configuration for the parameter sweep
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_sweep_parameters,
        num_simulations=num_simulations,
        distance_matrix=[distance_matrix],
        seed_adjacency_matrix=seed_adjacency_matrix
    )

    # List of criteria for evaluating binary network properties
    criteria = [
        evaluation.DegreeKS(), # Compare degree distributions
        evaluation.ClusteringKS(), # Compare clustering coefficients
        evaluation.BetweennessKS(), # Compare betweenness centrality
        evaluation.EdgeLengthKS(distance_matrix) # Compare edge length distributions
    ]

    # Use the maximum KS statistic across all evaluations to measure network similarity
    energy = evaluation.MaxCriteria(criteria)
    binary_evaluations = [energy]

    # Run the Parameter Sweep
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=binary_evaluations,
        real_binary_matrices=binary_network,  # Real network to compare synthetic networks against    save_model=True, # If True, saves the model in the experiment - set to False to save on memory
        save_run_history=False, # If True, saves the model in the experiment - set to False to save on memory
        verbose=True, # If True, displays a progress bar for the sweep
    )

    # Assess Parameter Combinations ----------
    # Initialize an empty list to store the results
    all_results = []

    # Iterate through each experiment
    for experiment in experiments:
        model = experiment.model
        eval_results = fitting.perform_evaluations(
            model=model,  # Model to evaluate
            binary_evaluations=binary_evaluations,
            real_binary_matrices=binary_network
        )
        
        # Extract eta and gamma
        eta = experiment.run_config.binary_parameters.eta.item()
        gamma = experiment.run_config.binary_parameters.gamma.item()
        
        # Extract the values for the key
        key = 'MaxCriteria(DegreeKS, ClusteringKS, BetweennessKS, EdgeLengthKS)'
        values = eval_results.binary_evaluations[key]
        
        # Compute the mean of the values
        mean_value = torch.mean(values).item()
        
        # Append the results to the list
        all_results.append({
            'eta': eta,
            'gamma': gamma,
            'mean_energy': mean_value,
        })

    # Find the optimal parameter combination
    optimal_experiments, optimal_energies = fitting.optimise_evaluation(
        experiments=experiments,
        criterion=energy,
    )

    # Convert optimal_results to DataFrame
    optimal_results = []
    for exp, energy in zip(optimal_experiments, optimal_energies):
        optimal_results.append({
            'eta': exp.run_config.binary_parameters.eta.item(),
            'gamma': exp.run_config.binary_parameters.gamma.item(),
            'energy': energy.item()
        })

    # Return
    return all_results, optimal_results

# t0-t1 parameter sweep
t0_t1_results = parameter_sweep(
    seed_binary_network=None,
    binary_network=modelling_data[0]['binary_network'],
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_nodes=modelling_data[0]['num_nodes'],
    num_connections=modelling_data[0]['num_connections']
    )

"""
########## Take this comment out when doing later parameter sweeps

# t1-t2 parameter sweep
t1_t2_optimal_results = parameter_sweep(
    seed_binary_network=modelling_data[0]['binary_network'],
    binary_network=modelling_data[1]['binary_network'], ########## change this to the simulated t1 network
    distance_matrix=modelling_data[1]['distance_matrix'], ########## change this to the simulated t1 dij
    num_nodes=modelling_data[1]['num_nodes'], ########## change this to the simulated t1 numnodes 
    num_connections=modelling_data[1]['num_connections'] ########## change this to the simulated t1 numconnections
    )

# t2-t3 parameter sweep
t2_t3_optimal_results = parameter_sweep(
    seed_binary_network=modelling_data[1]['binary_network'],
    binary_network=modelling_data[2]['binary_network'], ########## change this to the simulated t2 network
    distance_matrix=modelling_data[2]['distance_matrix'], ########## change this to the simulated t2 dij
    num_nodes=modelling_data[2]['num_nodes'], ########## change this to the simulated t2 numnodes 
    num_connections=modelling_data[2]['num_connections'] ########## change this to the simulated t2 numconnections
    )
"""

# %% Plot Energy Landscape --------------------------------------------------
def plot_energy_landscape(results, title='Energy Landscape', save_path=None):
    """
    Plots a 2D energy landscape from a list of dicts with 'eta', 'gamma', and 'mean_energy' keys.

    Parameters:
        results (list of dict): Output from parameter sweep with 'eta', 'gamma', and 'mean_energy'
        title (str): Title of the plot
        save_path (str): Optional path to save the figure
    """

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pivot the data to a matrix form (rows: gamma, columns: eta)
    energy_grid = df.pivot(index='gamma', columns='eta', values='mean_energy')

    # Reverse the gamma axis (index)
    energy_grid = energy_grid.sort_index(ascending=False)

    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        energy_grid,
        cmap='viridis_r',  # _r to have low energy = dark
        annot=False,  # Remove numbers within each box
        cbar_kws={'label': 'Mean Energy', 'ticks': [0, 1]},  # Color bar from 0 to 1
        linewidths=0,  # Remove white lines between boxes
        vmin=0,  # Set color bar minimum to 0
        vmax=1   # Set color bar maximum to 1
    )

    # Labeling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'$\eta$', fontsize=12)  # Use Greek letter for eta
    ax.set_ylabel(r'$\gamma$', fontsize=12)  # Use Greek letter for gamma

    # Explicitly set tick positions to align with the edges
    ax.set_xticks([0.5, len(energy_grid.columns) - 0.5])  # Tick positions at the edges
    ax.set_xticklabels([f"{energy_grid.columns[0]:.2f}", f"{energy_grid.columns[-1]:.2f}"], fontsize=10)
    ax.set_yticks([0.5, len(energy_grid.index) - 0.5])  # Tick positions at the edges
    ax.set_yticklabels([f"{energy_grid.index[0]:.2f}", f"{energy_grid.index[-1]:.2f}"], fontsize=10)

    # Adjust axis limits to ensure ticks are flush with the edges
    ax.set_xlim(0, len(energy_grid.columns))  # Align x-axis ticks with edges
    ax.set_ylim(len(energy_grid.index), 0)  # Align y-axis ticks with edges (reversed)

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

plot_energy_landscape(results=t0_t1_results[0], title='t0 → t1 Energy Landscape', save_path=None)


# %% Model --------------------------------------------------
def model():

# Set the parameters for the t0-t1 model
# Use the optimal parameters from the parameter sweep
t0_t1_binary_parameters = gnm.BinaryGenerativeParameters(
    eta=t1_optimal_results[0]['eta'],
    gamma=t1_optimal_results[0]['gamma'],
    lambdah=lambdah_value,
    distance_relationship_type='powerlaw',
    preferential_relationship_type='powerlaw',
    heterochronicity_relationship_type='powerlaw',
    generative_rule=MatchingIndex(divisor='mean'),
    num_iterations=timepoint_data_list[0]['num_connections'],
    prob_offset=1e-06,
    binary_updates_per_iteration=1
    )

# Define the model
gnm.GenerativeNetworkModel(
    binary_parameters=t0_t1_binary_parameters,
    num_simulations=num_simulations,
    num_nodes=timepoint_data_list[0]['num_nodes'],
    distance_matrix=timepoint_data_list[0]['distance_matrix_tensor'],
    )

# Run the model
t0_t1_model = model.run_model()

# Averaging across runs ----------
# Initialize a list to store the averaged matrices
averaged_matrices = []

# Iterate through each element in t0_t1_model[0]
for element in t0_t1_model[1]:
    # Convert the element to a NumPy array if it's a tensor
    matrices = element.numpy()  # Assuming each element contains multiple matrices

    # Compute the element-wise mean across all matrices
    avg_matrix = matrices.mean(axis=0)  # Average along the first axis

    # Append the averaged matrix to the list
    averaged_matrices.append(avg_matrix)

# Save the averaged matrices as a NumPy array or process further
averaged_matrices = np.array(averaged_matrices)

# Example: Save the averaged matrices to a file ########## add this later
#np.save('averaged_matrices.npy', averaged_matrices)


# %% Analysing simulated Network --------------------------------------------------
# Processing ----------
# Extract the adjM
t1_sim_adjM = averaged_matrices[-1]

# Compute graph metrics
degree = np.sum(t1_sim_adjM != 0, axis=0)

total_edge_length = np.sum(t1_sim_adjM, axis=0)

clustering = bct.clustering_coef_bu(t1_sim_adjM)

betweenness = bct.betweenness_wei(1 / (t1_sim_adjM + np.finfo(float).eps))

efficiency = bct.efficiency_wei(t1_sim_adjM, local=True)

N = t1_sim_adjM.shape[0]
matching_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i != j:
            min_weights = np.minimum(t1_sim_adjM[i, :], t1_sim_adjM[j, :])
            max_weights = np.maximum(t1_sim_adjM[i, :], t1_sim_adjM[j, :])
            if np.sum(max_weights) > 0:  # Avoid division by zero
                matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)

# Plot adjM ----------
plt.figure(figsize=(7, 6))
ax = sns.heatmap(t1_sim_adjM, cmap='RdBu_r', center=0, cbar=True)
ax.set_title('Thresholded Adjacency Matrix', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=10)
num_labels = 10
ax.set_xticks(np.linspace(0, t1_sim_adjM.shape[1] - 1, num_labels))
ax.set_yticks(np.linspace(0, t1_sim_adjM.shape[0] - 1, num_labels))
ax.set_xticklabels(np.linspace(0, t1_sim_adjM.shape[1] - 1, num_labels, dtype=int))
ax.set_yticklabels(np.linspace(0, t1_sim_adjM.shape[0] - 1, num_labels, dtype=int))
plt.show()

# Graph metrics visualisations ----------
# Create a 2x2 panel figure for the histograms
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Increase font size
font_size = 12

# Degree Distribution
sns.kdeplot(degree, color='#440154', ax=axes[0, 0], linewidth=3, alpha=0.7)

# Total Edge Length Distribution
sns.kdeplot(total_edge_length, color='#440154', ax=axes[0, 1], linewidth=3, alpha=0.7)

# Clustering Coefficient Distribution
sns.kdeplot(clustering, color='#440154', ax=axes[1, 0], linewidth=3, alpha=0.7)

# Betweenness Centrality Distribution
sns.kdeplot(betweenness, color='#440154', ax=axes[1, 1], linewidth=3, alpha=0.7)

# Set titles
axes[0, 0].set_title('Degree', fontsize=font_size)
axes[0, 1].set_title('Total Edge Length', fontsize=font_size)
axes[1, 0].set_title('Clustering Coefficient', fontsize=font_size)
axes[1, 1].set_title('Betweenness Centrality', fontsize=font_size)

# Adjust tick label sizes
for ax in axes.flat:
    ax.tick_params(axis='both', which='major', labelsize=font_size)

# Adjust spacing between subplots
fig.tight_layout(pad=2.0)
plt.show()

# 'Topological fingerprint' heatmap ----------
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
plt.show()


# %%
