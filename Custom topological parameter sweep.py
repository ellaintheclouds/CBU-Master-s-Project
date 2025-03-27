# %% Import packages --------------------------------------------------
import torch # for tensor operations
import time # for timing the parameter sweep
import scipy.io # for loading .mat files (MATLAB data)
import numpy as np # for numerical operations (e.g. matrix manipulation)
import pandas as pd # for saving results
import os # for file operations
import bct # brain Connectivity Toolbox (graph-theoretic analysis)
from scipy.spatial.distance import cdist # compute pairwise Euclidean distances
import matplotlib.pyplot as plt
import numpy as np


# %% Import GNM functions --------------------------------------------------
from gnm import fitting, evaluation
from gnm.generative_rules import MatchingIndex


# %% Load and Process Data --------------------------------------------------
# Set working directory (modify as needed)
os.chdir('/imaging/astle/kr01/organoid/OrgNets')

files = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

timepoint_data_list = []

for idx, file in enumerate(files):
    idx = idx + 1
    print(f'Processing {file} (timepoint {idx}):')

    # Load .mat file containing adjacency matrix
    mat_data = scipy.io.loadmat(file)

    # Load adjacency matrix and process it
    adjM = np.nan_to_num(mat_data['adjM'])

    # Extract spatial coordinates
    coords_channel = mat_data['coords']['channel'][0][0].flatten()
    coords_x = mat_data['coords']['x'][0][0].flatten()
    coords_y = mat_data['coords']['y'][0][0].flatten()

    # Extract spike time and associated channel vectors
    data_channel = mat_data['data']['channel'][0][0].flatten()

    # Remove channels with no coordinate information
    active_channel_idx = np.where(np.isin(coords_channel, np.unique(data_channel)))[0]
    x, y = coords_x[active_channel_idx], coords_y[active_channel_idx]

    # Remove channels with no corresponding coordinates from adjacency matrix
    difference = np.setdiff1d(np.unique(data_channel), coords_channel)
    indices = np.where(np.isin(np.unique(data_channel), difference))[0]
    adjM = np.delete(adjM, indices, axis=0)
    adjM = np.delete(adjM, indices, axis=1)

    # Compute pairwise Euclidean distance matrix
    dij = cdist(np.column_stack((x, y)), np.column_stack((x, y)))

    # Threshold adjacency matrix to retain top 5% strongest connections
    density_level = 0.05
    adjM_thresholded = bct.threshold_proportional(adjM, density_level)

    # Subset adjM and dij for testing
    adjM_thresholded = adjM_thresholded[:50, :50] ########## remove for full analysis
    dij = dij[:50, :50] ########## remove for full analysis

    # Get the number of nodes
    num_nodes = adjM_thresholded.shape[0]
    print(f'- number of nodes: {num_nodes}')

    # Get number of connections and density of the thresholded network
    num_connections = np.count_nonzero(adjM_thresholded) // 2
    print(f'- connections in thresholded matrix: {num_connections}')
    density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)
    print(f'- density of thresholded matrix: {density}')

    # Convert thresholded matrix to binary
    adjM_thresholded_binarized = (adjM_thresholded > 0).astype(int)

    # Convert matrices to PyTorch tensors
    binary_network = torch.tensor(adjM_thresholded_binarized, dtype=torch.float)
    binary_network = binary_network.unsqueeze(0)
    distance_matrix = torch.tensor(dij, dtype=torch.float)

    timepoint_data = {
        'file_name' : file,
        'adjM_thresholded' : adjM_thresholded,
        'dij' : dij,
        'num_nodes' : num_nodes,
        'num_connections' : num_connections,
        'binary_network_tensor' : binary_network,
        'distance_matrix_tensor' : distance_matrix
        }

    timepoint_data_list.append(timepoint_data)


# %% Plotting Timepoints -------------------------------------------------- # do not run this if the matrices are subsetted
os.chdir('/imaging/astle')

# Extract data for plotting
days = [0, 96, 153, 184]  # Include day 0
num_nodes = [0] + [data['num_nodes'] for data in timepoint_data_list]  # Add 0 for day 0
num_connections = [0] + [data['num_connections'] for data in timepoint_data_list]  # Add 0 for day 0

# Create a two-panel figure
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Smaller figure size

# Plot the number of nodes (dot plot)
axes[0].scatter(days, num_nodes, color='blue', s=150)  # Larger dots with 's'
axes[0].set_title('Number of Nodes', fontsize=18)  # Larger title
axes[0].set_xlabel('Day', fontsize=16)  # Larger x-axis label
axes[0].set_ylabel('Count', fontsize=16)  # Larger y-axis label
axes[0].tick_params(axis='both', which='major', labelsize=14)  # Larger tick labels
axes[0].yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce the number of y-axis markers
axes[0].xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce the number of x-axis markers

# Plot the number of connections (dot plot)
axes[1].scatter(days, num_connections, color='red', s=150)  # Larger dots with 's'
axes[1].set_title('Number of Connections', fontsize=18)  # Larger title
axes[1].set_xlabel('Day', fontsize=16)  # Larger x-axis label
axes[1].set_ylabel('Count', fontsize=16)  # Larger y-axis label
axes[1].tick_params(axis='both', which='major', labelsize=14)  # Larger tick labels
axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce the number of y-axis markers
axes[1].xaxis.set_major_locator(plt.MaxNLocator(5))  # Reduce the number of x-axis markers

# Adjust layout and show the plot
plt.tight_layout()
plt.savefig('er05/Organoid project scripts/Output/Chimpanzee/node_and_connection_number.png', dpi=300, bbox_inches='tight')
plt.close()


# %% t0-t1 Parameter Sweep --------------------------------------------------
# Define the parameterspace to explore ----------
# Set parameter ranges for eta and gamma
eta_values = torch.linspace(-3.606, -0.354, 3) ########## increase values to explore fully
gamma_values = torch.linspace(0.212, 0.495, 3) ########## increase values to explore fully
lambdah_value = 2.0  # Fixed lambda

# Define binary sweep parameters
t1_binary_sweep_parameters = fitting.BinarySweepParameters(
    eta=eta_values,
    gamma=gamma_values,
    lambdah=torch.tensor([lambdah_value]),
    distance_relationship_type=['powerlaw'],
    preferential_relationship_type=['powerlaw'],
    heterochronicity_relationship_type=['powerlaw'],
    generative_rule=[MatchingIndex(divisor='mean')],  # Use Matching Index rule
    num_iterations=[timepoint_data_list[0]['num_connections']],  # Match the number of connections in real network
)

# Number of networks to generate per parameter combination
num_simulations = 2 ########## Increase if needed

# Configuration for the parameter sweep
t1_sweep_config = fitting.SweepConfig(
    binary_sweep_parameters=t1_binary_sweep_parameters,
    num_simulations=num_simulations,
    distance_matrix=[timepoint_data_list[0]['distance_matrix_tensor']],
)

# Define Evaluation Criteria ----------
# List of criteria for evaluating binary network properties
t1_criteria = [
    evaluation.DegreeKS(), # Compare degree distributions
    evaluation.ClusteringKS(), # Compare clustering coefficients
    evaluation.BetweennessKS(), # Compare betweenness centrality
    evaluation.EdgeLengthKS(timepoint_data_list[0]['distance_matrix_tensor']), # Compare edge length distributions
]

# Use the maximum KS statistic across all evaluations to measure network similarity
t1_energy = evaluation.MaxCriteria(t1_criteria)
binary_evaluations = [t1_energy]

# Run the Parameter Sweep ----------
print('Running parameter sweep...')

t1_experiments = fitting.perform_sweep(
    sweep_config=t1_sweep_config,
    binary_evaluations=binary_evaluations,
    real_binary_matrices=timepoint_data_list[0]['binary_network_tensor'],  # Real network to compare synthetic networks against    save_model=True, # If True, saves the model in the experiment - set to False to save on memory
    save_run_history=False, # If True, saves the model in the experiment - set to False to save on memory
    verbose=True, # If True, displays a progress bar for the sweep
)

# Assess Parameter Combinations ----------
# Perform evaluations on the models
# Initialize an empty list to store the results
t1_results = []

# Iterate through each experiment
for experiment in t1_experiments:
    model = experiment.model
    eval_results = fitting.perform_evaluations(
        model=model,  # Model to evaluate
        binary_evaluations=binary_evaluations,
        real_binary_matrices=timepoint_data_list[0]['binary_network_tensor']
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
    t1_results.append({
        'eta': eta,
        'gamma': gamma,
        'mean_energy': mean_value,
    })

# Find the optimal parameter combination
t1_optimal_experiments, t1_optimal_energies = fitting.optimise_evaluation(
    experiments=t1_experiments,
    criterion=t1_energy,
)

# Save Results ----------
# Define the output directory
os.chdir('/imaging/astle')

# Convert optimal_results to DataFrame
t1_optimal_results = []
for exp, energy in zip(t1_optimal_experiments, t1_optimal_energies):
    t1_optimal_results.append({
        'eta': exp.run_config.binary_parameters.eta.item(),
        'gamma': exp.run_config.binary_parameters.gamma.item(),
        'energy': energy.item()
    })

pd.DataFrame(t1_optimal_results).to_csv('er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/custom_parameter_sweep_optimal.csv', index=False)
print(f'Optimal results saved.')

pd.DataFrame(t1_results).to_csv('er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/custom_parameter_sweep_all.csv', index=False)
print(f'All results saved.')

