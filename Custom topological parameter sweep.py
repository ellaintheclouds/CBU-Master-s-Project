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
import gnm
import seaborn as sns  # Data visualisation


# %% Import GNM functions --------------------------------------------------
from gnm import fitting, evaluation, BinaryGenerativeParameters, GenerativeNetworkModel
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

# %% Model t0-t1 ------------------------------------
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
