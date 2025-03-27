# %% Import packages --------------------------------------------------
import torch
import time
import scipy.io  # for loading .mat files (MATLAB data)
import numpy as np  # for numerical operations (e.g. matrix manipulation)
import bct  # Brain Connectivity Toolbox (graph-theoretic analysis)
from scipy.spatial.distance import cdist  # Pairwise Euclidean distances


# %% Import functions --------------------------------------------------
from gnm import fitting, generative_rules, weight_criteria, evaluation, defaults
from gnm.generative_rules import MatchingIndex


# %% Load data --------------------------------------------------
# # Set working directory for loading data
os.chdir('/imaging/astle/kr01/organoid/OrgNets')

# Load data
mat_data = scipy.io.loadmat('C_d96_s2_dt10') # .mat file containing adjacency matrix is loaded


# %% Process data --------------------------------------------------
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


# %% Process adjM and view its statistics
"""
# Subset adjM and dij
adjM = adjM[0:50,0:50] # Subset adjacency matrix to 50 nodes ##########
dij = dij[0:50,0:50] # Subset distance matrix to 50 nodes ##########
"""

# Get the number of nodes in the adjacency matrix
num_nodes = adjM.shape[0]
print(f'Number of nodes: {num_nodes}')

# Threshold adjacency matrix
density_level = 0.05
adjM_thresholded = bct.threshold_proportional(adjM, density_level)
print(f'Threshold: {density_level}')

# Get the number of connections in the thresholded matrix
num_connections = np.count_nonzero(adjM_thresholded) // 2
print(f'Number of connections in thresholded matrix: {num_connections}')

# Calculate the density of the thresholded matrix
density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)
print(f'Density: {density}')

# Binarize adjM
adjM_thresholded_binarized = (adjM_thresholded > 0).astype(int) ##########


# %% Set-Up the Parameter Sweep --------------------------------------------------
binary_consensus_network = adjM_thresholded_binarized
distance_matrix = dij

# Convert binary_consensus_network to a PyTorch tensor with dtype=float
binary_consensus_network = torch.tensor(binary_consensus_network, dtype=torch.float)
binary_consensus_network = binary_consensus_network.unsqueeze(0)  # Shape: (1, num_nodes, num_nodes)

# Convert distance_matrix to a PyTorch tensor with dtype=float
distance_matrix = torch.tensor(distance_matrix, dtype=torch.float)

# Set parameter ranges for the sweep
eta_values = torch.linspace(-3.606, -0.354, 5) ########## change to observe more values
gamma_values = torch.linspace(0.212, 0.495, 5) ########## change to observe more values

# Define binary sweep parameters
binary_sweep_parameters = fitting.BinarySweepParameters(
    eta=eta_values,
    gamma=gamma_values,
    lambdah=torch.Tensor([2.0]),  # Fixed at 2
    distance_relationship_type=['powerlaw'],
    preferential_relationship_type=['powerlaw'],
    heterochronicity_relationship_type=['powerlaw'],
    generative_rule=[MatchingIndex(divisor='mean')],
    num_iterations=[num_connections],  # Match the number of connections in the real network
)

"""
# Define weighting sweep (if needed)
weighted_sweep_parameters = fitting.WeightedSweepParameters(
    alpha=[0.01],
    optimisation_criterion=[
        weight_criteria.DistanceWeightedCommunicability(distance_matrix=distance_matrix)
    ],
)
"""

# Number of simulations per parameter set
num_simulations = 1 ##########

# Sweep configuration
sweep_config = fitting.SweepConfig(
    binary_sweep_parameters=binary_sweep_parameters,
    num_simulations=num_simulations,
    distance_matrix=[distance_matrix]
)

# Evaluation criteria
criteria = [
    evaluation.ClusteringKS(),
    evaluation.DegreeKS(),
    evaluation.EdgeLengthKS(distance_matrix),
]

# Evaluate how well the generated networks fit
energy = evaluation.MaxCriteria(criteria)  

binary_evaluations = [energy]
weighted_evaluations = [
    evaluation.WeightedNodeStrengthKS(normalise=True),
    evaluation.WeightedClusteringKS(),
]


# %% Run the Parameter Sweep --------------------------------------------------# --- RUNNING THE PARAMETER SWEEP ---
start_time = time.perf_counter()

experiments = fitting.perform_sweep(
    sweep_config=sweep_config,
    binary_evaluations=binary_evaluations,
    real_binary_matrices=binary_consensus_network,
    save_model=False,
    save_run_history=False,
)

end_time = time.perf_counter()

# --- PRINT RUN TIME ---
print(f'Sweep took {end_time - start_time:0.3f} seconds.')

total_simulations = num_simulations * len(eta_values) * len(gamma_values)
print(f'Total number of simulations: {total_simulations}')
print(f'Average time per simulation: {(end_time - start_time) / total_simulations:0.3f} seconds.')

# %% Find the Best Parameter Set --------------------------------------------------
optimal_experiments, optimal_energies = fitting.optimise_evaluation(
    experiments=experiments,
    criterion=energy,
)

optimal_experiment = optimal_experiments[0]
optimal_energy = optimal_energies[0]

print(f'Optimal energy: {optimal_energy:0.3f}')
print(f'Optimal value of eta: {optimal_experiment.run_config.binary_parameters.eta:0.2f}')
print(f'Optimal value of gamma: {optimal_experiment.run_config.binary_parameters.gamma:0.2f}')
