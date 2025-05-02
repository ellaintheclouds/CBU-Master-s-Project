# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import bct

# Operations
import numpy as np
import os
import pickle
import torch
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


# %% Import Custom Functions --------------------------------------------------
# Change directory to load in custom functions
os.chdir('/imaging/astle/er05/Organoid project scripts')

# Load custom functions
from Functions.binary_parameter_sweep import binary_parameter_sweep
from Functions.weighted_parameter_sweep import weighted_parameter_sweep
from Functions.plot_energy_landscape import plot_energy_landscape
from Functions.model import model
from Functions.view_model import view_model


# %% Load and Process Data --------------------------------------------------
# Set working directory
os.chdir('/imaging/astle/er05')
pickle_dir = '/imaging/astle/er05/Organoid project scripts/Output/processed_data.pkl'

# Load organoid slice data
with open(pickle_dir, 'rb') as f:
    raw_slice_data = pickle.load(f)

# Define the filenames to use for each timepoint
timepoint_slice_data = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

# Initialise storage list
modelling_data = []

# Loop through slices and process each one
for slice in raw_slice_data:
    if slice['file_name'] in timepoint_slice_data:
        # Extract adjacency and distance matrices
        adjM = np.maximum(slice['adjM'], 0)
        dij = slice['dij']

        # Threshold top 5% strongest connections
        adjM_thresholded = bct.threshold_proportional(adjM, 0.05)

        # Optional: Temporarily downsize for testing
        adjM_thresholded = adjM_thresholded[40:140, 40:140]
        dij = dij[40:140, 40:140]

        # Create binary matrix
        binary_adjM = (adjM_thresholded > 0).astype(int)

        # Convert to PyTorch tensors
        weighted_network = torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0)
        binary_network = torch.tensor(binary_adjM, dtype=torch.float).unsqueeze(0)
        distance_matrix = torch.tensor(dij, dtype=torch.float).unsqueeze(0)

        # Count nodes and connections
        num_nodes = adjM_thresholded.shape[0]
        total_connections = np.count_nonzero(adjM_thresholded) // 2

        # Compute number of new connections
        if not modelling_data:
            num_connections = total_connections
        else:
            prev_total = modelling_data[-1]['total_connections']
            num_connections = total_connections - prev_total

        # Store the results
        modelling_data.append({
            'file_name': slice['file_name'],
            'adjM_thresholded': adjM_thresholded,
            'weighted_network': weighted_network,
            'binary_network': binary_network,
            'distance_matrix': distance_matrix,
            'num_nodes': num_nodes,
            'num_connections': num_connections,
            'total_connections': total_connections
        })


# %% Set Parameterspace to Explore --------------------------------------------------
# Define parameter ranges
eta_values = torch.linspace(-10, 3, 4)
gamma_values = torch.linspace(-2, 8, 4)
alpha_values = torch.linspace(0, 1, 4)

# Define number of simulations
num_sweep = 2   # for parameter sweep
num_model = 10  # for final model run


# %% t0-t1 Parameter Sweep --------------------------------------------------
t0_t1_sweep_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t0_t1'

t0_t1_binary_results, t0_t1_optimal_binary_results = binary_parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_connections=modelling_data[0]['num_connections'],
    binary_network=modelling_data[0]['binary_network'],
    output_filepath=t0_t1_sweep_filepath
)

t0_t1_weighted_results, t0_t1_optimal_weighted_results = weighted_parameter_sweep(
    optimal_binary_parameters=t0_t1_optimal_binary_results,
    alpha_values=alpha_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_connections=modelling_data[0]['num_connections'],
    weighted_network=modelling_data[0]['weighted_network'],
    output_filepath=t0_t1_sweep_filepath
)

plot_energy_landscape(
    binary_results=t0_t1_binary_results,
    weighted_results=t0_t1_weighted_results,
    binary_title='t0 → t1 Binary Energy Landscape',
    weighted_title='t0 → t1 Weighted Energy Landscape',
    output_filepath=t0_t1_sweep_filepath
)


# %% t0-t1 Model --------------------------------------------------
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_model = model(
    optimal_results=pd.read_csv(f'{t0_t1_sweep_filepath}/optimal_weighted_parameters.csv'),
    num_simulations=num_model,
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_nodes=modelling_data[0]['num_nodes'],
    num_connections=modelling_data[0]['num_connections'],
    real_network=modelling_data[0]['weighted_network'],
    binary_network=modelling_data[0]['binary_network'],
    weighted_network=modelling_data[0]['weighted_network'],
    output_filepath=t0_t1_model_filepath
)


# %% t0-t1 View Model --------------------------------------------------
t0_t1_data = np.load(f'{t0_t1_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()
view_model(t0_t1_data['weight_snapshots'], output_filepath=t0_t1_model_filepath)


# %% t1-t2 Parameter Sweep --------------------------------------------------
# Get last timepoint from t0-t1 model as seed
t0_t1_last = t0_t1_data['weight_snapshots'][-1]
t0_t1_last_tensor = torch.tensor(t0_t1_last, dtype=torch.float).unsqueeze(0)

t0_t1_last_binary = t0_t1_data['adjacency_snapshots'][-1]
t0_t1_last_binary_tensor = torch.tensor(t0_t1_last_binary, dtype=torch.float).unsqueeze(0)

# Set output directory
t1_t2_sweep_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t1_t2'

# Binary parameter sweep
t1_t2_binary_results, t1_t2_optimal_binary_results = binary_parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_connections=modelling_data[1]['num_connections'],
    binary_network=modelling_data[1]['binary_network'],
    seed_binary_network=t0_t1_last_binary_tensor,
    output_filepath=t1_t2_sweep_filepath
)

# Weighted parameter sweep
t1_t2_weighted_results, t1_t2_optimal_weighted_results = weighted_parameter_sweep(
    optimal_binary_parameters=t1_t2_optimal_binary_results,
    alpha_values=alpha_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_connections=modelling_data[1]['num_connections'],
    weighted_network=modelling_data[1]['weighted_network'],
    seed_binary_network=t0_t1_last_binary_tensor,
    seed_weighted_network=t0_t1_last_tensor,
    output_filepath=t1_t2_sweep_filepath
)

# Plot energy landscape
plot_energy_landscape(
    binary_results=t1_t2_binary_results,
    weighted_results=t1_t2_weighted_results,
    binary_title='t1 → t2 Binary Energy Landscape',
    weighted_title='t1 → t2 Weighted Energy Landscape',
    output_filepath=t1_t2_sweep_filepath
)


# %% t1-t2 Model --------------------------------------------------
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_model = model(
    optimal_results=pd.read_csv(f'{t1_t2_sweep_filepath}/optimal_weighted_parameters.csv'),
    num_simulations=num_model,
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_nodes=modelling_data[1]['num_nodes'],
    num_connections=modelling_data[1]['num_connections'],
    real_network=modelling_data[1]['weighted_network'],
    seed_binary_network=t0_t1_last_binary_tensor,
    seed_weighted_network=t0_t1_last_tensor,
    binary_network=modelling_data[1]['binary_network'],
    weighted_network=modelling_data[1]['weighted_network'],
    output_filepath=t1_t2_model_filepath
)


# %% t1-t2 View Model --------------------------------------------------
t1_t2_data = np.load(f'{t1_t2_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()
view_model(t1_t2_data['weight_snapshots'], output_filepath=t1_t2_model_filepath)


# %% t2-t3 Parameter Sweep --------------------------------------------------
# Get last timepoint from t1-t2 model as seed
t1_t2_last = t1_t2_data['weight_snapshots'][-1]
t1_t2_last_tensor = torch.tensor(t1_t2_last, dtype=torch.float).unsqueeze(0)

t1_t2_last_binary = t1_t2_data['adjacency_snapshots'][-1]
t1_t2_last_binary_tensor = torch.tensor(t1_t2_last_binary, dtype=torch.float).unsqueeze(0)

# Set output directory
t2_t3_sweep_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t2_t3'

# Binary parameter sweep
t2_t3_binary_results, t2_t3_optimal_binary_results = binary_parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_connections=modelling_data[2]['num_connections'],
    binary_network=modelling_data[2]['binary_network'],
    seed_binary_network=t1_t2_last_binary_tensor,
    output_filepath=t2_t3_sweep_filepath
)

# Weighted parameter sweep
t2_t3_weighted_results, t2_t3_optimal_weighted_results = weighted_parameter_sweep(
    optimal_binary_parameters=t2_t3_optimal_binary_results,
    alpha_values=alpha_values,
    num_simulations=num_sweep,
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_connections=modelling_data[2]['num_connections'],
    weighted_network=modelling_data[2]['weighted_network'],
    seed_binary_network=t1_t2_last_binary_tensor,
    seed_weighted_network=t1_t2_last_tensor,
    output_filepath=t2_t3_sweep_filepath
)

# Plot energy landscape
plot_energy_landscape(
    binary_results=t2_t3_binary_results,
    weighted_results=t2_t3_weighted_results,
    binary_title='t2 → t3 Binary Energy Landscape',
    weighted_title='t2 → t3 Weighted Energy Landscape',
    output_filepath=t2_t3_sweep_filepath
)


# %% t2-t3 Model --------------------------------------------------
t2_t3_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3'
t2_t3_model = model(
    optimal_results=pd.read_csv(f'{t2_t3_sweep_filepath}/optimal_weighted_parameters.csv'),
    num_simulations=num_model,
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_nodes=modelling_data[2]['num_nodes'],
    num_connections=modelling_data[2]['num_connections'],
    real_network=modelling_data[2]['weighted_network'],
    seed_binary_network=t1_t2_last_binary_tensor,
    seed_weighted_network=t1_t2_last_tensor,
    binary_network=modelling_data[2]['binary_network'],
    weighted_network=modelling_data[2]['weighted_network'],
    output_filepath=t2_t3_model_filepath
)


# %% t2-t3 View Model --------------------------------------------------
t2_t3_data = np.load(f'{t2_t3_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()
view_model(t2_t3_data['weight_snapshots'], output_filepath=t2_t3_model_filepath)


# %%