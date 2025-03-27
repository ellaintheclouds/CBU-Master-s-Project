# %% Import packages --------------------------------------------------
import scipy.io  # for loading .mat files (MATLAB data)
import numpy as np  # for numerical operations (e.g. matrix manipulation)
import bct  # Brain Connectivity Toolbox (graph-theoretic analysis)
import itertools # For generating parameter combinations


# %% Import functions --------------------------------------------------
from gnm import BinaryGenerativeParameters, WeightedGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex


# %% Load data --------------------------------------------------
# # Set working directory for loading data
os.chdir('/imaging/astle/kr01/organoid/OrgNets')

# Load data
t1_data = scipy.io.loadmat('C_d96_s2_dt10') # .mat file containing adjacency matrix is loaded

# Extract and subset adjacency matrix
t1_adjM = np.nan_to_num(t1_data['adjM']) # Extract adjacency matrix from .mat file and convert NaN to 0
t1_adjM = t1_adjM[0:50,0:50] # Subset adjacency matrix to 50 nodes ##########
t1_num_nodes = t1_adjM.shape[0]
print(f"Number of nodes: {t1_num_nodes}")

# Threshold adjacency matrix
density_level = 0.05
t1_adjM_thresholded = bct.threshold_proportional(t1_adjM, density_level)
print(f"Threshold: {density_level}")

# Get the number of connections in the thresholded matrix
t1_num_connections = np.count_nonzero(t1_adjM_thresholded) // 2
print(f"Number of connections in thresholded matrix: {t1_num_connections}")

# Calculate the density of the thresholded matrix
t1_density = t1_num_connections / ((t1_num_nodes * (t1_num_nodes - 1)) / 2)
print(f"Density: {t1_density}")


# %% Define Parameterspace to Test --------------------------------------------------
# Energy cravasse suggested by Danyal Akarca
eta_values = np.linspace(-3.606, 0.254, 5)
gamma_values = np.linspace(0.212, 0.495, 5)

# Generate all possible combinations
parameter_combinations = list(itertools.product(eta_values, gamma_values))
parameter_combinations = parameter_combinations[0:1] # For testing, only run the first parameter set ##########


# %% Generative Modelling --------------------------------------------------
# Store results
results = []

# Iterate over parameter sets
for eta, gamma in parameter_combinations:
    print(f"Running model with eta={eta}, gamma={gamma}")

    binary_parameters = BinaryGenerativeParameters(
        eta=eta,
        gamma=gamma,
        lambdah=2,
        distance_relationship_type='exponential',
        preferential_relationship_type='powerlaw',
        heterochronicity_relationship_type='powerlaw',
        generative_rule=MatchingIndex(divisor='mean'),
        num_iterations=61,
        binary_updates_per_iteration=1,
    )

    model = GenerativeNetworkModel(
        binary_parameters=binary_parameters,
        num_nodes = 50,
        num_simulations=1,  # Run 10 networks in parallel ##########
    )

    model.run_model()
    
    model['weight_matrix']

    # Store the results (you might want to extract some error metric or network similarity measure)
    results.append({
        'eta': eta,
        'gamma': gamma,
        'model': model  # Store the model object for later analysis
    })
