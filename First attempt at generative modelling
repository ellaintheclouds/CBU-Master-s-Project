# %% Import packages --------------------------------------------------
import os  # for file and directory operations
import scipy.io  # for loading .mat files (MATLAB data)
import numpy as np  # for numerical operations (e.g. matrix manipulation)
from scipy.spatial.distance import cdist  # for computing pairwise Euclidean distances


# %% Define functions --------------------------------------------------
from gnm import BinaryGenerativeParameters, WeightedGenerativeParameters, GenerativeNetworkModel
from gnm.defaults import get_distance_matrix
from gnm.generative_rules import Neighbours
from gnm.weight_criteria import WeightedDistance
from gnm.utils import np_to_tensor


# %% Load data --------------------------------------------------
# # Set working directory for loading data
os.chdir("/imaging/astle/kr01/organoid/OrgNets")

"""
pip install /imaging/astle/er05/GenerativeNetworkModels

"""

# Load data
t1_data = scipy.io.loadmat("C_d96_s2_dt10") # .mat file containing adjacency matrix is loaded

# # Set working directory
os.chdir("/imaging/astle/er05")


# %% Preprocess data  --------------------------------------------------
preprocessed_data = []

# Load adjacency matrix
adjM = t1_data['adjM']

# Filter adjM and dij to only include channels with coordinates ----------
# Extract spike time and associated channel vectors from spike detection data
data_channel = t1_data['data']['channel'][0][0].flatten()
data_frameno = t1_data['data']['frameno'][0][0].flatten()

# Extract coordinates and channel IDs from spike sorting data
coords_channel = t1_data['coords']['channel'][0][0].flatten()
coords_x = t1_data['coords']['x'][0][0].flatten()
coords_y = t1_data['coords']['y'][0][0].flatten()

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


# %% --------------------------------------------------
tensor_dij = np_to_tensor(dij)


# %%
binary_parameters = BinaryGenerativeParameters(
    eta=1.0,
    gamma=-0.5,
    lambdah=1.0,
    distance_relationship_type='exponential',
    preferential_relationship_type='powerlaw',
    heterochronicity_relationship_type='powerlaw',
    generative_rule=Neighbours(),
    num_iterations=250,
    binary_updates_per_iteration=1,
    )

weighted_parameters = WeightedGenerativeParameters(
    alpha=0.003,
    optimisation_criterion=WeightedDistance(),
    weighted_updates_per_iteration=200,
    )

model = GenerativeNetworkModel(
    binary_parameters=binary_parameters,
    weighted_parameters=weighted_parameters,
    num_simulations=100, # Run 100 networks in parallel
    )

model.run_model()

# %% --------------------------------------------------
