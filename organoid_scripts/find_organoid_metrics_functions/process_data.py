# %% Import Packages and Functions --------------------------------------------------
import numpy as np
import os
import scipy.io
from scipy.spatial.distance import cdist


# %% Define the Function --------------------------------------------------
def process_data(file_path):
    """Load and process a single .mat file."""
    file_name = os.path.basename(file_path).replace('.mat', '')
    mat_data = scipy.io.loadmat(file_path)

    # Load adjacency matrix and process it
    adjM = np.nan_to_num(mat_data['adjM'])  # Replace NaNs with 0
    adjM[adjM < 0] = 0  # Clip negative values to zero

    # Extract spike detection data
    data_channel = mat_data['data']['channel'][0][0].flatten()
    data_frameno = mat_data['data']['frameno'][0][0].flatten()

    # Extract coordinates and channel IDs
    coords_channel = mat_data['coords']['channel'][0][0].flatten()
    coords_x = mat_data['coords']['x'][0][0].flatten()
    coords_y = mat_data['coords']['y'][0][0].flatten()

    # --- Get intersection of valid channels ---
    valid_channels = np.intersect1d(np.unique(data_channel), coords_channel)

    # --- Build index mapping of valid channels ---
    # Indices in adjM (rows/cols) correspond to unique(data_channel) sorted
    all_channels_sorted = np.sort(np.unique(data_channel))
    adjM_channel_indices = np.where(np.isin(all_channels_sorted, valid_channels))[0]

    # Make sure we clip adjM using the correct channel indices
    adjM = adjM[np.ix_(adjM_channel_indices, adjM_channel_indices)]

    # --- Filter coordinates to only valid channels ---
    coord_mask = np.isin(coords_channel, valid_channels)
    x = coords_x[coord_mask]
    y = coords_y[coord_mask]

    # Ensure dij is computed from same-order channels as used in adjM
    coords_filtered = coords_channel[coord_mask]
    sort_order = np.argsort(coords_filtered)
    x = x[sort_order]
    y = y[sort_order]

    dij = cdist(np.column_stack((x, y)), np.column_stack((x, y)))

    return file_name, adjM, dij

# &&
################################################################
# %% Import packages --------------------------------------------------
from glob import glob
import pandas as pd
import os
import pickle


# %% Import Custom Functions --------------------------------------------------
# Change directory to load in custom functions
os.chdir('/imaging/astle/er05/organoid_scripts')

# Load custom functions
from find_organoid_metrics_functions.process_data import process_data
from find_organoid_metrics_functions.sort_data import sort_data
from find_organoid_metrics_functions.compute_metrics import compute_metrics


# %% Things to Change ==================================================
# Define density levels
density_levels = [0.01, 0.05, 0.1, 0.2, 1]

# Should organoid metrics be re-computed if they already exist in the pickle file?
recompute_metrics = True


# %% Set Directories and Load Data ---------------------------------
# Define working directory
os.chdir('/imaging/astle')

# Set output directory
output_dir = 'er05/organoid_scripts/output'
os.makedirs(output_dir, exist_ok=True)

# Load data
matrix_files = (
    [file for file in glob('kr01/organoid/OrgNets/*.mat') 
     if ('C' in os.path.basename(file) or 'H' in os.path.basename(file)) and 'dt10' in os.path.basename(file)] + ########## change this to add in human data
    [file for file in glob('er05/H5 Analysis/Organoid Data/*.mat')]
)
if len(matrix_files) == 1:
    print(f'{len(matrix_files)} organoid file loaded: {matrix_files}', flush=True)
else:
    print(f'{len(matrix_files)} organoid files loaded: {matrix_files}', flush=True)


# Load existing processed data if available
pickle_dir = f'{output_dir}/processed_data.pkl'

if os.path.exists(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        processed_data = pickle.load(f)
        print(f'Loaded existing processed data with {len(processed_data)} entries.', flush=True)
else:
    print('No existing processed data found.', flush=True)


# %% Find Organoid Metrics ---------------------------------------------------
# Create empty list
processed_data = []

# Initialise dataframes
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()

for idx, file_path in enumerate(matrix_files):
    
    # Process data
    file_name, adjM, dij = process_data(file_path=file_path)

################################################################