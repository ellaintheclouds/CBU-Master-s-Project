# Import packages --------------------------------------------------
import os # for file and directory operations
import scipy.io # for loading .mat files (MATLAB data)
import matplotlib.pyplot as plt # for plotting - the "as plt part allows us to refer to the package as plt"
import numpy as np # for numerical operations (e.g. matrix manipulation)
import seaborn as sns # for heatmaps and enhanced data visualisation
import bct # for graph-theoretic analysis (from the brain connectivity toolbox)
from glob import glob # for finding files that match a certain pattern
from scipy.spatial.distance import cdist # for computing pairwise Euclidean distances


# Load data --------------------------------------------------
# Load individual data
# mat_data = scipy.io.loadmat("kr01/organoid/OrgNets/C_d95_s1_dt5") # .mat file containing adjacency matrix is loaded

# List all .mat files in the "matrices/" folder
matrix_files = [file for file in glob("kr01/organoid/OrgNets/*.mat") if ("C" in os.path.basename(file) or "H" in os.path.basename(file)) and "dt10" in os.path.basename(file)]

# Load the first file to check the keys
mat_data = scipy.io.loadmat(matrix_files[0])
print("Keys in the MATLAB file:", mat_data.keys())

# Check shapes of relevant matrices
for key in ['adjM', 'adjM_trimmed', 'dij', 'coords']:
    if key in mat_data:
        print(f"{key} shape: {np.shape(mat_data[key])}")


# Exploring coordinates --------------------------------------------------
# Extract the coordinates
coords_raw = mat_data['coords']
coords_raw = coords_raw[0, 0]  

# Extract x and y values
x_values = np.array(coords_raw['x']).astype(float).flatten()
y_values = np.array(coords_raw['y']).astype(float).flatten()

# Plot the electrode/node positions
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, c='blue', alpha=0.7)
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Electrode/Node Positions")
plt.grid(True)
plt.savefig("er05/Organoid project scripts/Output/Chimpanzee/Electrode positions.png")


# Sorting out matrix size mismatch --------------------------------------------------
# Extract relevant coordinates-----
coords = mat_data['coords']['channel'][0][0]  # Extract coordinates
active_channel_idx = mat_data['active_channel_idx']  # Extract active channel indices
active_coords = coords[active_channel_idx.flatten()] # Extract active coordinates

# Recompute the distance matrix-----
# Compute pairwise distances between active nodes
filtered_dij = cdist(active_coords[:, 1:], active_coords[:, 1:])  # Exclude channel column

# Filter adjM-----
adjM = mat_data['adjM']
# Extract vectors from the loaded .mat file
data_channel = mat_data['data']['channel'][0][0].flatten()  # Flatten to get a 1D array
coords_channel = mat_data['coords']['channel'][0][0].flatten()

# Get unique values
unique_data_channel = np.unique(data_channel)

# Perform set difference
difference = np.setdiff1d(unique_data_channel, coords_channel)

# Find indices in unique_data_channel that are in the set difference
indices = np.where(np.isin(unique_data_channel, difference))[0]

# Remove the indices from adjM on both dimensions
adjM = mat_data['adjM']
adjM = np.delete(adjM, indices, axis=0)
adjM = np.delete(adjM, indices, axis=1)

# Check shapes
print(adjM.shape, filtered_dij.shape)

