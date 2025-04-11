#%% Packages --------------------------------------------------
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


#%% Read in data -----------------------------------------------------
# Define File Paths
path = "/imaging/astle/kr01/organoid/OrgData/"
file = "Cd96s2.h5"  
full_path = path + file

# Load Data
with h5py.File(full_path, "r") as f:
    spike_data = f["/proc0/spikeTimes"]
    spike_times = spike_data["frameno"][:]  # Frame numbers
    spike_channels = spike_data["channel"][:]  # Channel IDs
    rec_duration = f["/assay/inputs/record_time"][()]

# Convert spike times to seconds
Hz = 20000  
spike_times_sec = spike_times / Hz # Convert to seconds
spike_times_sec = (spike_times - spike_times.min()) / Hz # Normalise to start from 0


#%% Compute STTC Adjacency Matrix -----------------------------
def compute_sttc(spike_times, spike_channels, lag, Hz):
    """
    Compute the Spike Time Tiling Coefficient (STTC).
    This function assumes `spike_times` are in seconds.
    """
    unique_channels, channel_indices = np.unique(spike_channels, return_inverse=True)
    n_channels = len(unique_channels)
    adjM = np.zeros((n_channels, n_channels))

    # Simulated STTC computation - replace with your real method
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                # Example computation: STTC depends on co-firing probability
                adjM[i, j] = np.random.uniform(-1, 1)  # Replace with real STTC logic

    return adjM, unique_channels, channel_indices

# Compute STTC
lag = 0.01  # STTC lag in seconds
adjM, unique_channels, channel_indices = compute_sttc(spike_times_sec, spike_channels, lag, Hz)

# Compute average STTC per channel
channel_sttc = np.mean(adjM, axis=1)  # Mean STTC for each channel

# Map raw channel IDs in spike_channels to the indices of unique_channels
color_values = channel_sttc[channel_indices]  # Correct mapping!

#%% Plot Raster with STTC Coloring -----------------------------
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# Filter data for the first 30 seconds
time_limit = 30  # Time limit in seconds
mask = spike_times_sec <= time_limit  # Boolean mask for filtering

# Apply the mask to filter spike times and channels
spike_times_sec_filtered = spike_times_sec[mask]
spike_channels_filtered = spike_channels[mask]
color_values_filtered = np.array(color_values)[mask]  # Filter color values as well

# Plot Raster for the First 30 Seconds
plt.figure(figsize=(7.2, 6))  # Increase figure size for better clarity

# Scatter plot with enhanced marker visibility and diverging colormap
norm = TwoSlopeNorm(vmin=np.min(color_values_filtered), vcenter=0, vmax=np.max(color_values_filtered))
plt.scatter(
    spike_times_sec_filtered,
    spike_channels_filtered,
    c=color_values_filtered,
    cmap="bwr",  # Use the diverging colormap
    norm=norm,  # Normalize to center at 0
    s=1,  # Larger marker size for better visibility
    alpha=0.8  # Slightly higher alpha for better contrast
)

# Add colorbar with 3 labels
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=12)  # Increase colorbar tick label size
cbar.set_ticks([np.min(color_values_filtered), 0, np.max(color_values_filtered)])  # Set 3 labels
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format colorbar ticks to 2 decimal places

# Adjust axis labels and title
plt.xlabel("Time (s)", fontsize=14)  # X-axis label in light grey
plt.ylabel("Electrode Channel", fontsize=14, )  # Y-axis label in light grey

# Adjust tick parameters
plt.xticks(fontsize=12)  # X-axis tick labels in light grey
plt.yticks(fontsize=12)  # Y-axis tick labels in light grey

# Limit both axes to 4 ticks
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(4))  # Limit x-axis to 4 ticks
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit y-axis to 4 ticks

# Adjust axis lines
ax = plt.gca()
ax.spines["top"].set_visible(False)  # Remove top boundary
ax.spines["right"].set_visible(False)  # Remove right boundary
ax.spines["left"].set_visible(False)  # Set left boundary to grey
ax.spines["bottom"].set_visible(False)  # Set bottom boundary to grey
ax.spines["left"].set_linewidth(1)
ax.spines["bottom"].set_linewidth(1)

# Set axis limits with padding
plt.xlim([0, time_limit])  # Set x-axis range to 30 seconds
plt.ylim(
    [np.min(spike_channels_filtered) - 1, np.max(spike_channels_filtered) + 1]
)  # Add padding to y-axis

# Save the refined plot with higher DPI
save_path_refined = "/imaging/astle/er05/Organoid project scripts/Output/activity_raster_colored_30s_refined.png"
#plt.savefig(save_path_refined, dpi=600, bbox_inches="tight")  # Higher DPI for publication

plt.show()

print(f"Refined raster plot (30s) saved to: {save_path_refined}")
# %%
