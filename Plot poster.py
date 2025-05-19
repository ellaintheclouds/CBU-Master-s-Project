# %% Import packages --------------------------------------------------
import os  # File operations
import scipy.io  # Load MATLAB .mat files
import matplotlib.pyplot as plt  # Plotting
import numpy as np  # Numerical operations
import seaborn as sns  # Data visualisation
import bct  # Brain Connectivity Toolbox (graph-theoretic analysis)
import pickle  # Save/load processed data


#%% Load
os.chdir('/imaging/astle')

with open('er05/Organoid project scripts/Output/processed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

# Create a 1x4 panel figure for the histograms
fig, axes = plt.subplots(1, 4, figsize=(16, 3))  # 1 row, 4 columns layout

# Increase font size for labels and ticks
font_size = 14  # Larger font size for labels
tick_size = 12  # Smaller font size for ticks

# Define base colors for each graph and generate shades for timepoints
graph_colours = {
    'degree': ['#beb0e8', '#6d4dcb', '#301e67'],  # Shades for Degree
    'clustering': ['#b7cce1', '#93b3d2', '#4b80b4'],  # Shades for Clustering
    'betweenness': ['#cae1e8', '#95c3d0', '#5fa6b9'],  # Shades for Betweenness
    'total_edge_length': ['#d7f4eb', '#afe9d8', '#73d9ba'],  # Shades for Edge length
}

# Define timepoints
timepoints = [2, 0, 6]  # Indices for t1, t2, t3
timepoint_labels = ['T1', 'T2', 'T3']  # Short labels for timepoints

# Degree Distribution (Plot 1)
for i, tp in enumerate(timepoints):
    sns.kdeplot(processed_data[tp]['metrics'][0]['degree'], color=graph_colours['degree'][i], ax=axes[0], linewidth=3, alpha=0.7)
    # Annotate with a line and label
    axes[0].annotate(
        timepoint_labels[i],
        xy=(50, 0.015 - i * 0.002),  # Position near the curve
        xytext=(70, 0.02 - i * 0.004 * i),  # Adjusted position for the label
        arrowprops=dict(arrowstyle='-', color=graph_colours['degree'][i], lw=1.5),
        fontsize=font_size,
        color=graph_colours['degree'][i],
        fontweight='bold'
    )
axes[0].set_xlabel('')  # Remove x-axis label
axes[0].set_ylabel('Density', fontsize=font_size)  # Only leftmost plot has "Density"
axes[0].tick_params(axis='both', labelsize=tick_size)  # Adjust tick size
axes[0].yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit to 4 y-axis labels

# Clustering Coefficient Distribution (Plot 2)
for i, tp in enumerate(timepoints):
    sns.kdeplot(processed_data[tp]['metrics'][0]['clustering'], color=graph_colours['clustering'][i], ax=axes[1], linewidth=3, alpha=0.7)
    # Annotate with a line and label
    axes[1].annotate(
        timepoint_labels[i],
        xy=(0.1, 8 - i * 1),  # Position near the curve
        xytext=(0.2, 10 - i * 1.5 * i),  # Adjusted position for the label
        arrowprops=dict(arrowstyle='-', color=graph_colours['clustering'][i], lw=1.5),
        fontsize=font_size,
        color=graph_colours['clustering'][i],
        fontweight='bold'
    )
axes[1].set_xlabel('')  # Remove x-axis label
axes[1].set_ylabel('')  # No "Density" label
axes[1].tick_params(axis='both', labelsize=tick_size)  # Adjust tick size
axes[1].yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit to 4 y-axis labels

# Betweenness Centrality Distribution (Plot 3)
for i, tp in enumerate(timepoints):
    sns.kdeplot(processed_data[tp]['metrics'][0]['betweenness'], color=graph_colours['betweenness'][i], ax=axes[2], linewidth=3, alpha=0.7)
    # Annotate with a line and label
    axes[2].annotate(
        timepoint_labels[i],
        xy=(10000, 0.0003 - i * 0.00003),  # Position near the curve
        xytext=(20000, 0.0004 - i * 0.00005 * i),  # Adjusted position for the label
        arrowprops=dict(arrowstyle='-', color=graph_colours['betweenness'][i], lw=1.5),
        fontsize=font_size,
        color=graph_colours['betweenness'][i],
        fontweight='bold'
    )
axes[2].set_xlabel('')  # Remove x-axis label
axes[2].set_ylabel('')  # No "Density" label
axes[2].tick_params(axis='both', labelsize=tick_size)  # Adjust tick size
axes[2].yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit to 4 y-axis labels

# Total Edge Length Distribution (Plot 4)
for i, tp in enumerate(timepoints):
    sns.kdeplot(processed_data[tp]['metrics'][0]['total_edge_length'], color=graph_colours['total_edge_length'][i], ax=axes[3], linewidth=3, alpha=0.7)
    # Annotate with a line and label
    axes[3].annotate(
        timepoint_labels[i],
        xy=(10, 0.1 - i * 0.01),  # Position near the curve
        xytext=(20, 0.12 - i * 0.02 * i),  # Adjusted position for the label
        arrowprops=dict(arrowstyle='-', color=graph_colours['total_edge_length'][i], lw=1.5),
        fontsize=font_size,
        color=graph_colours['total_edge_length'][i],
        fontweight='bold'
    )
axes[3].set_xlabel('')  # Remove x-axis label
axes[3].set_ylabel('')  # No "Density" label
axes[3].tick_params(axis='both', labelsize=tick_size)  # Adjust tick size
axes[3].yaxis.set_major_locator(plt.MaxNLocator(4))  # Limit to 4 y-axis labels

# Customize axes
for ax in axes:
    ax.spines['right'].set_visible(False)  # Remove right boundary
    ax.spines['top'].set_visible(False)    # Remove top boundary
    ax.spines['left'].set_color('lightgrey')  # Set left boundary to light grey
    ax.spines['bottom'].set_color('lightgrey')  # Set bottom boundary to light grey
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

# Adjust spacing between subplots
fig.tight_layout(pad=2.0, w_pad=3.0)  # Adjust spacing between plots

plt.show()


# %% adjM plot
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# Load adjacency matrix
adjM = processed_data[0]['adjM']  # Load adjacency matrix

# Remove negative values in adjM
adjM[adjM < 0] = 0

# Threshold adjM
threshold = 0.05  # Set threshold
adjM[adjM < threshold] = 0  # Set values below threshold to zero

# Plot adjacency matrix
plt.figure(figsize=(7.2, 6))
ax = sns.heatmap(adjM, cmap='RdPu', cbar=True)  # High-contrast colormap

# Adjust tick parameters
ax.tick_params(axis='both', which='major', labelsize=10)

# Limit x-axis and y-axis to 4 labels
ax.xaxis.set_major_locator(MaxNLocator(4))  # Limit x-axis to 4 labels
ax.yaxis.set_major_locator(MaxNLocator(4))  # Limit y-axis to 4 labels

# Adjust colorbar to have 3 labels
cbar = ax.collections[0].colorbar
cbar.set_ticks([adjM.min(), (adjM.min() + adjM.max()) / 2, adjM.max()])  # Set 3 labels: min, midpoint, max
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format colorbar ticks to 2 decimal places
cbar.ax.tick_params(labelsize=10)  # Adjust colorbar tick label size

# Adjust x-axis and y-axis labels
num_labels = 4  # Number of labels you want to show
ax.set_xticks(np.linspace(0, adjM.shape[1] - 1, num_labels))
ax.set_yticks(np.linspace(0, adjM.shape[0] - 1, num_labels))
ax.set_xticklabels(np.linspace(0, adjM.shape[1] - 1, num_labels, dtype=int))
ax.set_yticklabels(np.linspace(0, adjM.shape[0] - 1, num_labels, dtype=int))

plt.show()
# %%
