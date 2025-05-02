# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import bct

# Operations
import numpy as np
import os
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# %% Define the Function --------------------------------------------------
def view_model(matrices, output_filepath=None):
    # Ensure the output directory exists
    os.makedirs(output_filepath, exist_ok=True)
    
    if len(matrices)>10:

        # Visualise every 10th averaged matrix, starting from the 10th connection
        step = 10
        start_idx = 9
        selected_matrices = matrices[start_idx::step]

        # Set up figure layout
        num_matrices = len(selected_matrices)
        cols = 5 
        rows = (num_matrices + cols - 1) // cols

        # Create the figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()

        # Loop through the selected matrices and plot them
        for i, ax in enumerate(axes):
            if i < num_matrices:
                sns.heatmap(
                    selected_matrices[i],
                    cmap='viridis',
                    cbar=(i == 0),
                    ax=ax
                )

                # Set the title for each subplot
                ax.set_title(f'{(start_idx + i * step + 1)} Connections', fontsize=12)
                
                # Customise the first plot with axis labels and colourbar
                if i == 0:
                    num_labels = 4
                    ticks = np.linspace(0, selected_matrices[i].shape[0] - 1, num_labels, dtype=int)
                    ax.set_xticks(ticks)
                    ax.set_yticks(ticks)
                    ax.set_xticklabels(ticks, fontsize=10)
                    ax.set_yticklabels(ticks, fontsize=10)

                    # Customise colourbar
                    cbar = ax.collections[0].colorbar
                    vmin, vmax = selected_matrices[i].min(), selected_matrices[i].max()
                    cbar.set_ticks([vmin, vmax])
                    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.axis('off')

        # Save the averaged matrices plot
        fig.tight_layout(pad=3.0)
        plt.savefig(f'{output_filepath}/matrix_growth.png', dpi=300)
        plt.close(fig)

    # Analyse the final simulated timepoint
    sim_adjm = matrices[-1]

    # Compute graph metrics
    degree = np.sum(sim_adjm != 0, axis=0)
    total_edge_length = np.sum(sim_adjm, axis=0)
    clustering = bct.clustering_coef_bu(sim_adjm)
    betweenness = bct.betweenness_wei(1 / (sim_adjm + np.finfo(float).eps))
    efficiency = bct.efficiency_wei(sim_adjm, local=True)

    # Compute matching index
    N = sim_adjm.shape[0]
    matching_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            min_w = np.minimum(sim_adjm[i, :], sim_adjm[j, :])
            max_w = np.maximum(sim_adjm[i, :], sim_adjm[j, :])
            denom = np.sum(max_w)
            if denom > 0:
                mi = np.sum(min_w) / denom
                matching_matrix[i, j] = mi
                matching_matrix[j, i] = mi  # Symmetry

    matching_index = np.mean(matching_matrix, axis=1)

    # Save adjacency matrix plot
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(sim_adjm, cmap='viridis', cbar=True)
    ax.set_title('Thresholded Adjacency Matrix', fontsize=14)
    num_labels = 10
    ticks = np.linspace(0, sim_adjm.shape[0] - 1, num_labels, dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    plt.savefig(f'{output_filepath}/adjacency_matrix.png', dpi=300)
    plt.close()

    # Plot graph metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    font_size, tick_size = 16, 14
    colour = '#1f77b4'

    sns.kdeplot(degree, color=colour, ax=axes[0, 0], linewidth=4, alpha=0.7)
    axes[0, 0].set_xlabel('Degree', fontsize=font_size)
    axes[0, 0].set_ylabel('Density', fontsize=font_size)

    sns.kdeplot(total_edge_length, color=colour, ax=axes[0, 1], linewidth=4, alpha=0.7)
    axes[0, 1].set_xlabel('Total Edge Length', fontsize=font_size)
    axes[0, 1].set_ylabel('Density', fontsize=font_size)

    sns.kdeplot(clustering, color=colour, ax=axes[1, 0], linewidth=4, alpha=0.7)
    axes[1, 0].set_xlabel('Clustering Coefficient', fontsize=font_size)
    axes[1, 0].set_ylabel('Density', fontsize=font_size)

    sns.kdeplot(betweenness, color=colour, ax=axes[1, 1], linewidth=4, alpha=0.7)
    axes[1, 1].set_xlabel('Betweenness Centrality', fontsize=font_size)
    axes[1, 1].set_ylabel('Density', fontsize=font_size)

    for ax in axes.flatten():
        ax.tick_params(axis='both', labelsize=tick_size)

    fig.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(f'{output_filepath}/graph_metrics.png', dpi=300)
    plt.close(fig)

    # Compute and save correlation heatmap
    metrics_df = pd.DataFrame({
        'degree': degree,
        'clustering': clustering,
        'betweenness': betweenness,
        'total_edge_length': total_edge_length,
        'efficiency': efficiency,
        'matching_index': matching_index
    })

    correlation_matrix = metrics_df.corr()
    labels = ['Degree', 'Clustering', 'Betweenness', 'Total Edge Length', 'Efficiency', 'Matching Index']

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, cmap='RdBu_r', xticklabels=labels, yticklabels=labels, center=0, cbar=True)
    ax.set_title('Topological Fingerprint Heatmap', fontsize=14)
    ax.tick_params(axis='both', labelsize=10)
    plt.savefig(f'{output_filepath}/correlation_heatmap.png', dpi=300)
    plt.close()