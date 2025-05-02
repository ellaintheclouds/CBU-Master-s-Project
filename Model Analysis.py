# %% Import Packages and Functions --------------------------------------------------
# Core libraries
import os
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Graph theory metrics (Brain Connectivity Toolbox for Python)
from bct import clustering_coef_wu, betweenness_wei


# %% Find optimal iteration of model --------------------------------------------------
# Load all_runs.csv for each sweep
base_dir = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps'
t0_t1_runs = pd.read_csv(f'{base_dir}/t0_t1/all_runs.csv')
t1_t2_runs = pd.read_csv(f'{base_dir}/t1_t2/all_runs.csv')
t2_t3_runs = pd.read_csv(f'{base_dir}/t2_t3/all_runs.csv')

mean0 = pd.read_csv(f'{base_dir}/t0_t1/all_parameters.csv')
mean1 = pd.read_csv(f'{base_dir}/t1_t2/all_parameters.csv')
mean2 = pd.read_csv(f'{base_dir}/t2_t3/all_parameters.csv')

# Merge mean energies for easy lookup
merged_means = mean0.merge(mean1, on=['eta', 'gamma'], suffixes=('_0', '_1'))
merged_means = merged_means.merge(mean2, on=['eta', 'gamma'])
merged_means.rename(columns={'mean_energy': 'mean_energy_2'}, inplace=True)

# Add deviation per simulation
def get_deviation_df(df, mean_df, idx):
    df = df.copy()
    df = df.merge(mean_df, on=['eta', 'gamma'], suffixes=('', '_mean'))
    df['abs_dev'] = abs(df['energy'] - df[f'mean_energy_{idx}'])
    return df

dev0 = get_deviation_df(t0_t1_runs, mean0, 0)
dev1 = get_deviation_df(t1_t2_runs, mean1, 1)
dev2 = get_deviation_df(t2_t3_runs, mean2, 2)

# Sum absolute deviations across timepoints
deviation_sum = dev0[['eta', 'gamma', 'sim_idx', 'abs_dev']].rename(columns={'abs_dev': 'dev_0'})
deviation_sum['dev_1'] = dev1['abs_dev']
deviation_sum['dev_2'] = dev2['abs_dev']
deviation_sum['total_dev'] = deviation_sum[['dev_0', 'dev_1', 'dev_2']].sum(axis=1)

# Find the row with the lowest total deviation
best_sim = deviation_sum.loc[deviation_sum['total_dev'].idxmin()]
best_eta, best_gamma, best_idx = best_sim['eta'], best_sim['gamma'], int(best_sim['sim_idx'])
print(f"Most representative sim_idx: {best_idx} for eta={best_eta}, gamma={best_gamma}")

# Load the selected simulation adjacency matrices
def load_matrix_by_idx(path, sim_idx):
    raw_data = np.load(f"{path}/raw_model_outputs.npy", allow_pickle=True).item()
    return raw_data[sim_idx]['weight_snapshots']


# %% Extract Metrics Across Snapshots --------------------------------------------------
# Load the weighted snapshots
t0_t1_model = load_matrix_by_idx('/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1', best_idx)
t1_t2_model = load_matrix_by_idx('/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2', best_idx)
t2_t3_model = load_matrix_by_idx('/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3', best_idx)

for snapshot in to_t1_model & t1_t2_model & t2_t3_model:
    degree = np.sum(snapshot != 0, axis=0)

    total_edge_length = np.sum(snapshot, axis=0)

    clustering = bct.clustering_coef_bu(snapshot)

    betweenness = bct.betweenness_wei(1 / (snapshot + np.finfo(float).eps))

    efficiency = bct.efficiency_wei(snapshot, local=True)

    start_time = time.time()
    N = snapshot.shape[0]
    matching_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                min_weights = np.minimum(adjM[i, :], adjM[j, :])
                max_weights = np.maximum(adjM[i, :], adjM[j, :])
                if np.sum(max_weights) > 0:  # Avoid division by zero
                    matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)


# %% Plot Metric Distributions across three simulated timepoints -------------------------------------------------------------
# Compute metrics
metrics = []
for idx, adj in enumerate([t0_t1_model, t1_t2_model, t2_t3_model]):
    deg = np.sum((adj > 0).astype(int), axis=1)
    clustering = clustering_coef_wu(adj)
    betweenness = betweenness_wei(adj)
    edge_lengths = []

    #caluclate dij
    dij = timepoint_slice_data[idx]['dij'][0:50, 0:50] ########## change when alaysing full network
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] > 0:
                edge_lengths.append(dij[i, j])
    metrics.append({
        'degree': deg,
        'clustering': clustering,
        'betweenness': betweenness,
        'total_edge_length': edge_lengths
    })

# Plot KDE panel
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
font_size, tick_size = 18, 16
graph_colours = {
    'degree': ['#beb0e8', '#6d4dcb', '#301e67'],
    'clustering': ['#b7cce1', '#93b3d2', '#4b80b4'],
    'betweenness': ['#cae1e8', '#95c3d0', '#5fa6b9'],
    'total_edge_length': ['#d7f4eb', '#afe9d8', '#73d9ba']
}
timepoints = [0, 1, 2]

for i, tp in enumerate(timepoints):
    sns.kdeplot(metrics[tp]['degree'], color=graph_colours['degree'][i], ax=axes[0, 0], linewidth=3, alpha=0.7)
    sns.kdeplot(metrics[tp]['clustering'], color=graph_colours['clustering'][i], ax=axes[0, 1], linewidth=3, alpha=0.7)
    sns.kdeplot(metrics[tp]['betweenness'], color=graph_colours['betweenness'][i], ax=axes[1, 0], linewidth=3, alpha=0.7)
    sns.kdeplot(metrics[tp]['total_edge_length'], color=graph_colours['total_edge_length'][i], ax=axes[1, 1], linewidth=3, alpha=0.7)

axes[0, 0].set_xlabel('Degree', fontsize=font_size)
axes[0, 1].set_xlabel('Clustering Coefficient', fontsize=font_size)
axes[1, 0].set_xlabel('Betweenness Centrality', fontsize=font_size)
axes[1, 1].set_xlabel('Total Edge Length', fontsize=font_size)

for ax in axes.flatten():
    ax.set_ylabel('Density', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('lightgrey')
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
plt.savefig('/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/energy_landscape_across_timepoints.png', bbox_inches='tight', dpi=300)
plt.close()
