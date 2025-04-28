# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import bct
from bct import clustering_coef_wu, betweenness_wei, strengths_und
import gnm
from gnm import fitting, evaluation, BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex
from gnm.weight_criteria import Communicability

# Operations
import numpy as np
import os
import pickle
import scipy.io
import time
import torch
from scipy.spatial.distance import cdist

# Plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


# %% Define Sweep Functions --------------------------------------------------
def parameter_sweep(
    eta_values,
    gamma_values,
    lambdah_value,
    num_simulations,
    distance_matrix,
    num_nodes,
    num_connections,
    seed_weighted_network=None,
    seed_binary_network=None,
    weighted_network=None,
    binary_network=None,
    output_filepath=None
):

    # Determine mode and print what mode is being used
    is_weighted = weighted_network is not None
    print(f'Running parameter sweep in {'weighted' if is_weighted else 'binary'} mode.')

    # Define sweep parameters
    if is_weighted:
        weighted_sweep_parameters = fitting.WeightedSweepParameters(
            alpha=torch.tensor([0.05]),
            optimisation_criterion=[Communicability(omega=1.0)],
        )
    else:
        weighted_sweep_parameters = None

    binary_sweep_parameters = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([lambdah_value]),
        distance_relationship_type=['powerlaw'],
        preferential_relationship_type=['powerlaw'],
        heterochronicity_relationship_type=['powerlaw'],
        generative_rule=[MatchingIndex(divisor='mean')],
        num_iterations=[num_connections],
    )

    # Create the sweep configuration
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_sweep_parameters,
        weighted_sweep_parameters=weighted_sweep_parameters,
        num_simulations=num_simulations,
        distance_matrix=distance_matrix,
        seed_adjacency_matrix=seed_binary_network,
        seed_weight_matrix=seed_weighted_network
    )

    # Define the evaluation criteria
    if is_weighted:
        criteria = [
            evaluation.WeightedNodeStrengthKS(),
            evaluation.WeightedBetweennessKS(),
            evaluation.WeightedClusteringKS()
        ]
        eval_set = [evaluation.MaxCriteria(criteria)]
        eval_key = 'MaxCriteria(WeightedNodeStrengthKS, WeightedBetweennessKS, WeightedClusteringKS)'
    else:
        criteria = [
            evaluation.DegreeKS(),
            evaluation.ClusteringKS(),
            evaluation.BetweennessKS(),
            evaluation.EdgeLengthKS(distance_matrix)
        ]
        eval_set = [evaluation.MaxCriteria(criteria)]
        eval_key = 'MaxCriteria(DegreeKS, ClusteringKS, BetweennessKS, EdgeLengthKS)'

    # Perform the parameter sweep
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        weighted_evaluations=eval_set if is_weighted else None,
        binary_evaluations=eval_set if not is_weighted else None,
        real_binary_matrices=binary_network if not is_weighted else None,
        real_weighted_matrices=weighted_network if is_weighted else None,
        save_run_history=False,
        verbose=True
    )

    # Evaluate all parameter combinations
    all_results = []

    for experiment in experiments:
        model = experiment.model
        eval_results = fitting.perform_evaluations(
            model=model,
            weighted_evaluations=eval_set if is_weighted else None,
            binary_evaluations=eval_set if not is_weighted else None,
            real_binary_matrices=binary_network if not is_weighted else None,
            real_weighted_matrices=weighted_network if is_weighted else None
        )

        eta = experiment.run_config.binary_parameters.eta.item()
        gamma = experiment.run_config.binary_parameters.gamma.item()
        alpha = experiment.run_config.weighted_parameters.alpha.item() if is_weighted else None

        values = eval_results.weighted_evaluations[eval_key] if is_weighted else eval_results.binary_evaluations[eval_key]
        mean_energy = torch.mean(values).item()

        summary = {
            'eta': eta,
            'gamma': gamma,
            'mean_energy': mean_energy
        }
        
        if is_weighted:
            summary['alpha'] = alpha

        all_results.append(summary)



    # Find optimal parameter combinations
    optimal_experiments, optimal_energies = fitting.optimise_evaluation(
        experiments=experiments,
        criterion=eval_set[0]
    )

    optimal_results = []

    for exp, energy_val in zip(optimal_experiments, optimal_energies):
        eta = exp.run_config.binary_parameters.eta.item()
        gamma = exp.run_config.binary_parameters.gamma.item()
        alpha = exp.run_config.weighted_parameters.alpha.item() if is_weighted else None

        result = {
            'eta': eta,
            'gamma': gamma,
            'lambdah': lambdah_value,
            'energy': mean_energy
            }

        if is_weighted:
            result['alpha'] = alpha

        optimal_results.append(result)

    # Save results as csv files
    if output_filepath:
        pd.DataFrame(all_results).to_csv(f'{output_filepath}/all_parameters.csv', index=False)
        pd.DataFrame(optimal_results).to_csv(f'{output_filepath}/optimal_parameters.csv', index=False)

    return all_results, optimal_results

def plot_energy_landscape(
    results,
    title='Energy Landscape',
    output_filepath=None):

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Pivot the data to a matrix form (rows: gamma, columns: eta)
    energy_grid = df.pivot(index='gamma', columns='eta', values='mean_energy')

    # Reverse the gamma axis (index)
    energy_grid = energy_grid.sort_index(ascending=False)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        energy_grid,
        cmap='viridis_r',  # _r to have low energy = dark
        annot=False,  # Remove numbers within each box
        cbar_kws={'label': 'Mean Energy', 'ticks': [0, 1]},  # Color bar from 0 to 1
        linewidths=0,  # Remove white lines between boxes
        vmin=0,  # Set color bar minimum to 0
        vmax=1   # Set color bar maximum to 1
    )

    # Labeling
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(r'$\eta$', fontsize=12)  # Use Greek letter for eta
    ax.set_ylabel(r'$\gamma$', fontsize=12)  # Use Greek letter for gamma

    # Explicitly set tick positions to align with the edges
    ax.set_xticks([0.5, len(energy_grid.columns) - 0.5])  # Tick positions at the edges
    ax.set_xticklabels([f'{energy_grid.columns[0]:.2f}', f'{energy_grid.columns[-1]:.2f}'], fontsize=10)
    ax.set_yticks([0.5, len(energy_grid.index) - 0.5])  # Tick positions at the edges
    ax.set_yticklabels([f'{energy_grid.index[0]:.2f}', f'{energy_grid.index[-1]:.2f}'], fontsize=10)

    # Adjust axis limits to ensure ticks are flush with the edges
    ax.set_xlim(0, len(energy_grid.columns))  # Align x-axis ticks with edges
    ax.set_ylim(len(energy_grid.index), 0)  # Align y-axis ticks with edges (reversed)

    # Save or show
    if output_filepath:
        plt.savefig(f'{output_filepath}/energy_landscape.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


# %% Define Model Functions --------------------------------------------------
def model(
    optimal_results,
    num_simulations,
    distance_matrix,
    num_nodes,
    num_connections,
    real_network,
    seed_binary_network=None,
    seed_weighted_network=None,
    binary_network=None,
    weighted_network=None,
    output_filepath=None
    ):

    # Determine mode and print what mode is being used
    is_weighted = weighted_network is not None
    print(f'Running model in {'weighted' if is_weighted else 'binary'} mode.')

    # Set up sweep parameters ----------
    # Weighted mode
    if is_weighted:
        weighted_parameters = gnm.WeightedGenerativeParameters(
            alpha=torch.tensor(0.05),
            optimisation_criterion=Communicability(omega=1.0)
        )

    # Binary mode
    else:
        weighted_parameters = None
    
    # Binary parameters passed regardless of mode
    binary_parameters = gnm.BinaryGenerativeParameters(
        eta=float(optimal_results['eta'].iloc[0]),
        gamma=float(optimal_results['gamma'].iloc[0]),
        lambdah=optimal_results['lambdah'][0],
        distance_relationship_type='powerlaw',
        preferential_relationship_type='powerlaw',
        heterochronicity_relationship_type='powerlaw',
        generative_rule=MatchingIndex(divisor='mean'),
        num_iterations=num_connections,
        prob_offset=1e-06,
        binary_updates_per_iteration=1
        )

    # Repeat the dij along the first dimension to match num_simulations
    distance_matrix = distance_matrix.repeat(num_simulations, 1, 1)

    # Define the model
    model = gnm.GenerativeNetworkModel(
            binary_parameters=binary_parameters,
            num_simulations=num_simulations,
            num_nodes=num_nodes,
            seed_adjacency_matrix=seed_binary_network,
            distance_matrix=distance_matrix,
            verbose=True,
            weighted_parameters=weighted_parameters if is_weighted else None,
            seed_weight_matrix=seed_weighted_network
            )

    # Run the model
    run_model = model.run_model()

    # Unpack the run_model output ----------
    adjacency_snapshots = run_model[1]
    weight_snapshots = run_model[2] if is_weighted else None


    # Calculate the energy of each model run ----------
    # Stack generated networks
    synthetic_matrices = torch.stack([torch.tensor(matrix, dtype=torch.float32) for matrix in adjacency_snapshots])

    # Choose correct real network
    real_network = real_network.squeeze(0).unsqueeze(0)

    # Choose evaluation criteria
    if is_weighted:
        criteria = [
            evaluation.WeightedNodeStrengthKS(),
            evaluation.WeightedClusteringKS(),
            evaluation.WeightedBetweennessKS()
        ]
    else:
        criteria = [
            evaluation.DegreeKS(),
            evaluation.ClusteringKS(),
            evaluation.BetweennessKS(),
            evaluation.EdgeLengthKS(distance_matrix)
        ]
    eval_set = evaluation.MaxCriteria(criteria)

    # Evaluate
    ks_values = eval_set(synthetic_matrices, real_network)
    ks_values = ks_values.squeeze(1)

    # Find simulation closest to mean KS
    mean_ks = torch.mean(ks_values)
    best_sim_idx = torch.argmin(torch.abs(ks_values - mean_ks)).item()

    # Save raw matrices from the model ----------
    # Convert the best adjacency snapshot to numpy
    adjacency_numpy = adjacency_snapshots[best_sim_idx].numpy()

    # If weighted, also get best weight snapshot
    if weight_snapshots is not None:
        weight_numpy = weight_snapshots[best_sim_idx].numpy()
    else:
        weight_numpy = None

    # Create dictionary of raw outputs
    raw_output = {
        'adjacency_snapshots': adjacency_numpy,
        'weight_snapshots': weight_numpy
    }

    # Save best simulation
    if output_filepath:
        np.save(f'{output_filepath}/raw_model_outputs.npy', raw_output)

def analyse_model(matrices, output_filepath=None):
    # Ensure the output directory exists
    os.makedirs(output_filepath, exist_ok=True)

    # Visualize every 5th averaged matrix
    step = 10  # Visualize every 5th connection
    start_idx = 9  # Start from the 10th connection
    selected_matrices = matrices[start_idx::step] # Select every 5th matrix

    # Set up figure layout
    num_matrices = len(selected_matrices)
    cols = 5  # Number of columns for the grid
    rows = (num_matrices + cols - 1) // cols # Calculate number of rows for the grid

    # Create the figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    # Loop through the selected matrices and plot them
    for i, ax in enumerate(axes):
        if i < num_matrices:
            sns.heatmap(
                selected_matrices[i],
                cmap='viridis',
                cbar=(i == 0), # Only show the colourbar for the first plot
                ax=ax
            )

            # Set the title for each subplot
            ax.set_title(f'{(start_index + i * step + 1)} Connections', fontsize=12)
            
            # Customise the first plot with axis labels and colourbar
            if i == 0:
                num_labels = 4
                ax.set_xticks(np.linspace(0, selected_matrices[i].shape[1] - 1, num_labels))
                ax.set_yticks(np.linspace(0, selected_matrices[i].shape[0] - 1, num_labels))
                ax.set_xticklabels(np.linspace(0, selected_matrices[i].shape[1] - 1, num_labels, dtype=int), fontsize=10)
                ax.set_yticklabels(np.linspace(0, selected_matrices[i].shape[0] - 1, num_labels, dtype=int), fontsize=10)
                colorbar = ax.collections[0].colorbar
                colorbar.set_ticks([selected_matrices[i].min(), selected_matrices[i].max()])
                colorbar.set_ticklabels([f'{selected_matrices[i].min():.2f}', f'{selected_matrices[i].max():.2f}'])

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
    sim_adjm = matrix_series[-1]

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
        for j in range(N):
            if i != j:
                min_weights = np.minimum(sim_adjm[i, :], sim_adjm[j, :])
                max_weights = np.maximum(sim_adjm[i, :], sim_adjm[j, :])
                if np.sum(max_weights) > 0:  # Avoid division by zero
                    matching_matrix[i, j] = np.sum(min_weights) / np.sum(max_weights)

    # Save adjacency matrix plot
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(sim_adjm, cmap='viridis', cbar=True)
    ax.set_title('Thresholded Adjacency Matrix', fontsize=14)
    num_labels = 10
    ax.set_xticks(np.linspace(0, sim_adjm.shape[1] - 1, num_labels))
    ax.set_yticks(np.linspace(0, sim_adjm.shape[0] - 1, num_labels))
    ax.set_xticklabels(np.linspace(0, sim_adjm.shape[1] - 1, num_labels, dtype=int))
    ax.set_yticklabels(np.linspace(0, sim_adjm.shape[0] - 1, num_labels, dtype=int))
    plt.savefig(f'{output_filepath}/adjacency_matrix.png', dpi=300)
    plt.close()

    # Plot graph metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    font_size = 16
    tick_size = 14
    line_color = '#1f77b4'

    sns.kdeplot(degree, color=line_color, ax=axes[0, 0], linewidth=4, alpha=0.7)
    axes[0, 0].set_xlabel('Degree', fontsize=font_size)
    axes[0, 0].set_ylabel('Density', fontsize=font_size)
    axes[0, 0].tick_params(axis='both', labelsize=tick_size)

    sns.kdeplot(total_edge_length, color=line_color, ax=axes[0, 1], linewidth=4, alpha=0.7)
    axes[0, 1].set_xlabel('Total Edge Length', fontsize=font_size)
    axes[0, 1].set_ylabel('Density', fontsize=font_size)
    axes[0, 1].tick_params(axis='both', labelsize=tick_size)

    sns.kdeplot(clustering, color=line_color, ax=axes[1, 0], linewidth=4, alpha=0.7)
    axes[1, 0].set_xlabel('Clustering Coefficient', fontsize=font_size)
    axes[1, 0].set_ylabel('Density', fontsize=font_size)
    axes[1, 0].tick_params(axis='both', labelsize=tick_size)

    sns.kdeplot(betweenness, color=line_color, ax=axes[1, 1], linewidth=4, alpha=0.7)
    axes[1, 1].set_xlabel('Betweenness Centrality', fontsize=font_size)
    axes[1, 1].set_ylabel('Density', fontsize=font_size)
    axes[1, 1].tick_params(axis='both', labelsize=tick_size)

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
        'matching_index': np.mean(matching_matrix, axis=1)
    })
    correlation_matrix = metrics_df.corr()
    formatted_labels = ['Degree', 'Clustering', 'Betweenness', 'Total Edge Length', 'Efficiency', 'Matching Index']

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, cmap='RdBu_r', xticklabels=formatted_labels, yticklabels=formatted_labels, center=0, cbar=True)
    ax.set_title('Topological Fingerprint Heatmap', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=10)
    plt.savefig(f'{output_filepath}/correlation_heatmap.png', dpi=300)
    plt.close()


# %% Load and Process Data --------------------------------------------------
# Set working directory
os.chdir('/imaging/astle/er05')
pickle_dir = '/imaging/astle/er05/Organoid project scripts/Output/processed_data.pkl'

# Ensure the pickle directory exists
if not os.path.exists(os.path.dirname(pickle_dir)):
    os.makedirs(os.path.dirname(pickle_dir))

# Initialise lists
slice_data = []
modelling_data = []

# Load organoid slice data
with open(pickle_dir, 'rb') as f:
    slice_data = pickle.load(f)

# Define slices for each timepoint
timepoint_slice_data = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

for slice in slice_data:
    if slice['file_name'] in timepoint_slice_data:
        adjM = np.maximum(slice['adjM'], 0)
        dij = slice['dij']

        # Threshold to keep top 5% strongest connections
        adjM_thresholded = bct.threshold_proportional(adjM, 0.05)

        # Temporary downsize for testing
        adjM_thresholded = adjM_thresholded[40:140, 40:140] ########## change when using full network

        dij = dij[40:140, 40:140] ########## change when using full network

        # Calculate total connections at this timepoint
        total_connections = np.count_nonzero(adjM_thresholded) // 2
        num_nodes = adjM_thresholded.shape[0]

        # If it's the first timepoint, keep all connections
        if len(modelling_data) == 0:
            num_connections = total_connections
        else:
            prev_total = modelling_data[-1]['total_connections']
            num_connections = total_connections - prev_total
        
        # Binarise and convert to tensors
        binary_adjM = (adjM_thresholded > 0).astype(int)
        weighted_network = torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0)
        binary_network = torch.tensor(binary_adjM, dtype=torch.float).unsqueeze(0)
        distance_matrix = torch.tensor(dij, dtype=torch.float).unsqueeze(0)

        # Store in modelling_data
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
# Define parameters
eta_values = torch.linspace(-10, 3, 100) ########## may need to change this 
gamma_values = torch.linspace(-2, 8, 100) ########## may need to change this
lambdah_value = 2.0 ########## may need to change this

# Number of networks to generate per parameter combination
num_simulations = 10 ########## change this depending on the time available


# %% t0-t1 Parameter Sweep --------------------------------------------------
# Set output
t0_t1_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t0_t1'

# Ensure the directory exists
if not os.path.exists(t0_t1_sweep_filepath):
    os.makedirs(t0_t1_sweep_filepath)

# Parameter sweep
t0_t1_results, t0_t1_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_weighted_network=None,
    seed_binary_network=None,
    weighted_network=modelling_data[0]['weighted_network'],
    binary_network=modelling_data[0]['binary_network'],
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_nodes=modelling_data[0]['num_nodes'],
    num_connections=modelling_data[0]['num_connections'],
    output_filepath=t0_t1_sweep_filepath
)

# Plot energy landscape
plot_energy_landscape(
    results=t0_t1_results,
    title='t0 → t1 Energy Landscape',
    output_filepath=t0_t1_sweep_filepath
    )
    

# %% t0-t1 Model --------------------------------------------------
# Load the optimal results into a DataFrame
t0_t1_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t0_t1'
t0_t1_optimal_results = pd.read_csv(f'{t0_t1_sweep_filepath}/optimal_parameters.csv')

t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'

# Ensure the directory exists
if not os.path.exists(t0_t1_model_filepath):
    os.makedirs(t0_t1_model_filepath)

# Create the model
t0_t1_model = model(
    optimal_results=t0_t1_optimal_results,
    num_simulations=num_simulations,
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_connections=modelling_data[0]['num_connections'],
    num_nodes=modelling_data[0]['num_nodes'],
    real_network=modelling_data[0]['weighted_network'],
    seed_binary_network=None,
    seed_weighted_network=None,
    binary_network=modelling_data[0]['binary_network'],
    weighted_network=modelling_data[0]['weighted_network'],
    output_filepath=t0_t1_model_filepath
    )


# %% t0-t1 Assess Model --------------------------------------------------
# Load the raw model outputs
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_matrices = np.load(f'{t0_t1_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Extract weight snapshots
weight_snapshots = t0_t1_matrices['weight_snapshots']

# Analyse the t0-t1 model
analyse_model(matrices=weight_snapshots, output_filepath=t0_t1_model_filepath)


# %% t1-t2 Parameter Sweep --------------------------------------------------
# Load the model's averaged matrices for seed
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_matrices = np.load(f'{t0_t1_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Get last timepoints from t0-t1 model and convert to tensors
simulated_t1 = t0_t1_matrices['weight_snapshots'][-1]
simulated_t1 = torch.tensor(simulated_t1, dtype=torch.float)
simulated_t1 = simulated_t1.unsqueeze(0)

simulated_t1_binary = t0_t1_matrices['adjacency_snapshots'][-1]
simulated_t1_binary = torch.tensor(simulated_t1_binary, dtype=torch.float)
simulated_t1_binary = simulated_t1_binary.unsqueeze(0)

# Set output
t1_t2_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t1_t2'

# Ensure the directory exists
if not os.path.exists(t1_t2_sweep_filepath):
    os.makedirs(t1_t2_sweep_filepath)

# Parameter sweep
t1_t2_results, t1_t2_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_weighted_network=simulated_t1,
    seed_binary_network=simulated_t1_binary,
    weighted_network=modelling_data[1]['weighted_network'],
    binary_network=modelling_data[1]['binary_network'],
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_nodes=modelling_data[1]['num_nodes'],
    num_connections=modelling_data[1]['num_connections'],
    output_filepath=t1_t2_sweep_filepath
    )

# Plot energy landscape
plot_energy_landscape(results=t1_t2_results, title='t1 → t2 Energy Landscape', output_filepath=t1_t2_sweep_filepath)


# %% t1-t2 Model --------------------------------------------------
# Load the model's averaged matrices for seed
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_matrices = np.load(f'{t0_t1_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Get last timepoints from t0-t1 model and convert to tensors
simulated_t1 = t0_t1_matrices['weight_snapshots'][-1]
simulated_t1 = torch.tensor(simulated_t1, dtype=torch.float)

simulated_t1_binary = t0_t1_matrices['adjacency_snapshots'][-1]
simulated_t1_binary = torch.tensor(simulated_t1_binary, dtype=torch.float)

# Load the optimal results into a DataFrame
t1_t2_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t1_t2'
t1_t2_results = pd.read_csv(f'{t1_t2_sweep_filepath}/optimal_parameters.csv')

t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'

# Ensure the directory exists
if not os.path.exists(t1_t2_model_filepath):
    os.makedirs(t1_t2_model_filepath)

# Create the model
t1_t2_model = model(
    optimal_results=t1_t2_results,
    num_simulations=num_simulations,
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_connections=modelling_data[1]['num_connections'],
    num_nodes=modelling_data[1]['num_nodes'],
    real_network=modelling_data[1]['weighted_network'],    
    seed_binary_network=simulated_t1_binary,
    seed_weighted_network=simulated_t1,
    binary_network=modelling_data[1]['binary_network'],
    weighted_network=modelling_data[1]['weighted_network'],
    output_filepath=t1_t2_model_filepath
    )


# %% t1-t2 Assess Model --------------------------------------------------
# Load the model's averaged matrices for analysis
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_matrices = np.load(f'{t1_t2_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Extract weight snapshots
weight_snapshots = t1_t2_matrices['weight_snapshots']

# Analyse the t0-t1 model
analyse_model(matrices=weight_snapshots, output_filepath=t1_t2_model_filepath)


# %% t2-t3 Parameter Sweep --------------------------------------------------
# Load the model's averaged matrices for seed
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_matrices = np.load(f'{t1_t2_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Get last timepoints from t0-t1 model and convert to tensors
simulated_t2 = t1_t2_matrices['weight_snapshots'][-1]
simulated_t2 = torch.tensor(simulated_t2, dtype=torch.float)
simulated_t2 = simulated_t2.unsqueeze(0)

simulated_t2_binary = t1_t2_matrices['adjacency_snapshots'][-1]
simulated_t2_binary = torch.tensor(simulated_t2_binary, dtype=torch.float)
simulated_t2_binary = simulated_t2_binary.unsqueeze(0)

# Set output
t2_t3_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t2_t3'

# Ensure the directory exists
if not os.path.exists(t2_t3_sweep_filepath):
    os.makedirs(t2_t3_sweep_filepath)

# Parameter sweep
t2_t3_results, t2_t3_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_weighted_network=simulated_t2,
    seed_binary_network=simulated_t2_binary,
    weighted_network=modelling_data[2]['weighted_network'],
    binary_network=modelling_data[2]['binary_network'],
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_nodes=modelling_data[2]['num_nodes'],
    num_connections=modelling_data[2]['num_connections'],
    output_filepath=t2_t3_sweep_filepath
    )

# Plot energy landscape
plot_energy_landscape(results=t2_t3_results, title='t2 → t3 Energy Landscape', output_filepath=t2_t3_sweep_filepath)


# %% t2-t3 Model --------------------------------------------------
# Load the model's averaged matrices for seed
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_matrices = np.load(f'{t1_t2_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Get last timepoints from t0-t1 model and convert to tensors
simulated_t2 = t1_t2_matrices['weight_snapshots'][-1]
simulated_t2 = torch.tensor(simulated_t2, dtype=torch.float)

simulated_t2_binary = t1_t2_matrices['adjacency_snapshots'][-1]
simulated_t2_binary = torch.tensor(simulated_t2_binary, dtype=torch.float)

# Load the optimal results into a DataFrame
t2_t3_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t2_t3'
t2_t3_results = pd.read_csv(f'{t2_t3_sweep_filepath}/optimal_parameters.csv')

t2_t3_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3'

# Ensure the directory exists
if not os.path.exists(t2_t3_model_filepath):
    os.makedirs(t2_t3_model_filepath)

# Create the model
t2_t3_model = model(
    optimal_results=t2_t3_results,
    num_simulations=num_simulations,
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_connections=modelling_data[2]['num_connections'],
    num_nodes=modelling_data[2]['num_nodes'],
    real_network=modelling_data[2]['weighted_network'],
    seed_binary_network=simulated_t2_binary.unsqueeze(0)[:,:1,:,:],
    seed_weighted_network=simulated_t2.unsqueeze(0)[:,:1,:,:],
    binary_network=modelling_data[2]['binary_network'],
    weighted_network=modelling_data[2]['weighted_network'],
    output_filepath=t2_t3_model_filepath
    )


# %% t2-t3 Assess Model --------------------------------------------------
# Load the model's averaged matrices for analysis
t2_t3_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3'
t2_t3_matrices = np.load(f'{t2_t3_model_filepath}/raw_model_outputs.npy', allow_pickle=True).item()

# Extract weight snapshots
weight_snapshots = t2_t3_matrices['weight_snapshots']

# Analyse the t0-t1 model
analyse_model(matrices=weight_snapshots, output_filepath=t2_t3_model_filepath)


# %%