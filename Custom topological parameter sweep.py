# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import bct
import gnm
from gnm import fitting, evaluation, BinaryGenerativeParameters, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

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
import pandas as pd
import seaborn as sns


# %% Define Functions --------------------------------------------------
def parameter_sweep(eta_values, gamma_values, lambdah_value, num_simulations, seed_binary_network, binary_network, distance_matrix, num_nodes, num_connections, output_filepath=None):
    # Carry out parameter sweep ----------
    # Define binary sweep parameters
    binary_sweep_parameters = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([lambdah_value]),
        distance_relationship_type=['powerlaw'],
        preferential_relationship_type=['powerlaw'],
        heterochronicity_relationship_type=['powerlaw'],
        generative_rule=[MatchingIndex(divisor='mean')],  # Use Matching Index rule
        num_iterations=[num_connections],  # Match the number of connections in real network
    )

    if seed_binary_network is None:
        seed_adjacency_matrix = None
    else:
        seed_adjacency_matrix = [seed_binary_network]
    
    # Configuration for the parameter sweep
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_sweep_parameters,
        num_simulations=num_simulations,
        distance_matrix=[distance_matrix],
        seed_adjacency_matrix=seed_adjacency_matrix
    )

    # List of criteria for evaluating binary network properties
    criteria = [
        evaluation.DegreeKS(), # Compare degree distributions
        evaluation.ClusteringKS(), # Compare clustering coefficients
        evaluation.BetweennessKS(), # Compare betweenness centrality
        evaluation.EdgeLengthKS(distance_matrix) # Compare edge length distributions
    ]

    # Use the maximum KS statistic across all evaluations to measure network similarity
    energy = evaluation.MaxCriteria(criteria)
    binary_evaluations = [energy]

    # Run the Parameter Sweep
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=binary_evaluations,
        real_binary_matrices=binary_network,  # Real network to compare synthetic networks against
        save_run_history=False, # If True, saves the model in the experiment - set to False to save on memory
        verbose=True, # If True, displays a progress bar for the sweep
    )

    # Assess Parameter Combinations ----------
    # Initialize an empty list to store the results
    all_results = []

    # Iterate through each experiment
    for experiment in experiments:
        model = experiment.model
        eval_results = fitting.perform_evaluations(
            model=model,  # Model to evaluate
            binary_evaluations=binary_evaluations,
            real_binary_matrices=binary_network
        )
        
        # Extract eta and gamma
        eta = experiment.run_config.binary_parameters.eta.item()
        gamma = experiment.run_config.binary_parameters.gamma.item()
        
        # Extract the values for the key
        key = 'MaxCriteria(DegreeKS, ClusteringKS, BetweennessKS, EdgeLengthKS)'
        values = eval_results.binary_evaluations[key]
        
        # Compute the mean of the values
        mean_value = torch.mean(values).item()
        
        # Append the results to the list
        all_results.append({
            'eta': eta,
            'gamma': gamma,
            'mean_energy': mean_value,
        })

    # Find the optimal parameter combination
    optimal_experiments, optimal_energies = fitting.optimise_evaluation(
        experiments=experiments,
        criterion=energy,
    )

    # Convert optimal_results to DataFrame
    optimal_results = []
    for exp, energy in zip(optimal_experiments, optimal_energies):
        optimal_results.append({
            'eta': exp.run_config.binary_parameters.eta.item(),
            'gamma': exp.run_config.binary_parameters.gamma.item(),
            'energy': energy.item()
        })

    # Save results to CSV files if file paths are provided
    if output_filepath:
        pd.DataFrame(all_results).to_csv(f'{output_filepath}/all_parameters.csv', index=False)
        pd.DataFrame(optimal_results).to_csv(f'{output_filepath}/optimal_parameters.csv', index=False)

    # Return
    return all_results, optimal_results

def plot_energy_landscape(results, title='Energy Landscape', output_filepath=None):
    """
    Plots the energy landscape from the parameter sweep results.

    Parameters:
        results (pd.DataFrame): DataFrame containing the energy landscape data.
        title (str): Title of the plot.
        output_filepath (str): Filepath to save the plot. If None, the plot is shown.
    """
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
    ax.set_xticklabels([f"{energy_grid.columns[0]:.2f}", f"{energy_grid.columns[-1]:.2f}"], fontsize=10)
    ax.set_yticks([0.5, len(energy_grid.index) - 0.5])  # Tick positions at the edges
    ax.set_yticklabels([f"{energy_grid.index[0]:.2f}", f"{energy_grid.index[-1]:.2f}"], fontsize=10)

    # Adjust axis limits to ensure ticks are flush with the edges
    ax.set_xlim(0, len(energy_grid.columns))  # Align x-axis ticks with edges
    ax.set_ylim(len(energy_grid.index), 0)  # Align y-axis ticks with edges (reversed)

    # Save or show
    if output_filepath:
        plt.savefig(f'{output_filepath}/energy_landscape.png', bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def model(optimal_results, lambdah_value, num_simulations, seed_binary_network, binary_network, distance_matrix, num_nodes, num_connections, output_filepath=None):
    # Use the optimal parameters from the parameter sweep
    binary_parameters = gnm.BinaryGenerativeParameters(
        eta=float(optimal_results['eta'].iloc[0]),
        gamma=float(optimal_results['gamma'].iloc[0]),
        lambdah=lambdah_value,
        distance_relationship_type='powerlaw',
        preferential_relationship_type='powerlaw',
        heterochronicity_relationship_type='powerlaw',
        generative_rule=MatchingIndex(divisor='mean'),
        num_iterations=num_connections,
        prob_offset=1e-06,
        binary_updates_per_iteration=1
        )

    if seed_binary_network is None:
        seed_adjacency_matrix = None
    else:
        seed_adjacency_matrix = seed_binary_network

    # Define the model
    model = gnm.GenerativeNetworkModel(
            binary_parameters=binary_parameters,
            num_simulations=num_simulations,
            num_nodes=num_nodes,
            distance_matrix=distance_matrix,
            seed_adjacency_matrix=seed_adjacency_matrix
            )

    # Run the model
    run_model = model.run_model()

    # Averaging across runs ----------
    # Initialize a list to store the averaged matrices
    averaged_matrices = []

    # Iterate through each element in t0_t1_model[0]
    for element in run_model[1]:
        # Convert the element to a NumPy array if it's a tensor
        matrices = element.numpy()  # Assuming each element contains multiple matrices

        # Compute the element-wise mean across all matrices
        avg_matrix = matrices.mean(axis=0)  # Average along the first axis

        # Append the averaged matrix to the list
        averaged_matrices.append(avg_matrix)

    # Save the averaged matrices as a NumPy array
    averaged_matrices = np.array(averaged_matrices)

    # Save the averaged matrices
    if output_filepath:
        np.save(f'{output_filepath}/averaged_matrices.npy', averaged_matrices)

    # Save the model to a pickle file
    with open(f'{output_filepath}/model.pkl', 'wb') as f:
        pickle.dump(model, f)

def analyse_model(averaged_matrices, output_filepath=None):
    """
    Analyses a model by visualizing averaged matrices, computing graph metrics, 
    and saving distributions and a correlation heatmap to the specified filepath.

    Parameters:
        averaged_matrices (np.ndarray): Averaged matrices from the model.
        output_filepath (str): Filepath for saving the plots.
    """
    # Ensure the output directory exists
    os.makedirs(output_filepath, exist_ok=True)

    # Visualize all averaged matrices
    num_matrices = len(averaged_matrices)
    cols = 5  # Number of columns for the grid
    rows = (num_matrices + cols - 1) // cols  # Calculate rows for the grid

    # Create the figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(25, 5 * rows))  # Widen the figure for square plots
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_matrices:
            sns.heatmap(
                averaged_matrices[i],
                cmap='viridis',
                cbar=(i == 0),  # Only show the colorbar for the first plot
                ax=ax
            )
            ax.set_title(f'Connection {i}', fontsize=12)
            if i == 0:
                # Customize the first plot with axis labels and colorbar
                num_labels = 4
                ax.set_xticks(np.linspace(0, averaged_matrices[i].shape[1] - 1, num_labels))
                ax.set_yticks(np.linspace(0, averaged_matrices[i].shape[0] - 1, num_labels))
                ax.set_xticklabels(np.linspace(0, averaged_matrices[i].shape[1] - 1, num_labels, dtype=int), fontsize=10)
                ax.set_yticklabels(np.linspace(0, averaged_matrices[i].shape[0] - 1, num_labels, dtype=int), fontsize=10)
                colorbar = ax.collections[0].colorbar
                colorbar.set_ticks([averaged_matrices[i].min(), averaged_matrices[i].max()])
                colorbar.set_ticklabels([f'{averaged_matrices[i].min():.2f}', f'{averaged_matrices[i].max():.2f}'])
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.axis('off')

    # Save the averaged matrices plot
    fig.tight_layout(pad=3.0)
    plt.savefig(f'{output_filepath}/averaged_matrices.png', dpi=300)
    plt.close(fig)

    # Analyse the final simulated timepoint
    sim_adjm = averaged_matrices[-1]

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

# Initialise lists
slice_data = []
modelling_data = []  # Keep this as a list

# Load organoid slice data
with open(pickle_dir, 'rb') as f:
    slice_data = pickle.load(f)

# Define slices for each timepoint
timepoint_slice_data = ['C_d96_s2_dt10', 'C_d153_s7_dt10', 'C_d184_s8_dt10']

# Iterate through all slices
for slice in slice_data:
    # Process only the specified file names
    if slice['file_name'] in timepoint_slice_data:

        # Extract the adjacency matrix and distance matrix
        adjM = slice['adjM']
        dij = slice['dij']
        
        # Threshold the adjacency matrix to retain the top 5% strongest connections
        adjM_thresholded = bct.threshold_proportional(adjM, 0.05)
        
        # Create (50, 50) subsets for testing
        adjM_thresholded = adjM_thresholded[:50, :50]  ########## Remove when testing larger network
        dij = dij[:50, :50]  ########## Remove when testing larger network

        # Get the number of nodes and connections
        num_nodes = adjM_thresholded.shape[0]
        num_connections = np.count_nonzero(adjM_thresholded) // 2

        # Binarise the adjacency matrix
        adjM_thresholded = (adjM_thresholded > 0).astype(int) ########## Remove when changing to weighted

        # Convert matrices to PyTorch tensors
        binary_network = torch.tensor(adjM_thresholded, dtype=torch.float)
        binary_network = binary_network.unsqueeze(0)
        distance_matrix = torch.tensor(dij, dtype=torch.float)
        
        # Add the data to the modelling_data list
        modelling_data.append({
            'file_name': slice['file_name'],
            'adjM_thresholded': adjM_thresholded,  # Change to adjM_thresholded when testing larger network
            'binary_network': binary_network,
            'dij': dij,  # Change to dij when testing larger network
            'distance_matrix': distance_matrix,
            'num_nodes': num_nodes,
            'num_connections': num_connections
        })

# %% Set Parameterspace to Explore --------------------------------------------------
# Define parameters
eta_values = torch.linspace(-5, 1.7, 50) ########## increase when testing large parameterspace
gamma_values = torch.linspace(-5, 1.7, 50) ########## increase when testing large parameterspace
lambdah_value = 2.0  # Fixed lambda

# Number of networks to generate per parameter combination
num_simulations = 4 ########## increase when testing large parameterspace

# %% t0-t1 Parameter Sweep --------------------------------------------------
# Set output
t0_t1_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t0_t1'

# Parameter sweep
t0_t1_results, t0_t1_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=None,
    binary_network=modelling_data[0]['binary_network'],
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_nodes=modelling_data[0]['num_nodes'],
    num_connections=modelling_data[0]['num_connections'],
    output_filepath=t0_t1_sweep_filepath
    )

# Plot energy landscape
plot_energy_landscape(results=t0_t1_results, title='t0 → t1 Energy Landscape', output_filepath=t0_t1_sweep_filepath)


# %% t0-t1 Model --------------------------------------------------
# Load the optimal results into a DataFrame
t0_t1_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t0_t1'
t0_t1_results = pd.read_csv(f'{t0_t1_sweep_filepath}/optimal_parameters.csv')

t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'

# Create the model
t0_t1_model = model(
    optimal_results=t0_t1_results,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=None,
    binary_network=modelling_data[0]['binary_network'],
    distance_matrix=modelling_data[0]['distance_matrix'],
    num_nodes=modelling_data[0]['num_nodes'],
    num_connections=modelling_data[0]['num_connections'],
    output_filepath=t0_t1_model_filepath)


# %% t0-t1 Assess Model --------------------------------------------------
# Load the model's averaged matrices
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_averaged_matrices = np.load(f'{t0_t1_model_filepath}/averaged_matrices.npy')

# Analyse the t0-t1 model
analyse_model(averaged_matrices=t0_t1_averaged_matrices, output_filepath=t0_t1_model_filepath)


# %% t1-t2 Parameter Sweep --------------------------------------------------
# Load the model's averaged matrices
t0_t1_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t0_t1'
t0_t1_averaged_matrices = np.load(f'{t0_t1_model_filepath}/averaged_matrices.npy')

# Get last timepoint from t0-t1 model
simulated_t1 = t0_t1_averaged_matrices[-1]

# Threshold the adjacency matrix to retain the top 5% strongest connections
simulated_t1 = bct.threshold_proportional(simulated_t1, 0.05)

# Binarise
simulated_t1 = (simulated_t1 > 0).astype(int)  ########## Remove when changing to weighted

# Convert to PyTorch tensor
simulated_t1 = torch.tensor(simulated_t1, dtype=torch.float)

# Set output
t1_t2_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t1_t2'

# Parameter sweep
t1_t2_results, t1_t2_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=simulated_t1,
    binary_network=modelling_data[1]['binary_network'],
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_nodes=modelling_data[1]['num_nodes'],
    num_connections=modelling_data[1]['num_connections'],
    output_filepath=t1_t2_sweep_filepath
    )

# Plot energy landscape
plot_energy_landscape(results=t1_t2_results, title='t1 → t2 Energy Landscape', output_filepath=t1_t2_sweep_filepath)


# %% t1-t2 Model --------------------------------------------------
# Load the optimal results into a DataFrame
t1_t2_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t1_t2'
t1_t2_results = pd.read_csv(f'{t1_t2_sweep_filepath}/optimal_parameters.csv')

t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'

# Create the model
t1_t2_model = model(
    optimal_results=t1_t2_results,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=simulated_t1,
    binary_network=modelling_data[1]['binary_network'],
    distance_matrix=modelling_data[1]['distance_matrix'],
    num_nodes=modelling_data[1]['num_nodes'],
    num_connections=modelling_data[1]['num_connections'],
    output_filepath=t1_t2_model_filepath
    )


# %% t1-t2 Assess Model --------------------------------------------------
# Load the model's averaged matrices
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_averaged_matrices = np.load(f'{t1_t2_model_filepath}/averaged_matrices.npy')

# Analyse the t0-t1 model
analyse_model(averaged_matrices=t1_t2_averaged_matrices, output_filepath=t1_t2_model_filepath)


# %% t2-t3 Parameter Sweep --------------------------------------------------
# Load the model's averaged matrices
t1_t2_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t1_t2'
t1_t2_averaged_matrices = np.load(f'{t1_t2_model_filepath}/averaged_matrices.npy')

# Get last timepoint from t0-t1 model
simulated_t2 = t1_t2_averaged_matrices[-1]

# Threshold the adjacency matrix to retain the top 5% strongest connections
simulated_t2 = bct.threshold_proportional(simulated_t2, 0.05)

# Binarise
simulated_t2 = (simulated_t2 > 0).astype(int)  ########## Remove when changing to weighted

# Convert to PyTorch tensor
simulated_t2 = torch.tensor(simulated_t2, dtype=torch.float)

# Set output
t2_t3_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t2_t3'

# Parameter sweep
t2_t3_results, t2_t3_optimal_results = parameter_sweep(
    eta_values=eta_values,
    gamma_values=gamma_values,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=simulated_t2,
    binary_network=modelling_data[2]['binary_network'],
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_nodes=modelling_data[2]['num_nodes'],
    num_connections=modelling_data[2]['num_connections'],
    output_filepath=t2_t3_sweep_filepath
    )

# Plot energy landscape
plot_energy_landscape(results=t2_t3_results, title='t2 → t3 Energy Landscape', output_filepath=t2_t3_sweep_filepath)


# %% t2-t3 Model --------------------------------------------------
# Load the optimal results into a DataFrame
t2_t3_sweep_filepath='/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Parameter sweeps/t2_t3'
t2_t3_results = pd.read_csv(f'{t2_t3_sweep_filepath}/optimal_parameters.csv')

t2_t3_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3'

# Create the model
t2_t3_model = model(
    optimal_results=t2_t3_results,
    lambdah_value=lambdah_value,
    num_simulations=num_simulations,
    seed_binary_network=simulated_t2,
    binary_network=modelling_data[2]['binary_network'],
    distance_matrix=modelling_data[2]['distance_matrix'],
    num_nodes=modelling_data[2]['num_nodes'],
    num_connections=modelling_data[2]['num_connections'],
    output_filepath=t2_t3_model_filepath
    )


# %% t2-t3 Assess Model --------------------------------------------------
# Load the model's averaged matrices
t2_t3_model_filepath = '/imaging/astle/er05/Organoid project scripts/Output/Chimpanzee/Models/t2_t3'
t2_t3_averaged_matrices = np.load(f'{t2_t3_model_filepath}/averaged_matrices.npy')

# Analyse the t0-t1 model
analyse_model(averaged_matrices=t2_t3_averaged_matrices, output_filepath=t2_t3_model_filepath)


# %%