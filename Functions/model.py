# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import gnm
from gnm import evaluation
from gnm.generative_rules import MatchingIndex
from gnm.weight_criteria import Communicability

# Operations
import numpy as np
import os
import torch


# %% Define the Function --------------------------------------------------
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


    # Set up sweep parameters ----------
    # Weighted parameters
    weighted_parameters = gnm.WeightedGenerativeParameters(
        alpha=float(optimal_results['alpha'].iloc[0]),
        optimisation_criterion=Communicability(omega=1.0)
    )

    # Binary parameters
    binary_parameters = gnm.BinaryGenerativeParameters(
        eta=float(optimal_results['eta'].iloc[0]),
        gamma=float(optimal_results['gamma'].iloc[0]),
        lambdah=0,
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
    seed_binary_network = seed_binary_network.repeat(num_simulations, 1, 1) if seed_binary_network is not None else None
    seed_weighted_network = seed_weighted_network.repeat(num_simulations, 1, 1) if seed_weighted_network is not None else None

    # Model ----------
    # Define the model
    model = gnm.GenerativeNetworkModel(
        binary_parameters=binary_parameters,
        num_simulations=num_simulations,
        num_nodes=num_nodes,
        seed_adjacency_matrix=seed_binary_network,
        distance_matrix=distance_matrix,
        verbose=True,
        weighted_parameters=weighted_parameters,
        seed_weight_matrix=seed_weighted_network
    )

    # Run the model
    run_model = model.run_model()

    # Unpack the run_model output ----------
    adjacency_snapshots = run_model[1]
    weight_snapshots = run_model[2]

    # Extract the final timepoint to compare the real network against
    synthetic_network = weight_snapshots[-1]

    # Choose evaluation criteria ----------
    criteria = [
        evaluation.WeightedNodeStrengthKS(),
        evaluation.WeightedClusteringKS(),
        evaluation.WeightedBetweennessKS()
    ]
    
    # Combine into a single evaluation set
    eval_set = evaluation.MaxCriteria(criteria)

    # Compute energy (KS values)
    ks_values = eval_set(synthetic_network, real_network).squeeze(1)

    # Find simulation closest to the 5th percentile KS
    percentile_5th = torch.quantile(ks_values, 0.05)
    best_sim_idx = torch.argmin(torch.abs(ks_values - percentile_5th)).item()

    # Save raw matrices from the model ----------
    # Convert the best snapshot to numpy

    adjacency_numpy = adjacency_snapshots[:, best_sim_idx, :, :].numpy()
    weight_numpy = weight_snapshots[:, best_sim_idx, :, :].numpy()

    # Create dictionary of raw outputs
    raw_output = {
        'adjacency_snapshots': adjacency_numpy,
        'weight_snapshots': weight_numpy
    }

    # Save best simulation
    if output_filepath:
        os.makedirs(output_filepath, exist_ok=True)
        np.save(f'{output_filepath}/raw_model_outputs.npy', raw_output)

    return raw_output