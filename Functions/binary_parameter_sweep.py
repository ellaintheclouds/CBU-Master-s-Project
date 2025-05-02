# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import gnm
from gnm import fitting, evaluation, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex

# Operations
import os
import pandas as pd
import torch


# %% Define the Function --------------------------------------------------
# Set up the function
def binary_parameter_sweep(
    eta_values,
    gamma_values,
    num_simulations,
    distance_matrix,
    num_connections,
    binary_network,
    seed_binary_network=None,
    output_filepath=None
):

    # Set up the parametr sweep ----------
    # Define the binary sweep parameters
    binary_params = fitting.BinarySweepParameters(
        eta=eta_values,
        gamma=gamma_values,
        lambdah=torch.tensor([0]),
        distance_relationship_type=['powerlaw'],
        preferential_relationship_type=['powerlaw'],
        heterochronicity_relationship_type=['powerlaw'],
        generative_rule=[MatchingIndex(divisor='mean')],
        num_iterations=[num_connections]
    )

    # Create the sweep configuration
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_params,
        num_simulations=num_simulations,
        distance_matrix=distance_matrix,
        seed_adjacency_matrix=seed_binary_network
    )

    # Define the evaluation criteria
    criteria = [
        evaluation.DegreeKS(),
        evaluation.ClusteringKS(),
        evaluation.BetweennessKS(),
        evaluation.EdgeLengthKS(distance_matrix.squeeze(0))
    ]
    
    # Use the maximum KS statistic across all evaluations to measure network similarity
    energy_fn = evaluation.MaxCriteria(criteria)
    key = str(energy_fn)

    # Perform the parameter sweep ----------
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        binary_evaluations=[energy_fn],
        real_binary_matrices=binary_network,
        save_run_history=False,
        verbose=True
    )

    # Evaluate all parameter combinations ----------
    # Initialise list to store results
    all_results = []
    optimal_result = None
    best_energy = float('inf')

    # Go through each 'experiment' from the parameter sweep
    for experiment in experiments:

        # Extract the model from the experiment
        model = experiment.model

        # Perform evaluation
        eval_results = fitting.perform_evaluations(
            model=model,
            binary_evaluations=[energy_fn],
            real_binary_matrices=binary_network
        )

        # Extract the parameters and mean energy
        eta = experiment.run_config.binary_parameters.eta.item()
        gamma = experiment.run_config.binary_parameters.gamma.item()
        values = eval_results.binary_evaluations[key]
        mean_energy = torch.mean(values).item()

        # Save the parameters and results
        all_results.append({
            'eta': eta,
            'gamma': gamma,
            'mean_energy': mean_energy
            })

        # Track optimal
        if mean_energy < best_energy:
            best_energy = mean_energy
            optimal_result = {
                'eta': eta,
                'gamma': gamma,
                'energy': best_energy
            }

    # Save results ----------
    if output_filepath:
        os.makedirs(output_filepath, exist_ok=True)
        pd.DataFrame(all_results).to_csv(f'{output_filepath}/all_binary_parameters.csv', index=False)
        pd.DataFrame([optimal_result]).to_csv(f'{output_filepath}/optimal_binary_parameters.csv', index=False)

    return all_results, [optimal_result]