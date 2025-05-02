# %% Import Packages and Functions --------------------------------------------------
# Network Neuroscience
import gnm
from gnm import fitting, evaluation, GenerativeNetworkModel
from gnm.generative_rules import MatchingIndex
from gnm.weight_criteria.optimisation_criteria import Communicability

# Operations
import os
import pandas as pd
import torch


# %% Define the Function --------------------------------------------------
def weighted_parameter_sweep(
    optimal_binary_parameters,
    alpha_values,
    num_simulations,
    distance_matrix,
    num_connections,
    weighted_network,
    seed_weighted_network=None,
    seed_binary_network=None,
    output_filepath=None
):

    # Unpack optimal binary parameters
    eta = torch.tensor([optimal_binary_parameters[0]["eta"]])
    gamma = torch.tensor([optimal_binary_parameters[0]["gamma"]])

    # Set up the sweep parameters ----------
    weighted_params = fitting.WeightedSweepParameters(
        alpha=alpha_values,
        optimisation_criterion=[Communicability(omega=1.0)]
    )

    binary_params = fitting.BinarySweepParameters(
        eta=eta,
        gamma=gamma,
        lambdah=torch.tensor([0]),
        distance_relationship_type=["powerlaw"],
        preferential_relationship_type=["powerlaw"],
        heterochronicity_relationship_type=["powerlaw"],
        generative_rule=[MatchingIndex(divisor="mean")],
        num_iterations=[num_connections]
    )

    # Set up sweep configuration ----------
    sweep_config = fitting.SweepConfig(
        binary_sweep_parameters=binary_params,
        weighted_sweep_parameters=weighted_params,
        num_simulations=num_simulations,
        distance_matrix=distance_matrix,
        seed_adjacency_matrix=seed_binary_network,
        seed_weight_matrix=seed_weighted_network
    )

    # Define the evaluation criteria ----------
    criteria = [
        evaluation.WeightedNodeStrengthKS(),
        evaluation.WeightedBetweennessKS(),
        evaluation.WeightedClusteringKS()
    ]

    energy_fn = evaluation.MaxCriteria(criteria)
    key = str(energy_fn)

    # Perform the parameter sweep ----------
    experiments = fitting.perform_sweep(
        sweep_config=sweep_config,
        weighted_evaluations=[energy_fn],
        real_weighted_matrices=weighted_network,
        save_run_history=False,
        verbose=True
    )

    # Evaluate all parameter combinations ----------
    all_results = []
    optimal_result = None
    best_energy = float("inf")

    for exp in experiments:
        model = exp.model

        # Evaluate
        eval_results = fitting.perform_evaluations(
            model=model,
            weighted_evaluations=[energy_fn],
            real_weighted_matrices=weighted_network
        )

        # Extract parameters and mean energy
        alpha = exp.run_config.weighted_parameters.alpha.item()
        values = eval_results.weighted_evaluations[key]
        mean_energy = torch.mean(values).item()

        # Save results
        all_results.append({
            "eta": eta.item(),
            "gamma": gamma.item(),
            "alpha": alpha,
            "mean_energy": mean_energy
        })

        # Track optimal
        if mean_energy < best_energy:
            best_energy = mean_energy
            optimal_result = {
                "eta": eta.item(),
                "gamma": gamma.item(),
                "alpha": alpha,
                "energy": best_energy
            }

    # Save results ----------
    if output_filepath:
        os.makedirs(output_filepath, exist_ok=True)
        pd.DataFrame(all_results).to_csv(f"{output_filepath}/all_weighted_parameters.csv", index=False)
        pd.DataFrame([optimal_result]).to_csv(f"{output_filepath}/optimal_weighted_parameters.csv", index=False)

    return all_results, [optimal_result]
