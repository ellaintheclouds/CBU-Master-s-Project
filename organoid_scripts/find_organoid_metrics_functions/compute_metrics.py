# %% Import Packages and Functions --------------------------------------------------
import bct
import gnm
from gnm import utils
import networkx as nx
from networkx.algorithms.community import louvain_communities
import numpy as np
import pandas as pd
import time
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew


# %% Define the Function --------------------------------------------------
def compute_metrics(file_name, species, day_number, adjM, dij, density_levels, human_metrics_df, chimpanzee_metrics_df):
    """Compute graph theory metrics at different thresholds."""
    
    # Create empty list
    metrics_list = []
    
    for density_level in density_levels:

        # Process ----------
        # Threshold the adjacency matrix
        adjM_thresholded = bct.threshold_proportional(adjM, density_level)
        print(f'  Processing {density_level * 100:.0f}% density level:', flush=True)
        adjM_thresholded = np.maximum(adjM_thresholded, adjM_thresholded.T) #ensure symmetry (undirected graph)

        # Compute and save statistics for each metric
        # Compute density
        num_connections = np.count_nonzero(adjM_thresholded) // 2
        num_nodes = adjM_thresholded.shape[0]
        density = num_connections / ((num_nodes * (num_nodes - 1)) / 2)

        # Compute graph metrics ----------
        # Betweenness centrality (weighted)
        start_time = time.time()
        betweenness_centrality = bct.betweenness_wei(1 / (adjM_thresholded + np.finfo(float).eps))
        end_time = time.time()
        print(f'    - betweenness centrality computed in {end_time - start_time:.1f} seconds')

        # Clustering coefficient (weighted)
        start_time = time.time()
        clustering_coefficient = gnm.utils.weighted_clustering_coefficients(torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0))
        end_time = time.time()
        print(f'    - clustering coefficient computed in {end_time - start_time:.1f} seconds')

        # Communicability (weighted)
        start_time = time.time()
        communicability = gnm.utils.communicability(torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0))
        end_time = time.time()
        print(f'    - communicability computed in {end_time - start_time:.1f} seconds')

        # Degree
        start_time = time.time()
        degree = np.sum(adjM_thresholded != 0, axis=0)
        end_time = time.time()
        print(f'    - degree computed in {end_time - start_time:.1f} seconds')

        # Edge length and total edge length (weighted)
        start_time = time.time()
        euclidean_dists = squareform(pdist(dij, metric='euclidean')) # compute pairwise Euclidean distances and convert to (n, n) symmetric matrix
        edge_length = euclidean_dists * adjM_thresholded # edge length scaled by connection strength
        total_edge_length = np.sum(edge_length) / (2 if np.allclose(adjM_thresholded, adjM_thresholded.T) else 1) # sum all the edge lengths in the network and divide by 2 if the adjacency matrix is symmetric (undirected graph)
        end_time = time.time()
        print(f'    - edge length computed in {end_time - start_time:.1f} seconds')

        # Global efficiency (weighted)
        start_time = time.time()
        global_efficiency = bct.efficiency_wei(adjM_thresholded, local=False)
        end_time = time.time()
        print(f'    - global efficiency computed in {end_time - start_time:.1f} seconds')

        # Local efficiency (weighted)
        start_time = time.time()
        local_efficiency = bct.efficiency_wei(adjM_thresholded, local=True)
        end_time = time.time()
        print(f'    - local efficiency computed in {end_time - start_time:.1f} seconds')

        # Homophily/Matching index (weighted)
        start_time = time.time()
        # Expand dimensions for broadcasting
        A = adjM_thresholded[:, np.newaxis, :] # row vector of connections from node i shape: (N, 1, N)
        B = adjM_thresholded[np.newaxis, :, :] #  row vector of connections from node j shape: (1, N, N)
        # For each node pair (i, j) compare how strongly both are connected to every other node k
        min_weights = np.minimum(A, B) # shared connection strength (if both are connected)
        max_weights = np.maximum(A, B) # total possible connection strength between i or j and k
        # Sum over all other nodes
        sum_min = np.sum(min_weights, axis=2) # sum of shared weights to mutual neighbors
        sum_max = np.sum(max_weights, axis=2) # sum of all weights to any neighbor of i or j
        # Compute the matching index
        with np.errstate(divide='ignore', invalid='ignore'):
            matching_index = np.true_divide(sum_min, sum_max)
            matching_index[sum_max == 0] = 0.0 # avoid division by zero
        # Zero out the diagonal (i == j)
        np.fill_diagonal(matching_index, 0.0) # remove self-comparison
        end_time = time.time()
        print(f'    - matching index computed in {end_time - start_time:.1f} seconds')

        # Modularity (weighted)
        start_time = time.time()
        adjM_thresholded_nx = nx.from_numpy_array(adjM_thresholded, create_using=nx.Graph) # convert to NetworkX graph
        communities = louvain_communities(adjM_thresholded_nx, weight='weight') # find communities of neurones that are more strongly connected to eachother than neurones outside group
        modularity_score = nx.community.modularity(adjM_thresholded_nx, communities, weight='weight') # calculate modularity score
        end_time = time.time()
        print(f'    - modularity computed in {end_time - start_time:.1f} seconds')

        # Small-worldness (weighted)
        adjM_thresholded_nx = nx.from_numpy_array(adjM_thresholded)
        if nx.is_connected(adjM_thresholded_nx):
            start_time = time.time()
            small_worldness_score = gnm.utils.weighted_small_worldness(torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0))
            end_time = time.time()
        else: 
            small_worldness_score = np.nan
        print(f'    - small-worldness computed in {end_time - start_time:.1f} seconds')

        # Strength (weighted)
        start_time = time.time()
        strength = gnm.utils.node_strengths(torch.tensor(adjM_thresholded, dtype=torch.float).unsqueeze(0))
        end_time = time.time()
        print(f'    - strength computed in {end_time - start_time:.1f} seconds')

        # Save ----------
        # Save graph metrics
        metrics = {
            'density_level': density_level,
            'adjM_thresholded': adjM_thresholded,
            'betweenness_centrality': betweenness_centrality,
            'clustering_coefficient': clustering_coefficient,
            'communicability': communicability,
            'degree': degree,
            'edge_length': edge_length,
            'global_efficiency': global_efficiency,
            'local_efficiency': local_efficiency,
            'matching_index': matching_index,
            'modularity_score': modularity_score,
            'small_worldness_score': small_worldness_score,
            'strength': strength,
        }
        metrics_list.append(metrics)

        # Save statistics
        metrics_stats = pd.DataFrame([{
            'file_name': file_name,
            'species': species,
            'day_number': day_number,
            'density_level': density_level,
            'num_nodes': num_nodes,
            'num_connections': num_connections,
            'density': density,
            'betweenness_centrality_mean': np.mean(betweenness_centrality),
            'betweenness_centrality_skew': skew(betweenness_centrality),
            'clustering_coefficient_mean': np.mean(clustering_coefficient.numpy()),
            'clustering_coefficient_skew': skew(clustering_coefficient.numpy()),
            'communicability_mean': np.mean(communicability.numpy()),
            'communicability_skew': skew(communicability.numpy()),
            'degree_mean': np.mean(degree),
            'degree_skew': skew(degree),
            'edge_length_mean': np.mean(edge_length),
            'edge_length_skew': skew(edge_length),
            'global_efficiency': global_efficiency,
            'local_efficiency_mean': np.mean(local_efficiency),
            'local_efficiency_skew': skew(local_efficiency),
            'matching_index_mean': np.mean(matching_index),
            'matching_index_skew': skew(matching_index),
            'modularity_score': modularity_score,
            'small_worldness_score': small_worldness_score,
            'strength_mean': np.mean(strength.numpy()),
            'strength_skew': skew(strength.numpy())
        }])

        if species == 'Chimpanzee':
            
            chimpanzee_metrics_df = pd.concat([chimpanzee_metrics_df, metrics_stats], ignore_index=True)
        elif species == 'Human':
            human_metrics_df = pd.concat([human_metrics_df, metrics_stats], ignore_index=True)

    return metrics_list, chimpanzee_metrics_df, human_metrics_df


# %%
