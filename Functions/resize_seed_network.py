def resize_seed_network(seed_tensor, target_num_nodes, seed=None, verbose=True):
    """
    Resize a square seed tensor to match target_num_nodes.
    Pads with zero-connected nodes (in random positions) or removes nodes.
    
    Args:
        seed_tensor (torch.Tensor): shape (1, N, N), square adjacency matrix.
        target_num_nodes (int): number of desired nodes.
        seed (int): optional, for reproducibility.
        verbose (bool): print details if True.
    
    Returns:
        torch.Tensor: shape (1, target_num_nodes, target_num_nodes)
    """
    current_num_nodes = seed_tensor.shape[-1]

    # If no resizing needed
    if current_num_nodes == target_num_nodes:

        if verbose:
            print(f"No resizing needed: current nodes = {current_num_nodes}, target = {target_num_nodes}")
        
        return seed_tensor

    # If padding needed
    if current_num_nodes < target_num_nodes:

        if seed is not None:
            np.random.seed(seed)
        
        # Create a blank target-sized matrix
        new_tensor = torch.zeros(1, target_num_nodes, target_num_nodes, dtype=seed_tensor.dtype)
        
        # Select random positions where the old nodes will be inserted
        all_indices = np.arange(target_num_nodes)
        old_node_indices = np.sort(np.random.choice(all_indices, size=current_num_nodes, replace=False))

        # Fill in the original connectivity structure at these positions
        for i, old_i in enumerate(old_node_indices):

            for j, old_j in enumerate(old_node_indices):
                new_tensor[0, old_i, old_j] = seed_tensor[0, i, j]

        if verbose:
            print(f"Resized by adding {target_num_nodes - current_num_nodes} nodes (random insertion)")
            
        return new_tensor

    # If reduction needed
    else:
        if seed is not None:
            np.random.seed(seed)

        keep_indices = np.sort(np.random.choice(current_num_nodes, target_num_nodes, replace=False))
        reduced_tensor = seed_tensor[:, keep_indices][:, :, keep_indices]

        if verbose:
            print(f"Resized by removing {current_num_nodes - target_num_nodes} nodes")

        return reduced_tensor
