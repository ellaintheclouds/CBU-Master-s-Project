import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

import pickle  # Save/load processed data
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
with open('D:/WIP/CBU Project Presentation/#3 output/preprocessed_data.pkl', 'rb') as f:
    processed_data = pickle.load(f)

# Function to create and save a 3D network plot
def create_3d_network(data_index, output_path):
    # Define adjM
    adj_matrix = processed_data[data_index]['adjM']

    # Create a thresholded binary graph (optional, to remove weak connections)
    threshold = 0.05  # Adjust based on your data
    adj_binary = (adj_matrix > threshold).astype(int)

    # Create a NetworkX graph
    G = nx.from_numpy_array(adj_binary)

    # 2D layout with adjusted spring constant for more spread
    pos_2d = nx.spring_layout(G, seed=42, dim=2, k=0.5)  # Increase `k` for more spread

    # Convert to 3D by scaling Z-coordinates
    pos_3d = {k: (v[0], v[1], np.random.uniform(-1, 1)) for k, v in pos_2d.items()}  # Spread Z-coordinates

    # Create a 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Remove background cube, grid, and axes
    ax.axis('off')  # Completely removes the axes

    # Draw nodes (balls) with smaller size and lighter color
    for node, (x, y, z) in pos_3d.items():
        ax.scatter(x, y, z, color="#A1D6E2", s=25, edgecolors="#253494", linewidth=0.8)

    # Draw edges (sticks) with more transparency and thinner lines
    for edge in G.edges():
        x_vals = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
        y_vals = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
        z_vals = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
        ax.plot(x_vals, y_vals, z_vals, color="#31688E", linewidth=1.0, alpha=0.2)

    # Save the figure in high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show(fig)

# Generate and save the plots for the specified indices
create_3d_network(2, 'D:/WIP/CBU Project Presentation/Presentation/network_t1.png')
create_3d_network(0, 'D:/WIP/CBU Project Presentation/Presentation/network_t2.png')
create_3d_network(6, 'D:/WIP/CBU Project Presentation/Presentation/network_t3.png')