# %% Import Packages and Functions --------------------------------------------------
# Operations
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


# %% Define the Function --------------------------------------------------
def plot_energy_landscape(
    binary_results=None,
    weighted_results=None,
    binary_title=None,
    weighted_title=None,
    output_filepath=None
):

    # Binary Energy Landscape Plot ----------
    if binary_results:
        # Convert binary results to DataFrame
        binary_df = pd.DataFrame(binary_results)

        # Pivot the data to a matrix form (rows: gamma, columns: eta)
        binary_energy_grid = binary_df.pivot(
            index='gamma',
            columns='eta',
            values='mean_energy'
        )

        # Reverse the gamma axis (index)
        binary_energy_grid = binary_energy_grid.sort_index(ascending=False)

        # Create the binary energy landscape plot
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            binary_energy_grid,
            cmap='viridis_r',
            annot=False,
            cbar_kws={'label': 'Mean Energy'},
            linewidths=0,
            vmin=0,
            vmax=1
        )

        # Labeling for binary plot
        ax.set_title('Binary Energy Landscape', fontsize=14)
        ax.set_xlabel(r'$\eta$', fontsize=12)  # Use Greek letter for eta
        ax.set_ylabel(r'$\gamma$', fontsize=12)  # Use Greek letter for gamma

        # Save or show binary plot
        if output_filepath:
            plt.savefig(f'{output_filepath}/binary_energy_landscape.png', bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    # Alpha Weighted Energy Plot -------------------------------------------
    if weighted_results:
        # Convert weighted results to DataFrame
        weighted_df = pd.DataFrame(weighted_results)

        # Extract alpha and mean energy values
        alpha_values = weighted_df['alpha']
        mean_energies = weighted_df['mean_energy']

        # Create the alpha-weighted energy plot
        plt.figure(figsize=(6, 4))  # Smaller plot size
        plt.plot(alpha_values, mean_energies, marker='o', linestyle='-', color='purple')  # Viridis-inspired color

        # Labeling for weighted plot
        plt.title('Alpha Weighted Energy Plot', fontsize=16)  # Bigger text
        plt.xlabel(r'$\alpha$', fontsize=14)  # Use Greek letter for alpha
        plt.ylabel('Mean Energy', fontsize=14)

        # Set x-axis and y-axis ticks to only 4 labels
        x_ticks = np.linspace(alpha_values.min(), alpha_values.max(), 4)  # 4 evenly spaced ticks
        y_ticks = np.linspace(mean_energies.min(), mean_energies.max(), 4)  # 4 evenly spaced ticks
        plt.xticks(x_ticks, [f'{tick:.2f}' for tick in x_ticks], fontsize=12)  # Format x-tick labels
        plt.yticks(y_ticks, [f'{tick:.2f}' for tick in y_ticks], fontsize=12)  # Format y-tick labels

        # Remove grid and top/right borders
        sns.despine()

        # Save or show weighted plot
        if output_filepath:
            plt.savefig(f'{output_filepath}/weighted_energy_landscape.png', bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()