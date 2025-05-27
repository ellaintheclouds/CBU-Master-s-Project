# %% Import packages --------------------------------------------------
from glob import glob
import pandas as pd
import os
import pickle


# %% Import Custom Functions --------------------------------------------------
# Change directory to load in custom functions
os.chdir('/imaging/astle/er05/organoid_scripts')

# Load custom functions
from find_organoid_metrics_functions.process_data import process_data
from find_organoid_metrics_functions.sort_data import sort_data
from find_organoid_metrics_functions.compute_metrics import compute_metrics


# %% Things to Change ==================================================
# Define density levels
density_levels = [0.01, 0.05, 0.1, 0.2, 1]

# Should organoid metrics be re-computed if they already exist in the pickle file?
recompute_metrics = True


# %% Set Directories and Load Data ---------------------------------
# Define working directory
os.chdir('/imaging/astle')

# Set output directory
output_dir = 'er05/organoid_scripts/output'
os.makedirs(output_dir, exist_ok=True)

# Load data
matrix_files = (
    [file for file in glob('kr01/organoid/OrgNets/*.mat') 
     if ('C' in os.path.basename(file) or 'H' in os.path.basename(file)) and 'dt10' in os.path.basename(file)] + ########## change this to add in human data
    [file for file in glob('er05/H5 Analysis/Organoid Data/*.mat')]
)
if len(matrix_files) == 1:
    print(f'{len(matrix_files)} organoid file loaded: {matrix_files}', flush=True)
else:
    print(f'{len(matrix_files)} organoid files loaded: {matrix_files}', flush=True)


# Load existing processed data if available
pickle_dir = f'{output_dir}/processed_data.pkl'

if os.path.exists(pickle_dir):
    with open(pickle_dir, 'rb') as f:
        processed_data = pickle.load(f)
        print(f'Loaded existing processed data with {len(processed_data)} entries.', flush=True)
else:
    print('No existing processed data found.', flush=True)


# %% Find Organoid Metrics ---------------------------------------------------
# Create empty list
processed_data = []

# Initialise dataframes
chimpanzee_metrics_df = pd.DataFrame()
human_metrics_df = pd.DataFrame()

for idx, file_path in enumerate(matrix_files):
    
    # Process data
    file_name, adjM, dij = process_data(file_path=file_path)

    print(f"{file_name} shape match: {adjM.shape == dij.shape} ✅" if adjM.shape == dij.shape else f"{file_name} ❌ adjM: {adjM.shape}, dij: {dij.shape}") # check that adjM and dij are the same size

    # Sort data
    species, day_number, timepoint, cell_line = sort_data(file_name)
    print(f'Processing {file_name} ({idx + 1}/{len(matrix_files)}):', flush=True)

    # Check if this organoid's data is already processed
    existing_entry = next((entry for entry in processed_data if entry['file_name'] == file_name), None)

    # If the data is already processed and recompute_metrics is False, load the existing metrics
    if existing_entry and recompute_metrics is False:
        # Load existing metrics
        metrics_list = existing_entry['metrics']
        print(f'- Data for {file_name} already processed. Using loaded data.', flush=True)

   # If the data is not processed or recompute_metrics is True, compute the metrics
    else:
        # Compute metrics
        metrics_list, chimpanzee_metrics_df, human_metrics_df = compute_metrics(
            file_name=file_name,
            species=species,
            day_number=day_number,
            adjM=adjM,
            dij=dij,
            density_levels=density_levels,
            human_metrics_df=human_metrics_df,
            chimpanzee_metrics_df=chimpanzee_metrics_df
        )
        print('- Metrics computed', flush=True)
        
        # Append new data to processed_data
        processed_data.append({
            'file_name': file_name,
            'species': species,
            'timepoint': timepoint,
            'cell_line': cell_line,
            'adjM': adjM,
            'dij': dij,
            'metrics': metrics_list
        })
        
        # Save ----------
        # Ensure output directories exist
        os.makedirs(f'{output_dir}/chimpanzee', exist_ok=True)
        os.makedirs(f'{output_dir}/human', exist_ok=True)

        # Save after each new computation
        chimpanzee_metrics_df.to_csv(f'{output_dir}/chimpanzee/chimpanzee_metrics_summary.csv', index=False)
        human_metrics_df.to_csv(f'{output_dir}/human/human_metrics_summary.csv', index=False)
        with open(pickle_dir, 'wb') as f:
            pickle.dump(processed_data, f)
            print('- Data saved', flush=True)

print('Processing complete.')


# %%
