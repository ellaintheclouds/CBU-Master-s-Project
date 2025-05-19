#!/bin/bash
#SBATCH -A ASTLE-SL3-CPU
#SBATCH -p ampere
--gres=gpu:32
#SBATCH --job-name=create_env
#SBATCH --output=create_env.out

# Load conda
module load conda
source activate

# Create environment from file
conda env create --name ella_hpc_env --file ella_organoid_env.yml
