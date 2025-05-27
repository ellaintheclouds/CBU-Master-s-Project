#!/bin/bash
#SBATCH --job-name=find_organoid_metrics
#SBATCH -o find_organoid_metrics_output_%j.txt

module load conda
conda activate ella_organoid

cd "/imaging/astle/er05/organoid_scripts"
python "find_organoid_metrics.py"