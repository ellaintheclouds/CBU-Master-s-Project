#!/bin/bash
#SBATCH --job-name=s_elle_org_metrics

module load conda
conda activate ella_organoid

cd "/imaging/astle/er05/Organoid project scripts"
python "Organoid metrics.py"