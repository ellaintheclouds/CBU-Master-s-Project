#!/bin/bash
#SBATCH --job-name=subset_elle_job_trial

module load conda
conda activate ella_organoid

cd "/imaging/astle/er05/Organoid project scripts"
python "Custom topological parameter sweep.py"