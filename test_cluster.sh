
module load conda

conda activate ella_organoid

conda init bash

cd "/imaging/astle/er05/Organoid project scripts"

python "Custom topological parameter sweep.py"