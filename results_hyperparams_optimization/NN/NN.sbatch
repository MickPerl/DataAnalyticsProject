#!/bin/bash
#SBATCH --job-name=0out_of5
#SBATCH --error=0out_of5.txt
#SBATCH --output=0out_of5.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_nn.py  --nr_sets $1 --idx_set $2
