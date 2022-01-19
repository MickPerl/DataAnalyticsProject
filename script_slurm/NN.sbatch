#!/bin/bash
#SBATCH --job-name=NN
#SBATCH --mail-type=END
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=err_nn.txt
#SBATCH --output=nn.txt
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_nn.py  --nr_sets 6 --idx_set 1