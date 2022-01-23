#!/bin/bash
#SBATCH --job-name=regres
#SBATCH --error=err_regr.txt
#SBATCH --output=regres.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_nn_regression.py