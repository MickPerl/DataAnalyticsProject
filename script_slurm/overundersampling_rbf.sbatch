#!/bin/bash
#SBATCH --job-name=overUndersampleRbf
#SBATCH --mail-type=END
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --time=500:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=err_overundersampling_rbf.txt
#SBATCH --output=overundersampling_rbf.txt
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_OverUnderSampling_rbf.py
