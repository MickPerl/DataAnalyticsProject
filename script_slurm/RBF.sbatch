#!/bin/bash
#SBATCH --job-name=RBF
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --time=500:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=erroroutput.txt
#SBATCH --output=output.txt
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_RBF.py
