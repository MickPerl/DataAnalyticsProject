#!/bin/bash
#SBATCH --job-name=RBF_LDA
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --time=300:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --error=LDAerroroutput.txt
#SBATCH --output=LDAoutput.txt
#SBATCH --gres=gpu:1

source ../venv/bin/activate
which python

pip install --no-cache-dir -r ../requirements.txt

python -u script_RBF_LDA.py
