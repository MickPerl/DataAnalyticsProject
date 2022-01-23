#!/bin/bash
#SBATCH --job-name=new_ft
#SBATCH --error=err_new_ft.txt
#SBATCH --output=new_ft.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=michele.perlino@studio.unibo.it
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source ../venv/bin/activate

pip install --no-cache-dir -r ../requirements.txt

python3 -u script_nn_second.py