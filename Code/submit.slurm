#!/bin/bash

#SBATCH --job-name=rashomon_mcmc
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=40:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=span18@uw.edu

set -euo pipefail

echo "Running on $(hostname)"
echo "Started at $(date)"

# Move to project root
cd $HOME/RHS-X/Code

# Activate venv
source venv/bin/activate

echo "Using python: $(which python)"
python --version

# Run script
python run3.py

echo "Finished at $(date)"
