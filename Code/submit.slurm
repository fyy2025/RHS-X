#!/bin/bash

#SBATCH --job-name=simulation_scenario2
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH --array=1-5                      # Spawns 5 tasks indexed 1 to 5
#SBATCH --output=log/%x_%A_%a.out        # %A is the master Job ID, %a is the array task ID
#SBATCH --error=log/%x_%A_%a.err         # Keeps your 5 error logs separate
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=span18@uw.edu

set -euo pipefail

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"

# Move to project root
cd $HOME/RHS-X/Code

# Activate venv
source venv/bin/activate

echo "Using python: $(which python)"
python --version

# Run script and pass the array ID to the --epoch argument
python simulation_scenario2.py --epoch $SLURM_ARRAY_TASK_ID

echo "Finished at $(date)"