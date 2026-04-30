#!/bin/bash

#SBATCH --job-name=sim2
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

# --- Define Simulation Parameters ---
EPS1=0.4
EPS2=0.7

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}"

# Move to project root
cd $HOME/RHS-X/Code

# Create directories if they don't exist
mkdir -p output2 output_files2

# Activate venv
source venv/bin/activate

echo "Using python: $(which python)"
python --version

# Run script
python simulation_scenario2.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2

# 3. Create ONE master zip
# This grabs the folder and all its contents
zip -r "results_${SLURM_JOB_ID}.zip" output2/ output_files2/

echo "Job $SLURM_JOB_ID finished and zipped."
