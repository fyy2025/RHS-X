#!/bin/bash

#SBATCH --job-name=sim_pb
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=80:00:00
#SBATCH --array=1-5
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=span18@uw.edu

set -euo pipefail

# --- Define Simulation Parameters ---
EPS1=0.4
EPS2=0.7
PB_STEPS=20
PB_DELTA=0.05
FRONTIER_CAP=500

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}, PB_STEPS=${PB_STEPS}, PB_DELTA=${PB_DELTA}, FRONTIER_CAP=${FRONTIER_CAP}"

# Move to project root
cd $HOME/RHS-X/Code

# Create directories if they don't exist
mkdir -p output_pb log

# Activate venv
source venv/bin/activate

echo "Using python: $(which python)"
python --version

# Run script
python simulation_scenario1_pb.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --pb_steps $PB_STEPS \
    --pb_delta $PB_DELTA \
    --frontier_cap $FRONTIER_CAP

# Create one zip per array task output bundle
zip -r "results_pb_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip" output_pb/

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished and zipped."
