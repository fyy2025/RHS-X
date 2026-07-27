#!/bin/bash

#SBATCH --job-name=sim2_combined
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --array=1-10
#SBATCH --output=/mmfs1/gscratch/escience/span18/output/RHS-X/log/%x_%A_%a.out
#SBATCH --error=/mmfs1/gscratch/escience/span18/output/RHS-X/log/%x_%A_%a.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

# --- Simulation Parameters ---
EPS1=0.4
EPS2=0.7
N_LADDER=4    # -> 5 rungs (a beta=0 rung is prepended in the script)
EPS_GRID="0.002,0.003,0.004,0.006,0.009"   # RELATIVE thresholds: theta = Q_min*(1+eps) per replication.
                                     # Max |RPS| over 10 reps: 16, 37, 72, 176, 455 (all <500).
                                     # Both conditions hold for RPS coverage 0.32-0.59.

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}, N_LADDER=${N_LADDER}, EPS_GRID=${EPS_GRID}"

cd $HOME/RHS-X/Code

mkdir -p /mmfs1/gscratch/escience/span18/output/RHS-X/output2_combined /mmfs1/gscratch/escience/span18/output/RHS-X/log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python simulation_scenario2_combined.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --n_ladder $N_LADDER \
    --eps "$EPS_GRID"

zip -r "/mmfs1/gscratch/escience/span18/output/RHS-X/results_sim2_combined_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip" /mmfs1/gscratch/escience/span18/output/RHS-X/output2_combined/

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished at $(date)."
