#!/bin/bash

#SBATCH --job-name=sim1_all
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

# --- Simulation Parameters ---
EPS1=0.4
EPS2=0.7
N_SSS_STEPS=200
PB_STEPS=20
PB_DELTA=0.05
FRONTIER_CAP=500

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}, N_SSS_STEPS=${N_SSS_STEPS}, PB_STEPS=${PB_STEPS}"

cd $HOME/RHS-X/Code

mkdir -p output_all log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python simulation1_all.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --n_sss_steps $N_SSS_STEPS \
    --pb_steps $PB_STEPS \
    --pb_delta $PB_DELTA \
    --frontier_cap $FRONTIER_CAP

zip -r "results_sim1_all_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip" output_all/

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished at $(date)."
