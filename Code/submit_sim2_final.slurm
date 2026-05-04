#!/bin/bash

#SBATCH --job-name=sim2_final
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --array=1-10
#SBATCH --output=log/%x_%A_%a.out
#SBATCH --error=log/%x_%A_%a.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

# --- Simulation Parameters ---
EPS1=0.4
EPS2=0.7
PB_STEPS=300
PB_DELTA=0.05
FRONTIER_CAP=500
THETA_INIT=13.0
EPSILONS="0.16,0.17,0.18,0.19,0.20"

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}, PB_STEPS=${PB_STEPS}, PB_DELTA=${PB_DELTA}, FRONTIER_CAP=${FRONTIER_CAP}, THETA_INIT=${THETA_INIT}, EPSILONS=${EPSILONS}"

cd $HOME/RHS-X/Code

mkdir -p output2 log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python simulation_scenario2_final.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --pb_steps $PB_STEPS \
    --pb_delta $PB_DELTA \
    --frontier_cap $FRONTIER_CAP \
    --theta_init $THETA_INIT \
    --epsilons "$EPSILONS"

zip -r "results_sim2_final_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip" output2/

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished at $(date)."
