#!/bin/bash

#SBATCH --job-name=sim1_combined
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
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
N_LADDER=4
THETAS="1.02,1.03,1.04,1.05,1.06"   # g-prior theta_RA thresholds (scenario 1 sits higher than s2)

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Executing Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameters: EPS1=${EPS1}, EPS2=${EPS2}, N_LADDER=${N_LADDER}, THETAS=${THETAS}"

cd $HOME/RHS-X/Code

mkdir -p /mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined /mmfs1/gscratch/escience/span18/output/RHS-X/log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python simulation_scenario1_combined.py \
    --epoch $SLURM_ARRAY_TASK_ID \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --n_ladder $N_LADDER \
    --thetas "$THETAS"

zip -r "/mmfs1/gscratch/escience/span18/output/RHS-X/results_sim1_combined_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.zip" /mmfs1/gscratch/escience/span18/output/RHS-X/output1_combined/

echo "Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID finished at $(date)."
