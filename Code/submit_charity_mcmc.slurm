#!/bin/bash

#SBATCH --job-name=charity_mcmc
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

echo "Running on $(hostname)"
echo "Started at $(date)"

cd $HOME/RHS-X/Code

mkdir -p log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python charity_mcmc_comparison.py \
    --data_dta    ../Data/Does_Price_Matter_AER_merged.dta \
    --out_dir     ./output_charity_mcmc \
    --mcmc_steps  50000 \
    --mcmc_burnin 20000 \
    --mcmc_thin   10 \
    --mcmc_chains 8 \
    --seed        42

echo "Job $SLURM_JOB_ID finished at $(date)."
