#!/bin/bash

#SBATCH --job-name=sim2_ais_cfg
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --time=80:00:00
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

# --- Simulation Parameters ---
ITER=1
EPS1=0.4
EPS2=0.7
THETA_INIT=13.0
EPSILONS="0.16,0.17,0.18,0.19,0.20"
AIS_CFG_GRID="300:3:5,300:5:5,300:12:5,300:3:10,300:5:10,300:12:10"
LADDER_MIN_LOG10=-0.9

MCMC_STEPS=50000
MCMC_BURNIN=20000
MCMC_THIN=10
N_PER_POL=500

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "Parameters: ITER=${ITER}, EPS1=${EPS1}, EPS2=${EPS2}, THETA_INIT=${THETA_INIT}, EPSILONS=${EPSILONS}"
echo "AIS config grid: ${AIS_CFG_GRID}"

cd $HOME/RHS-X/Code

mkdir -p output2 log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python simulation_scenario2_ais_cfg.py \
    --iter $ITER \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --theta_init $THETA_INIT \
    --epsilons "$EPSILONS" \
    --ais_cfg_grid "$AIS_CFG_GRID" \
    --ladder_min_log10 $LADDER_MIN_LOG10 \
    --mcmc_steps $MCMC_STEPS \
    --mcmc_burnin $MCMC_BURNIN \
    --mcmc_thin $MCMC_THIN \
    --n_per_pol $N_PER_POL \
    --out_dir ./output2 \
    --out_name "sim2_ais_cfg_iter${ITER}.pkl"

zip -r "results_sim2_ais_cfg_${SLURM_JOB_ID}.zip" output2/

echo "Job $SLURM_JOB_ID finished at $(date)."
