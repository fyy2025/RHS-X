#!/bin/bash

#SBATCH --job-name=sim_eff_s2
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

# --- scenario 2 (65536-partition) AIS efficiency ---
SCENARIO=2
N_PATHS=6000
REPS=5
MOVES_PER_LEVEL=5
SCHEDULE_POWER=3.0
NUM_SAMPLES=20            # last run gave perplexity ~11 (spread, not a point mass)
MU_SCALE=0.25
TARGET_PERPLEXITY=0      # faithful (data-driven spread)
# Focused on the ACTIVE range: RPS grows 1 -> a few thousand. Stays below RPS=all
# (which happened at gap>=128 and produced the chi^2=0 / sqrt crash), so no degenerate rows.
THETA_GAPS="1,2,3,4,6,8,10,12,16,24"
T_GRID="1,2,4,8,16,32,64,128,256"        # capped at 256 (last run's max useful T_min was ~253)

# Dedicated output directory on the cluster (not ../Figures)
OUTDIR=$HOME/RHS-X/Code/output_sim_efficiency

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "scenario=${SCENARIO} n_paths=${N_PATHS} reps=${REPS} num_samples=${NUM_SAMPLES} mu_scale=${MU_SCALE}"

cd $HOME/RHS-X/Code

mkdir -p $OUTDIR/s2_eps_prod $OUTDIR/s2_eps_clean log

source venv/bin/activate
echo "Using python: $(which python)"; python --version

# ---- Run 1: PRODUCTION 3-region proposal (eps1=0.4, eps2=0.7) ----
# The real make_p0_buckets_weighted_S0 proposal. Expect chi^2 to FLOOR at
# 1/eps1 - 1 = 1.5 (only 40% of proposal mass on the RPS) -> capped / non-monotone.
echo "=== Run 1: eps1=0.4 eps2=0.7 (production 3-region) ==="
python sim_efficiency.py \
    --scenario $SCENARIO --num_samples $NUM_SAMPLES --mu_scale $MU_SCALE \
    --n_paths $N_PATHS --reps $REPS --moves_per_level $MOVES_PER_LEVEL \
    --schedule_power $SCHEDULE_POWER --target_perplexity $TARGET_PERPLEXITY \
    --eps1 0.4 --eps2 0.7 \
    --theta_gaps "$THETA_GAPS" --T_grid "$T_GRID" \
    --outdir $OUTDIR/s2_eps_prod

# ---- Run 2: CLEAN 2-region proposal (eps1=eps2=0.8) ----
# eps2==eps1 -> RPS + uniform rest. chi^2 floor drops to 1/eps1 - 1 = 0.25 -> the
# clean monotone sqrt(chi^2) efficiency law.
echo "=== Run 2: eps1=0.8 eps2=0.8 (clean 2-region) ==="
python sim_efficiency.py \
    --scenario $SCENARIO --num_samples $NUM_SAMPLES --mu_scale $MU_SCALE \
    --n_paths $N_PATHS --reps $REPS --moves_per_level $MOVES_PER_LEVEL \
    --schedule_power $SCHEDULE_POWER --target_perplexity $TARGET_PERPLEXITY \
    --eps1 0.8 --eps2 0.9 \
    --theta_gaps "$THETA_GAPS" --T_grid "$T_GRID" \
    --outdir $OUTDIR/s2_eps_clean

echo "Job $SLURM_JOB_ID finished at $(date)."
