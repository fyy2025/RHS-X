#!/bin/bash

#SBATCH --job-name=sim_efficiency
#SBATCH --account=escience
#SBATCH --partition=cpu-g2-mem2x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=log/%x_%A.out
#SBATCH --error=log/%x_%A.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=fyy2030@uw.edu

set -euo pipefail

# --- AIS efficiency study parameters ---
SCENARIO=2                 # 2 = M=3, R=[4,3,3] (65536-partition space = 2^16)
N_PATHS=6000
REPS=6
MOVES_PER_LEVEL=5
SCHEDULE_POWER=3.0
EPS1=0.8                   # mass on RPS (~posterior). eps2==eps1 -> 2-region proposal (RPS+uniform
EPS2=0.8                   # rest); set EPS2>EPS1 (e.g. 0.7 vs 0.4) for the production 3-region bucket.
NUM_SAMPLES=20            # samples/policy. LOW spreads the posterior off a point mass (n=500 is
                          # near point mass). Check the .out perplexity; raise/lower to taste.
MU_SCALE=0.25             # shrink pool-mean separations (helps spread the posterior)
TARGET_PERPLEXITY=0       # 0 = faithful (data-driven spread via NUM_SAMPLES/MU_SCALE, not tempering)
THETA_GAPS="0.5,1,2,4,8,16,32,64,128,256"   # Rashomon thresholds -> RPS via RAggregate(reg*)
T_GRID="1,2,4,8,16,32,64,128,256,512"

echo "Running on $(hostname)"
echo "Started at $(date)"
echo "scenario=${SCENARIO} n_paths=${N_PATHS} reps=${REPS} num_samples=${NUM_SAMPLES} mu_scale=${MU_SCALE}"

cd $HOME/RHS-X/Code

mkdir -p ../Figures log

source venv/bin/activate

echo "Using python: $(which python)"
python --version

python sim_efficiency.py \
    --scenario $SCENARIO \
    --num_samples $NUM_SAMPLES \
    --mu_scale $MU_SCALE \
    --n_paths $N_PATHS \
    --reps $REPS \
    --moves_per_level $MOVES_PER_LEVEL \
    --schedule_power $SCHEDULE_POWER \
    --eps1 $EPS1 \
    --eps2 $EPS2 \
    --target_perplexity $TARGET_PERPLEXITY \
    --theta_gaps "$THETA_GAPS" \
    --T_grid "$T_GRID" \
    --outdir ../Figures

echo "Job $SLURM_JOB_ID finished at $(date)."
