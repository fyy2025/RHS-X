#!/bin/bash
#SBATCH --job-name=trial
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err

set -euo pipefail

# 1. Setup
cd "$HOME/RHS-X/Code"
mkdir -p log trial_output
source venv/bin/activate

# 2. Run Python
python trial.py

# 3. Create ONE master zip
# This grabs the folder and all its contents
zip -r "results_${SLURM_JOB_ID}.zip" trial_output/

echo "Job $SLURM_JOB_ID finished and zipped."