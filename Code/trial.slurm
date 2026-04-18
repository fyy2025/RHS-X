#!/bin/bash

#SBATCH --job-name=trial
#SBATCH --account=escience
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=log/%x_%j.out        # %j is the unique Job ID
#SBATCH --error=log/%x_%j.err
#SBATCH --mail-type=FAIL,KILL,END
#SBATCH --mail-user=span18@uw.edu

set -euo pipefail

echo "Running on $(hostname)"
echo "Started at $(date)"

# Move to project root
cd "$HOME/RHS-X/Code"

# Create directories if they don't exist
mkdir -p log trial_output

# Activate venv
source venv/bin/activate

echo "Using python: $(which python)"
python --version

# Run the test script
python trial.py

echo "Finished python script at $(date)"

# --- Compression and Emailing ---

ZIP_NAME="trial_results_${SLURM_JOB_ID}.zip"

# Zip the trial_output directory
# -r for recursive, -j to keep the zip clean (don't include folder path)
zip -rj "$ZIP_NAME" trial_output/

# Send the email with the attachment
echo "Trial job $SLURM_JOB_ID finished. Results are attached in $ZIP_NAME." | \
mail -s "Klone Trial Results: $SLURM_JOB_ID" -a "$ZIP_NAME" span18@uw.edu

# Clean up zip file
rm "$ZIP_NAME"

echo "Job completed and email sent at $(date)"