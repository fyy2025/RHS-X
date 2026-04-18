#!/bin/bash

# 1. Submit the main simulation job and capture its ID
# --parsable makes it so the variable 'ID' only contains the numbers
ID=$(sbatch --parsable trial.slurm)

echo "Simulation submitted with Job ID: $ID"
echo "The email will be sent automatically once Job $ID finishes."

# 2. Schedule the email script to run ONLY after the simulation job finishes successfully
# We use --partition=ckpt because those nodes usually have internet access
sbatch --dependency=afterok:$ID \
       --partition=ckpt \
       --job-name=email_job \
       --wrap="./email.sh $ID"