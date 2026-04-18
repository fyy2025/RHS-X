#!/bin/bash

# Move to your code directory
cd "$HOME/RHS-X/Code"

# Get the Job ID of the job that just finished
JOB_ID=$1

# Identify the zip file created by that job
ZIP_FILE="results_${JOB_ID}.zip"

# Send the email
if [ -f "$ZIP_FILE" ]; then
    echo "Simulation $JOB_ID complete. Results attached." | \
    mail -s "Trial Results: $JOB_ID" -a "$ZIP_FILE" span18@uw.edu
    echo "Email sent for Job $JOB_ID"
else
    echo "Error: $ZIP_FILE not found."
fi