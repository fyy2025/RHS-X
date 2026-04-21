#!/bin/bash

# 1. Submit simulation
ID=$(sbatch --parsable submit.slurm)
echo "Job $ID submitted."

# 2. Run a "silent" background process on the login node
# 'nohup' keeps it alive even if you close your laptop.
nohup bash -c "
    # Wait until the job is no longer in the queue
    while squeue -j $ID 2>/dev/null | grep -q $ID; do
        sleep 60
    done

    # Now that it's done, we are STILL on the login node.
    # Send the email.
    mail -s 'Results $ID' -a results_${ID}.zip span18@uw.edu fyy2030@uw.edu
" > log/email_watcher.log 2>&1 &

echo "Watcher started. You can log out. The login node will email you later."
