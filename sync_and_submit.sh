#!/bin/bash

# Mother script to synchronize files and submit SLURM job
# This script should be run from the local machine

# Exit on any error
set -e

echo "=== Contacts Project - Sync and Submit ==="
echo "Started at: $(date)"

# Step 1: Synchronize files to remote server
echo ""
echo "Step 1: Synchronizing files to remote server..."
./update_files.sh

if [ $? -ne 0 ]; then
    echo "❌ File synchronization failed!"
    exit 1
fi

echo "✅ Files synchronized successfully"

# Step 2: Submit SLURM job
echo ""
echo "Step 2: Submitting SLURM job..."

# Submit the job and capture the job ID
JOB_OUTPUT=$(ssh server "cd contacts && sbatch slurm_job.sh")
echo "$JOB_OUTPUT"

# Extract job ID from output
JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+' | tail -1)

if [ -n "$JOB_ID" ]; then
    echo "✅ Job submitted successfully with ID: $JOB_ID"
    echo ""
    echo "You can monitor the job with:"
    echo "  ssh server 'squeue -j $JOB_ID'"
    echo ""
    echo "View job output with:"
    echo "  ssh server 'cat contacts/tests_${JOB_ID}.out'"
    echo "  ssh server 'cat contacts/tests_${JOB_ID}.err'"
    echo ""
    echo "Email notifications will be sent to yair.daon@gmail.com"
else
    echo "❌ Failed to extract job ID from sbatch output"
    exit 1
fi

echo ""
echo "=== Process completed at: $(date) ==="
