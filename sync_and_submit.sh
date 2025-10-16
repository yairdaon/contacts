#!/bin/bash

# Mother script to synchronize files and submit SLURM job
# This script should be run from the local machine
# Usage: ./sync_and_submit.sh [server_number]
# Example: ./sync_and_submit.sh 12  (uses dml12)
# Default: ./sync_and_submit.sh     (uses dml21/server)

# Exit on any error
set -e

# Parse command line arguments
SERVER_NUM=${1:-21}  # Default to 21 if no argument provided

# Determine target server
if [ "$SERVER_NUM" = "21" ]; then
    TARGET_SERVER="server"  # Use existing alias for dml21
else
    TARGET_SERVER="dml$SERVER_NUM"
fi

echo "=== Contacts Project - Sync and Submit ==="
echo "Target server: $TARGET_SERVER (dml$SERVER_NUM)"
echo "Started at: $(date)"

echo ""
echo "Step 1: Synchronizing current directory to server via rsync"

rsync -avz \
    --delete \
    --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    --exclude='poster/' \
    --exclude='pix/' \
    ./ \
    head:contacts/

if [ $? -ne 0 ]; then
    echo "File synchronization failed!"
    exit 1
fi

echo "Files synchronized successfully"

# Step 2: Submit SLURM job
echo ""
echo "Step 2: Submitting SLURM job"

# Submit the job and capture the job ID
JOB_OUTPUT=$(ssh "$TARGET_SERVER" "cd contacts && sbatch slurm_job.sh")
echo "$JOB_OUTPUT"

# Extract job ID from output
JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]\+' | tail -1)

if [ -n "$JOB_ID" ]; then
    echo "Job submitted successfully with ID: $JOB_ID on $TARGET_SERVER"
    echo ""
    echo "You can monitor the job with:"
    echo "  ssh $TARGET_SERVER 'squeue -j $JOB_ID'"
    echo ""
    echo "View job output with:"
    echo "  ssh $TARGET_SERVER 'cat contacts/tests_${JOB_ID}.out'"
    echo "  ssh $TARGET_SERVER 'cat contacts/tests_${JOB_ID}.err'"
    echo ""
    echo "Email notifications will be sent to yair.daon@gmail.com"
else
    echo "Failed to extract job ID from sbatch output"
    exit 1
fi

echo ""
echo "=== Process completed at: $(date) ==="
