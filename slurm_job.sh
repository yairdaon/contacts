#!/bin/bash

#SBATCH --job-name=contacts_tests
#SBATCH --output=tests_%j.out
#SBATCH --error=tests_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yair.daon@gmail.com

# Script to synchronize files and run tests via SLURM

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load conda environment
echo "Loading conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yair

echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Synchronize files from local machine (if needed)
echo "Synchronizing files..."
# Note: This assumes rsync is run from the local machine before job submission
# The files should already be present in ~/contacts/

# Change to the contacts directory
cd ~/contacts/

echo "Current directory: $(pwd)"
echo "Directory contents:"
ls -la

# Verify conda environment has required packages
echo "Verifying Python environment..."
python -c "import numpy, pandas, scipy, matplotlib, pytest; print('All required packages imported successfully')"

# Run the specific tests
echo "Running tests..."
# Force unbuffered output so we can see results in real-time
PYTHONUNBUFFERED=1 python -u -m pytest tests/ -v -s --tb=long --capture=no

# Check exit status
if [ $? -eq 0 ]; then
    echo "Tests completed successfully!"
else
    echo "Tests failed!"
    exit 1
fi

echo "Job completed at: $(date)"
