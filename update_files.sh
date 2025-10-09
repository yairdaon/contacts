#!/bin/bash

# Rsync synchronization script for contacts project
# This script efficiently syncs only changed files to the remote server

# Local source directory
LOCAL_DIR="/Users/yairdaon/contacts/"

# Remote destination
REMOTE_HOST="server"
REMOTE_DIR="contacts/"

# Rsync options:
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose output
# -z: compress data during transfer
# --delete: delete files on remote that don't exist locally
# --exclude: exclude certain files/directories
# --progress: show progress during transfer

echo "Synchronizing $LOCAL_DIR to $REMOTE_HOST:$REMOTE_DIR"
echo "Using rsync with differential updates..."

rsync -avz \
    --delete \
    --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    "$LOCAL_DIR" \
    "$REMOTE_HOST:$REMOTE_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Synchronization completed successfully"
else
    echo "✗ Synchronization failed"
    exit 1
fi