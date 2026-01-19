#!/bin/bash
# Wrapper script to run swarm with CUDA enabled

# Use MATLAB's CUDA libraries if available
MATLAB_CUDA="/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
MATLAB_LIB="/usr/local/MATLAB/R2025b/bin/glnxa64"

if [ -d "$MATLAB_CUDA" ]; then
    export CUDA_PATH="$MATLAB_CUDA"
    export LD_LIBRARY_PATH="$MATLAB_LIB:$MATLAB_CUDA/lib64:$LD_LIBRARY_PATH"
    echo "[CUDA] Using MATLAB's CUDA libraries"
fi

# Run the swarm manager
cd "$(dirname "$0")"
python3 swarm_manager.py "$@"
