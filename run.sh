#!/bin/bash
# GPU Swarm Simulation Launcher
# Sets CUDA environment for CuPy to work with MATLAB's CUDA libraries

export CUDA_PATH="/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
export LD_LIBRARY_PATH="/usr/local/MATLAB/R2025b/bin/glnxa64:$LD_LIBRARY_PATH"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "Starting GPU Swarm Simulation..."
echo "CUDA_PATH: $CUDA_PATH"

python3 run_with_sensors.py "$@"
