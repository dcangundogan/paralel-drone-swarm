#!/bin/bash
# Source this file to set up CUDA environment for GPU acceleration
# Usage: source setup_cuda.sh

export CUDA_PATH="/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
export LD_LIBRARY_PATH="/usr/local/MATLAB/R2025b/bin/glnxa64:$LD_LIBRARY_PATH"

echo "CUDA environment configured for GPU acceleration"
echo "  CUDA_PATH: $CUDA_PATH"
echo ""
echo "Now run: python run_with_sensors.py --drones 9"
