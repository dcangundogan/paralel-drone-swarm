#!/bin/bash
# GPU Swarm Simulation Launcher
# Updated to use CUDA from /hey/cuda

set -e

# Set CUDA environment
export CUDA_HOME=/hey/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Project directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Activate virtual environment
source .venv/bin/activate

echo "======================================"
echo "  GPU Drone Swarm Simulation"
echo "======================================"
echo "CUDA: $CUDA_HOME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Check which simulation to run
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: $0 [options]"
    echo ""
    echo "Available simulations:"
    echo "  --gpu           GPU-accelerated swarm (default)"
    echo "  --sensors       GPU swarm with sensors"
    echo "  --large         Large swarm simulation"
    echo "  --drones N      Number of drones (default: 10)"
    echo ""
    echo "Examples:"
    echo "  $0 --gpu --drones 50"
    echo "  $0 --sensors --drones 25"
    echo "  $0 --large"
    exit 0
fi

# Parse arguments
SCRIPT="run_gpu_simulation.py"
DRONES=10

while [ $# -gt 0 ]; do
    case "$1" in
        --sensors)
            SCRIPT="run_with_sensors.py"
            shift
            ;;
        --large)
            SCRIPT="run_large_swarm.py"
            shift
            ;;
        --gpu)
            SCRIPT="run_gpu_simulation.py"
            shift
            ;;
        --drones)
            DRONES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Running: python3 $SCRIPT --drones $DRONES"
echo ""

# Run simulation
python3 "$SCRIPT" --drones $DRONES
