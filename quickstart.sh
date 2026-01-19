#!/bin/bash
# =============================================================================
# QUICK START - GPU SWARM WITH HEADLESS GAZEBO
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
NUM_DRONES=${1:-5}

echo "=============================================="
echo "  GPU SWARM QUICK START"
echo "=============================================="
echo ""

# Check dependencies
echo "[1/5] Checking dependencies..."

check_dep() {
    if ! command -v $1 &> /dev/null; then
        echo "  ✗ $1 not found"
        return 1
    else
        echo "  ✓ $1"
        return 0
    fi
}

check_dep python3 || exit 1
check_dep gz || { echo "Install Gazebo: sudo apt install gz-harmonic"; exit 1; }

# Check Python packages
echo ""
echo "[2/5] Checking Python packages..."

pip_install() {
    if pip3 install --help 2>/dev/null | grep -q -- '--break-system-packages'; then
        pip3 install "$@" --break-system-packages
    else
        pip3 install "$@"
    fi
}

python3 -c "import numpy" 2>/dev/null || { echo "Installing numpy..."; pip_install numpy; }
python3 -c "import pygame" 2>/dev/null || { echo "Installing pygame..."; pip_install pygame; }

# Check CUDA
echo ""
echo "[3/5] Checking GPU..."
if python3 - <<'PY' 2>/dev/null; then
import cupy as cp
device = cp.cuda.Device()
name = ""
try:
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props.get("name", b"").decode("utf-8", errors="ignore")
except Exception:
    pass
cp.arange(1)
print(f"  ✓ CuPy with CUDA: {name or 'unknown'}")
PY
    GPU_OK=true
else
    echo "  ○ CuPy not available (will use CPU)"
    echo "  Install for GPU: pip3 install cupy-cuda12x"
    GPU_OK=false
fi

# Start Gazebo in background
echo ""
echo "[4/5] Starting Gazebo headless..."

cd "$PROJECT_DIR"

# Set environment
export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR:$PROJECT_DIR/models:$GZ_SIM_RESOURCE_PATH"

# Kill old instances
pkill -9 -f "gz sim" 2>/dev/null || true
sleep 2

# Start headless
gz sim -s -r --headless-rendering "$PROJECT_DIR/swarm_world.sdf" &
GZ_PID=$!
echo "  Gazebo PID: $GZ_PID"

sleep 8

if ! ps -p $GZ_PID > /dev/null 2>&1; then
    echo "  ✗ Gazebo failed to start"
    echo ""
    echo "Try running Gazebo manually:"
    echo "  cd $PROJECT_DIR"
    echo "  gz sim -s $PROJECT_DIR/swarm_world.sdf --headless-rendering"
    exit 1
fi

echo "  ✓ Gazebo running"

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    pkill -9 -f "gz sim" 2>/dev/null || true
    pkill -9 -f "ruby.*gz" 2>/dev/null || true
    pkill -9 -f "swarm_manager" 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Run swarm manager
echo ""
echo "[5/5] Starting swarm controller..."
echo ""

cd "$PROJECT_DIR"
python3 swarm_manager.py --drones $NUM_DRONES
