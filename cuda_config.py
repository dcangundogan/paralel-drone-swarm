#!/usr/bin/env python3
"""
CUDA Configuration and Auto-Detection
======================================

Bu modül:
1. CUDA versiyonunu otomatik tespit eder
2. Doğru CuPy paketini önerir
3. GPU/CPU modunu yönetir

Kullanım:
    from cuda_config import get_array_module, GPU_AVAILABLE, CUDA_VERSION

    xp = get_array_module()  # cupy veya numpy döner

    # GPU varsa cupy, yoksa numpy kullanır
    arr = xp.zeros((100, 3))
"""

import os
import subprocess
import sys
from typing import Tuple, Optional

# ============================================================
# CUDA VERSION DETECTION
# ============================================================

def detect_cuda_version() -> Optional[str]:
    """
    Sistemde kurulu CUDA versiyonunu tespit et.

    Returns:
        CUDA version string (e.g., "11.8", "12.1") or None
    """
    # Method 1: nvcc --version
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse output: "Cuda compilation tools, release 11.8, V11.8.89"
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    parts = line.split('release')[-1].strip()
                    version = parts.split(',')[0].strip()
                    return version
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: CUDA_PATH environment variable
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if cuda_path:
        # Extract version from path like "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
        import re
        match = re.search(r'v?(\d+\.\d+)', cuda_path)
        if match:
            return match.group(1)

    # Method 3: nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse "CUDA Version: 11.8"
            import re
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def get_cupy_package_name(cuda_version: str) -> str:
    """
    CUDA versiyonuna göre doğru CuPy paket adını döndür.

    Args:
        cuda_version: CUDA version (e.g., "11.8", "12.1")

    Returns:
        CuPy package name (e.g., "cupy-cuda11x", "cupy-cuda12x")
    """
    major = int(cuda_version.split('.')[0])
    minor = int(cuda_version.split('.')[1]) if '.' in cuda_version else 0

    if major == 11:
        return "cupy-cuda11x"
    elif major == 12:
        return "cupy-cuda12x"
    elif major == 10:
        # Specific versions for CUDA 10
        return f"cupy-cuda{major}{minor}"
    else:
        return "cupy-cuda11x"  # Default


# ============================================================
# GPU AVAILABILITY CHECK
# ============================================================

def check_gpu_available() -> Tuple[bool, str]:
    """
    GPU ve CuPy kullanılabilirliğini kontrol et.

    Returns:
        (is_available, message)
    """
    # Check if CuPy is installed
    try:
        import cupy as cp
    except ImportError:
        cuda_ver = detect_cuda_version()
        if cuda_ver:
            pkg = get_cupy_package_name(cuda_ver)
            return False, f"CuPy kurulu değil. CUDA {cuda_ver} için: pip install {pkg}"
        else:
            return False, "CuPy kurulu değil ve CUDA bulunamadı. CPU modu kullanılacak."

    # Check if GPU is accessible
    try:
        cp.cuda.Device(0).compute_capability
        device_name = cp.cuda.Device(0).name
        mem_total = cp.cuda.Device(0).mem_info[1] / (1024**3)
        return True, f"GPU aktif: {device_name} ({mem_total:.1f} GB)"
    except Exception as e:
        return False, f"CuPy kurulu ama GPU erişilemiyor: {str(e)}"


# ============================================================
# ARRAY MODULE (xp)
# ============================================================

def get_array_module(prefer_gpu: bool = True):
    """
    GPU varsa CuPy, yoksa NumPy döndür.

    Args:
        prefer_gpu: GPU tercih edilsin mi? False ise hep NumPy kullan.

    Returns:
        cupy or numpy module

    Usage:
        xp = get_array_module()
        arr = xp.zeros((100, 3))
    """
    if not prefer_gpu:
        import numpy as np
        return np

    try:
        import cupy as cp
        # Test GPU access
        cp.cuda.Device(0).compute_capability
        _ = cp.array([1, 2, 3])
        return cp
    except:
        import numpy as np
        return np


# ============================================================
# GLOBAL STATE
# ============================================================

# Detect at import time
CUDA_VERSION = detect_cuda_version()
GPU_AVAILABLE, GPU_MESSAGE = check_gpu_available()

# Default array module
xp = get_array_module()


# ============================================================
# INSTALLATION HELPER
# ============================================================

def install_cupy():
    """CuPy'yi otomatik kur (kullanıcı izniyle)."""
    cuda_ver = detect_cuda_version()

    if not cuda_ver:
        print("HATA: CUDA bulunamadı!")
        print("CUDA Toolkit'i yükleyin: https://developer.nvidia.com/cuda-downloads")
        return False

    pkg = get_cupy_package_name(cuda_ver)

    print(f"CUDA {cuda_ver} tespit edildi.")
    print(f"CuPy paketi: {pkg}")

    response = input(f"\n{pkg} yüklensin mi? [e/h]: ").strip().lower()

    if response in ['e', 'y', 'yes', 'evet']:
        print(f"\nYükleniyor: pip install {pkg}")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', pkg])
        if result.returncode == 0:
            print("\nCuPy başarıyla yüklendi!")
            print("Programı yeniden başlatın.")
            return True
        else:
            print("\nYükleme başarısız.")
            return False

    return False


# ============================================================
# CLI INTERFACE
# ============================================================

def print_status():
    """GPU durumunu yazdır."""
    print("=" * 60)
    print("GPU CONFIGURATION STATUS")
    print("=" * 60)

    if CUDA_VERSION:
        print(f"CUDA Version  : {CUDA_VERSION}")
        print(f"CuPy Package  : {get_cupy_package_name(CUDA_VERSION)}")
    else:
        print("CUDA Version  : Not detected")

    print(f"GPU Available : {GPU_AVAILABLE}")
    print(f"Status        : {GPU_MESSAGE}")

    if GPU_AVAILABLE:
        import cupy as cp
        print(f"Device        : {cp.cuda.Device(0).name}")
        mem_free, mem_total = cp.cuda.Device(0).mem_info
        print(f"Memory        : {mem_free/1024**3:.1f} / {mem_total/1024**3:.1f} GB")

    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CUDA/CuPy Configuration")
    parser.add_argument('--status', action='store_true', help='Show GPU status')
    parser.add_argument('--install', action='store_true', help='Install CuPy')

    args = parser.parse_args()

    if args.install:
        install_cupy()
    else:
        print_status()
