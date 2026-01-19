#!/usr/bin/env python3
"""
GPU-ACCELERATED SWARM CONTROLLER
================================
Uses CuPy (CUDA) for parallel computation of:
- Collision avoidance (Reynolds separation)
- Trajectory following
- Formation control
- Velocity blending

All N drones computed in parallel on GPU.
"""

import numpy as np
import time
import os
from dataclasses import dataclass, field
from typing import Callable, Tuple, List, Optional

# Configure CUDA paths before importing CuPy
# Try MATLAB's CUDA installation if system CUDA is missing
_MATLAB_CUDA_PATH = "/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
_MATLAB_LIB_PATH = "/usr/local/MATLAB/R2025b/bin/glnxa64"

if os.path.exists(_MATLAB_CUDA_PATH) and "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = _MATLAB_CUDA_PATH
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    _new_paths = f"{_MATLAB_LIB_PATH}:{_MATLAB_CUDA_PATH}/lib64"
    if _new_paths not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_new_paths}:{_ld_path}"

# GPU acceleration
GPU_AVAILABLE = False
try:
    import cupy as cp
    device = cp.cuda.Device()
    # Test CUDA functionality
    cp.arange(1)  # Trigger kernel compilation to verify NVRTC
    GPU_AVAILABLE = True
    print("[GPU] CUDA acceleration enabled")
    try:
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        name = props.get("name", b"").decode("utf-8", errors="ignore")
    except Exception:
        name = "unknown"
    print(f"  Device: {name}")
    print(f"  Memory: {device.mem_info[1] / 1e9:.1f} GB")
except ImportError:
    cp = np
    print("[GPU] CuPy not installed - running on CPU (numpy)")
    print("  To enable GPU: pip install cupy-cuda12x")
except Exception as e:
    cp = np
    print(f"[GPU] CUDA init failed: {type(e).__name__}")
    print(f"  {str(e)[:100]}")
    print("  Falling back to CPU (numpy)")
    print("  Check: nvidia-smi, nvcc --version, and CuPy/CUDA version match")


@dataclass
class SwarmConfig:
    """Swarm controller parameters"""
    num_drones: int = 25
    
    # Control rates
    control_rate: float = 50.0      # Hz
    
    # Velocity limits
    max_velocity_xy: float = 3.0    # m/s horizontal
    max_velocity_z: float = 2.0     # m/s vertical
    max_acceleration: float = 2.0   # m/s²
    
    # Collision avoidance
    collision_radius: float = 0.8   # metre - hard collision zone
    avoidance_radius: float = 2.5   # metre - soft avoidance starts
    avoidance_strength: float = 3.0 # repulsion multiplier
    
    # Formation
    cohesion_strength: float = 0.3  # attraction to swarm center
    alignment_strength: float = 0.2 # velocity alignment
    
    # Target tracking
    target_gain_p: float = 1.5      # proportional gain
    target_gain_d: float = 0.3      # derivative gain
    
    # Height control
    default_altitude: float = 5.0   # metres
    altitude_tolerance: float = 0.3 # metres


class GPUSwarmController:
    """
    GPU-accelerated swarm controller.
    All vector operations run on CUDA for parallel computation.
    """
    
    def __init__(self, config: SwarmConfig = None):
        self.config = config or SwarmConfig()
        self.n = self.config.num_drones
        
        # Use CuPy arrays for GPU computation
        self.xp = cp if GPU_AVAILABLE else np
        
        # State arrays on GPU
        self.positions = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        self.velocities = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        self.targets = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        self.cmd_velocities = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        
        # Previous errors for derivative control
        self.prev_errors = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        
        # Status
        self.active = self.xp.zeros(self.n, dtype=bool)
        self.armed = self.xp.zeros(self.n, dtype=bool)

        # Store initial positions for visualization
        self.initial_positions = self.xp.zeros((self.n, 3), dtype=self.xp.float32)
        
        # Timing
        self.last_update = time.time()
        self.dt = 1.0 / self.config.control_rate
        
        # Trajectory function (if following)
        self.trajectory_func: Optional[Callable] = None
        self.trajectory_offsets = self.xp.linspace(0, 5, self.n, dtype=self.xp.float32)
        
        # Performance tracking
        self.compute_times = []
        
    def update_positions(self, positions: np.ndarray):
        """Update drone positions from Gazebo (CPU->GPU transfer)"""
        if GPU_AVAILABLE:
            self.positions = cp.asarray(positions, dtype=cp.float32)
        else:
            self.positions = positions.astype(np.float32)
    
    def update_velocities(self, velocities: np.ndarray):
        """Update drone velocities from Gazebo"""
        if GPU_AVAILABLE:
            self.velocities = cp.asarray(velocities, dtype=cp.float32)
        else:
            self.velocities = velocities.astype(np.float32)
    
    def _compute_separation(self) -> 'xp.ndarray':
        """
        Reynolds separation (collision avoidance)
        GPU parallel: O(n²) but all computed simultaneously
        """
        xp = self.xp
        cfg = self.config
        
        # Pairwise differences: [n, n, 3]
        diff = self.positions[:, xp.newaxis, :] - self.positions[xp.newaxis, :, :]
        
        # Pairwise distances: [n, n]
        dist = xp.linalg.norm(diff, axis=2)
        
        # Avoid self-comparison
        xp.fill_diagonal(dist, xp.inf)
        
        # Mask for drones within avoidance radius
        avoid_mask = dist < cfg.avoidance_radius
        
        # Normalized direction (away from neighbor)
        safe_dist = xp.maximum(dist, 0.001)[:, :, xp.newaxis]
        direction = diff / safe_dist
        
        # Separation magnitude: stronger when closer
        # Linear falloff from avoidance_radius to collision_radius
        magnitude = xp.where(
            avoid_mask,
            xp.clip(
                (cfg.avoidance_radius - dist) / (cfg.avoidance_radius - cfg.collision_radius),
                0.0, 2.0  # Cap at 2x for very close
            ),
            0.0
        )[:, :, xp.newaxis]
        
        # Sum all separation vectors for each drone
        separation = xp.sum(direction * magnitude, axis=1) * cfg.avoidance_strength
        
        return separation
    
    def _compute_cohesion(self) -> 'xp.ndarray':
        """
        Cohesion: attraction toward swarm center
        Keeps swarm together
        """
        xp = self.xp
        
        # Swarm center
        center = xp.mean(self.positions, axis=0)
        
        # Direction to center
        to_center = center - self.positions
        dist = xp.linalg.norm(to_center, axis=1, keepdims=True)
        dist = xp.maximum(dist, 0.001)
        
        # Normalized, scaled by distance
        cohesion = (to_center / dist) * xp.minimum(dist * 0.3, 1.0)
        
        return cohesion * self.config.cohesion_strength
    
    def _compute_alignment(self) -> 'xp.ndarray':
        """
        Alignment: match velocity with neighbors
        Smoother swarm movement
        """
        xp = self.xp
        
        # Average velocity
        avg_vel = xp.mean(self.velocities, axis=0)
        
        # Difference from average
        alignment = avg_vel - self.velocities
        
        return alignment * self.config.alignment_strength
    
    def _compute_target_tracking(self) -> 'xp.ndarray':
        """
        PD controller for target tracking
        """
        xp = self.xp
        cfg = self.config
        
        # Position error
        error = self.targets - self.positions
        
        # Derivative (change in error)
        d_error = (error - self.prev_errors) / self.dt
        self.prev_errors = error.copy()
        
        # PD control
        tracking = error * cfg.target_gain_p + d_error * cfg.target_gain_d
        
        return tracking
    
    def _apply_velocity_limits(self, velocities: 'xp.ndarray') -> 'xp.ndarray':
        """Apply velocity and acceleration limits"""
        xp = self.xp
        cfg = self.config
        
        # Separate XY and Z
        vel_xy = velocities[:, :2]
        vel_z = velocities[:, 2:3]
        
        # Limit XY speed
        speed_xy = xp.linalg.norm(vel_xy, axis=1, keepdims=True)
        speed_xy = xp.maximum(speed_xy, 0.001)
        vel_xy = xp.where(
            speed_xy > cfg.max_velocity_xy,
            vel_xy / speed_xy * cfg.max_velocity_xy,
            vel_xy
        )
        
        # Limit Z speed
        vel_z = xp.clip(vel_z, -cfg.max_velocity_z, cfg.max_velocity_z)
        
        # Recombine
        limited = xp.concatenate([vel_xy, vel_z], axis=1)
        
        # Acceleration limiting (smooth changes)
        accel = (limited - self.cmd_velocities) / self.dt
        accel_mag = xp.linalg.norm(accel, axis=1, keepdims=True)
        accel_mag = xp.maximum(accel_mag, 0.001)
        
        accel = xp.where(
            accel_mag > cfg.max_acceleration,
            accel / accel_mag * cfg.max_acceleration,
            accel
        )
        
        return self.cmd_velocities + accel * self.dt
    
    def compute_control(self) -> np.ndarray:
        """
        Main control computation - runs on GPU
        Returns velocity commands for all drones
        """
        t_start = time.perf_counter()
        
        xp = self.xp
        
        # Update trajectory targets if following
        if self.trajectory_func is not None:
            self._update_trajectory_targets()
        
        # Compute all control components (parallel on GPU)
        separation = self._compute_separation()
        cohesion = self._compute_cohesion()
        alignment = self._compute_alignment()
        tracking = self._compute_target_tracking()
        
        # Blend control signals
        # Priority: separation > tracking > cohesion > alignment
        desired_vel = (
            separation * 2.0 +      # Highest priority - safety
            tracking * 1.5 +        # Follow targets
            cohesion * 0.5 +        # Stay together
            alignment * 0.3         # Smooth movement
        )
        
        # Apply only to active drones
        active_mask = self.active[:, xp.newaxis]
        desired_vel = desired_vel * active_mask
        
        # Apply limits
        self.cmd_velocities = self._apply_velocity_limits(desired_vel)
        
        # Transfer back to CPU for Gazebo
        if GPU_AVAILABLE:
            result = cp.asnumpy(self.cmd_velocities)
        else:
            result = self.cmd_velocities.copy()
        
        # Track compute time
        t_elapsed = time.perf_counter() - t_start
        self.compute_times.append(t_elapsed)
        if len(self.compute_times) > 100:
            self.compute_times.pop(0)
        
        return result
    
    def _update_trajectory_targets(self):
        """Update targets from trajectory function"""
        xp = self.xp
        current_time = time.time()
        
        # Compute targets on CPU (trajectory func is Python)
        targets_cpu = np.zeros((self.n, 3), dtype=np.float32)
        for i in range(self.n):
            t = current_time + float(self.trajectory_offsets[i] if GPU_AVAILABLE else self.trajectory_offsets[i])
            targets_cpu[i] = self.trajectory_func(t)
        
        # Transfer to GPU
        if GPU_AVAILABLE:
            self.targets = cp.asarray(targets_cpu)
        else:
            self.targets = targets_cpu
    
    # ============ HIGH-LEVEL COMMANDS ============
    
    def arm_all(self):
        """Arm all drones"""
        self.armed[:] = True
        print(f"[GPU] Armed {self.n} drones")

    def disarm_all(self):
        """Disarm all drones"""
        self.armed[:] = False
        self.active[:] = False
        self.cmd_velocities[:] = 0
        print(f"[GPU] Disarmed {self.n} drones")

    def takeoff(self, altitude: float = None):
        """Command takeoff to specified altitude - auto-arms drones"""
        alt = altitude or self.config.default_altitude

        # Auto-arm on takeoff
        if not self.xp.any(self.armed):
            self.arm_all()

        # Store initial positions for visualization
        self.initial_positions = self.positions.copy()

        # Set targets to current XY, target Z
        self.targets[:, 0] = self.positions[:, 0]
        self.targets[:, 1] = self.positions[:, 1]
        self.targets[:, 2] = alt

        self.active[:] = True
        self.trajectory_func = None
        print(f"[GPU] Takeoff to {alt}m ({self.n} drones)")
    
    def land(self):
        """Command landing"""
        self.targets[:, 2] = 0.0
        self.trajectory_func = None
        print("[GPU] Landing")
    
    def set_formation_grid(self, center: Tuple[float, float, float], spacing: float = 3.0):
        """Grid formation"""
        xp = self.xp
        cx, cy, cz = center
        cols = int(np.ceil(np.sqrt(self.n)))
        
        targets_cpu = np.zeros((self.n, 3), dtype=np.float32)
        for i in range(self.n):
            row, col = divmod(i, cols)
            targets_cpu[i] = [
                cx + (col - cols/2) * spacing,
                cy + (row - cols/2) * spacing,
                cz
            ]
        
        if GPU_AVAILABLE:
            self.targets = cp.asarray(targets_cpu)
        else:
            self.targets = targets_cpu
        
        self.trajectory_func = None
        print(f"[GPU] Grid formation at ({cx}, {cy}, {cz})")
    
    def set_formation_circle(self, center: Tuple[float, float, float], radius: float = 10.0):
        """Circle formation"""
        cx, cy, cz = center
        angles = np.linspace(0, 2*np.pi, self.n, endpoint=False)
        
        targets_cpu = np.zeros((self.n, 3), dtype=np.float32)
        targets_cpu[:, 0] = cx + radius * np.cos(angles)
        targets_cpu[:, 1] = cy + radius * np.sin(angles)
        targets_cpu[:, 2] = cz
        
        if GPU_AVAILABLE:
            self.targets = cp.asarray(targets_cpu)
        else:
            self.targets = targets_cpu
        
        self.trajectory_func = None
        print(f"[GPU] Circle formation r={radius}m")
    
    def set_formation_v(self, tip: Tuple[float, float, float], spacing: float = 2.5, angle: float = 45.0):
        """V formation (bird-like)"""
        tx, ty, tz = tip
        angle_rad = np.radians(angle)
        
        targets_cpu = np.zeros((self.n, 3), dtype=np.float32)
        for i in range(self.n):
            side = 1 if i % 2 == 0 else -1
            depth = (i + 1) // 2
            targets_cpu[i] = [
                tx - depth * spacing * np.cos(angle_rad),
                ty + side * depth * spacing * np.sin(angle_rad),
                tz
            ]
        
        if GPU_AVAILABLE:
            self.targets = cp.asarray(targets_cpu)
        else:
            self.targets = targets_cpu
        
        self.trajectory_func = None
        print("[GPU] V formation")
    
    def follow_trajectory(self, traj_func: Callable[[float], Tuple[float, float, float]], 
                         spacing: float = 1.0):
        """Follow a trajectory function"""
        self.trajectory_func = traj_func
        self.trajectory_offsets = self.xp.linspace(0, spacing * self.n, self.n, dtype=self.xp.float32)
        print("[GPU] Trajectory following enabled")
    
    def stop_trajectory(self):
        """Stop trajectory following, hold current targets"""
        self.trajectory_func = None
        print("[GPU] Trajectory stopped")
    
    # ============ STATS ============
    
    def get_min_distance(self) -> float:
        """Get minimum distance between any two drones"""
        xp = self.xp
        
        diff = self.positions[:, xp.newaxis, :] - self.positions[xp.newaxis, :, :]
        dist = xp.linalg.norm(diff, axis=2)
        xp.fill_diagonal(dist, xp.inf)
        
        min_dist = float(xp.min(dist))
        return min_dist
    
    def get_center(self) -> np.ndarray:
        """Get swarm center position"""
        center = self.xp.mean(self.positions, axis=0)
        if GPU_AVAILABLE:
            return cp.asnumpy(center)
        return center
    
    def get_avg_compute_time(self) -> float:
        """Average computation time per cycle"""
        if not self.compute_times:
            return 0.0
        return sum(self.compute_times) / len(self.compute_times)
    
    def get_compute_rate(self) -> float:
        """Computation rate in Hz"""
        avg_time = self.get_avg_compute_time()
        if avg_time <= 0:
            return 0.0
        return 1.0 / avg_time


# ============ TRAJECTORY FUNCTIONS ============

def trajectory_circle(t: float, radius: float = 10.0, altitude: float = 5.0, 
                     speed: float = 0.3) -> Tuple[float, float, float]:
    """Circular trajectory"""
    return (
        radius * np.cos(t * speed),
        radius * np.sin(t * speed),
        altitude
    )

def trajectory_figure8(t: float, scale: float = 10.0, altitude: float = 5.0,
                      speed: float = 0.2) -> Tuple[float, float, float]:
    """Figure-8 (lemniscate) trajectory"""
    return (
        scale * np.sin(t * speed),
        scale * np.sin(t * speed * 2) / 2,
        altitude
    )

def trajectory_helix(t: float, radius: float = 8.0, climb_rate: float = 0.3,
                    speed: float = 0.4) -> Tuple[float, float, float]:
    """Helical (spiral) trajectory"""
    return (
        radius * np.cos(t * speed),
        radius * np.sin(t * speed),
        3.0 + (t * climb_rate) % 10  # 3-13m altitude
    )

def trajectory_square(t: float, size: float = 15.0, altitude: float = 5.0,
                     speed: float = 0.3) -> Tuple[float, float, float]:
    """Square trajectory"""
    period = 4.0 / speed
    phase = (t % period) / period * 4  # 0-4
    
    if phase < 1:
        return (size/2 - phase * size, -size/2, altitude)
    elif phase < 2:
        return (-size/2, -size/2 + (phase-1) * size, altitude)
    elif phase < 3:
        return (-size/2 + (phase-2) * size, size/2, altitude)
    else:
        return (size/2, size/2 - (phase-3) * size, altitude)


# ============ TEST ============

def test_gpu_controller():
    """Test GPU controller performance"""
    print("\n" + "="*60)
    print("  GPU SWARM CONTROLLER TEST")
    print("="*60 + "\n")
    
    # Create controller
    config = SwarmConfig(num_drones=25)
    controller = GPUSwarmController(config)
    
    # Initialize random positions
    init_pos = np.random.uniform(-10, 10, (config.num_drones, 3)).astype(np.float32)
    init_pos[:, 2] = np.abs(init_pos[:, 2]) + 2  # Positive altitude
    controller.update_positions(init_pos)
    
    # Arm and set formation
    controller.arm_all()
    controller.takeoff(altitude=5.0)
    controller.active[:] = True
    
    # Run benchmark
    print("Running 1000 control cycles...")
    
    t_start = time.time()
    for _ in range(1000):
        # Simulate slight position changes
        if GPU_AVAILABLE:
            controller.positions += cp.random.uniform(-0.01, 0.01, (config.num_drones, 3), dtype=cp.float32)
        else:
            controller.positions += np.random.uniform(-0.01, 0.01, (config.num_drones, 3)).astype(np.float32)
        
        # Compute control
        cmd = controller.compute_control()
    
    t_total = time.time() - t_start
    
    print(f"\nResults:")
    print(f"  Total time: {t_total:.3f}s")
    print(f"  Per cycle: {t_total/1000*1000:.3f}ms")
    print(f"  Rate: {1000/t_total:.0f} Hz")
    print(f"  Min distance: {controller.get_min_distance():.2f}m")
    
    # Test formations
    print("\nTesting formations...")
    controller.set_formation_grid((0, 0, 5), spacing=3.0)
    for _ in range(100):
        controller.compute_control()
    print(f"  Grid min dist: {controller.get_min_distance():.2f}m")
    
    controller.set_formation_circle((0, 0, 5), radius=12.0)
    for _ in range(100):
        controller.compute_control()
    print(f"  Circle min dist: {controller.get_min_distance():.2f}m")
    
    print("\n✓ GPU controller test complete!")


if __name__ == "__main__":
    test_gpu_controller()
