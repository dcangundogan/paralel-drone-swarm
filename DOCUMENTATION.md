<<<<<<< HEAD
# GPU-Accelerated Drone Swarm Control System

## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Drone Hardware & Mechanics](#3-drone-hardware--mechanics)
4. [Control Algorithms](#4-control-algorithms)
5. [Software Components](#5-software-components)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Usage Guide](#7-usage-guide)
8. [Performance Characteristics](#8-performance-characteristics)

---

## 1. Project Overview

### What is This Project?

This is a **GPU-accelerated drone swarm control system** that simulates and controls multiple quadcopter drones simultaneously. The system uses CUDA/GPU acceleration to compute swarm behaviors in real-time, enabling coordination of 25+ drones with collision avoidance, formation flying, and trajectory following.

### Key Features

- **GPU Acceleration**: Uses CuPy/CUDA for parallel computation of swarm algorithms
- **Gazebo Simulation**: Realistic physics simulation using Gazebo Ignition with DART physics engine
- **Headless Mode**: Supports running without GUI for maximum performance
- **Real-time 2D Visualization**: Pygame-based top-down view of swarm operations
- **Multiple Formations**: Grid, Circle, and V-formation support
- **Trajectory Following**: Circle, Figure-8, Helix, and Square trajectories
- **Collision Avoidance**: Reynolds-based separation algorithm running in parallel on GPU

### Project Structure

```
deneme7/
├── gpu_controller.py      # GPU-accelerated swarm algorithms (597 lines)
├── gz_bridge.py           # Gazebo simulator interface (476 lines)
├── swarm_manager.py       # Main orchestrator/manager (569 lines)
├── visualizer_2d.py       # 2D real-time visualization (278 lines)
├── run_swarm.py           # Integrated system launcher
├── run_swarm.sh           # Bash wrapper with CUDA setup
├── run_gazebo_2d.py       # Gazebo + 2D GUI launcher
├── quickstart.sh          # Quick start script
├── model.sdf              # Quadcopter drone model definition
├── swarm_world.sdf        # Gazebo world configuration
└── models/quadcopter/     # Model directory
```

---

## 2. System Architecture

### Three-Layer Control Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          LAYER 1: GPU-Accelerated Control (CuPy/CUDA)       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Collision  │ │  Trajectory │ │  Formation Control  │   │
│  │  Avoidance  │ │  Following  │ │  (Cohesion/Align)   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                    ↓ Velocity Commands                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          LAYER 2: Gazebo Bridge (gz-transport)              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Spawn/Delete│ │  Velocity   │ │   Pose Updates      │   │
│  │   Drones    │ │  Commands   │ │   (Feedback)        │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          LAYER 3: Physics Simulation (Gazebo + DART)        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Headless  │ │   Physics   │ │   State Tracking    │   │
│  │  Rendering  │ │ Computation │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Commands (takeoff, formation, trajectory)
         ↓
    SwarmManager (Orchestrator)
         ↓
    GPUSwarmController (Compute velocities on GPU)
         ↓
    GazeboBridge (Send velocity commands)
         ↓
    Gazebo Simulation (Physics execution)
         ↓
    Position Feedback (20 Hz)
         ↓
    GPUSwarmController (Update state)
         ↓
    Visualizer2D (Display to user)
```

---

## 3. Drone Hardware & Mechanics

### Drone Type: Quadcopter (4-Rotor UAV)

### Physical Specifications

| Property | Value |
|----------|-------|
| **Total Mass** | 1.5 kg |
| **Body Dimensions** | 0.4 x 0.4 x 0.1 m (box) |
| **Arm Length** | 0.3 m per arm |
| **Motor Size** | 0.05 m radius cylinders |
| **Inertia (Ixx, Iyy)** | 0.015 kg·m² |
| **Inertia (Izz)** | 0.025 kg·m² |
| **Initial Height** | 0.2 m above ground |

### Motor Configuration

```
        Front
          │
    ┌─────┼─────┐
    │  M2 │ M1  │    M1: Front-Right (+0.2, +0.2)  [Red]
    │  ●  │  ●  │    M2: Front-Left  (-0.2, +0.2)  [Green]
    ├─────┼─────┤    M3: Back-Left   (-0.2, -0.2)  [Red]
    │  ●  │  ●  │    M4: Back-Right  (+0.2, -0.2)  [Green]
    │  M3 │ M4  │
    └─────┴─────┘
```

### Control Interface

- **Control Type**: Velocity Control (not direct motor RPM)
- **Plugin**: `gz-sim-velocity-control-system`
- **Input**: Linear velocity vectors (vx, vy, vz) + angular velocity
- **Output**: Pose published at 30 Hz via `PosePublisher` plugin

### Physics Engine

- **Engine**: DART (Dynamic Animation and Robotics Toolkit)
- **Time Step**: 1 ms (0.001 seconds)
- **Real-time Factor**: 1.0

---

## 4. Control Algorithms

### 4.1 Reynolds Boids-Based Swarm Control

The system implements a modified Reynolds flocking algorithm with four main behaviors:

#### A. Separation (Collision Avoidance)

**Purpose**: Prevent drones from colliding with each other

**Algorithm**:
```
For each drone pair (i, j):
    distance = ||position_i - position_j||

    if distance < avoidance_radius (2.5m):
        direction = normalize(position_i - position_j)  # Away from neighbor

        if distance < collision_radius (0.8m):
            strength = avoidance_strength (3.0)  # Maximum repulsion
        else:
            # Linear falloff between collision and avoidance radius
            strength = (avoidance_radius - distance) / (avoidance_radius - collision_radius)

        separation_velocity += direction * strength
```

**Key Parameters**:
- Collision Radius: 0.8 m (hard boundary)
- Avoidance Radius: 2.5 m (soft repulsion starts)
- Avoidance Strength: 3.0x multiplier

**GPU Implementation**: O(n²) pairwise distances computed in parallel using CuPy

#### B. Cohesion (Swarm Unity)

**Purpose**: Keep the swarm unified by attracting drones toward the center

**Algorithm**:
```
swarm_center = mean(all_drone_positions)
for each drone:
    direction = normalize(swarm_center - drone_position)
    magnitude = min(distance_to_center, 1.0)
    cohesion_velocity = direction * magnitude * cohesion_strength (0.3)
```

#### C. Alignment (Velocity Matching)

**Purpose**: Smooth, coordinated swarm movement

**Algorithm**:
```
average_velocity = mean(all_drone_velocities)
for each drone:
    alignment_velocity = (average_velocity - drone_velocity) * alignment_strength (0.2)
```

#### D. Target Tracking (PD Controller)

**Purpose**: Guide drones to their target positions smoothly

**Algorithm**:
```
position_error = target_position - current_position
derivative_error = (position_error - previous_error) / dt

tracking_velocity = error * P_gain (1.5) + d_error * D_gain (0.3)
```

### 4.2 Velocity Blending

All behaviors are combined using weighted blending:

```
Final Velocity = 2.0 × Separation     (Highest Priority - Safety)
               + 1.5 × Tracking       (Goal Achievement)
               + 0.5 × Cohesion       (Swarm Unity)
               + 0.3 × Alignment      (Smoothing)
```

### 4.3 Velocity Limiting

After blending, velocities are constrained:

| Limit | Value |
|-------|-------|
| Max XY Velocity | 3.0 m/s |
| Max Z Velocity | 2.0 m/s |
| Max Acceleration | 2.0 m/s² |

### 4.4 Formation Algorithms

#### Grid Formation
```
columns = ceil(sqrt(num_drones))
for drone i:
    row = i // columns
    col = i % columns
    target_x = col * spacing
    target_y = row * spacing
    target_z = default_altitude
```

#### Circle Formation
```
for drone i:
    angle = (2 * π / num_drones) * i
    target_x = center_x + radius * cos(angle)
    target_y = center_y + radius * sin(angle)
    target_z = default_altitude
```

#### V-Formation (Bird-like)
```
for drone i:
    if i == 0:
        position = tip_position  # Lead drone
    else:
        side = (i % 2) * 2 - 1   # Alternating left/right
        depth = (i + 1) // 2      # Increasing depth
        target_x = tip_x - depth * spacing * cos(angle)
        target_y = tip_y + side * depth * spacing * sin(angle)
```

### 4.5 Trajectory Functions

| Trajectory | Mathematical Function |
|------------|----------------------|
| **Circle** | x = radius × cos(t × speed), y = radius × sin(t × speed) |
| **Figure-8** | x = sin(t), y = sin(2t) / 2 (Lemniscate) |
| **Helix** | x = radius × cos(t), y = radius × sin(t), z = 3 + climb_rate × t |
| **Square** | Parameterized 4-segment path |

---

## 5. Software Components

### 5.1 gpu_controller.py - GPU-Accelerated Swarm Controller

**Main Class**: `GPUSwarmController`

**Responsibilities**:
- Maintains drone positions, velocities, and targets on GPU arrays
- Computes all swarm behaviors in parallel
- Handles formation and trajectory generation
- Provides arm/disarm, takeoff/land commands

**Key Methods**:
```python
compute_control()           # Main control computation (50 Hz)
update_positions()          # Receive position updates from Gazebo
takeoff(altitude)           # Auto-arm and set takeoff targets
land()                      # Set landing targets
set_formation_grid()        # Grid formation
set_formation_circle()      # Circle formation
set_formation_v()           # V formation
follow_trajectory()         # Enable trajectory following
```

**GPU Detection**:
```python
# Attempts to use CuPy (CUDA) first
# Falls back to NumPy (CPU) if CUDA unavailable
# Can detect MATLAB's CUDA installation as fallback
```

### 5.2 gz_bridge.py - Gazebo Interface

**Main Class**: `GazeboBridge`

**Responsibilities**:
- Starts Gazebo in headless mode
- Spawns drone models from SDF files
- Sends velocity commands to drones
- Receives position feedback
- Manages simulation lifecycle

**Communication Protocol**:
- **Velocity Commands**: `/model/drone_{id}/cmd_vel` (Twist messages)
- **Pose Feedback**: `/model/drone_{id}/pose` (Pose messages)
- **Update Rate**: 20 Hz pose updates, command rate as needed

### 5.3 swarm_manager.py - Main Orchestrator

**Main Class**: `SwarmManager`

**Responsibilities**:
- Coordinates GPU controller and Gazebo bridge
- Runs main control loop (30 Hz)
- Provides interactive CLI interface
- Manages visualization
- Handles startup/shutdown

**Operating Modes**:
1. **Interactive CLI**: User types commands at prompt
2. **Demo Mode**: Automated sequence demonstration
3. **Pure Simulation**: CPU-only mode without Gazebo

### 5.4 visualizer_2d.py - Real-time Visualization

**Main Class**: `SimpleSwarmVisualizer`

**Features**:
- Pygame-based 2D top-down view
- Altitude-based color coding
- Velocity vector arrows
- Target position indicators
- Collision risk highlighting (red when < 1.5m)
- Interactive pan/zoom controls

---

## 6. Configuration & Parameters

### SwarmConfig (GPU Controller)

```python
@dataclass
class SwarmConfig:
    num_drones: int = 25
    control_rate: float = 50.0          # Hz

    # Velocity Limits
    max_velocity_xy: float = 3.0        # m/s
    max_velocity_z: float = 2.0         # m/s
    max_acceleration: float = 2.0       # m/s²

    # Collision Avoidance
    collision_radius: float = 0.8       # m
    avoidance_radius: float = 2.5       # m
    avoidance_strength: float = 3.0

    # Swarm Behavior
    cohesion_strength: float = 0.3
    alignment_strength: float = 0.2
    target_gain_p: float = 1.5          # PD proportional
    target_gain_d: float = 0.3          # PD derivative

    # Altitude
    default_altitude: float = 5.0       # m
    altitude_tolerance: float = 0.3     # m
```

### SwarmManagerConfig

```python
@dataclass
class SwarmManagerConfig:
    num_drones: int = 5
    control_rate: float = 30.0          # Hz
    pose_update_rate: float = 20.0      # Hz
    auto_start_gazebo: bool = True
    auto_spawn_drones: bool = True
    enable_2d_viz: bool = True
    viz_update_rate: float = 10.0       # Hz
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing (CPU fallback) |
| cupy | CUDA GPU acceleration |
| pygame | 2D visualization |
| psutil | Resource monitoring (optional) |

### System Requirements

- **Python**: 3.10+
- **Gazebo**: Ignition (gz-sim)
- **CUDA**: 12.x (for GPU acceleration)
- **OS**: Linux recommended (Windows with WSL2)

---

## 7. Usage Guide

### Quick Start

```bash
# Using the shell script (recommended)
./run_swarm.sh

# Or directly with Python
python3 swarm_manager.py --drones 25
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `spawn [spacing]` | Create drones in Gazebo (default spacing: 3.0m) |
| `start` | Begin control loop |
| `stop` | Pause control |
| `takeoff [alt]` | Auto-arm and ascend (default: 5.0m) |
| `land` | Descend to ground |
| `arm` | Manually arm all drones |
| `disarm` | Manually disarm all drones |
| `grid [spacing]` | Set grid formation |
| `circle [radius]` | Set circle formation |
| `v [spacing]` | Set V formation |
| `traj <type>` | Start trajectory (circle/figure8/helix/square) |
| `notraj` | Stop trajectory following |
| `status` | Show statistics |
| `quit` | Exit program |

### Visualization Controls (Pygame Window)

| Key | Action |
|-----|--------|
| SPACE | Takeoff |
| C | Circle formation |
| G | Grid formation |
| V | V formation |
| T | Toggle trajectory mode |
| L | Land |
| +/- | Zoom in/out |
| Arrows | Pan view |
| R | Reset view |
| Q/ESC | Quit |

### Example Session

```bash
$ python3 swarm_manager.py --drones 25

> spawn          # Spawn 25 drones in grid
> start          # Start control loop
> takeoff 5      # Takeoff to 5 meters
> circle 12      # Form circle with 12m radius
> traj figure8   # Follow figure-8 trajectory
> notraj         # Stop trajectory
> grid 3         # Return to grid formation
> land           # Land all drones
> quit           # Exit
```

---

## 8. Performance Characteristics

### Computational Performance

| Metric | Value |
|--------|-------|
| GPU Control Rate | 50 Hz |
| Typical Cycle Time | < 5 ms (25 drones on GPU) |
| Algorithm Complexity | O(n²) for collision (parallelized) |
| CPU Fallback | Functional but slower |

### Simulation Performance

| Metric | Value |
|--------|-------|
| Physics Step | 1 ms |
| Pose Update Rate | 30 Hz |
| Control Update Rate | 30 Hz (manager) |
| Total Latency | ~30-50 ms per command |

### Scalability

| Drone Count | GPU Performance | CPU Performance |
|-------------|-----------------|-----------------|
| 5 | < 1 ms | < 5 ms |
| 25 | < 5 ms | ~20 ms |
| 100 | ~15 ms | ~200 ms |
| 500+ | ~50 ms | Not recommended |

### Memory Usage

- **Base Memory**: ~50 MB
- **GPU Memory**: Minimal (N × 3 floats per array)
- **Scales linearly** with drone count

---

## Summary

This GPU-accelerated drone swarm system represents a sophisticated approach to multi-agent UAV control, combining:

1. **Real-time GPU computation** for scalable collision avoidance
2. **Reynolds flocking algorithms** for natural swarm behavior
3. **PD controllers** for smooth trajectory tracking
4. **Gazebo physics simulation** for realistic drone dynamics
5. **Modular architecture** with clean separation of concerns
6. **Graceful degradation** to CPU when GPU unavailable

The system is designed for research, development, and testing of swarm algorithms with support for 5-100+ drones in coordinated flight scenarios.

---

*Documentation generated for the GPU-Accelerated Drone Swarm project*
=======
# GPU-Accelerated Drone Swarm Control System

## Complete Technical Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Drone Hardware & Mechanics](#3-drone-hardware--mechanics)
4. [Control Algorithms](#4-control-algorithms)
5. [Software Components](#5-software-components)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Usage Guide](#7-usage-guide)
8. [Performance Characteristics](#8-performance-characteristics)

---

## 1. Project Overview

### What is This Project?

This is a **GPU-accelerated drone swarm control system** that simulates and controls multiple quadcopter drones simultaneously. The system uses CUDA/GPU acceleration to compute swarm behaviors in real-time, enabling coordination of 25+ drones with collision avoidance, formation flying, and trajectory following.

### Key Features

- **GPU Acceleration**: Uses CuPy/CUDA for parallel computation of swarm algorithms
- **Gazebo Simulation**: Realistic physics simulation using Gazebo Ignition with DART physics engine
- **Headless Mode**: Supports running without GUI for maximum performance
- **Real-time 2D Visualization**: Pygame-based top-down view of swarm operations
- **Multiple Formations**: Grid, Circle, and V-formation support
- **Trajectory Following**: Circle, Figure-8, Helix, and Square trajectories
- **Collision Avoidance**: Reynolds-based separation algorithm running in parallel on GPU

### Project Structure

```
deneme7/
├── gpu_controller.py      # GPU-accelerated swarm algorithms (597 lines)
├── gz_bridge.py           # Gazebo simulator interface (476 lines)
├── swarm_manager.py       # Main orchestrator/manager (569 lines)
├── visualizer_2d.py       # 2D real-time visualization (278 lines)
├── run_swarm.py           # Integrated system launcher
├── run_swarm.sh           # Bash wrapper with CUDA setup
├── run_gazebo_2d.py       # Gazebo + 2D GUI launcher
├── quickstart.sh          # Quick start script
├── model.sdf              # Quadcopter drone model definition
├── swarm_world.sdf        # Gazebo world configuration
└── models/quadcopter/     # Model directory
```

---

## 2. System Architecture

### Three-Layer Control Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          LAYER 1: GPU-Accelerated Control (CuPy/CUDA)       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │  Collision  │ │  Trajectory │ │  Formation Control  │   │
│  │  Avoidance  │ │  Following  │ │  (Cohesion/Align)   │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
│                    ↓ Velocity Commands                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          LAYER 2: Gazebo Bridge (gz-transport)              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │ Spawn/Delete│ │  Velocity   │ │   Pose Updates      │   │
│  │   Drones    │ │  Commands   │ │   (Feedback)        │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│          LAYER 3: Physics Simulation (Gazebo + DART)        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │   Headless  │ │   Physics   │ │   State Tracking    │   │
│  │  Rendering  │ │ Computation │ │                     │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Commands (takeoff, formation, trajectory)
         ↓
    SwarmManager (Orchestrator)
         ↓
    GPUSwarmController (Compute velocities on GPU)
         ↓
    GazeboBridge (Send velocity commands)
         ↓
    Gazebo Simulation (Physics execution)
         ↓
    Position Feedback (20 Hz)
         ↓
    GPUSwarmController (Update state)
         ↓
    Visualizer2D (Display to user)
```

---

## 3. Drone Hardware & Mechanics

### Drone Type: Quadcopter (4-Rotor UAV)

### Physical Specifications

| Property | Value |
|----------|-------|
| **Total Mass** | 1.5 kg |
| **Body Dimensions** | 0.4 x 0.4 x 0.1 m (box) |
| **Arm Length** | 0.3 m per arm |
| **Motor Size** | 0.05 m radius cylinders |
| **Inertia (Ixx, Iyy)** | 0.015 kg·m² |
| **Inertia (Izz)** | 0.025 kg·m² |
| **Initial Height** | 0.2 m above ground |

### Motor Configuration

```
        Front
          │
    ┌─────┼─────┐
    │  M2 │ M1  │    M1: Front-Right (+0.2, +0.2)  [Red]
    │  ●  │  ●  │    M2: Front-Left  (-0.2, +0.2)  [Green]
    ├─────┼─────┤    M3: Back-Left   (-0.2, -0.2)  [Red]
    │  ●  │  ●  │    M4: Back-Right  (+0.2, -0.2)  [Green]
    │  M3 │ M4  │
    └─────┴─────┘
```

### Control Interface

- **Control Type**: Velocity Control (not direct motor RPM)
- **Plugin**: `gz-sim-velocity-control-system`
- **Input**: Linear velocity vectors (vx, vy, vz) + angular velocity
- **Output**: Pose published at 30 Hz via `PosePublisher` plugin

### Physics Engine

- **Engine**: DART (Dynamic Animation and Robotics Toolkit)
- **Time Step**: 1 ms (0.001 seconds)
- **Real-time Factor**: 1.0

---

## 4. Control Algorithms

### 4.1 Reynolds Boids-Based Swarm Control

The system implements a modified Reynolds flocking algorithm with four main behaviors:

#### A. Separation (Collision Avoidance)

**Purpose**: Prevent drones from colliding with each other

**Algorithm**:
```
For each drone pair (i, j):
    distance = ||position_i - position_j||

    if distance < avoidance_radius (2.5m):
        direction = normalize(position_i - position_j)  # Away from neighbor

        if distance < collision_radius (0.8m):
            strength = avoidance_strength (3.0)  # Maximum repulsion
        else:
            # Linear falloff between collision and avoidance radius
            strength = (avoidance_radius - distance) / (avoidance_radius - collision_radius)

        separation_velocity += direction * strength
```

**Key Parameters**:
- Collision Radius: 0.8 m (hard boundary)
- Avoidance Radius: 2.5 m (soft repulsion starts)
- Avoidance Strength: 3.0x multiplier

**GPU Implementation**: O(n²) pairwise distances computed in parallel using CuPy

#### B. Cohesion (Swarm Unity)

**Purpose**: Keep the swarm unified by attracting drones toward the center

**Algorithm**:
```
swarm_center = mean(all_drone_positions)
for each drone:
    direction = normalize(swarm_center - drone_position)
    magnitude = min(distance_to_center, 1.0)
    cohesion_velocity = direction * magnitude * cohesion_strength (0.3)
```

#### C. Alignment (Velocity Matching)

**Purpose**: Smooth, coordinated swarm movement

**Algorithm**:
```
average_velocity = mean(all_drone_velocities)
for each drone:
    alignment_velocity = (average_velocity - drone_velocity) * alignment_strength (0.2)
```

#### D. Target Tracking (PD Controller)

**Purpose**: Guide drones to their target positions smoothly

**Algorithm**:
```
position_error = target_position - current_position
derivative_error = (position_error - previous_error) / dt

tracking_velocity = error * P_gain (1.5) + d_error * D_gain (0.3)
```

### 4.2 Velocity Blending

All behaviors are combined using weighted blending:

```
Final Velocity = 2.0 × Separation     (Highest Priority - Safety)
               + 1.5 × Tracking       (Goal Achievement)
               + 0.5 × Cohesion       (Swarm Unity)
               + 0.3 × Alignment      (Smoothing)
```

### 4.3 Velocity Limiting

After blending, velocities are constrained:

| Limit | Value |
|-------|-------|
| Max XY Velocity | 3.0 m/s |
| Max Z Velocity | 2.0 m/s |
| Max Acceleration | 2.0 m/s² |

### 4.4 Formation Algorithms

#### Grid Formation
```
columns = ceil(sqrt(num_drones))
for drone i:
    row = i // columns
    col = i % columns
    target_x = col * spacing
    target_y = row * spacing
    target_z = default_altitude
```

#### Circle Formation
```
for drone i:
    angle = (2 * π / num_drones) * i
    target_x = center_x + radius * cos(angle)
    target_y = center_y + radius * sin(angle)
    target_z = default_altitude
```

#### V-Formation (Bird-like)
```
for drone i:
    if i == 0:
        position = tip_position  # Lead drone
    else:
        side = (i % 2) * 2 - 1   # Alternating left/right
        depth = (i + 1) // 2      # Increasing depth
        target_x = tip_x - depth * spacing * cos(angle)
        target_y = tip_y + side * depth * spacing * sin(angle)
```

### 4.5 Trajectory Functions

| Trajectory | Mathematical Function |
|------------|----------------------|
| **Circle** | x = radius × cos(t × speed), y = radius × sin(t × speed) |
| **Figure-8** | x = sin(t), y = sin(2t) / 2 (Lemniscate) |
| **Helix** | x = radius × cos(t), y = radius × sin(t), z = 3 + climb_rate × t |
| **Square** | Parameterized 4-segment path |

---

## 5. Software Components

### 5.1 gpu_controller.py - GPU-Accelerated Swarm Controller

**Main Class**: `GPUSwarmController`

**Responsibilities**:
- Maintains drone positions, velocities, and targets on GPU arrays
- Computes all swarm behaviors in parallel
- Handles formation and trajectory generation
- Provides arm/disarm, takeoff/land commands

**Key Methods**:
```python
compute_control()           # Main control computation (50 Hz)
update_positions()          # Receive position updates from Gazebo
takeoff(altitude)           # Auto-arm and set takeoff targets
land()                      # Set landing targets
set_formation_grid()        # Grid formation
set_formation_circle()      # Circle formation
set_formation_v()           # V formation
follow_trajectory()         # Enable trajectory following
```

**GPU Detection**:
```python
# Attempts to use CuPy (CUDA) first
# Falls back to NumPy (CPU) if CUDA unavailable
# Can detect MATLAB's CUDA installation as fallback
```

### 5.2 gz_bridge.py - Gazebo Interface

**Main Class**: `GazeboBridge`

**Responsibilities**:
- Starts Gazebo in headless mode
- Spawns drone models from SDF files
- Sends velocity commands to drones
- Receives position feedback
- Manages simulation lifecycle

**Communication Protocol**:
- **Velocity Commands**: `/model/drone_{id}/cmd_vel` (Twist messages)
- **Pose Feedback**: `/model/drone_{id}/pose` (Pose messages)
- **Update Rate**: 20 Hz pose updates, command rate as needed

### 5.3 swarm_manager.py - Main Orchestrator

**Main Class**: `SwarmManager`

**Responsibilities**:
- Coordinates GPU controller and Gazebo bridge
- Runs main control loop (30 Hz)
- Provides interactive CLI interface
- Manages visualization
- Handles startup/shutdown

**Operating Modes**:
1. **Interactive CLI**: User types commands at prompt
2. **Demo Mode**: Automated sequence demonstration
3. **Pure Simulation**: CPU-only mode without Gazebo

### 5.4 visualizer_2d.py - Real-time Visualization

**Main Class**: `SimpleSwarmVisualizer`

**Features**:
- Pygame-based 2D top-down view
- Altitude-based color coding
- Velocity vector arrows
- Target position indicators
- Collision risk highlighting (red when < 1.5m)
- Interactive pan/zoom controls

---

## 6. Configuration & Parameters

### SwarmConfig (GPU Controller)

```python
@dataclass
class SwarmConfig:
    num_drones: int = 25
    control_rate: float = 50.0          # Hz

    # Velocity Limits
    max_velocity_xy: float = 3.0        # m/s
    max_velocity_z: float = 2.0         # m/s
    max_acceleration: float = 2.0       # m/s²

    # Collision Avoidance
    collision_radius: float = 0.8       # m
    avoidance_radius: float = 2.5       # m
    avoidance_strength: float = 3.0

    # Swarm Behavior
    cohesion_strength: float = 0.3
    alignment_strength: float = 0.2
    target_gain_p: float = 1.5          # PD proportional
    target_gain_d: float = 0.3          # PD derivative

    # Altitude
    default_altitude: float = 5.0       # m
    altitude_tolerance: float = 0.3     # m
```

### SwarmManagerConfig

```python
@dataclass
class SwarmManagerConfig:
    num_drones: int = 5
    control_rate: float = 30.0          # Hz
    pose_update_rate: float = 20.0      # Hz
    auto_start_gazebo: bool = True
    auto_spawn_drones: bool = True
    enable_2d_viz: bool = True
    viz_update_rate: float = 10.0       # Hz
```

### Dependencies

| Package | Purpose |
|---------|---------|
| numpy | Numerical computing (CPU fallback) |
| cupy | CUDA GPU acceleration |
| pygame | 2D visualization |
| psutil | Resource monitoring (optional) |

### System Requirements

- **Python**: 3.10+
- **Gazebo**: Ignition (gz-sim)
- **CUDA**: 12.x (for GPU acceleration)
- **OS**: Linux recommended (Windows with WSL2)

---

## 7. Usage Guide

### Quick Start

```bash
# Using the shell script (recommended)
./run_swarm.sh

# Or directly with Python
python3 swarm_manager.py --drones 25
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `spawn [spacing]` | Create drones in Gazebo (default spacing: 3.0m) |
| `start` | Begin control loop |
| `stop` | Pause control |
| `takeoff [alt]` | Auto-arm and ascend (default: 5.0m) |
| `land` | Descend to ground |
| `arm` | Manually arm all drones |
| `disarm` | Manually disarm all drones |
| `grid [spacing]` | Set grid formation |
| `circle [radius]` | Set circle formation |
| `v [spacing]` | Set V formation |
| `traj <type>` | Start trajectory (circle/figure8/helix/square) |
| `notraj` | Stop trajectory following |
| `status` | Show statistics |
| `quit` | Exit program |

### Visualization Controls (Pygame Window)

| Key | Action |
|-----|--------|
| SPACE | Takeoff |
| C | Circle formation |
| G | Grid formation |
| V | V formation |
| T | Toggle trajectory mode |
| L | Land |
| +/- | Zoom in/out |
| Arrows | Pan view |
| R | Reset view |
| Q/ESC | Quit |

### Example Session

```bash
$ python3 swarm_manager.py --drones 25

> spawn          # Spawn 25 drones in grid
> start          # Start control loop
> takeoff 5      # Takeoff to 5 meters
> circle 12      # Form circle with 12m radius
> traj figure8   # Follow figure-8 trajectory
> notraj         # Stop trajectory
> grid 3         # Return to grid formation
> land           # Land all drones
> quit           # Exit
```

---

## 8. Performance Characteristics

### Computational Performance

| Metric | Value |
|--------|-------|
| GPU Control Rate | 50 Hz |
| Typical Cycle Time | < 5 ms (25 drones on GPU) |
| Algorithm Complexity | O(n²) for collision (parallelized) |
| CPU Fallback | Functional but slower |

### Simulation Performance

| Metric | Value |
|--------|-------|
| Physics Step | 1 ms |
| Pose Update Rate | 30 Hz |
| Control Update Rate | 30 Hz (manager) |
| Total Latency | ~30-50 ms per command |

### Scalability

| Drone Count | GPU Performance | CPU Performance |
|-------------|-----------------|-----------------|
| 5 | < 1 ms | < 5 ms |
| 25 | < 5 ms | ~20 ms |
| 100 | ~15 ms | ~200 ms |
| 500+ | ~50 ms | Not recommended |

### Memory Usage

- **Base Memory**: ~50 MB
- **GPU Memory**: Minimal (N × 3 floats per array)
- **Scales linearly** with drone count

---

## Summary

This GPU-accelerated drone swarm system represents a sophisticated approach to multi-agent UAV control, combining:

1. **Real-time GPU computation** for scalable collision avoidance
2. **Reynolds flocking algorithms** for natural swarm behavior
3. **PD controllers** for smooth trajectory tracking
4. **Gazebo physics simulation** for realistic drone dynamics
5. **Modular architecture** with clean separation of concerns
6. **Graceful degradation** to CPU when GPU unavailable

The system is designed for research, development, and testing of swarm algorithms with support for 5-100+ drones in coordinated flight scenarios.

--
