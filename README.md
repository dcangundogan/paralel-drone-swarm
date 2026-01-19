# GPU-Accelerated Drone Swarm with Headless Gazebo

## Mimari Özeti

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GPU SWARM CONTROLLER (Python/CuPy)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Collision   │  │  Trajectory  │  │  Formation   │              │
│  │  Avoidance   │  │  Planner     │  │  Controller  │              │
│  │  (CUDA)      │  │  (CUDA)      │  │  (CUDA)      │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                           │                                         │
│              GPU Parallel Velocity Commands                         │
└─────────────────────────────────────────────────────────────────────┘
                            │
                    ROS 2 / Gazebo Transport
                            │
┌─────────────────────────────────────────────────────────────────────┐
│              HEADLESS GAZEBO IGNITION (gz-sim)                      │
│                     --headless-rendering                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        ┌─────────┐           │
│  │ Drone 0 │ │ Drone 1 │ │ Drone 2 │  ...   │Drone N  │           │
│  │ (SDF)   │ │ (SDF)   │ │ (SDF)   │        │ (SDF)   │           │
│  └─────────┘ └─────────┘ └─────────┘        └─────────┘           │
│                                                                     │
│  Physics: DART/Bullet (CPU) | Minimal rendering overhead            │
└─────────────────────────────────────────────────────────────────────┘
```

## Dosya Yapısı

```
gz_swarm/
├── models/
│   └── custom_drone/
│       ├── model.config
│       └── model.sdf
├── worlds/
│   └── swarm_world.sdf
├── launch/
│   └── swarm.launch.py
├── src/
│   ├── gpu_controller.py      # CUDA swarm algorithms
│   ├── gz_bridge.py           # Gazebo communication
│   ├── swarm_manager.py       # Main orchestrator
│   └── visualizer_2d.py       # Pygame 2D view
├── config/
│   └── swarm_params.yaml
├── scripts/
│   ├── spawn_swarm.sh
│   └── run_headless.sh
└── requirements.txt
```
