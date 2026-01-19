#!/usr/bin/env python3
"""
SWARM MANAGER
=============
Main orchestrator for GPU-accelerated swarm control with Gazebo.
Coordinates:
- GPU controller (collision avoidance, trajectories)
- Gazebo bridge (physics simulation)
- 2D Visualizer (monitoring)
"""

import numpy as np
import time
import threading
import signal
import sys
import os
from typing import Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_controller import (
    GPUSwarmController, SwarmConfig,
    trajectory_circle, trajectory_figure8, trajectory_helix, trajectory_square
)
from gz_bridge import GazeboBridge


@dataclass
class SwarmManagerConfig:
    """Manager configuration"""
    num_drones: int = 5
    control_rate: float = 30.0      # Hz - control loop rate
    pose_update_rate: float = 20.0  # Hz - pose from Gazebo
    
    # Gazebo
    world_file: str = "swarm_world.sdf"
    drone_spacing: float = 3.0
    
    # Auto-start options
    auto_start_gazebo: bool = True
    auto_spawn_drones: bool = True
    
    # Visualization
    enable_2d_viz: bool = True
    viz_update_rate: float = 10.0


class SwarmManager:
    """
    Main swarm manager - coordinates all components.
    """
    
    def __init__(self, config: SwarmManagerConfig = None):
        self.config = config or SwarmManagerConfig()
        
        # Components
        self.controller: Optional[GPUSwarmController] = None
        self.bridge: Optional[GazeboBridge] = None
        self.visualizer = None
        self.visualizer_thread: Optional[threading.Thread] = None
        
        # State
        self.running = False
        self.control_thread: Optional[threading.Thread] = None
        
        # Stats
        self.loop_count = 0
        self.start_time = 0.0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n[MANAGER] Shutdown signal received...")
        self.stop()
    
    def initialize(self) -> bool:
        """Initialize all components"""
        print("\n" + "="*60)
        print("  SWARM MANAGER INITIALIZATION")
        print("="*60 + "\n")
        
        cfg = self.config
        
        # 1. Create GPU controller
        print("[1/4] Initializing GPU controller...")
        swarm_cfg = SwarmConfig(
            num_drones=cfg.num_drones,
            control_rate=cfg.control_rate
        )
        self.controller = GPUSwarmController(swarm_cfg)
        print(f"  Controller ready for {cfg.num_drones} drones")
        
        # 2. Create Gazebo bridge
        print("\n[2/4] Initializing Gazebo bridge...")
        self.bridge = GazeboBridge(
            num_drones=cfg.num_drones,
            world_name="swarm_world"
        )
        
        # 3. Start Gazebo (if needed)
        if cfg.auto_start_gazebo:
            print("\n[3/4] Starting Gazebo headless...")
            world_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                cfg.world_file
            )
            
            if not os.path.exists(world_path):
                print(f"  WARNING: World file not found: {world_path}")
                print("  Please start Gazebo manually:")
                print(f"    gz sim -s {cfg.world_file} --headless-rendering")
                cfg.auto_start_gazebo = False
            else:
                if not self.bridge.start_gazebo_headless(world_path):
                    print("  Gazebo auto-start failed. Please start manually.")
                    cfg.auto_start_gazebo = False
        else:
            print("\n[3/4] Skipping Gazebo auto-start (manual mode)")
        
        # 4. Spawn drones
        if cfg.auto_spawn_drones:
            print("\n[4/4] Spawning drone swarm...")
            spawned = self.bridge.spawn_swarm_grid(spacing=cfg.drone_spacing)
            
            if spawned < cfg.num_drones:
                print(f"  WARNING: Only {spawned}/{cfg.num_drones} drones spawned")
            
            # Enable motors
            self.bridge.enable_all_drones()
            
            # Initialize controller positions
            init_pos = self.bridge.get_all_positions()
            self.controller.update_positions(init_pos)
        else:
            print("\n[4/4] Skipping drone spawn (manual mode)")
        
        # Start pose updates
        self.bridge.start_pose_updates(rate=cfg.pose_update_rate)
        
        print("\n" + "="*60)
        print("  INITIALIZATION COMPLETE")
        print("="*60 + "\n")
        
        return True
    
    def _control_loop(self):
        """Main control loop - runs in separate thread"""
        interval = 1.0 / self.config.control_rate

        # Check if any drones are connected to Gazebo
        gazebo_connected = any(d.connected for d in self.bridge.drones.values())
        if not gazebo_connected:
            print("[MANAGER] No Gazebo connection - running in SIMULATION mode")

        while self.running:
            t_start = time.time()

            if gazebo_connected:
                # Get positions from Gazebo
                positions = self.bridge.get_all_positions()
                self.controller.update_positions(positions)

            # Compute control (GPU)
            cmd_velocities = self.controller.compute_control()

            if gazebo_connected:
                # Send commands to Gazebo
                self.bridge.send_velocity_commands_batch(cmd_velocities)
            else:
                # SIMULATION MODE: Update positions based on velocities
                xp = self.controller.xp
                dt = interval
                # Get current positions
                if hasattr(self.controller.positions, 'get'):
                    pos = self.controller.positions.get()
                    vel = self.controller.cmd_velocities.get()
                else:
                    pos = np.array(self.controller.positions)
                    vel = np.array(self.controller.cmd_velocities)

                # Simple physics: position += velocity * dt
                pos = pos + vel * dt
                # Ground constraint
                pos[:, 2] = np.maximum(pos[:, 2], 0.0)
                # Update controller
                self.controller.update_positions(pos)

            # Stats
            self.loop_count += 1

            # Rate limiting
            elapsed = time.time() - t_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def start(self):
        """Start the control loop"""
        if self.running:
            print("[MANAGER] Already running")
            return
        
        print("[MANAGER] Starting control loop...")
        self.running = True
        self.start_time = time.time()
        self.loop_count = 0
        
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        print(f"[MANAGER] Control loop running at {self.config.control_rate}Hz")
    
    def stop(self):
        """Stop all operations"""
        print("[MANAGER] Stopping...")
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        if self.visualizer:
            self.visualizer.running = False
        
        if self.bridge:
            self.bridge.shutdown()
        
        # Print stats
        if self.loop_count > 0:
            elapsed = time.time() - self.start_time
            avg_rate = self.loop_count / elapsed
            print(f"[MANAGER] Stats: {self.loop_count} loops in {elapsed:.1f}s ({avg_rate:.1f}Hz)")
        
        print("[MANAGER] Stopped")
    
    # ============ COMMANDS ============
    
    def takeoff(self, altitude: float = 5.0):
        """Command all drones to takeoff"""
        if not self.controller:
            print("[MANAGER] No controller initialized")
            return

        # Auto-start control loop if not running
        if not self.running:
            print("[MANAGER] Starting control loop first...")
            self.start()

        self.controller.takeoff(altitude)
    
    def land(self):
        """Command all drones to land"""
        if self.controller:
            self.controller.land()
    
    def formation_grid(self, spacing: float = 3.0):
        """Set grid formation"""
        if self.controller:
            self.controller.set_formation_grid((0, 0, 5), spacing=spacing)
    
    def formation_circle(self, radius: float = 12.0):
        """Set circle formation"""
        if self.controller:
            self.controller.set_formation_circle((0, 0, 5), radius=radius)
    
    def formation_v(self, spacing: float = 2.5):
        """Set V formation"""
        if self.controller:
            self.controller.set_formation_v((10, 0, 5), spacing=spacing)
    
    def trajectory_start(self, traj_type: str = "circle"):
        """Start trajectory following"""
        if not self.controller:
            return
        
        traj_map = {
            "circle": trajectory_circle,
            "figure8": trajectory_figure8,
            "helix": trajectory_helix,
            "square": trajectory_square
        }
        
        if traj_type in traj_map:
            self.controller.follow_trajectory(traj_map[traj_type])
        else:
            print(f"Unknown trajectory: {traj_type}")
            print(f"Available: {list(traj_map.keys())}")
    
    def trajectory_stop(self):
        """Stop trajectory following"""
        if self.controller:
            self.controller.stop_trajectory()
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        stats = {
            'running': self.running,
            'loop_count': self.loop_count,
            'uptime': time.time() - self.start_time if self.start_time else 0,
        }
        
        if self.controller:
            stats['min_distance'] = self.controller.get_min_distance()
            stats['center'] = self.controller.get_center().tolist()
            stats['compute_rate'] = self.controller.get_compute_rate()
        
        return stats
    
    def print_status(self):
        """Print current status"""
        stats = self.get_stats()

        print("\n" + "-"*40)
        print("SWARM STATUS")
        print("-"*40)
        print(f"  Control Loop: {'RUNNING' if stats['running'] else 'STOPPED'}")
        print(f"  Loops: {stats['loop_count']}")
        print(f"  Uptime: {stats['uptime']:.1f}s")

        if self.controller:
            xp = self.controller.xp
            armed_count = int(xp.sum(self.controller.armed))
            active_count = int(xp.sum(self.controller.active))
            print(f"  Armed: {armed_count}/{self.config.num_drones}")
            print(f"  Active: {active_count}/{self.config.num_drones}")

        if 'min_distance' in stats:
            print(f"  Min Distance: {stats['min_distance']:.2f}m")
            print(f"  Center: ({stats['center'][0]:.1f}, {stats['center'][1]:.1f}, {stats['center'][2]:.1f})")
            print(f"  Compute Rate: {stats['compute_rate']:.0f}Hz")
        print("-"*40 + "\n")


def run_interactive(num_drones: int):
    """Interactive CLI mode"""
    print("\n" + "="*60)
    print("  GPU SWARM MANAGER - INTERACTIVE MODE")
    print("="*60)

    # Config
    config = SwarmManagerConfig(
        num_drones=num_drones,
        auto_start_gazebo=False,  # Manual Gazebo start recommended
        auto_spawn_drones=False
    )

    manager = SwarmManager(config)

    print("\nNOTE: Start Gazebo manually first:")
    print("  cd deneme7")
    print("  gz sim -s swarm_world.sdf --headless-rendering\n")

    # Initialize
    manager.initialize()

    # Set initial grid positions so drones are visible in visualizer immediately
    if manager.controller:
        spacing = 3.0
        cols = int(np.ceil(np.sqrt(num_drones)))
        positions = np.zeros((num_drones, 3), dtype=np.float32)
        for i in range(num_drones):
            row, col = divmod(i, cols)
            positions[i] = [
                (col - cols / 2 + 0.5) * spacing,
                (row - cols / 2 + 0.5) * spacing,
                0.0  # Ground level
            ]
        manager.controller.update_positions(positions)
        xp = manager.controller.xp
        manager.controller.targets = xp.asarray(positions, dtype=xp.float32)
        manager.controller.initial_positions = xp.asarray(positions, dtype=xp.float32)
        print(f"[INIT] Drones positioned in {cols}x{cols} grid")
    
    # Simple CLI
    print("\nCommands:")
    print("  spawn    - Spawn drones in Gazebo")
    print("  start    - Start control loop")
    print("  stop     - Stop control loop")
    print("  takeoff [alt] - Takeoff (auto-arms, default 5m)")
    print("  land     - Land all drones")
    print("  arm      - Arm drones manually")
    print("  disarm   - Disarm all drones")
    print("  grid     - Grid formation")
    print("  circle   - Circle formation")
    print("  v        - V formation")
    print("  traj <type> - Trajectory (circle/figure8/helix/square)")
    print("  notraj   - Stop trajectory")
    print("  status   - Show status")
    print("  quit     - Exit")
    print()
    print("Typical flow: spawn -> start -> takeoff -> [formations] -> land")

    stop_event = threading.Event()

    def cli_loop():
        try:
            while not stop_event.is_set():
                try:
                    cmd = input("swarm> ").strip().lower()
                except EOFError:
                    break

                if not cmd:
                    continue

                parts = cmd.split()
                action = parts[0]

                if action == "quit" or action == "exit" or action == "q":
                    stop_event.set()
                    break
                elif action == "start":
                    manager.start()
                elif action == "stop":
                    manager.running = False
                elif action == "takeoff":
                    alt = float(parts[1]) if len(parts) > 1 else 5.0
                    manager.takeoff(alt)
                elif action == "land":
                    manager.land()
                elif action == "grid":
                    spacing = float(parts[1]) if len(parts) > 1 else 3.0
                    manager.formation_grid(spacing)
                elif action == "circle":
                    radius = float(parts[1]) if len(parts) > 1 else 12.0
                    manager.formation_circle(radius)
                elif action == "v":
                    manager.formation_v()
                elif action == "traj":
                    traj_type = parts[1] if len(parts) > 1 else "circle"
                    manager.trajectory_start(traj_type)
                elif action == "notraj":
                    manager.trajectory_stop()
                elif action == "status":
                    manager.print_status()
                elif action == "spawn":
                    spacing = float(parts[1]) if len(parts) > 1 else 3.0
                    manager.bridge.spawn_swarm_grid(spacing=spacing)
                    manager.bridge.enable_all_drones()
                    if manager.controller:
                        cols = int(np.ceil(np.sqrt(manager.controller.n)))
                        positions = np.zeros((manager.controller.n, 3), dtype=np.float32)
                        for i in range(manager.controller.n):
                            row, col = divmod(i, cols)
                            positions[i] = [
                                (col - cols / 2) * spacing,
                                (row - cols / 2) * spacing,
                                0.2,
                            ]
                        manager.controller.update_positions(positions)
                        xp = manager.controller.xp
                        manager.controller.targets = xp.asarray(positions, dtype=xp.float32)
                        # Store initial positions for visualization
                        manager.controller.initial_positions = xp.asarray(positions, dtype=xp.float32)
                    print(f"  Drones spawned. Ready for: start, takeoff")
                elif action == "arm":
                    if manager.controller:
                        manager.controller.arm_all()
                elif action == "disarm":
                    if manager.controller:
                        manager.controller.disarm_all()
                else:
                    print(f"Unknown command: {action}")

        except KeyboardInterrupt:
            pass

    if config.enable_2d_viz:
        try:
            from visualizer_2d import SwarmVisualizer2D
            manager.visualizer = SwarmVisualizer2D(num_drones=config.num_drones)
            print("  2D visualizer started")
        except Exception as exc:
            print(f"  WARNING: 2D visualizer not started: {exc}")
            manager.visualizer = None

    if manager.visualizer:
        cli_thread = threading.Thread(target=cli_loop, daemon=True)
        cli_thread.start()
        try:
            manager.visualizer.run_standalone(manager.controller)
        except Exception as exc:
            print(f"[VIZ] Error: {exc}")
        stop_event.set()
    else:
        cli_loop()

    manager.stop()
    print("\nGoodbye!")


def run_demo():
    """Automated demo sequence"""
    print("\n" + "="*60)
    print("  GPU SWARM DEMO")
    print("="*60)
    
    config = SwarmManagerConfig(
        num_drones=25,
        auto_start_gazebo=True,
        auto_spawn_drones=True
    )
    
    manager = SwarmManager(config)
    
    if not manager.initialize():
        print("Initialization failed!")
        return
    
    try:
        # Start control
        manager.start()
        time.sleep(2)
        
        # Demo sequence
        print("\n[DEMO] Takeoff...")
        manager.takeoff(5.0)
        time.sleep(5)
        
        print("[DEMO] Grid formation...")
        manager.formation_grid(3.0)
        time.sleep(5)
        manager.print_status()
        
        print("[DEMO] Circle formation...")
        manager.formation_circle(12.0)
        time.sleep(5)
        manager.print_status()
        
        print("[DEMO] Trajectory: circle...")
        manager.trajectory_start("circle")
        time.sleep(10)
        manager.print_status()
        
        print("[DEMO] Trajectory: figure-8...")
        manager.trajectory_start("figure8")
        time.sleep(10)
        manager.print_status()
        
        print("[DEMO] Landing...")
        manager.trajectory_stop()
        manager.land()
        time.sleep(5)
        
        print("\n[DEMO] Complete!")
        
    except KeyboardInterrupt:
        print("\n[DEMO] Interrupted")
    
    manager.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Swarm Manager")
    parser.add_argument("--demo", action="store_true", help="Run automated demo")
    parser.add_argument("--drones", type=int, default=25, help="Number of drones")
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    else:
        run_interactive(args.drones)
