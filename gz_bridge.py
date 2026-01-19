#!/usr/bin/env python3
"""
GAZEBO IGNITION BRIDGE
======================
Handles communication between GPU controller and Gazebo sim.
Uses gz-transport (Python bindings) or subprocess for topics.
"""

import subprocess
import threading
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import re
import os


@dataclass
class DroneState:
    """State of a single drone from Gazebo"""
    drone_id: int
    name: str
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qw, qx, qy, qz]
    velocity: np.ndarray  # [vx, vy, vz]
    angular_velocity: np.ndarray  # [wx, wy, wz]
    connected: bool = False


class GazeboBridge:
    """
    Bridge between Python controller and Gazebo Ignition.
    Supports headless operation.
    """
    
    def __init__(self, num_drones: int = 25, world_name: str = "swarm_world"):
        self.num_drones = num_drones
        self.world_name = world_name
        
        # Drone states
        self.drones: Dict[int, DroneState] = {}
        for i in range(num_drones):
            name = f"drone_{i}"
            self.drones[i] = DroneState(
                drone_id=i,
                name=name,
                position=np.zeros(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                velocity=np.zeros(3),
                angular_velocity=np.zeros(3)
            )
        
        # State arrays for efficient batch operations
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)
        
        # Threading
        self.running = False
        self.pose_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Gazebo process
        self.gz_process: Optional[subprocess.Popen] = None
        
    def start_gazebo_headless(self, world_file: str) -> bool:
        """Start Gazebo in headless mode"""
        print("[GZ] Starting Gazebo headless...")
        
        try:
            # Kill any existing Gazebo
            subprocess.run(["pkill", "-9", "gz"], capture_output=True)
            subprocess.run(["pkill", "-9", "ruby"], capture_output=True)
            time.sleep(2)
            
            # Start headless
            cmd = [
                "gz", "sim", "-s", "-r",  # Server only, run immediately
                "--headless-rendering",
                world_file
            ]
            
            env = os.environ.copy()
            world_dir = os.path.dirname(os.path.abspath(world_file))
            existing = env.get("GZ_SIM_RESOURCE_PATH", "")
            env["GZ_SIM_RESOURCE_PATH"] = f"{world_dir}:{existing}" if existing else world_dir

            self.gz_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Wait for startup
            print("[GZ] Waiting for Gazebo to initialize...")
            time.sleep(5)
            
            # Check if running
            if self.gz_process.poll() is not None:
                print("[GZ] ERROR: Gazebo failed to start")
                stderr = self.gz_process.stderr.read().decode()
                print(f"  {stderr[:500]}")
                return False
            
            print("[GZ] Gazebo headless started successfully")
            return True
            
        except Exception as e:
            print(f"[GZ] ERROR starting Gazebo: {e}")
            return False
    
    def spawn_drone(self, drone_id: int, x: float, y: float, z: float = 0.2) -> bool:
        """Spawn a single drone at position"""
        name = f"drone_{drone_id}"
        
        # Use gz service to spawn
        sdf_path = "model.sdf"
        
        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000",
            "--req", f'sdf_filename: "{sdf_path}", name: "{name}", pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                self.drones[drone_id].connected = True
                self.drones[drone_id].position = np.array([x, y, z])
                self.positions[drone_id] = np.array([x, y, z], dtype=np.float32)
                return True
            else:
                print(f"[GZ] Failed to spawn {name}: {result.stderr.decode()[:200]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"[GZ] Timeout spawning {name}")
            return False
        except Exception as e:
            print(f"[GZ] Error spawning {name}: {e}")
            return False
    
    def spawn_swarm_grid(self, spacing: float = 3.0) -> int:
        """Spawn all drones in grid formation"""
        print(f"[GZ] Spawning {self.num_drones} drones...")
        
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        spawned = 0
        
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            x = (col - cols/2) * spacing
            y = (row - cols/2) * spacing
            
            if self.spawn_drone(i, x, y, 0.2):
                spawned += 1
                print(f"  Spawned drone_{i} at ({x:.1f}, {y:.1f})")
            
            time.sleep(0.3)  # Small delay between spawns
        
        print(f"[GZ] Spawned {spawned}/{self.num_drones} drones")
        return spawned
    
    def send_velocity_command(self, drone_id: int, vx: float, vy: float, vz: float):
        """Send velocity command to a single drone"""
        name = f"drone_{drone_id}"
        topic = f"/model/{name}/cmd_vel"

        # Format as Twist message (Gazebo Sim format)
        msg = f'linear: {{x: {vx:.4f}, y: {vy:.4f}, z: {vz:.4f}}}, angular: {{x: 0, y: 0, z: 0}}'

        cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Twist", "-p", msg]

        try:
            subprocess.run(cmd, capture_output=True, timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            pass
    
    def send_velocity_commands_batch(self, velocities: np.ndarray):
        """Send velocity commands to all drones efficiently"""
        # Use threading for parallel sends
        threads = []

        for i in range(self.num_drones):
            if self.drones[i].connected:
                vx, vy, vz = velocities[i]
                t = threading.Thread(
                    target=self.send_velocity_command,
                    args=(i, float(vx), float(vy), float(vz))
                )
                threads.append(t)
                t.start()

        # Wait for all sends
        for t in threads:
            t.join(timeout=0.1)

    def set_drone_pose(self, drone_id: int, x: float, y: float, z: float):
        """Set drone position directly using Gazebo service"""
        name = f"drone_{drone_id}"

        # Use gz service to set pose
        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "100",
            "--req", f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}'
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=0.3)
            return True
        except:
            return False

    def set_all_poses(self, positions: np.ndarray):
        """Set all drone positions directly"""
        threads = []
        for i in range(self.num_drones):
            if self.drones[i].connected:
                x, y, z = positions[i]
                t = threading.Thread(
                    target=self.set_drone_pose,
                    args=(i, float(x), float(y), float(z))
                )
                threads.append(t)
                t.start()

        for t in threads:
            t.join(timeout=0.2)
    
    def get_drone_pose(self, drone_id: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get pose of a single drone"""
        name = f"drone_{drone_id}"
        topic = f"/model/{name}/pose"
        
        cmd = ["gz", "topic", "-e", "-t", topic, "-n", "1"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=0.5)
            output = result.stdout.decode()
            
            # Parse pose from output (simplified)
            pos_match = re.search(r'position.*?x:\s*([-\d.]+).*?y:\s*([-\d.]+).*?z:\s*([-\d.]+)', 
                                 output, re.DOTALL)
            if pos_match:
                x, y, z = map(float, pos_match.groups())
                return np.array([x, y, z]), np.array([1, 0, 0, 0])
        except:
            pass

        # Fallback: read from world pose info
        return self._get_pose_from_world_info(name)

    def _get_pose_from_world_info(self, name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Parse pose from /world/<world>/pose/info when model pose topic isn't published."""
        topic = f"/world/{self.world_name}/pose/info"
        cmd = ["gz", "topic", "-e", "-t", topic, "-n", "1"]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=0.5)
            output = result.stdout.decode()
            if not output:
                return None

            pattern = (
                r'name:\s*"' + re.escape(name) + r'".*?position\s*\{'
                r'.*?x:\s*([-\d.]+).*?y:\s*([-\d.]+).*?z:\s*([-\d.]+)'
            )
            pos_match = re.search(pattern, output, re.DOTALL)
            if pos_match:
                x, y, z = map(float, pos_match.groups())
                return np.array([x, y, z]), np.array([1, 0, 0, 0])
        except:
            pass
        
        return None
    
    def start_pose_updates(self, rate: float = 20.0):
        """Start background thread for pose updates"""
        self.running = True
        self.pose_thread = threading.Thread(target=self._pose_update_loop, args=(rate,))
        self.pose_thread.daemon = True
        self.pose_thread.start()
        print(f"[GZ] Pose updates started at {rate}Hz")
    
    def _pose_update_loop(self, rate: float):
        """Background loop to update poses"""
        interval = 1.0 / rate
        
        while self.running:
            t_start = time.time()
            
            # Update all drones
            for i in range(self.num_drones):
                if self.drones[i].connected:
                    result = self.get_drone_pose(i)
                    if result:
                        pos, _ = result
                        with self.lock:
                            self.drones[i].position = pos
                            self.positions[i] = pos
            
            # Rate limiting
            elapsed = time.time() - t_start
            if elapsed < interval:
                time.sleep(interval - elapsed)
    
    def stop_pose_updates(self):
        """Stop background pose updates"""
        self.running = False
        if self.pose_thread:
            self.pose_thread.join(timeout=1.0)
    
    def get_all_positions(self) -> np.ndarray:
        """Get positions of all drones"""
        with self.lock:
            return self.positions.copy()
    
    def get_all_velocities(self) -> np.ndarray:
        """Get velocities of all drones (estimated)"""
        with self.lock:
            return self.velocities.copy()
    
    def enable_drone(self, drone_id: int):
        """Enable a drone's motors"""
        name = f"drone_{drone_id}"
        topic = f"/model/{name}/enable"
        
        cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Boolean", "-p", "data: true"]
        subprocess.run(cmd, capture_output=True)
    
    def enable_all_drones(self):
        """Enable all drone motors"""
        print("[GZ] Enabling all drones...")
        for i in range(self.num_drones):
            if self.drones[i].connected:
                self.enable_drone(i)
        time.sleep(0.5)
    
    def shutdown(self):
        """Clean shutdown"""
        print("[GZ] Shutting down...")
        self.stop_pose_updates()
        
        if self.gz_process:
            self.gz_process.terminate()
            self.gz_process.wait(timeout=5)
        
        # Kill any remaining processes
        subprocess.run(["pkill", "-9", "gz"], capture_output=True)


class GazeboTransportBridge(GazeboBridge):
    """
    Advanced bridge using gz-transport Python bindings (if available).
    Falls back to subprocess method if not available.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Try to import gz-transport
        self.use_transport = False
        try:
            from gz.transport import Node
            from gz.msgs import Twist, Pose, Boolean
            self.Node = Node
            self.Twist = Twist
            self.Pose = Pose
            self.Boolean = Boolean
            self.use_transport = True
            print("[GZ] Using gz-transport Python bindings")
        except ImportError:
            print("[GZ] gz-transport not available, using subprocess")
        
        if self.use_transport:
            self.node = self.Node()
            self.publishers: Dict[int, any] = {}
            self.subscribers: Dict[int, any] = {}
    
    def _setup_publishers(self):
        """Setup topic publishers for each drone"""
        if not self.use_transport:
            return
        
        for i in range(self.num_drones):
            name = f"drone_{i}"
            topic = f"/model/{name}/cmd_vel"
            self.publishers[i] = self.node.advertise(topic, self.Twist)
    
    def _setup_subscribers(self):
        """Setup pose subscribers for each drone"""
        if not self.use_transport:
            return
        
        def make_callback(drone_id):
            def callback(msg):
                pos = np.array([msg.position.x, msg.position.y, msg.position.z])
                with self.lock:
                    self.drones[drone_id].position = pos
                    self.positions[drone_id] = pos
            return callback
        
        for i in range(self.num_drones):
            name = f"drone_{i}"
            topic = f"/model/{name}/pose"
            self.subscribers[i] = self.node.subscribe(
                topic, self.Pose, make_callback(i)
            )
    
    def send_velocity_commands_batch(self, velocities: np.ndarray):
        """Send velocity commands using transport (if available)"""
        if self.use_transport:
            for i in range(self.num_drones):
                if i in self.publishers and self.drones[i].connected:
                    msg = self.Twist()
                    msg.linear.x = float(velocities[i, 0])
                    msg.linear.y = float(velocities[i, 1])
                    msg.linear.z = float(velocities[i, 2])
                    self.publishers[i].publish(msg)
        else:
            super().send_velocity_commands_batch(velocities)


# ============ TEST ============

def test_bridge():
    """Test Gazebo bridge"""
    print("\n" + "="*60)
    print("  GAZEBO BRIDGE TEST")
    print("="*60 + "\n")
    
    bridge = GazeboBridge(num_drones=5)
    
    # Check if Gazebo is running
    result = subprocess.run(["pgrep", "-f", "gz sim"], capture_output=True)
    
    if result.returncode != 0:
        print("Gazebo not running. Start with:")
        print("  gz sim -s swarm_world.sdf --headless-rendering")
        return
    
    print("Gazebo detected!")
    
    # Try spawning
    print("\nSpawning drones...")
    spawned = bridge.spawn_swarm_grid(spacing=3.0)
    
    if spawned > 0:
        print(f"\nEnabling drones...")
        bridge.enable_all_drones()
        
        print("\nSending test velocity commands...")
        vel = np.array([[0, 0, 1]] * 5, dtype=np.float32)  # Up
        bridge.send_velocity_commands_batch(vel)
        
        time.sleep(2)
        
        print("\nGetting poses...")
        for i in range(5):
            pose = bridge.get_drone_pose(i)
            if pose:
                print(f"  Drone {i}: {pose[0]}")
    
    bridge.shutdown()


if __name__ == "__main__":
    test_bridge()
