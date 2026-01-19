#!/usr/bin/env python3
"""
GAZEBO + 2D VISUALIZER
======================
- Gazebo runs headless (server only) for REAL physics
- 2D window shows real Gazebo drone positions
- All physics happen in Gazebo, 2D is just a viewer

Usage: python3 run_gazebo_2d.py --drones 9
"""

import numpy as np
import subprocess
import threading
import time
import sys
import os
import signal

# CUDA setup
MATLAB_CUDA = "/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
MATLAB_LIB = "/usr/local/MATLAB/R2025b/bin/glnxa64"
if os.path.exists(MATLAB_CUDA):
    os.environ["CUDA_PATH"] = MATLAB_CUDA
    os.environ["LD_LIBRARY_PATH"] = f"{MATLAB_LIB}:{MATLAB_CUDA}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import pygame
pygame.init()

# Colors
BLACK = (20, 20, 30)
WHITE = (255, 255, 255)
GRAY = (50, 50, 60)
GREEN = (0, 255, 100)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
RED = (255, 50, 50)
CYAN = (0, 220, 255)
PINK = (255, 100, 200)


class GazeboHeadless:
    """Manages headless Gazebo instance"""

    def __init__(self, world_file, num_drones=9):
        self.world_file = world_file
        self.num_drones = num_drones
        self.process = None
        self.world_name = "swarm_world"
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.running = False
        self.connected = [False] * num_drones

    def start(self):
        """Start Gazebo in headless mode"""
        print("[GZ] Starting Gazebo headless...")

        # Kill any existing
        subprocess.run(["pkill", "-9", "gz"], capture_output=True)
        time.sleep(2)

        # Start headless server
        cmd = ["gz", "sim", "-s", "-r", self.world_file]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        # Wait for Gazebo to start
        print("[GZ] Waiting for Gazebo to initialize...")
        time.sleep(5)

        # Check if running by looking for process
        result = subprocess.run(["pgrep", "-f", "gz sim"], capture_output=True)
        if result.returncode != 0:
            print("[GZ] ERROR: Gazebo failed to start")
            return False

        print("[GZ] Gazebo headless running (PID check passed)")
        self.running = True
        return True

    def spawn_drone(self, drone_id, x, y, z=0.3):
        """Spawn a drone in Gazebo"""
        name = f"drone_{drone_id}"
        model_file = os.path.join(os.path.dirname(__file__), "model.sdf")

        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/create",
            "--reqtype", "gz.msgs.EntityFactory",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", f'sdf_filename: "{model_file}", name: "{name}", pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}'
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                self.connected[drone_id] = True
                self.positions[drone_id] = [x, y, z]
                return True
        except:
            pass
        return False

    def spawn_grid(self, spacing=3.0):
        """Spawn all drones in grid"""
        print(f"[GZ] Spawning {self.num_drones} drones...")
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        spawned = 0

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            x = (col - cols/2 + 0.5) * spacing
            y = (row - cols/2 + 0.5) * spacing

            if self.spawn_drone(i, x, y, 0.3):
                spawned += 1
                print(f"  Drone {i} at ({x:.1f}, {y:.1f})")
            time.sleep(0.2)

        print(f"[GZ] Spawned {spawned}/{self.num_drones}")
        return spawned

    def get_pose(self, drone_id):
        """Get drone pose from Gazebo"""
        name = f"drone_{drone_id}"
        topic = f"/world/{self.world_name}/pose/info"

        cmd = ["gz", "topic", "-e", "-t", topic, "-n", "1"]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=0.3)
            output = result.stdout.decode()

            # Parse pose
            import re
            pattern = rf'name:\s*"{name}".*?position\s*\{{.*?x:\s*([-\d.]+).*?y:\s*([-\d.]+).*?z:\s*([-\d.]+)'
            match = re.search(pattern, output, re.DOTALL)

            if match:
                x, y, z = map(float, match.groups())
                return np.array([x, y, z])
        except:
            pass
        return None

    def update_all_poses(self):
        """Update all drone positions from Gazebo"""
        for i in range(self.num_drones):
            if self.connected[i]:
                pose = self.get_pose(i)
                if pose is not None:
                    self.positions[i] = pose

    def set_velocity(self, drone_id, vx, vy, vz):
        """Send velocity command to drone"""
        name = f"drone_{drone_id}"
        topic = f"/model/{name}/cmd_vel"
        msg = f'linear: {{x: {vx:.3f}, y: {vy:.3f}, z: {vz:.3f}}}'

        cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Twist", "-p", msg]

        try:
            subprocess.run(cmd, capture_output=True, timeout=0.2)
        except:
            pass

    def set_pose(self, drone_id, x, y, z):
        """Set drone position directly"""
        name = f"drone_{drone_id}"

        cmd = [
            "gz", "service", "-s", f"/world/{self.world_name}/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "100",
            "--req", f'name: "{name}", position: {{x: {x:.3f}, y: {y:.3f}, z: {z:.3f}}}'
        ]

        try:
            subprocess.run(cmd, capture_output=True, timeout=0.2)
        except:
            pass

    def stop(self):
        """Stop Gazebo"""
        print("[GZ] Stopping Gazebo...")
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=3)
        subprocess.run(["pkill", "-9", "gz"], capture_output=True)


class Drone2DViewer:
    """2D viewer that shows Gazebo drone positions"""

    def __init__(self, num_drones, width=1000, height=700):
        self.num_drones = num_drones
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Gazebo Drone Swarm - 2D View")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 14)
        self.font_big = pygame.font.SysFont('arial', 18, bold=True)

        # View
        self.scale = 20
        self.center_x = width // 2
        self.center_y = height // 2

        # Data from Gazebo
        self.positions = np.zeros((num_drones, 3))
        self.targets = np.zeros((num_drones, 3))
        self.velocities = np.zeros((num_drones, 3))

        self.running = True
        self.armed = False
        self.status = "INIT"

    def world_to_screen(self, x, y):
        sx = self.center_x + int(x * self.scale)
        sy = self.center_y - int(y * self.scale)
        return sx, sy

    def draw(self):
        self.screen.fill(BLACK)

        # Grid
        for i in range(-25, 26, 5):
            sx, _ = self.world_to_screen(i, 0)
            pygame.draw.line(self.screen, GRAY, (sx, 0), (sx, self.height), 1)
            _, sy = self.world_to_screen(0, i)
            pygame.draw.line(self.screen, GRAY, (0, sy), (self.width, sy), 1)

        # Center
        pygame.draw.circle(self.screen, WHITE, (self.center_x, self.center_y), 5, 1)

        # Targets
        for i in range(self.num_drones):
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            pygame.draw.circle(self.screen, PINK, (tx, ty), 8, 2)

        # Drones
        min_dist = 999
        for i in range(self.num_drones):
            x, y, z = self.positions[i]
            sx, sy = self.world_to_screen(x, y)

            # Check distances
            for j in range(self.num_drones):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    min_dist = min(min_dist, d)

            # Color by altitude
            if z < 0.5:
                color = ORANGE
            elif z < 2:
                color = YELLOW
            else:
                color = GREEN

            # Size by altitude
            size = max(8, int(8 + z * 2))

            # Draw
            pygame.draw.circle(self.screen, color, (sx, sy), size)
            pygame.draw.circle(self.screen, WHITE, (sx, sy), size, 2)

            # ID
            label = self.font.render(str(i), True, WHITE)
            self.screen.blit(label, (sx - 4, sy - 5))

            # Velocity
            vx, vy = self.velocities[i, 0], self.velocities[i, 1]
            if abs(vx) > 0.05 or abs(vy) > 0.05:
                ex = sx + int(vx * 20)
                ey = sy - int(vy * 20)
                pygame.draw.line(self.screen, CYAN, (sx, sy), (ex, ey), 2)

        # Info
        avg_alt = self.positions[:, 2].mean()
        info = [
            f"[GAZEBO PHYSICS]",
            f"",
            f"Drones: {self.num_drones}",
            f"Armed: {'YES' if self.armed else 'NO'}",
            f"Altitude: {avg_alt:.1f}m",
            f"Min Dist: {min_dist:.1f}m",
            f"Status: {self.status}",
            f"",
            f"CONTROLS:",
            f"SPACE - Takeoff",
            f"C - Circle",
            f"G - Grid",
            f"L - Land",
            f"+/- Zoom",
            f"Q - Quit",
        ]

        y = 10
        for text in info:
            color = GREEN if "YES" in text else (CYAN if "GAZEBO" in text else WHITE)
            label = self.font.render(text, True, color)
            self.screen.blit(label, (10, y))
            y += 18

        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
                    return None
                elif event.key == pygame.K_SPACE:
                    return "takeoff"
                elif event.key == pygame.K_c:
                    return "circle"
                elif event.key == pygame.K_g:
                    return "grid"
                elif event.key == pygame.K_l:
                    return "land"
                elif event.key == pygame.K_EQUALS:
                    self.scale = min(50, self.scale + 2)
                elif event.key == pygame.K_MINUS:
                    self.scale = max(5, self.scale - 2)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, default=9)
    args = parser.parse_args()

    num_drones = args.drones
    spacing = 3.0

    print("=" * 50)
    print("  GAZEBO HEADLESS + 2D VIEWER")
    print("=" * 50)

    # Get world file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    world_file = os.path.join(script_dir, "swarm_world.sdf")

    if not os.path.exists(world_file):
        print(f"ERROR: World file not found: {world_file}")
        return

    # Start Gazebo headless
    gazebo = GazeboHeadless(world_file, num_drones)
    if not gazebo.start():
        print("Failed to start Gazebo")
        return

    # Spawn drones
    spawned = gazebo.spawn_grid(spacing=spacing)
    if spawned == 0:
        print("Failed to spawn drones")
        gazebo.stop()
        return

    time.sleep(1)

    # Create viewer
    viewer = Drone2DViewer(num_drones)

    # Initialize targets
    cols = int(np.ceil(np.sqrt(num_drones)))
    for i in range(num_drones):
        row, col = divmod(i, cols)
        viewer.targets[i] = [
            (col - cols/2 + 0.5) * spacing,
            (row - cols/2 + 0.5) * spacing,
            0.3
        ]

    viewer.status = "READY"
    print("\n[READY] Press SPACE to takeoff\n")

    # Control parameters
    kp = 1.5  # Position gain

    # Main loop
    try:
        while viewer.running:
            # Handle input
            cmd = viewer.handle_input()

            if cmd == "takeoff":
                print("TAKEOFF!")
                viewer.armed = True
                viewer.targets[:, 2] = 5.0
                viewer.status = "TAKEOFF"

            elif cmd == "circle":
                print("CIRCLE")
                angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
                viewer.targets[:, 0] = 8 * np.cos(angles)
                viewer.targets[:, 1] = 8 * np.sin(angles)
                viewer.targets[:, 2] = 5.0
                viewer.status = "CIRCLE"

            elif cmd == "grid":
                print("GRID")
                for i in range(num_drones):
                    row, col = divmod(i, cols)
                    viewer.targets[i, 0] = (col - cols/2 + 0.5) * spacing
                    viewer.targets[i, 1] = (row - cols/2 + 0.5) * spacing
                viewer.targets[:, 2] = 5.0
                viewer.status = "GRID"

            elif cmd == "land":
                print("LAND")
                viewer.targets[:, 2] = 0.3
                viewer.status = "LANDING"

            # Update positions from Gazebo
            gazebo.update_all_poses()
            viewer.positions = gazebo.positions.copy()

            # Compute and send velocities
            if viewer.armed:
                for i in range(num_drones):
                    if gazebo.connected[i]:
                        # Simple P controller
                        error = viewer.targets[i] - viewer.positions[i]
                        vel = error * kp

                        # Limit velocity
                        speed = np.linalg.norm(vel)
                        if speed > 2.0:
                            vel = vel / speed * 2.0

                        viewer.velocities[i] = vel

                        # Send to Gazebo
                        gazebo.set_velocity(i, vel[0], vel[1], vel[2])

            # Draw
            viewer.draw()
            viewer.clock.tick(20)

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        gazebo.stop()
        pygame.quit()
        print("Done!")


if __name__ == "__main__":
    main()
