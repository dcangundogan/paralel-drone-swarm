#!/usr/bin/env python3
"""
INTEGRATED DRONE SWARM SYSTEM
- Connects to Gazebo (real physics)
- GPU-accelerated control with collision avoidance
- 2D real-time visualization

Usage:
  1. Start Gazebo: gz sim -r swarm_world.sdf
  2. Run this: python3 run_swarm.py
"""

import numpy as np
import threading
import time
import sys
import os

# Set CUDA paths for GPU
MATLAB_CUDA = "/usr/local/MATLAB/R2025b/sys/cuda/glnxa64/cuda"
MATLAB_LIB = "/usr/local/MATLAB/R2025b/bin/glnxa64"
if os.path.exists(MATLAB_CUDA):
    os.environ["CUDA_PATH"] = MATLAB_CUDA
    os.environ["LD_LIBRARY_PATH"] = f"{MATLAB_LIB}:{MATLAB_CUDA}/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

import pygame
pygame.init()

from gpu_controller import GPUSwarmController, SwarmConfig
from gz_bridge import GazeboBridge

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


class DroneSwarmSystem:
    def __init__(self, num_drones=9):
        self.num_drones = num_drones
        self.running = False

        # Components
        self.controller = None
        self.bridge = None

        # Visualization
        self.width = 1000
        self.height = 700
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Drone Swarm Control - {num_drones} drones")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 14)
        self.font_big = pygame.font.SysFont('arial', 20, bold=True)

        # View
        self.scale = 15
        self.center_x = self.width // 2
        self.center_y = self.height // 2

        # State
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.targets = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)
        self.armed = False
        self.mode = "INIT"
        self.gazebo_connected = False
        self.use_simulation = False

    def initialize(self):
        """Initialize controller and try to connect to Gazebo"""
        print("=" * 50)
        print("  DRONE SWARM SYSTEM")
        print("=" * 50)

        # Create GPU controller
        print("\n[1] Initializing GPU controller...")
        config = SwarmConfig(num_drones=self.num_drones)
        self.controller = GPUSwarmController(config)

        # Create Gazebo bridge
        print("[2] Connecting to Gazebo...")
        self.bridge = GazeboBridge(num_drones=self.num_drones)

        # Check if Gazebo is running
        import subprocess
        result = subprocess.run(["pgrep", "-f", "gz sim"], capture_output=True)

        if result.returncode == 0:
            print("    Gazebo detected!")
            self.gazebo_connected = True
            self.mode = "GAZEBO"
        else:
            print("    Gazebo not running - using SIMULATION mode")
            print("    (Start Gazebo with: gz sim -r swarm_world.sdf)")
            self.use_simulation = True
            self.mode = "SIM"

        # Initialize drone positions in grid
        print("[3] Setting up drone grid...")
        spacing = 3.0
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.positions[i] = [
                (col - cols/2 + 0.5) * spacing,
                (row - cols/2 + 0.5) * spacing,
                0.2
            ]
            self.targets[i] = self.positions[i].copy()

        self.controller.update_positions(self.positions)
        xp = self.controller.xp
        self.controller.targets = xp.asarray(self.targets, dtype=xp.float32)
        self.controller.initial_positions = xp.asarray(self.positions, dtype=xp.float32)

        # Spawn in Gazebo if connected
        if self.gazebo_connected:
            print("[4] Spawning drones in Gazebo...")
            spawned = self.bridge.spawn_swarm_grid(spacing=spacing)
            if spawned > 0:
                self.bridge.enable_all_drones()
                print(f"    Spawned {spawned} drones")
            else:
                print("    Spawn failed - switching to simulation")
                self.use_simulation = True
                self.mode = "SIM"

        print("\n" + "=" * 50)
        print("  READY - Press SPACE to takeoff")
        print("=" * 50 + "\n")

    def world_to_screen(self, x, y):
        sx = self.center_x + int(x * self.scale)
        sy = self.center_y - int(y * self.scale)
        return sx, sy

    def update_physics(self, dt):
        """Update drone physics - either from Gazebo or simulation"""

        # Compute control velocities (with collision avoidance)
        if self.armed:
            self.velocities = self.controller.compute_control()

            # Get current positions
            if hasattr(self.controller.positions, 'get'):
                pos = self.controller.positions.get()
                vel = self.controller.cmd_velocities.get()
            else:
                pos = np.array(self.controller.positions)
                vel = np.array(self.controller.cmd_velocities)

            # Update positions
            pos = pos + vel * dt
            pos[:, 2] = np.maximum(pos[:, 2], 0.0)
            self.positions = pos
            self.controller.update_positions(pos)

            # Send to Gazebo if connected
            if self.gazebo_connected and not self.use_simulation:
                self.bridge.set_all_poses(pos)

        # Get current state from controller
        if hasattr(self.controller.positions, 'get'):
            self.positions = self.controller.positions.get()
            self.targets = self.controller.targets.get()
            self.velocities = self.controller.cmd_velocities.get()
        else:
            self.positions = np.array(self.controller.positions)
            self.targets = np.array(self.controller.targets)
            self.velocities = np.array(self.controller.cmd_velocities)

    def draw(self):
        self.screen.fill(BLACK)

        # Grid
        for i in range(-20, 21, 5):
            sx, _ = self.world_to_screen(i, 0)
            pygame.draw.line(self.screen, GRAY, (sx, 0), (sx, self.height), 1)
            _, sy = self.world_to_screen(0, i)
            pygame.draw.line(self.screen, GRAY, (0, sy), (self.width, sy), 1)

        # Center
        pygame.draw.circle(self.screen, WHITE, (self.center_x, self.center_y), 5, 1)

        # Targets
        for i in range(self.num_drones):
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            pygame.draw.circle(self.screen, PINK, (tx, ty), 10, 2)

        # Lines to targets
        for i in range(self.num_drones):
            dx, dy = self.world_to_screen(self.positions[i, 0], self.positions[i, 1])
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            pygame.draw.line(self.screen, (80, 40, 50), (dx, dy), (tx, ty), 1)

        # Drones
        min_dist = float('inf')
        for i in range(self.num_drones):
            x, y, z = self.positions[i]
            sx, sy = self.world_to_screen(x, y)

            # Check collision risk
            collision_risk = False
            for j in range(self.num_drones):
                if i != j:
                    d = np.linalg.norm(self.positions[i] - self.positions[j])
                    min_dist = min(min_dist, d)
                    if d < 1.5:
                        collision_risk = True

            # Color
            if collision_risk:
                color = RED
            elif z < 0.5:
                color = ORANGE
            elif z < 3:
                color = YELLOW
            else:
                color = GREEN

            # Size
            size = max(8, int(10 + z * 1.5))

            # Draw
            pygame.draw.circle(self.screen, color, (sx, sy), size)
            pygame.draw.circle(self.screen, WHITE, (sx, sy), size, 2)

            # ID
            label = self.font.render(str(i), True, WHITE)
            self.screen.blit(label, (sx - 4, sy - 5))

            # Velocity arrow
            vx, vy = self.velocities[i, 0], self.velocities[i, 1]
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                ex = sx + int(vx * 15)
                ey = sy - int(vy * 15)
                pygame.draw.line(self.screen, CYAN, (sx, sy), (ex, ey), 2)

        # Info panel
        info = [
            f"Mode: {self.mode}",
            f"Drones: {self.num_drones}",
            f"Armed: {'YES' if self.armed else 'NO'}",
            f"Altitude: {self.positions[:, 2].mean():.1f}m",
            f"Min Dist: {min_dist:.1f}m" if min_dist < 100 else "Min Dist: --",
            "",
            "CONTROLS:",
            "SPACE - Arm & Takeoff",
            "C - Circle formation",
            "G - Grid formation",
            "V - V formation",
            "T - Trajectory",
            "L - Land",
            "+/- - Zoom",
            "Q - Quit",
        ]

        y = 10
        for text in info:
            color = GREEN if "YES" in text else (RED if "NO" in text else WHITE)
            label = self.font.render(text, True, color)
            self.screen.blit(label, (10, y))
            y += 18

        # Mode indicator
        mode_color = GREEN if self.mode == "GAZEBO" else YELLOW
        mode_text = self.font_big.render(f"[{self.mode}]", True, mode_color)
        self.screen.blit(mode_text, (self.width - 100, 10))

        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    if not self.armed:
                        print("ARMING & TAKEOFF")
                        self.armed = True
                        self.controller.arm_all()
                    self.controller.takeoff(5.0)

                elif event.key == pygame.K_c:
                    print("CIRCLE formation")
                    self.controller.set_formation_circle((0, 0, 5), radius=10.0)

                elif event.key == pygame.K_g:
                    print("GRID formation")
                    self.controller.set_formation_grid((0, 0, 5), spacing=3.0)

                elif event.key == pygame.K_v:
                    print("V formation")
                    self.controller.set_formation_v((0, 0, 5), spacing=2.5)

                elif event.key == pygame.K_t:
                    print("TRAJECTORY - circle")
                    from gpu_controller import trajectory_circle
                    self.controller.follow_trajectory(trajectory_circle)

                elif event.key == pygame.K_l:
                    print("LANDING")
                    self.controller.land()

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.scale = min(50, self.scale + 2)
                elif event.key == pygame.K_MINUS:
                    self.scale = max(5, self.scale - 2)

                elif event.key == pygame.K_LEFT:
                    self.center_x += 30
                elif event.key == pygame.K_RIGHT:
                    self.center_x -= 30
                elif event.key == pygame.K_UP:
                    self.center_y += 30
                elif event.key == pygame.K_DOWN:
                    self.center_y -= 30

    def run(self):
        """Main loop"""
        self.initialize()
        self.running = True

        while self.running:
            dt = 1/30

            self.handle_input()
            self.update_physics(dt)
            self.draw()

            self.clock.tick(30)

        # Cleanup
        print("\nShutting down...")
        if self.bridge:
            self.bridge.shutdown()
        pygame.quit()
        print("Done!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, default=9, help="Number of drones")
    args = parser.parse_args()

    system = DroneSwarmSystem(num_drones=args.drones)
    system.run()


if __name__ == "__main__":
    main()
