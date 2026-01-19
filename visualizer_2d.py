#!/usr/bin/env python3
"""
SIMPLE 2D SWARM VISUALIZER - REWRITTEN FOR RELIABILITY
"""

import numpy as np
import time
import sys
import os

try:
    import pygame
    pygame.init()
except ImportError:
    print("Install pygame: pip install pygame")
    sys.exit(1)

# Colors
BLACK = (20, 20, 30)
WHITE = (255, 255, 255)
GRAY = (60, 60, 70)
GREEN = (0, 255, 100)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
RED = (255, 50, 50)
CYAN = (0, 220, 255)
PINK = (255, 100, 200)
PURPLE = (180, 80, 255)


class SimpleSwarmVisualizer:
    def __init__(self, num_drones=9, width=1000, height=700):
        self.num_drones = num_drones
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"Drone Swarm - {num_drones} drones")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 14)
        self.font_big = pygame.font.SysFont('arial', 18, bold=True)

        # Drone data
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.targets = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)

        # View settings
        self.scale = 12  # pixels per meter
        self.center_x = width // 2
        self.center_y = height // 2

        self.running = True

    def world_to_screen(self, x, y):
        """Convert world coords to screen coords"""
        sx = self.center_x + int(x * self.scale)
        sy = self.center_y - int(y * self.scale)  # Y flipped
        return sx, sy

    def draw(self):
        self.screen.fill(BLACK)

        # Draw grid
        for i in range(-20, 21, 5):
            sx, sy = self.world_to_screen(i, 0)
            pygame.draw.line(self.screen, GRAY, (sx, 0), (sx, self.height), 1)
            sx, sy = self.world_to_screen(0, i)
            pygame.draw.line(self.screen, GRAY, (0, sy), (self.width, sy), 1)

        # Draw center cross
        pygame.draw.line(self.screen, WHITE, (self.center_x - 20, self.center_y),
                        (self.center_x + 20, self.center_y), 2)
        pygame.draw.line(self.screen, WHITE, (self.center_x, self.center_y - 20),
                        (self.center_x, self.center_y + 20), 2)

        # Draw targets
        for i in range(self.num_drones):
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            pygame.draw.circle(self.screen, PINK, (tx, ty), 8, 2)

        # Draw lines from drones to targets
        for i in range(self.num_drones):
            dx, dy = self.world_to_screen(self.positions[i, 0], self.positions[i, 1])
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            pygame.draw.line(self.screen, (100, 50, 70), (dx, dy), (tx, ty), 1)

        # Draw drones
        for i in range(self.num_drones):
            x, y, z = self.positions[i]
            sx, sy = self.world_to_screen(x, y)

            # Color based on altitude
            if z < 0.5:
                color = ORANGE  # Ground
            elif z < 3:
                color = YELLOW  # Low
            else:
                color = GREEN   # Flying

            # Size based on altitude
            size = max(8, int(10 + z))

            # Draw drone
            pygame.draw.circle(self.screen, color, (sx, sy), size)
            pygame.draw.circle(self.screen, WHITE, (sx, sy), size, 2)

            # Draw ID
            label = self.font.render(str(i), True, WHITE)
            self.screen.blit(label, (sx - 4, sy - 6))

            # Draw velocity arrow
            vx, vy = self.velocities[i, 0], self.velocities[i, 1]
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                ex = sx + int(vx * 20)
                ey = sy - int(vy * 20)
                pygame.draw.line(self.screen, CYAN, (sx, sy), (ex, ey), 2)

        # Draw info panel
        info_y = 10
        info = [
            f"Drones: {self.num_drones}",
            f"Avg Alt: {self.positions[:, 2].mean():.1f}m",
            f"Scale: {self.scale} px/m",
            "",
            "Controls:",
            "+/- : Zoom",
            "Arrows: Pan",
            "R: Reset view",
            "Q: Quit"
        ]
        for text in info:
            label = self.font.render(text, True, WHITE)
            self.screen.blit(label, (10, info_y))
            info_y += 20

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self.scale = min(50, self.scale + 2)
                elif event.key == pygame.K_MINUS:
                    self.scale = max(4, self.scale - 2)
                elif event.key == pygame.K_LEFT:
                    self.center_x += 50
                elif event.key == pygame.K_RIGHT:
                    self.center_x -= 50
                elif event.key == pygame.K_UP:
                    self.center_y += 50
                elif event.key == pygame.K_DOWN:
                    self.center_y -= 50
                elif event.key == pygame.K_r:
                    self.center_x = self.width // 2
                    self.center_y = self.height // 2
                    self.scale = 12


def run_demo():
    """Standalone demo - no external dependencies"""
    print("=" * 50)
    print("  DRONE SWARM VISUALIZER")
    print("=" * 50)

    num_drones = 9
    viz = SimpleSwarmVisualizer(num_drones=num_drones)

    # Initialize grid positions
    spacing = 4.0
    cols = int(np.ceil(np.sqrt(num_drones)))
    for i in range(num_drones):
        row, col = divmod(i, cols)
        viz.positions[i] = [(col - cols/2 + 0.5) * spacing,
                           (row - cols/2 + 0.5) * spacing,
                           0.0]
        viz.targets[i] = viz.positions[i].copy()

    print(f"Drones initialized in {cols}x{cols} grid")
    print("Press SPACE to takeoff, C for circle, G for grid, L to land")

    phase = "ground"
    sim_time = 0.0

    while viz.running:
        viz.handle_events()

        # Check for commands
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] and phase == "ground":
            print("TAKEOFF!")
            viz.targets[:, 2] = 5.0
            phase = "takeoff"
        elif keys[pygame.K_c]:
            print("CIRCLE formation")
            angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
            viz.targets[:, 0] = 10 * np.cos(angles)
            viz.targets[:, 1] = 10 * np.sin(angles)
            viz.targets[:, 2] = 5.0
            phase = "circle"
        elif keys[pygame.K_g]:
            print("GRID formation")
            for i in range(num_drones):
                row, col = divmod(i, cols)
                viz.targets[i, 0] = (col - cols/2 + 0.5) * spacing
                viz.targets[i, 1] = (row - cols/2 + 0.5) * spacing
            viz.targets[:, 2] = 5.0
            phase = "grid"
        elif keys[pygame.K_l]:
            print("LANDING")
            viz.targets[:, 2] = 0.0
            phase = "land"
        elif keys[pygame.K_t]:
            print("TRAJECTORY - rotating circle")
            phase = "trajectory"

        # Update trajectory targets if in trajectory mode
        if phase == "trajectory":
            angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False) + sim_time * 0.5
            viz.targets[:, 0] = 10 * np.cos(angles)
            viz.targets[:, 1] = 10 * np.sin(angles)

        # Physics with collision avoidance
        dt = 1/30
        sim_time += dt

        # 1. Attraction to target
        diff = viz.targets - viz.positions
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 0.01)
        speed = np.minimum(dist * 2, 3.0)
        vel_attract = diff / dist * speed

        # 2. Collision avoidance - repel from nearby drones
        vel_repel = np.zeros_like(viz.positions)
        safe_distance = 2.0  # meters
        repel_strength = 5.0

        for i in range(num_drones):
            for j in range(num_drones):
                if i != j:
                    delta = viz.positions[i] - viz.positions[j]
                    d = np.linalg.norm(delta)
                    if d < safe_distance and d > 0.01:
                        # Repulsive force inversely proportional to distance
                        force = (safe_distance - d) / safe_distance * repel_strength
                        vel_repel[i] += (delta / d) * force

        # 3. Combine velocities
        viz.velocities = vel_attract + vel_repel

        # 4. Limit max speed
        speeds = np.linalg.norm(viz.velocities, axis=1, keepdims=True)
        max_speed = 4.0
        viz.velocities = np.where(speeds > max_speed,
                                   viz.velocities / speeds * max_speed,
                                   viz.velocities)

        # Update positions
        viz.positions += viz.velocities * dt

        # Ground constraint
        viz.positions[:, 2] = np.maximum(viz.positions[:, 2], 0.0)

        # Draw
        viz.draw()
        viz.clock.tick(30)

    pygame.quit()
    print("Done!")


if __name__ == "__main__":
    run_demo()
