#!/usr/bin/env python3
"""
GAZEBO + 2D VISUALIZER WITH GUI
===============================
- Gazebo headless for REAL physics
- 2D view with clickable buttons
- Resource monitoring (CPU, RAM, GPU)

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

# Try to import psutil for resource monitoring
try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    print("Note: pip install psutil for resource monitoring")

# Colors
BLACK = (20, 20, 30)
DARK_GRAY = (35, 35, 45)
GRAY = (60, 60, 70)
LIGHT_GRAY = (100, 100, 110)
WHITE = (255, 255, 255)
GREEN = (0, 220, 100)
DARK_GREEN = (0, 150, 70)
YELLOW = (255, 220, 0)
ORANGE = (255, 140, 0)
RED = (255, 60, 60)
DARK_RED = (180, 40, 40)
CYAN = (0, 200, 240)
PINK = (255, 100, 180)
BLUE = (60, 120, 220)
DARK_BLUE = (40, 80, 160)


class Button:
    """Clickable button"""
    def __init__(self, x, y, w, h, text, color, hover_color, text_color=WHITE):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hovered = False
        self.enabled = True

    def draw(self, screen, font):
        color = self.hover_color if self.hovered else self.color
        if not self.enabled:
            color = GRAY

        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, WHITE, self.rect, 2, border_radius=5)

        text_surf = font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered

    def check_click(self, pos):
        return self.enabled and self.rect.collidepoint(pos)


class ResourceMonitor:
    """Monitor CPU, RAM, GPU usage"""
    def __init__(self):
        self.cpu_percent = 0
        self.ram_percent = 0
        self.ram_used = 0
        self.gpu_percent = 0
        self.gpu_mem_percent = 0
        self.gpu_mem_used = 0
        self.last_update = 0
        self.update_interval = 1.0  # seconds

    def update(self):
        now = time.time()
        if now - self.last_update < self.update_interval:
            return

        self.last_update = now

        # CPU and RAM
        if PSUTIL_OK:
            self.cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            self.ram_percent = mem.percent
            self.ram_used = mem.used / (1024**3)  # GB

        # GPU (nvidia-smi)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, timeout=1
            )
            if result.returncode == 0:
                parts = result.stdout.decode().strip().split(",")
                self.gpu_percent = float(parts[0].strip())
                self.gpu_mem_used = float(parts[1].strip()) / 1024  # GB
                gpu_total = float(parts[2].strip()) / 1024
                self.gpu_mem_percent = (self.gpu_mem_used / gpu_total) * 100
        except:
            pass


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
        print("[GZ] Starting Gazebo headless...")
        subprocess.run(["pkill", "-9", "gz"], capture_output=True)
        time.sleep(2)

        cmd = ["gz", "sim", "-s", "-r", self.world_file]
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print("[GZ] Waiting for Gazebo...")
        time.sleep(5)

        result = subprocess.run(["pgrep", "-f", "gz sim"], capture_output=True)
        if result.returncode != 0:
            print("[GZ] ERROR: Gazebo failed to start")
            return False

        print("[GZ] Gazebo running")
        self.running = True
        return True

    def spawn_drone(self, drone_id, x, y, z=0.3):
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
        print(f"[GZ] Spawning {self.num_drones} drones...")
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        spawned = 0

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            x = (col - cols/2 + 0.5) * spacing
            y = (row - cols/2 + 0.5) * spacing

            if self.spawn_drone(i, x, y, 0.3):
                spawned += 1
            time.sleep(0.15)

        print(f"[GZ] Spawned {spawned}/{self.num_drones}")
        return spawned

    def get_pose(self, drone_id):
        name = f"drone_{drone_id}"
        topic = f"/world/{self.world_name}/pose/info"
        cmd = ["gz", "topic", "-e", "-t", topic, "-n", "1"]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=0.3)
            output = result.stdout.decode()
            import re
            pattern = rf'name:\s*"{name}".*?position\s*\{{.*?x:\s*([-\d.]+).*?y:\s*([-\d.]+).*?z:\s*([-\d.]+)'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                return np.array([float(x) for x in match.groups()])
        except:
            pass
        return None

    def update_all_poses(self):
        for i in range(self.num_drones):
            if self.connected[i]:
                pose = self.get_pose(i)
                if pose is not None:
                    self.positions[i] = pose

    def set_velocity(self, drone_id, vx, vy, vz):
        name = f"drone_{drone_id}"
        topic = f"/model/{name}/cmd_vel"
        msg = f'linear: {{x: {vx:.3f}, y: {vy:.3f}, z: {vz:.3f}}}'
        cmd = ["gz", "topic", "-t", topic, "-m", "gz.msgs.Twist", "-p", msg]
        try:
            subprocess.run(cmd, capture_output=True, timeout=0.2)
        except:
            pass

    def stop(self):
        print("[GZ] Stopping Gazebo...")
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except:
                pass
        subprocess.run(["pkill", "-9", "gz"], capture_output=True)


class DroneSwarmGUI:
    """Main GUI application"""

    def __init__(self, num_drones=9):
        self.num_drones = num_drones
        self.width = 1200
        self.height = 800

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Drone Swarm Control Panel")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_small = pygame.font.SysFont('arial', 12)
        self.font = pygame.font.SysFont('arial', 14)
        self.font_med = pygame.font.SysFont('arial', 16, bold=True)
        self.font_big = pygame.font.SysFont('arial', 20, bold=True)

        # View settings
        self.map_x = 250
        self.map_y = 50
        self.map_size = 500
        self.scale = 25  # pixels per meter
        self.center_x = self.map_x + self.map_size // 2
        self.center_y = self.map_y + self.map_size // 2

        # Drone data
        self.positions = np.zeros((num_drones, 3))
        self.targets = np.zeros((num_drones, 3))
        self.velocities = np.zeros((num_drones, 3))

        # State
        self.armed = False
        self.status = "INITIALIZING"
        self.running = True

        # Create buttons
        btn_x = 20
        btn_w = 100
        btn_h = 40
        btn_spacing = 50

        self.buttons = {
            'arm': Button(btn_x, 100, btn_w, btn_h, "ARM", DARK_GREEN, GREEN),
            'takeoff': Button(btn_x + 110, 100, btn_w, btn_h, "TAKEOFF", DARK_BLUE, BLUE),
            'land': Button(btn_x, 160, btn_w, btn_h, "LAND", DARK_RED, RED),
            'disarm': Button(btn_x + 110, 160, btn_w, btn_h, "DISARM", GRAY, LIGHT_GRAY),
            'circle': Button(btn_x, 240, btn_w, btn_h, "CIRCLE", DARK_BLUE, BLUE),
            'grid': Button(btn_x + 110, 240, btn_w, btn_h, "GRID", DARK_BLUE, BLUE),
            'v_form': Button(btn_x, 300, btn_w, btn_h, "V-FORM", DARK_BLUE, BLUE),
            'line': Button(btn_x + 110, 300, btn_w, btn_h, "LINE", DARK_BLUE, BLUE),
            'zoom_in': Button(btn_x, 380, 50, 35, "+", GRAY, LIGHT_GRAY),
            'zoom_out': Button(btn_x + 60, 380, 50, 35, "-", GRAY, LIGHT_GRAY),
            'quit': Button(btn_x, 700, 210, 40, "QUIT", DARK_RED, RED),
        }

        # Resource monitor
        self.resource_monitor = ResourceMonitor()

        # Log messages
        self.log_messages = []
        self.max_log = 15

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_messages.append(f"[{timestamp}] {msg}")
        if len(self.log_messages) > self.max_log:
            self.log_messages.pop(0)
        print(f"[{timestamp}] {msg}")

    def world_to_screen(self, x, y):
        sx = self.center_x + int(x * self.scale)
        sy = self.center_y - int(y * self.scale)
        return sx, sy

    def draw_map(self):
        # Map background
        pygame.draw.rect(self.screen, DARK_GRAY,
                        (self.map_x, self.map_y, self.map_size, self.map_size))
        pygame.draw.rect(self.screen, WHITE,
                        (self.map_x, self.map_y, self.map_size, self.map_size), 2)

        # Grid lines
        for i in range(-20, 21, 5):
            sx, sy = self.world_to_screen(i, 0)
            if self.map_x < sx < self.map_x + self.map_size:
                pygame.draw.line(self.screen, GRAY,
                               (sx, self.map_y), (sx, self.map_y + self.map_size), 1)
            sx, sy = self.world_to_screen(0, i)
            if self.map_y < sy < self.map_y + self.map_size:
                pygame.draw.line(self.screen, GRAY,
                               (self.map_x, sy), (self.map_x + self.map_size, sy), 1)

        # Center cross
        pygame.draw.line(self.screen, WHITE,
                        (self.center_x - 10, self.center_y),
                        (self.center_x + 10, self.center_y), 2)
        pygame.draw.line(self.screen, WHITE,
                        (self.center_x, self.center_y - 10),
                        (self.center_x, self.center_y + 10), 2)

        # Targets
        for i in range(self.num_drones):
            tx, ty = self.world_to_screen(self.targets[i, 0], self.targets[i, 1])
            if self.map_x < tx < self.map_x + self.map_size and \
               self.map_y < ty < self.map_y + self.map_size:
                pygame.draw.circle(self.screen, PINK, (tx, ty), 8, 2)

        # Drones
        for i in range(self.num_drones):
            x, y, z = self.positions[i]
            sx, sy = self.world_to_screen(x, y)

            if not (self.map_x < sx < self.map_x + self.map_size and
                    self.map_y < sy < self.map_y + self.map_size):
                continue

            # Color by altitude
            if z < 0.5:
                color = ORANGE
            elif z < 2:
                color = YELLOW
            else:
                color = GREEN

            size = max(6, int(6 + z * 1.5))

            pygame.draw.circle(self.screen, color, (sx, sy), size)
            pygame.draw.circle(self.screen, WHITE, (sx, sy), size, 2)

            # ID
            label = self.font_small.render(str(i), True, WHITE)
            self.screen.blit(label, (sx - 3, sy - 4))

            # Velocity arrow
            vx, vy = self.velocities[i, 0], self.velocities[i, 1]
            if abs(vx) > 0.05 or abs(vy) > 0.05:
                ex = sx + int(vx * 15)
                ey = sy - int(vy * 15)
                pygame.draw.line(self.screen, CYAN, (sx, sy), (ex, ey), 2)

    def draw_sidebar(self):
        # Title
        title = self.font_big.render("DRONE SWARM", True, CYAN)
        self.screen.blit(title, (20, 20))

        subtitle = self.font.render("Control Panel", True, WHITE)
        self.screen.blit(subtitle, (20, 45))

        # Draw buttons
        for btn in self.buttons.values():
            btn.draw(self.screen, self.font_med)

        # Status section
        pygame.draw.line(self.screen, GRAY, (20, 440), (220, 440), 1)

        status_y = 450
        status_color = GREEN if self.armed else RED
        self.screen.blit(self.font_med.render("STATUS", True, CYAN), (20, status_y))
        status_y += 25

        armed_text = "ARMED" if self.armed else "DISARMED"
        self.screen.blit(self.font.render(f"State: {armed_text}", True, status_color), (20, status_y))
        status_y += 20

        avg_alt = self.positions[:, 2].mean()
        self.screen.blit(self.font.render(f"Altitude: {avg_alt:.1f}m", True, WHITE), (20, status_y))
        status_y += 20

        self.screen.blit(self.font.render(f"Drones: {self.num_drones}", True, WHITE), (20, status_y))
        status_y += 20

        self.screen.blit(self.font.render(f"Mode: {self.status}", True, YELLOW), (20, status_y))

    def draw_resource_panel(self):
        """Draw resource monitoring panel"""
        panel_x = 780
        panel_y = 50
        panel_w = 400
        panel_h = 150

        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), 2)

        self.screen.blit(self.font_med.render("SYSTEM RESOURCES", True, CYAN), (panel_x + 10, panel_y + 10))

        # CPU
        y = panel_y + 40
        self.draw_progress_bar(panel_x + 10, y, 180, 20, self.resource_monitor.cpu_percent, "CPU")

        # RAM
        y += 30
        ram_text = f"RAM ({self.resource_monitor.ram_used:.1f}GB)"
        self.draw_progress_bar(panel_x + 10, y, 180, 20, self.resource_monitor.ram_percent, ram_text)

        # GPU
        y += 30
        self.draw_progress_bar(panel_x + 10, y, 180, 20, self.resource_monitor.gpu_percent, "GPU")

        # GPU Memory
        y = panel_y + 40
        gpu_mem_text = f"VRAM ({self.resource_monitor.gpu_mem_used:.1f}GB)"
        self.draw_progress_bar(panel_x + 210, y, 180, 20, self.resource_monitor.gpu_mem_percent, gpu_mem_text)

    def draw_progress_bar(self, x, y, w, h, percent, label):
        """Draw a progress bar with label"""
        # Background
        pygame.draw.rect(self.screen, GRAY, (x, y, w, h))

        # Fill
        fill_w = int(w * percent / 100)
        if percent < 50:
            color = GREEN
        elif percent < 80:
            color = YELLOW
        else:
            color = RED
        pygame.draw.rect(self.screen, color, (x, y, fill_w, h))

        # Border
        pygame.draw.rect(self.screen, WHITE, (x, y, w, h), 1)

        # Label
        label_surf = self.font_small.render(f"{label}: {percent:.0f}%", True, WHITE)
        self.screen.blit(label_surf, (x, y - 15))

    def draw_log_panel(self):
        """Draw log messages panel"""
        panel_x = 780
        panel_y = 220
        panel_w = 400
        panel_h = 330

        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), 2)

        self.screen.blit(self.font_med.render("LOG", True, CYAN), (panel_x + 10, panel_y + 10))

        y = panel_y + 35
        for msg in self.log_messages[-self.max_log:]:
            label = self.font_small.render(msg[:55], True, LIGHT_GRAY)
            self.screen.blit(label, (panel_x + 10, y))
            y += 18

    def draw_legend(self):
        """Draw color legend"""
        panel_x = 780
        panel_y = 570
        panel_w = 400
        panel_h = 180

        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x, panel_y, panel_w, panel_h))
        pygame.draw.rect(self.screen, WHITE, (panel_x, panel_y, panel_w, panel_h), 2)

        self.screen.blit(self.font_med.render("LEGEND", True, CYAN), (panel_x + 10, panel_y + 10))

        legends = [
            (GREEN, "Flying (>2m)"),
            (YELLOW, "Low altitude (<2m)"),
            (ORANGE, "Ground (<0.5m)"),
            (PINK, "Target position"),
            (CYAN, "Velocity vector"),
        ]

        y = panel_y + 40
        for color, text in legends:
            pygame.draw.circle(self.screen, color, (panel_x + 20, y + 6), 8)
            label = self.font.render(text, True, WHITE)
            self.screen.blit(label, (panel_x + 40, y))
            y += 25

    def draw(self):
        self.screen.fill(BLACK)
        self.draw_sidebar()
        self.draw_map()
        self.draw_resource_panel()
        self.draw_log_panel()
        self.draw_legend()
        pygame.display.flip()

    def handle_events(self):
        mouse_pos = pygame.mouse.get_pos()

        # Update button hover states
        for btn in self.buttons.values():
            btn.check_hover(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    for name, btn in self.buttons.items():
                        if btn.check_click(mouse_pos):
                            return name

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "quit"
                elif event.key == pygame.K_SPACE:
                    return "takeoff"

        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, default=9)
    args = parser.parse_args()

    num_drones = args.drones
    spacing = 3.0

    # Create GUI
    gui = DroneSwarmGUI(num_drones)
    gui.log("Starting Drone Swarm System...")

    # Get world file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    world_file = os.path.join(script_dir, "swarm_world.sdf")

    if not os.path.exists(world_file):
        gui.log(f"ERROR: World file not found")
        gui.status = "ERROR"
        return

    # Start Gazebo
    gui.log("Starting Gazebo headless...")
    gui.status = "STARTING GAZEBO"
    gui.draw()

    gazebo = GazeboHeadless(world_file, num_drones)
    if not gazebo.start():
        gui.log("Failed to start Gazebo!")
        gui.status = "ERROR"
        return

    gui.log("Gazebo started successfully")

    # Spawn drones
    gui.log("Spawning drones...")
    gui.status = "SPAWNING"
    gui.draw()

    spawned = gazebo.spawn_grid(spacing=spacing)
    if spawned == 0:
        gui.log("Failed to spawn drones!")
        gazebo.stop()
        return

    gui.log(f"Spawned {spawned} drones")
    time.sleep(1)

    # Initialize targets
    cols = int(np.ceil(np.sqrt(num_drones)))
    for i in range(num_drones):
        row, col = divmod(i, cols)
        gui.targets[i] = [
            (col - cols/2 + 0.5) * spacing,
            (row - cols/2 + 0.5) * spacing,
            0.3
        ]

    gui.status = "READY"
    gui.log("System ready - click ARM then TAKEOFF")

    # Control parameters
    kp = 1.5

    # Main loop
    try:
        while gui.running:
            # Handle input
            cmd = gui.handle_events()

            if cmd == "quit":
                gui.log("Shutting down...")
                break

            elif cmd == "arm":
                gui.armed = True
                gui.log("ARMED")
                gui.status = "ARMED"

            elif cmd == "disarm":
                gui.armed = False
                gui.log("DISARMED")
                gui.status = "DISARMED"

            elif cmd == "takeoff":
                if not gui.armed:
                    gui.armed = True
                    gui.log("Auto-armed")
                gui.targets[:, 2] = 5.0
                gui.log("TAKEOFF to 5m")
                gui.status = "TAKEOFF"

            elif cmd == "land":
                gui.targets[:, 2] = 0.3
                gui.log("LANDING")
                gui.status = "LANDING"

            elif cmd == "circle":
                angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
                gui.targets[:, 0] = 8 * np.cos(angles)
                gui.targets[:, 1] = 8 * np.sin(angles)
                gui.targets[:, 2] = 5.0
                gui.log("CIRCLE formation")
                gui.status = "CIRCLE"

            elif cmd == "grid":
                for i in range(num_drones):
                    row, col = divmod(i, cols)
                    gui.targets[i, 0] = (col - cols/2 + 0.5) * spacing
                    gui.targets[i, 1] = (row - cols/2 + 0.5) * spacing
                gui.targets[:, 2] = 5.0
                gui.log("GRID formation")
                gui.status = "GRID"

            elif cmd == "v_form":
                for i in range(num_drones):
                    side = i % 2
                    depth = i // 2
                    gui.targets[i, 0] = depth * 2
                    gui.targets[i, 1] = (side * 2 - 1) * (depth + 1) * 1.5
                gui.targets[:, 2] = 5.0
                gui.log("V formation")
                gui.status = "V-FORM"

            elif cmd == "line":
                for i in range(num_drones):
                    gui.targets[i, 0] = (i - num_drones/2) * 2
                    gui.targets[i, 1] = 0
                gui.targets[:, 2] = 5.0
                gui.log("LINE formation")
                gui.status = "LINE"

            elif cmd == "zoom_in":
                gui.scale = min(50, gui.scale + 3)

            elif cmd == "zoom_out":
                gui.scale = max(10, gui.scale - 3)

            # Update positions from Gazebo
            gazebo.update_all_poses()
            gui.positions = gazebo.positions.copy()

            # Send velocity commands
            if gui.armed:
                for i in range(num_drones):
                    if gazebo.connected[i]:
                        error = gui.targets[i] - gui.positions[i]
                        vel = error * kp
                        speed = np.linalg.norm(vel)
                        if speed > 2.0:
                            vel = vel / speed * 2.0
                        gui.velocities[i] = vel
                        gazebo.set_velocity(i, vel[0], vel[1], vel[2])

            # Update resource monitor
            gui.resource_monitor.update()

            # Draw
            gui.draw()
            gui.clock.tick(20)

    except KeyboardInterrupt:
        gui.log("Interrupted")

    finally:
        gazebo.stop()
        pygame.quit()
        print("Done!")


if __name__ == "__main__":
    main()
