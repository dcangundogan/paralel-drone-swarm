#!/usr/bin/env python3
"""
Performance Monitoring System
==============================

Monitors and displays real-time performance metrics for the drone swarm simulation:
- FPS (Frames Per Second)
- CPU and RAM usage
- GPU utilization and memory
- Timing breakdown for different simulation sections
- FPS history graph

Author: Ali (Visualization & UI Module)
Date: 2024-01-20
"""

import numpy as np
import time
import psutil
import subprocess
from collections import deque
from typing import Dict, Optional
import pygame


class PerformanceMonitor:
    """
    Real-time performance monitoring and visualization.

    Tracks frame timing, system resources, and provides rendering
    for Pygame-based visualization.

    Attributes:
        history_size: Number of frames to keep in history (default: 60)
        fps_history: Deque of recent FPS values
        timing_sections: Dict of timing data for different sections

    Example:
        >>> monitor = PerformanceMonitor(history_size=60)
        >>> monitor.start_frame()
        >>>
        >>> monitor.start_section('physics')
        >>> # ... physics code ...
        >>> monitor.end_section('physics')
        >>>
        >>> monitor.end_frame()
        >>> stats = monitor.get_stats()
        >>> print(f"FPS: {stats['fps']:.1f}")
    """

    def __init__(self, history_size: int = 60):
        """
        Initialize performance monitor.

        Args:
            history_size: Number of frames to track in history for graphing
        """
        self.history_size = history_size

        # Frame timing
        self.frame_start = 0
        self.frame_count = 0
        self.fps_history = deque(maxlen=history_size)

        # Section timing
        self.section_start = 0
        self.current_section = None
        self.section_times = {}  # Current frame section times

        # Historical timing data
        self.physics_time = deque(maxlen=history_size)
        self.control_time = deque(maxlen=history_size)
        self.sensors_time = deque(maxlen=history_size)
        self.render_time = deque(maxlen=history_size)

        # System stats
        self.cpu_percent = 0.0
        self.ram_used_gb = 0.0
        self.gpu_percent = 0
        self.gpu_memory_mb = 0.0

        # System stats update rate limiting (update every 1 second)
        self.last_system_update = 0
        self.system_update_interval = 1.0

        # GPU availability check
        self.gpu_available = self._check_gpu_available()

        print("[PERF] Performance monitor initialized")
        if self.gpu_available:
            print("[PERF] GPU monitoring enabled")
        else:
            print("[PERF] GPU monitoring disabled (not available)")

    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import cupy as cp

            # Also check if nvidia-smi is available
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=1,
            )
            return result.returncode == 0
        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def start_frame(self):
        """Mark the start of a new frame."""
        self.frame_start = time.perf_counter()
        self.section_times = {}
        self.frame_count += 1

    def end_frame(self):
        """
        Mark the end of frame and calculate FPS.

        Call this at the end of your main loop iteration.
        """
        frame_time = (time.perf_counter() - self.frame_start) * 1000  # ms

        # Calculate FPS (avoid division by zero)
        fps = 1000.0 / max(frame_time, 0.1)
        self.fps_history.append(fps)

        # Store section times in history
        if "physics" in self.section_times:
            self.physics_time.append(self.section_times["physics"])
        if "control" in self.section_times:
            self.control_time.append(self.section_times["control"])
        if "sensors" in self.section_times:
            self.sensors_time.append(self.section_times["sensors"])
        if "render" in self.section_times:
            self.render_time.append(self.section_times["render"])

        # Update system stats periodically
        current_time = time.time()
        if current_time - self.last_system_update >= self.system_update_interval:
            self.update_system_stats()
            self.last_system_update = current_time

    def start_section(self, name: str):
        """
        Start timing a section of code.

        Args:
            name: Section name (e.g., 'physics', 'control', 'sensors', 'render')
        """
        self.current_section = name
        self.section_start = time.perf_counter()

    def end_section(self, name: str):
        """
        End timing a section and record the elapsed time.

        Args:
            name: Section name (must match start_section call)
        """
        if self.current_section != name:
            print(
                f"[PERF] Warning: Section mismatch. Expected {self.current_section}, got {name}"
            )

        elapsed_ms = (time.perf_counter() - self.section_start) * 1000
        self.section_times[name] = elapsed_ms
        self.current_section = None

    def update_system_stats(self):
        """
        Update system statistics (CPU, RAM, GPU).

        This is called automatically by end_frame() at a rate-limited interval
        to avoid excessive system calls.
        """
        # CPU usage (percentage)
        self.cpu_percent = psutil.cpu_percent(interval=None)

        # RAM usage (GB)
        mem = psutil.virtual_memory()
        self.ram_used_gb = mem.used / (1024**3)

        # GPU monitoring (if available)
        if self.gpu_available:
            try:
                # GPU utilization via nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=0.5,
                )
                if result.returncode == 0:
                    self.gpu_percent = int(result.stdout.strip())

                # GPU memory via CuPy
                import cupy as cp

                mempool = cp.get_default_memory_pool()
                self.gpu_memory_mb = mempool.used_bytes() / (1024**2)

            except (subprocess.TimeoutExpired, ValueError, Exception) as e:
                # Silently fail - GPU stats will show last known value
                pass

    def get_stats(self) -> Dict:
        """
        Get current performance statistics.

        Returns:
            Dictionary containing:
                - fps: Current FPS
                - fps_avg: Average FPS over history
                - fps_min: Minimum FPS in history
                - fps_max: Maximum FPS in history
                - fps_history: List of recent FPS values
                - physics_ms: Latest physics timing
                - control_ms: Latest control timing
                - sensors_ms: Latest sensors timing
                - render_ms: Latest render timing
                - total_ms: Sum of all section times
                - cpu_percent: CPU usage percentage
                - ram_gb: RAM usage in GB
                - gpu_percent: GPU utilization percentage
                - gpu_mb: GPU memory usage in MB
        """
        fps = self.fps_history[-1] if self.fps_history else 0
        fps_avg = (
            sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        )
        fps_min = min(self.fps_history) if self.fps_history else 0
        fps_max = max(self.fps_history) if self.fps_history else 0

        physics_ms = self.physics_time[-1] if self.physics_time else 0
        control_ms = self.control_time[-1] if self.control_time else 0
        sensors_ms = self.sensors_time[-1] if self.sensors_time else 0
        render_ms = self.render_time[-1] if self.render_time else 0
        total_ms = physics_ms + control_ms + sensors_ms + render_ms

        return {
            "fps": fps,
            "fps_avg": fps_avg,
            "fps_min": fps_min,
            "fps_max": fps_max,
            "fps_history": list(self.fps_history),
            "physics_ms": physics_ms,
            "control_ms": control_ms,
            "sensors_ms": sensors_ms,
            "render_ms": render_ms,
            "total_ms": total_ms,
            "cpu_percent": self.cpu_percent,
            "ram_gb": self.ram_used_gb,
            "gpu_percent": self.gpu_percent,
            "gpu_mb": self.gpu_memory_mb,
        }

    def draw(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        width: int,
        font_normal,
        font_small,
    ) -> int:
        """
        Draw performance panel on Pygame surface.

        This method renders the performance metrics as part of the sensor panel.

        Args:
            surface: Pygame surface to draw on
            x: X position of panel
            y: Y position to start drawing
            width: Width of the panel
            font_normal: Pygame font for normal text
            font_small: Pygame font for small text

        Returns:
            Y position after drawing (for stacking panels)
        """
        stats = self.get_stats()

        # Section header
        y = self._draw_section_header(
            surface, "PERFORMANCE [P to hide]", x, y, width, (150, 200, 255)
        )

        # === TOP ROW: FPS | CPU | GPU ===
        fps_color = (100, 255, 100) if stats["fps"] >= 30 else (255, 100, 100)

        line1 = f"FPS: {stats['fps']:.0f}  CPU: {stats['cpu_percent']:.0f}%  GPU: {stats['gpu_percent']}%"
        text = font_normal.render(line1, True, fps_color)
        surface.blit(text, (x, y))
        y += 18

        # === SECOND ROW: RAM | GPU Memory ===
        line2 = f"RAM: {stats['ram_gb']:.1f}GB  GPU Mem: {stats['gpu_mb']:.2f}MB"
        text = font_small.render(line2, True, (180, 180, 180))
        surface.blit(text, (x, y))
        y += 20

        # === TIMING BARS ===
        label = font_small.render("Timing (ms):", True, (150, 150, 150))
        surface.blit(label, (x, y))
        y += 18

        # Bar parameters
        bar_max_width = width - 140
        bar_height = 10
        max_time_ms = 20  # 20ms = full bar

        # Define sections with colors
        sections = [
            ("Physics", stats["physics_ms"], (100, 150, 255)),
            ("Control", stats["control_ms"], (150, 255, 100)),
            ("Sensors", stats["sensors_ms"], (255, 200, 100)),
            ("Render", stats["render_ms"], (255, 150, 150)),
        ]

        for section_name, time_ms, color in sections:
            # Label
            label_text = font_small.render(f"{section_name}:", True, (160, 160, 160))
            surface.blit(label_text, (x + 5, y))

            # Value
            value_text = font_small.render(f"{time_ms:.1f}", True, (200, 200, 200))
            surface.blit(value_text, (x + 75, y))

            # Bar
            bar_width = min(bar_max_width, int((time_ms / max_time_ms) * bar_max_width))
            bar_x = x + 115
            bar_y = y + 2

            # Background bar (outline)
            pygame.draw.rect(
                surface, (60, 60, 70), (bar_x, bar_y, bar_max_width, bar_height), 1
            )

            # Filled bar
            if bar_width > 0:
                pygame.draw.rect(surface, color, (bar_x, bar_y, bar_width, bar_height))

            y += 15

        y += 5

        # === FPS HISTORY GRAPH ===
        if stats["fps_history"] and len(stats["fps_history"]) > 1:
            graph_label = font_small.render(
                f"FPS History ({len(stats['fps_history'])} frames):",
                True,
                (150, 150, 150),
            )
            surface.blit(graph_label, (x, y))
            y += 18

            # Draw mini line graph
            graph_width = width - 20
            graph_height = 40
            graph_x = x + 10
            graph_y = y

            # Background
            pygame.draw.rect(
                surface, (25, 25, 35), (graph_x, graph_y, graph_width, graph_height)
            )
            pygame.draw.rect(
                surface, (60, 60, 70), (graph_x, graph_y, graph_width, graph_height), 1
            )

            # Plot FPS values
            fps_values = stats["fps_history"]
            if len(fps_values) >= 2:
                # Normalize values
                min_fps = max(0, stats["fps_min"] - 5)
                max_fps = stats["fps_max"] + 5
                fps_range = max_fps - min_fps

                if fps_range > 0:
                    points = []
                    for i, fps_val in enumerate(fps_values):
                        px = graph_x + int((i / (len(fps_values) - 1)) * graph_width)
                        normalized = (fps_val - min_fps) / fps_range
                        py = graph_y + graph_height - int(normalized * graph_height)
                        points.append((px, py))

                    # Draw line
                    if len(points) > 1:
                        pygame.draw.lines(surface, (100, 200, 100), False, points, 2)

                # Draw reference lines
                # 30 FPS line (minimum acceptable)
                if min_fps < 30 < max_fps:
                    ref_y = (
                        graph_y
                        + graph_height
                        - int(((30 - min_fps) / fps_range) * graph_height)
                    )
                    pygame.draw.line(
                        surface,
                        (255, 150, 150),
                        (graph_x, ref_y),
                        (graph_x + graph_width, ref_y),
                        1,
                    )

                # Min/Max labels
                min_label = font_small.render(f"{min_fps:.0f}", True, (120, 120, 120))
                max_label = font_small.render(f"{max_fps:.0f}", True, (120, 120, 120))
                surface.blit(
                    min_label, (graph_x + graph_width + 5, graph_y + graph_height - 10)
                )
                surface.blit(max_label, (graph_x + graph_width + 5, graph_y))

            y += graph_height + 10

        # Add some spacing at the bottom
        y += 10

        return y

    def _draw_section_header(
        self, surface, title: str, x: int, y: int, width: int, color: tuple
    ) -> int:
        """Draw a section header with separator line."""
        # Separator line
        pygame.draw.line(surface, (60, 65, 70), (x, y), (x + width - 30, y), 1)
        y += 8

        # Title
        font = pygame.font.SysFont("consolas", 14, bold=True)
        text = font.render(title, True, color)
        surface.blit(text, (x, y))
        y += 22

        return y


# ============================================================================
# STANDALONE DEMO
# ============================================================================


def demo_performance_monitor():
    """
    Standalone demo of the performance monitor.

    Shows how to use the monitor in a typical game loop.
    """
    print("=" * 60)
    print("PERFORMANCE MONITOR DEMO")
    print("=" * 60)
    print("A simple demo showing performance monitoring in action.")
    print("Press Q or ESC to quit")
    print("=" * 60)

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Performance Monitor Demo")
    clock = pygame.time.Clock()

    # Fonts
    font_normal = pygame.font.SysFont("consolas", 12)
    font_small = pygame.font.SysFont("consolas", 10)

    # Create monitor
    monitor = PerformanceMonitor(history_size=60)

    # Demo variables
    show_performance = True
    running = True

    while running:
        monitor.start_frame()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_p:
                    show_performance = not show_performance

        # Simulate some work
        monitor.start_section("physics")
        time.sleep(0.002)  # 2ms of "physics"
        monitor.end_section("physics")

        monitor.start_section("control")
        time.sleep(0.001)  # 1ms of "control"
        monitor.end_section("control")

        monitor.start_section("sensors")
        time.sleep(0.0005)  # 0.5ms of "sensors"
        monitor.end_section("sensors")

        # Render
        monitor.start_section("render")
        screen.fill((20, 25, 30))

        # Draw performance panel if enabled
        if show_performance:
            monitor.draw(screen, 20, 20, 360, font_normal, font_small)

        # Instructions
        instructions = [
            "Press P to toggle performance panel",
            "Press Q or ESC to quit",
        ]
        y = 300
        for instruction in instructions:
            text = font_normal.render(instruction, True, (150, 150, 150))
            screen.blit(text, (20, y))
            y += 20

        pygame.display.flip()
        monitor.end_section("render")

        monitor.end_frame()
        clock.tick(60)  # Target 60 FPS

    pygame.quit()
    print("[DEMO] Performance monitor demo finished")


if __name__ == "__main__":
    demo_performance_monitor()
