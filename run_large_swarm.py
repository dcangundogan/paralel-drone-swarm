#!/usr/bin/env python3
"""
Large Swarm Simulator - 100+ Drones
====================================

Gazebo KULLANMADAN 100-1000 drone simülasyonu.
Tümü GPU üzerinde çalışır.

Kullanım:
    python run_large_swarm.py --drones 100
    python run_large_swarm.py --drones 500
    python run_large_swarm.py --drones 1000
"""

import argparse
import numpy as np
import time
import threading
from typing import Optional

# Modüller
from lightweight_physics import LightweightPhysicsEngine, QuadcopterParams, EnvironmentParams
from gpu_sensors import GPUSensorSimulator, SensorConfig
from visualizer_with_sensors import AdvancedSwarmVisualizer, VisualizerConfig


class LargeSwarmSimulator:
    """
    100+ drone için tam simülasyon sistemi.

    Gazebo yerine lightweight GPU physics kullanır.
    """

    def __init__(self, num_drones: int = 100,
                 enable_sensors: bool = True,
                 enable_wind: bool = True):

        self.num_drones = num_drones
        self.enable_sensors = enable_sensors
        self.running = False

        print("\n" + "="*70)
        print(f"  LARGE SWARM SIMULATOR - {num_drones} DRONES")
        print("="*70)

        # === PHYSICS ENGINE ===
        quad_params = QuadcopterParams(
            mass=1.5,
            max_thrust=10.0,
            motor_time_constant=0.05,
            drag_coefficient_xy=0.5,
        )

        env_params = EnvironmentParams(
            wind_enabled=enable_wind,
            wind_gust_strength=0.5 if enable_wind else 0.0,
            wind_gust_frequency=0.05,
        )

        self.physics = LightweightPhysicsEngine(
            num_drones,
            quad_params,
            env_params,
            dt=0.004  # 250 Hz physics
        )
        self.physics.reset()

        # === SENSORS (optional) ===
        if enable_sensors:
            sensor_config = SensorConfig(
                gps_horizontal_std=1.0,
                gps_dropout_prob=0.01,
                imu_rate=200.0,  # Düşür, performans için
            )
            self.sensors = GPUSensorSimulator(num_drones, sensor_config)
            print(f"[SYSTEM] Sensors: ENABLED")
        else:
            self.sensors = None
            print(f"[SYSTEM] Sensors: DISABLED (faster)")

        # === STATE ===
        self.targets = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocity_commands = np.zeros((num_drones, 3), dtype=np.float32)

        # Control params
        self.target_gain_p = 1.5
        self.target_gain_d = 0.3
        self.prev_error = np.zeros((num_drones, 3), dtype=np.float32)

        self.collision_radius = 0.8
        self.avoidance_radius = 2.0
        self.avoidance_strength = 2.5

        self.default_altitude = 5.0

        # === VISUALIZER ===
        viz_config = VisualizerConfig(
            window_width=1400,
            window_height=900,
            meters_per_pixel=0.1,  # Daha uzak görünüm
        )
        self.visualizer = AdvancedSwarmVisualizer(num_drones, viz_config)

        # Init positions
        self._init_grid_positions()

        print(f"[SYSTEM] Physics rate: {1/self.physics.dt:.0f} Hz")
        print(f"[SYSTEM] Ready!")
        print("="*70)

    def _init_grid_positions(self):
        """Grid başlangıç pozisyonları"""
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        spacing = 2.5

        positions = np.zeros((self.num_drones, 3), dtype=np.float32)
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            positions[i] = [
                (col - cols/2) * spacing,
                (row - cols/2) * spacing,
                0.1
            ]

        self.physics.positions[:] = positions
        self.targets = positions.copy()

    def arm_all(self):
        self.physics.arm()
        print(f"[SYSTEM] {self.num_drones} drones ARMED")

    def disarm_all(self):
        self.physics.disarm()
        print(f"[SYSTEM] {self.num_drones} drones DISARMED")

    def takeoff(self, altitude: float = None):
        alt = altitude or self.default_altitude
        self.arm_all()
        self.targets[:, 2] = alt
        print(f"[SYSTEM] Takeoff to {alt}m")

    def land(self):
        self.targets[:, 2] = 0.1
        print(f"[SYSTEM] Landing")

    def set_formation_grid(self, spacing: float = 3.0):
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        center = np.mean(self.physics.get_positions()[:, :2], axis=0)

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.targets[i, 0] = center[0] + (col - cols/2) * spacing
            self.targets[i, 1] = center[1] + (row - cols/2) * spacing

        print(f"[SYSTEM] Grid formation ({spacing}m spacing)")

    def set_formation_circle(self, radius: float = None):
        if radius is None:
            radius = max(10.0, self.num_drones * 0.3)  # Auto-scale

        center = np.mean(self.physics.get_positions()[:, :2], axis=0)
        angles = np.linspace(0, 2 * np.pi, self.num_drones, endpoint=False)

        for i in range(self.num_drones):
            self.targets[i, 0] = center[0] + radius * np.cos(angles[i])
            self.targets[i, 1] = center[1] + radius * np.sin(angles[i])

        print(f"[SYSTEM] Circle formation (radius={radius:.1f}m)")

    def set_formation_spiral(self, turns: float = 3.0):
        """Spiral formasyon - büyük sürüler için güzel görünür"""
        center = np.mean(self.physics.get_positions()[:, :2], axis=0)

        for i in range(self.num_drones):
            t = i / self.num_drones
            angle = t * turns * 2 * np.pi
            radius = 5 + t * 30  # İçten dışa büyüyen

            self.targets[i, 0] = center[0] + radius * np.cos(angle)
            self.targets[i, 1] = center[1] + radius * np.sin(angle)

        print(f"[SYSTEM] Spiral formation")

    def set_formation_heart(self):
        """Kalp şekli formasyonu"""
        center = np.mean(self.physics.get_positions()[:, :2], axis=0)
        scale = max(15, self.num_drones * 0.2)

        for i in range(self.num_drones):
            t = i / self.num_drones * 2 * np.pi

            # Heart parametric equation
            x = 16 * np.sin(t)**3
            y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)

            self.targets[i, 0] = center[0] + x * scale / 16
            self.targets[i, 1] = center[1] + y * scale / 16

        print(f"[SYSTEM] Heart formation")

    def _compute_separation(self, positions: np.ndarray) -> np.ndarray:
        """Basitleştirilmiş collision avoidance - büyük sürüler için optimize"""
        separation = np.zeros_like(positions)

        # Spatial hashing yerine brute force (GPU'da hala hızlı)
        for i in range(self.num_drones):
            diff = positions[i] - positions
            dist = np.linalg.norm(diff, axis=1)

            # Sadece yakın olanları hesapla
            mask = (dist < self.avoidance_radius) & (dist > 0.01)

            if np.any(mask):
                directions = diff[mask] / dist[mask, None]
                strengths = (self.avoidance_radius - dist[mask]) / self.avoidance_radius
                separation[i] = np.sum(directions * strengths[:, None], axis=0) * self.avoidance_strength

        return separation

    def _compute_control(self, dt: float):
        """Control hesapla"""
        positions = self.physics.get_positions()
        velocities = self.physics.get_velocities()

        # Eğer sensörler varsa, tahminleri kullan
        if self.sensors:
            self.sensors.update_ground_truth(positions, velocities)
            positions, velocities = self.sensors.update(dt)

        # Separation
        separation = self._compute_separation(positions)

        # Target tracking (PD)
        error = self.targets - positions
        d_error = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error.copy()

        tracking = error * self.target_gain_p + d_error * self.target_gain_d

        # Combine
        self.velocity_commands = separation + tracking

        # Velocity limit
        for i in range(self.num_drones):
            speed = np.linalg.norm(self.velocity_commands[i, :2])
            if speed > 3.0:
                self.velocity_commands[i, :2] *= 3.0 / speed
            self.velocity_commands[i, 2] = np.clip(self.velocity_commands[i, 2], -2.0, 2.0)

    def _control_loop(self):
        """Background control thread"""
        control_rate = 50.0  # Hz
        physics_substeps = int(1.0 / control_rate / self.physics.dt)

        dt = 1.0 / control_rate
        last_time = time.time()

        while self.running:
            current_time = time.time()
            actual_dt = current_time - last_time
            last_time = current_time

            # Control
            self._compute_control(actual_dt)

            # Physics'e komut ver ve simüle et
            self.physics.set_velocity_commands(self.velocity_commands)
            self.physics.step(num_substeps=physics_substeps)

            # Rate limit
            elapsed = time.time() - current_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        """Ana çalıştırma"""
        self.running = True

        # Control thread
        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        control_thread.start()

        print("\n" + "="*70)
        print("CONTROLS:")
        print("  CLICK MAP    - Tüm drone'lar oraya gider")
        print("  1-9          - Drone seç (sensör görünümü)")
        print("  SPACE        - Kalkış")
        print("  L            - İniş")
        print("  G            - Grid formasyon")
        print("  C            - Daire formasyon")
        print("  S            - Spiral formasyon")
        print("  H            - Kalp formasyon")
        print("  +/-          - Zoom")
        print("  Q/ESC        - Çıkış")
        print("="*70)

        # Initial takeoff
        time.sleep(0.5)
        self.takeoff()

        # Visualization loop
        try:
            while self.running and self.visualizer.running:
                if not self.visualizer.handle_events():
                    break

                # Extra keyboard handling
                self._handle_extra_keys()

                # Waypoint
                waypoint = self.visualizer.get_waypoint()
                if waypoint is not None:
                    selected = self.visualizer.get_selected_drones()
                    for drone_id in selected:
                        self.targets[drone_id, 0] = waypoint[0]
                        self.targets[drone_id, 1] = waypoint[1]
                        if self.targets[drone_id, 2] < 1.0:
                            self.targets[drone_id, 2] = self.default_altitude

                # Update visualizer
                positions = self.physics.get_positions()
                velocities = self.physics.get_velocities()
                armed = self.physics.armed

                # Numpy'a çevir (GPU'daysa)
                try:
                    import cupy as cp
                    if isinstance(armed, cp.ndarray):
                        armed = cp.asnumpy(armed)
                except:
                    pass

                self.visualizer.update_state(
                    positions,
                    velocities,
                    self.targets,
                    armed
                )

                # Sensor data
                if self.sensors:
                    sensor_data = self.sensors.get_raw_sensors()
                    errors = self.sensors.get_estimation_error()
                    self.visualizer.update_sensor_data(sensor_data, errors)

                self.visualizer.draw()

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted")

        finally:
            self.running = False
            self.visualizer.quit()
            print("[SYSTEM] Shutdown complete")

    def _handle_extra_keys(self):
        """Ek klavye kontrolleri"""
        import pygame
        keys = pygame.key.get_pressed()

        if keys[pygame.K_SPACE]:
            self.takeoff()
            time.sleep(0.3)
        elif keys[pygame.K_l]:
            self.land()
            time.sleep(0.3)
        elif keys[pygame.K_g]:
            self.set_formation_grid()
            self.visualizer.clear_waypoint()
            time.sleep(0.3)
        elif keys[pygame.K_c]:
            self.set_formation_circle()
            self.visualizer.clear_waypoint()
            time.sleep(0.3)
        elif keys[pygame.K_s]:
            self.set_formation_spiral()
            self.visualizer.clear_waypoint()
            time.sleep(0.3)
        elif keys[pygame.K_h]:
            self.set_formation_heart()
            self.visualizer.clear_waypoint()
            time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="Large Swarm Simulator")
    parser.add_argument('--drones', type=int, default=100, help='Number of drones (default: 100)')
    parser.add_argument('--no-sensors', action='store_true', help='Disable sensor simulation')
    parser.add_argument('--no-wind', action='store_true', help='Disable wind')
    args = parser.parse_args()

    sim = LargeSwarmSimulator(
        num_drones=args.drones,
        enable_sensors=not args.no_sensors,
        enable_wind=not args.no_wind
    )

    sim.run()


if __name__ == "__main__":
    main()
