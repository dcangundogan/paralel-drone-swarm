#!/usr/bin/env python3
"""
Integrated Swarm Runner with Sensors and Click-to-Navigate
===========================================================

Bu script tüm sistemi birleştirir:
- GPU Controller (sensör destekli)
- GPU Sensör Simülasyonu (IMU, GPS, Baro, Mag + Kalman Filter)
- Gelişmiş Visualizer (sensör paneli + tıkla-git)
- Opsiyonel Gazebo entegrasyonu

Kullanım:
    python run_with_sensors.py --drones 15
    python run_with_sensors.py --drones 25 --no-gazebo
"""

import argparse
import numpy as np
import time
import threading
import os
from typing import Optional

# Configure CUDA paths before importing CuPy
_MATLAB_LIB_PATH = "/usr/local/MATLAB/R2025b/bin/glnxa64"

if os.path.exists(_MATLAB_LIB_PATH):
    _ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _MATLAB_LIB_PATH not in _ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{_MATLAB_LIB_PATH}:{_ld_path}"
    import ctypes

    try:
        ctypes.CDLL(f"{_MATLAB_LIB_PATH}/libnvrtc.so.12", mode=ctypes.RTLD_GLOBAL)
    except:
        pass

# Modülleri import et
from gpu_sensors import GPUSensorSimulator, SensorConfig, GPU_AVAILABLE as SENSORS_GPU
from visualizer_with_sensors import AdvancedSwarmVisualizer, VisualizerConfig
from performance_monitor import PerformanceMonitor


# GPU kontrolü
GPU_AVAILABLE = False
try:
    import cupy as cp

    _test = cp.arange(1)
    del _test
    GPU_AVAILABLE = True
except (ImportError, Exception):
    import numpy as cp


class IntegratedSwarmSystem:
    """
    Tam entegre swarm sistemi:
    - Sensör simülasyonu
    - Swarm kontrolü
    - Görselleştirme
    - Tıkla-git navigasyonu
    """

    def __init__(self, num_drones: int = 10, use_gazebo: bool = False):
        self.num_drones = num_drones
        self.use_gazebo = use_gazebo
        self.running = False
        self.control_rate = 50.0  # Hz

        print("\n" + "=" * 60)
        print("INTEGRATED SWARM SYSTEM WITH SENSORS")
        print("=" * 60)
        print(f"Drones: {num_drones}")
        print(f"GPU: {GPU_AVAILABLE}")
        print(f"Gazebo: {use_gazebo}")
        print("=" * 60)

        # === SENSÖR SİMÜLASYONU ===
        sensor_config = SensorConfig(
            imu_rate=400.0,
            gps_rate=10.0,
            gps_horizontal_std=1.5,
            gps_vertical_std=3.0,
            gps_dropout_prob=0.02,  # %2 GPS kaybı
            baro_noise_std=0.5,
        )
        self.sensors = GPUSensorSimulator(num_drones, sensor_config)

        # === STATE ARRAYS ===
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)
        self.targets = np.zeros((num_drones, 3), dtype=np.float32)
        self.armed = np.zeros(num_drones, dtype=bool)

        # Estimated state (sensörlerden)
        self.estimated_positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.estimated_velocities = np.zeros((num_drones, 3), dtype=np.float32)

        # PD Controller için
        self.prev_error = np.zeros((num_drones, 3), dtype=np.float32)

        # === CONTROLLER PARAMETRELERİ ===
        self.max_velocity_xy = 3.0
        self.max_velocity_z = 2.0
        self.max_acceleration = 2.0
        self.collision_radius = 0.8
        self.avoidance_radius = 2.5
        self.avoidance_strength = 3.0
        self.target_gain_p = 1.5
        self.target_gain_d = 0.3
        self.default_altitude = 5.0

        # === VİSUALİZER ===
        viz_config = VisualizerConfig(
            window_width=1400,
            window_height=900,
        )
        self.visualizer = AdvancedSwarmVisualizer(num_drones, viz_config)

        # === PERFORMANCE MONITOR ===
        self.performance_monitor = PerformanceMonitor(history_size=60)
        self.visualizer.set_performance_monitor(self.performance_monitor)
        print("[SYSTEM] Performance monitor initialized")

        # === GAZEBO (opsiyonel) ===
        self.gazebo_bridge = None
        if use_gazebo:
            try:
                from gz_bridge import GazeboBridge

                self.gazebo_bridge = GazeboBridge()
                print("[SYSTEM] Gazebo bridge initialized")
            except Exception as e:
                print(f"[SYSTEM] Gazebo not available: {e}")
                self.use_gazebo = False

        # Başlangıç pozisyonları (grid)
        self._init_positions()

        print("[SYSTEM] Initialization complete")

    def _init_positions(self):
        """Başlangıç pozisyonlarını grid olarak ayarla"""
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        spacing = 3.0

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.positions[i] = [
                col * spacing - (cols * spacing / 2),
                row * spacing - (cols * spacing / 2),
                0.2,
            ]
            self.targets[i] = self.positions[i].copy()

        self.estimated_positions = self.positions.copy()

    def arm_all(self):
        """Tüm drone'ları arm et"""
        self.armed[:] = True
        print("[SYSTEM] All drones armed")

    def disarm_all(self):
        """Tüm drone'ları disarm et"""
        self.armed[:] = False
        print("[SYSTEM] All drones disarmed")

    def takeoff(self, altitude: float = None):
        """Kalkış"""
        alt = altitude or self.default_altitude
        self.arm_all()
        self.targets[:, 2] = alt
        print(f"[SYSTEM] Takeoff to {alt}m")

    def land(self):
        """İniş"""
        self.targets[:, 2] = 0.1
        print("[SYSTEM] Landing")

    def set_formation_grid(self, spacing: float = 4.0):
        """Grid formasyonu"""
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        center = np.mean(self.positions[:, :2], axis=0)

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.targets[i, 0] = center[0] + (col - cols / 2) * spacing
            self.targets[i, 1] = center[1] + (row - cols / 2) * spacing

        print(f"[SYSTEM] Grid formation (spacing={spacing}m)")

    def set_formation_circle(self, radius: float = 10.0):
        """Daire formasyonu"""
        center = np.mean(self.positions[:, :2], axis=0)
        angles = np.linspace(0, 2 * np.pi, self.num_drones, endpoint=False)

        for i in range(self.num_drones):
            self.targets[i, 0] = center[0] + radius * np.cos(angles[i])
            self.targets[i, 1] = center[1] + radius * np.sin(angles[i])

        print(f"[SYSTEM] Circle formation (radius={radius}m)")

    def _compute_separation(self, positions: np.ndarray) -> np.ndarray:
        """Collision avoidance (CPU version)"""
        separation = np.zeros_like(positions)

        for i in range(self.num_drones):
            for j in range(self.num_drones):
                if i == j:
                    continue

                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)

                if dist < self.avoidance_radius and dist > 0.001:
                    direction = diff / dist

                    if dist < self.collision_radius:
                        strength = self.avoidance_strength
                    else:
                        strength = (self.avoidance_radius - dist) / (
                            self.avoidance_radius - self.collision_radius
                        )

                    separation[i] += direction * strength

        return separation

    def _compute_control(self, dt: float) -> np.ndarray:
        """Ana kontrol hesaplaması"""

        # Sensörlerden gelen tahminleri kullan
        positions = self.estimated_positions
        velocities = self.estimated_velocities

        # 1. Collision avoidance
        separation = self._compute_separation(positions)

        # 2. Target tracking (PD)
        error = self.targets - positions
        d_error = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error.copy()

        tracking = error * self.target_gain_p + d_error * self.target_gain_d

        # 3. Blend
        command = self.avoidance_strength * separation + tracking

        # 4. Velocity limits
        for i in range(self.num_drones):
            # XY limit
            xy_speed = np.linalg.norm(command[i, :2])
            if xy_speed > self.max_velocity_xy:
                command[i, :2] *= self.max_velocity_xy / xy_speed

            # Z limit
            command[i, 2] = np.clip(
                command[i, 2], -self.max_velocity_z, self.max_velocity_z
            )

            # Disarmed = no movement
            if not self.armed[i]:
                command[i] = 0

        return command

    def _physics_step(self, command: np.ndarray, dt: float):
        """Basit fizik simülasyonu (Gazebo yoksa)"""
        # Hız güncelle (basit 1st order)
        self.velocities = command * 0.8 + self.velocities * 0.2

        # Pozisyon güncelle
        self.positions += self.velocities * dt

        # Yer sınırı
        self.positions[:, 2] = np.maximum(self.positions[:, 2], 0.05)

    def _control_loop(self):
        """Ana kontrol döngüsü"""
        dt = 1.0 / self.control_rate
        last_time = time.time()

        while self.running:
            current_time = time.time()
            actual_dt = current_time - last_time
            last_time = current_time

            # === 1. GROUND TRUTH GÜNCELLE ===
            if self.use_gazebo and self.gazebo_bridge:
                # Gazebo'dan pozisyon al
                for i in range(self.num_drones):
                    pose = self.gazebo_bridge.get_drone_pose(i)
                    if pose:
                        self.positions[i] = [pose["x"], pose["y"], pose["z"]]
            # else: pozisyonlar physics_step'te güncelleniyor

            # === 2. SENSÖR SİMÜLASYONU ===
            self.performance_monitor.start_section("sensors")
            self.sensors.update_ground_truth(self.positions, self.velocities)
            self.estimated_positions, self.estimated_velocities = self.sensors.update(
                actual_dt
            )
            self.performance_monitor.end_section("sensors")

            # === 3. KONTROL HESAPLA ===
            self.performance_monitor.start_section("control")
            command = self._compute_control(actual_dt)
            self.performance_monitor.end_section("control")

            # === 4. KOMUT UYGULA ===
            self.performance_monitor.start_section("physics")
            if self.use_gazebo and self.gazebo_bridge:
                # Gazebo'ya gönder
                for i in range(self.num_drones):
                    self.gazebo_bridge.send_velocity_command(i, command[i])
            else:
                # Yerel fizik simülasyonu
                self._physics_step(command, actual_dt)
            self.performance_monitor.end_section("physics")

            # Rate limit
            elapsed = time.time() - current_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        """Ana çalıştırma döngüsü"""
        self.running = True

        # Kontrol thread'ini başlat
        control_thread = threading.Thread(target=self._control_loop, daemon=True)
        control_thread.start()

        print("\n[SYSTEM] Starting visualization...")
        print("=" * 60)
        print("CONTROLS:")
        print("  CLICK MAP    - Set waypoint (drones go there)")
        print("  1-9          - Select drone for sensor view")
        print("  A            - Select all drones")
        print("  P            - Toggle performance panel")
        print("  SPACE        - Takeoff")
        print("  L            - Land")
        print("  G            - Grid formation")
        print("  C            - Circle formation")
        print("  +/-          - Zoom in/out")
        print("  Arrow keys   - Pan view")
        print("  Q/ESC        - Quit")
        print("=" * 60)

        # İlk takeoff
        time.sleep(0.5)
        self.takeoff()

        # Ana görselleştirme döngüsü
        try:
            while self.running and self.visualizer.running:
                # Start performance frame
                self.performance_monitor.start_frame()

                # Event işle
                if not self.visualizer.handle_events():
                    break

                # Keyboard shortcuts
                keys = self._check_keyboard()
                if keys:
                    self._handle_keys(keys)

                # Waypoint kontrolü
                waypoint = self.visualizer.get_waypoint()
                selected_drones = self.visualizer.get_selected_drones()

                if waypoint is not None:
                    for drone_id in selected_drones:
                        self.targets[drone_id, 0] = waypoint[0]
                        self.targets[drone_id, 1] = waypoint[1]
                        # Z'yi mevcut hedeften koru
                        if self.targets[drone_id, 2] < 1.0:
                            self.targets[drone_id, 2] = self.default_altitude

                # Visualizer'ı güncelle
                self.visualizer.update_state(
                    self.estimated_positions,  # Sensörden gelen tahmin
                    self.estimated_velocities,
                    self.targets,
                    self.armed,
                )

                # Sensör verilerini gönder
                sensor_data = self.sensors.get_raw_sensors()
                errors = self.sensors.get_estimation_error()
                self.visualizer.update_sensor_data(sensor_data, errors)

                # Çiz (timed)
                self.performance_monitor.start_section("render")
                self.visualizer.draw()
                self.performance_monitor.end_section("render")

                # End performance frame
                self.performance_monitor.end_frame()

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted")

        finally:
            self.running = False
            self.visualizer.quit()
            print("[SYSTEM] Shutdown complete")

    def _check_keyboard(self):
        """Pygame'den klavye durumunu kontrol et"""
        import pygame

        return pygame.key.get_pressed()

    def _handle_keys(self, keys):
        """Klavye kısayolları"""
        import pygame

        if keys[pygame.K_SPACE]:
            self.takeoff()
            time.sleep(0.3)  # Debounce

        elif keys[pygame.K_l]:
            self.land()
            time.sleep(0.3)

        elif keys[pygame.K_g]:
            self.set_formation_grid()
            time.sleep(0.3)

        elif keys[pygame.K_c]:
            self.set_formation_circle()
            self.visualizer.clear_waypoint()
            time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser(description="Integrated Swarm System with Sensors")
    parser.add_argument("--drones", type=int, default=10, help="Number of drones")
    parser.add_argument("--no-gazebo", action="store_true", help="Disable Gazebo")
    args = parser.parse_args()

    system = IntegratedSwarmSystem(
        num_drones=args.drones, use_gazebo=not args.no_gazebo
    )

    system.run()


if __name__ == "__main__":
    main()
