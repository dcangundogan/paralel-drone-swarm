#!/usr/bin/env python3
"""
GPU Swarm Simulation Runner
============================

25 drone test için:
    python run_gpu_simulation.py

Başka sayı için:
    python run_gpu_simulation.py --drones 50
    python run_gpu_simulation.py --drones 100

Sensörsüz (daha hızlı):
    python run_gpu_simulation.py --no-sensors
"""

import argparse
import numpy as np
import time

# Simülasyon
from gpu_swarm_simulation import GPUSwarmSimulation

# Visualizer
from visualizer_with_sensors import AdvancedSwarmVisualizer, VisualizerConfig

# Performance Monitor
from performance_monitor import PerformanceMonitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drones', type=int, default=25, help='Drone sayısı')
    parser.add_argument('--no-sensors', action='store_true', help='Sensör simülasyonu kapat')
    args = parser.parse_args()

    # ========================================================================
    # SİMÜLASYON OLUŞTUR
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"   STARTING SIMULATION WITH {args.drones} DRONES")
    print("=" * 70)

    sim = GPUSwarmSimulation(
        num_drones=args.drones,
        enable_sensors=not args.no_sensors
    )

    # ========================================================================
    # VİSUALİZER OLUŞTUR
    # ========================================================================
    viz_config = VisualizerConfig(
        window_width=1400,
        window_height=900,
        meters_per_pixel=0.1 if args.drones > 50 else 0.05,
    )
    viz = AdvancedSwarmVisualizer(args.drones, viz_config)

    # Performance Monitor
    perf_monitor = PerformanceMonitor(history_size=60)
    viz.set_performance_monitor(perf_monitor)
    print("[SYSTEM] Performance monitor initialized")

    # ========================================================================
    # BAŞLAT
    # ========================================================================
    sim.reset()
    time.sleep(0.5)
    sim.takeoff(5.0)

    print("\n" + "=" * 70)
    print("   CONTROLS")
    print("=" * 70)
    print("   CLICK       - Set waypoint (all drones go there)")
    print("   SPACE       - Takeoff")
    print("   L           - Land")
    print("   G           - Grid formation")
    print("   C           - Circle formation")
    print("   1-9         - Select drone for sensor view")
    print("   +/-         - Zoom")
    print("   Q/ESC       - Quit")
    print("=" * 70)

    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    frame_count = 0
    last_fps_time = time.time()
    fps = 0

    try:
        while viz.running:
            frame_start = time.time()

            # Start performance frame
            perf_monitor.start_frame()

            # --- Event handling ---
            if not viz.handle_events():
                break

            # --- Keyboard ---
            import pygame
            keys = pygame.key.get_pressed()

            if keys[pygame.K_SPACE]:
                sim.takeoff(5.0)
                time.sleep(0.2)
            elif keys[pygame.K_l]:
                sim.land()
                time.sleep(0.2)
            elif keys[pygame.K_g]:
                sim.formation_grid(spacing=3.0)
                viz.clear_waypoint()
                time.sleep(0.2)
            elif keys[pygame.K_c]:
                sim.formation_circle(radius=max(8, args.drones * 0.3))
                viz.clear_waypoint()
                time.sleep(0.2)

            # --- Waypoint from visualizer ---
            waypoint = viz.get_waypoint()
            if waypoint is not None:
                sim.set_waypoint(waypoint[0], waypoint[1], waypoint[2])

            # --- Simulation step (includes physics, sensors, control) ---
            perf_monitor.start_section("physics")
            sim.step()
            perf_monitor.end_section("physics")

            # --- Get state for visualization ---
            state = sim.get_state()

            # --- Update visualizer ---
            viz.update_state(
                state['positions'],
                state['velocities'],
                state['targets'],
                state['armed']
            )

            # Sensor data (if available)
            if 'sensor_data' in state:
                # Estimation errors hesapla
                errors = {
                    'position_rmse': np.sqrt(np.mean(
                        (state['positions'] - state['sensor_data']['gps_positions'])**2,
                        axis=0
                    ))
                }
                viz.update_sensor_data(state['sensor_data'], errors)

            # --- Draw ---
            perf_monitor.start_section("render")
            viz.draw()
            perf_monitor.end_section("render")

            # End performance frame
            perf_monitor.end_frame()

            # --- FPS calculation ---
            frame_count += 1
            if time.time() - last_fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                last_fps_time = time.time()

            # --- Frame rate limit ---
            elapsed = time.time() - frame_start
            sleep_time = 1/60 - elapsed  # 60 FPS target
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SIM] Interrupted")

    finally:
        viz.quit()
        print(f"\n[SIM] Final time: {sim.time:.1f}s")
        print("[SIM] Shutdown complete")


if __name__ == "__main__":
    main()
