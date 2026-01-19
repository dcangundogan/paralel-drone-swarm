"""
Advanced Swarm Visualizer with Sensor Panel & Click-to-Navigate
================================================================

Features:
- 2D top-down view of drone swarm
- Real-time sensor data panel (IMU, GPS, Baro, Mag)
- Click on map to set waypoint - drones go there
- Sensor error visualization
- GPS dropout indicators
- Kalman filter uncertainty ellipses
"""

import pygame
import numpy as np
import time
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    """Visualizer ayarları"""
    window_width: int = 1400          # Daha geniş (sensör paneli için)
    window_height: int = 900
    map_width: int = 1000             # Harita alanı
    panel_width: int = 400            # Sensör paneli
    background_color: Tuple = (20, 25, 30)
    grid_color: Tuple = (40, 45, 50)
    text_color: Tuple = (200, 200, 200)
    fps: int = 60                     # 60 FPS for smooth rendering

    # Harita
    meters_per_pixel: float = 0.05    # 1 pixel = 5 cm
    grid_spacing: float = 5.0         # 5 metre grid

    # Drone görünümü
    drone_radius: int = 12
    target_radius: int = 8
    waypoint_radius: int = 15


class SensorPanelRenderer:
    """Sensör verilerini gösteren panel"""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)
        self.fonts = {}

    def init_fonts(self):
        self.fonts = {
            'title': pygame.font.SysFont('consolas', 18, bold=True),
            'header': pygame.font.SysFont('consolas', 14, bold=True),
            'normal': pygame.font.SysFont('consolas', 12),
            'small': pygame.font.SysFont('consolas', 10),
        }

    def draw(self, surface: pygame.Surface, sensor_data: dict, selected_drone: int,
             estimation_errors: dict, num_drones: int):
        """Sensör panelini çiz"""

        # Panel arka planı
        pygame.draw.rect(surface, (30, 35, 40), self.rect)
        pygame.draw.rect(surface, (60, 65, 70), self.rect, 2)

        x = self.rect.x + 15
        y = self.rect.y + 15

        # Başlık
        title = self.fonts['title'].render("SENSOR MONITOR", True, (100, 200, 255))
        surface.blit(title, (x, y))
        y += 30

        # Seçili drone
        drone_text = self.fonts['header'].render(f"Drone #{selected_drone} / {num_drones-1}", True, (255, 200, 100))
        surface.blit(drone_text, (x, y))
        y += 25

        # Çizgi
        pygame.draw.line(surface, (60, 65, 70), (x, y), (x + self.rect.width - 30, y), 1)
        y += 15

        if not sensor_data:
            no_data = self.fonts['normal'].render("Sensör verisi yok", True, (150, 150, 150))
            surface.blit(no_data, (x, y))
            return

        # === IMU SECTION ===
        y = self._draw_section(surface, "IMU (400 Hz)", x, y, (100, 255, 150))

        accel = sensor_data.get('imu_accel', np.zeros(3))
        gyro = sensor_data.get('imu_gyro', np.zeros(3))

        if len(accel) > selected_drone:
            a = accel[selected_drone]
            g = gyro[selected_drone]

            self._draw_labeled_value(surface, "Accel X:", f"{a[0]:+.2f} m/s²", x, y)
            y += 18
            self._draw_labeled_value(surface, "Accel Y:", f"{a[1]:+.2f} m/s²", x, y)
            y += 18
            self._draw_labeled_value(surface, "Accel Z:", f"{a[2]:+.2f} m/s²", x, y)
            y += 22

            self._draw_labeled_value(surface, "Gyro X:", f"{g[0]:+.3f} rad/s", x, y)
            y += 18
            self._draw_labeled_value(surface, "Gyro Y:", f"{g[1]:+.3f} rad/s", x, y)
            y += 18
            self._draw_labeled_value(surface, "Gyro Z:", f"{g[2]:+.3f} rad/s", x, y)
            y += 25

        # === GPS SECTION ===
        y = self._draw_section(surface, "GPS (10 Hz)", x, y, (255, 200, 100))

        gps_pos = sensor_data.get('gps_position', np.zeros(3))
        gps_vel = sensor_data.get('gps_velocity', np.zeros(3))
        gps_valid = sensor_data.get('gps_valid', np.ones(1, dtype=bool))

        if len(gps_pos) > selected_drone:
            pos = gps_pos[selected_drone]
            vel = gps_vel[selected_drone]
            valid = gps_valid[selected_drone] if len(gps_valid) > selected_drone else True

            # GPS durumu
            if valid:
                status_color = (100, 255, 100)
                status_text = "VALID"
            else:
                status_color = (255, 100, 100)
                status_text = "NO SIGNAL"

            status = self.fonts['header'].render(f"Status: {status_text}", True, status_color)
            surface.blit(status, (x, y))
            y += 22

            self._draw_labeled_value(surface, "Lat (X):", f"{pos[0]:+.2f} m", x, y)
            y += 18
            self._draw_labeled_value(surface, "Lon (Y):", f"{pos[1]:+.2f} m", x, y)
            y += 18
            self._draw_labeled_value(surface, "Alt (Z):", f"{pos[2]:+.2f} m", x, y)
            y += 22

            speed = np.linalg.norm(vel)
            self._draw_labeled_value(surface, "Speed:", f"{speed:.2f} m/s", x, y)
            y += 25

        # === BAROMETER SECTION ===
        y = self._draw_section(surface, "BAROMETER (50 Hz)", x, y, (200, 150, 255))

        baro_alt = sensor_data.get('baro_altitude', np.zeros(1))
        if len(baro_alt) > selected_drone:
            alt = baro_alt[selected_drone]
            self._draw_labeled_value(surface, "Altitude:", f"{alt:.2f} m", x, y)
            y += 25

        # === MAGNETOMETER SECTION ===
        y = self._draw_section(surface, "MAGNETOMETER (75 Hz)", x, y, (255, 150, 200))

        mag_heading = sensor_data.get('mag_heading', np.zeros(1))
        if len(mag_heading) > selected_drone:
            heading_rad = mag_heading[selected_drone]
            heading_deg = np.degrees(heading_rad) % 360

            self._draw_labeled_value(surface, "Heading:", f"{heading_deg:.1f}°", x, y)
            y += 20

            # Pusula göstergesi
            self._draw_compass(surface, x + 100, y + 40, 35, heading_rad)
            y += 90

        # === ESTIMATION ERROR SECTION ===
        y = self._draw_section(surface, "KALMAN FILTER ERROR", x, y, (255, 100, 100))

        if estimation_errors:
            pos_rmse = estimation_errors.get('position_rmse', np.zeros(3))
            vel_rmse = estimation_errors.get('velocity_rmse', np.zeros(3))

            self._draw_labeled_value(surface, "Pos RMSE:", f"{np.mean(pos_rmse):.3f} m", x, y)
            y += 18
            self._draw_labeled_value(surface, "Vel RMSE:", f"{np.mean(vel_rmse):.3f} m/s", x, y)
            y += 25

        # === CONTROLS ===
        y = self._draw_section(surface, "CONTROLS", x, y, (150, 150, 150))

        controls = [
            "Click map: Set waypoint",
            "1-9: Select drone",
            "A: Select all drones",
            "SPACE: Takeoff",
            "L: Land",
            "G: Grid formation",
            "C: Circle formation",
            "+/-: Zoom",
            "Arrow keys: Pan",
            "Q/ESC: Quit"
        ]

        for ctrl in controls:
            text = self.fonts['small'].render(ctrl, True, (150, 150, 150))
            surface.blit(text, (x, y))
            y += 15

    def _draw_section(self, surface, title: str, x: int, y: int, color: Tuple) -> int:
        """Section başlığı çiz"""
        pygame.draw.line(surface, (60, 65, 70), (x, y), (x + self.rect.width - 30, y), 1)
        y += 8
        text = self.fonts['header'].render(title, True, color)
        surface.blit(text, (x, y))
        y += 22
        return y

    def _draw_labeled_value(self, surface, label: str, value: str, x: int, y: int):
        """Label: Value formatında çiz"""
        label_surf = self.fonts['normal'].render(label, True, (150, 150, 150))
        value_surf = self.fonts['normal'].render(value, True, (220, 220, 220))
        surface.blit(label_surf, (x, y))
        surface.blit(value_surf, (x + 100, y))

    def _draw_compass(self, surface, cx: int, cy: int, radius: int, heading: float):
        """Mini pusula çiz"""
        # Daire
        pygame.draw.circle(surface, (60, 65, 70), (cx, cy), radius, 2)

        # N, E, S, W işaretleri
        font = self.fonts['small']
        dirs = [('N', 0), ('E', 90), ('S', 180), ('W', 270)]
        for label, angle in dirs:
            rad = np.radians(angle - 90)
            tx = cx + (radius + 10) * np.cos(rad)
            ty = cy + (radius + 10) * np.sin(rad)
            text = font.render(label, True, (150, 150, 150))
            surface.blit(text, (tx - 4, ty - 6))

        # Heading iğnesi
        needle_len = radius - 5
        nx = cx + needle_len * np.cos(heading - np.pi/2)
        ny = cy + needle_len * np.sin(heading - np.pi/2)
        pygame.draw.line(surface, (255, 100, 100), (cx, cy), (int(nx), int(ny)), 3)

        # Merkez nokta
        pygame.draw.circle(surface, (200, 200, 200), (cx, cy), 3)


class AdvancedSwarmVisualizer:
    """Gelişmiş Swarm Visualizer - Sensör paneli ve tıkla-git özelliği"""

    def __init__(self, num_drones: int, config: Optional[VisualizerConfig] = None):
        self.num_drones = num_drones
        self.config = config or VisualizerConfig()

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Drone Swarm - Sensor Monitor & Navigation")
        self.clock = pygame.time.Clock()

        # Fonts
        self.fonts = {
            'large': pygame.font.SysFont('consolas', 16, bold=True),
            'normal': pygame.font.SysFont('consolas', 12),
            'small': pygame.font.SysFont('consolas', 10),
        }

        # Sensor panel
        self.sensor_panel = SensorPanelRenderer(
            self.config.map_width, 0,
            self.config.panel_width, self.config.window_height
        )
        self.sensor_panel.init_fonts()

        # Drone state
        self.positions = np.zeros((num_drones, 3), dtype=np.float32)
        self.velocities = np.zeros((num_drones, 3), dtype=np.float32)
        self.targets = np.zeros((num_drones, 3), dtype=np.float32)
        self.armed = np.zeros(num_drones, dtype=bool)

        # Sensor data (from GPUSensorSimulator)
        self.sensor_data = {}
        self.estimation_errors = {}

        # View settings
        self.zoom = 1.0
        self.offset = np.array([self.config.map_width / 2, self.config.window_height / 2])

        # Selection & waypoints
        self.selected_drone = 0           # Sensör panelinde gösterilen drone
        self.selected_drones = set(range(num_drones))  # Waypoint'e gidecek drone'lar
        self.waypoint = None              # (x, y, z) - tıklanan nokta
        self.waypoint_history = []        # Geçmiş waypoint'ler

        # GPS validity tracking
        self.gps_valid = np.ones(num_drones, dtype=bool)

        self.running = True

    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Dünya koordinatlarını ekran koordinatlarına çevir"""
        scale = 1.0 / (self.config.meters_per_pixel * self.zoom)
        screen_x = self.offset[0] + world_pos[0] * scale
        screen_y = self.offset[1] - world_pos[1] * scale  # Y ters
        return int(screen_x), int(screen_y)

    def screen_to_world(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        """Ekran koordinatlarını dünya koordinatlarına çevir"""
        scale = 1.0 / (self.config.meters_per_pixel * self.zoom)
        world_x = (screen_pos[0] - self.offset[0]) / scale
        world_y = (self.offset[1] - screen_pos[1]) / scale  # Y ters
        return np.array([world_x, world_y])

    def update_state(self, positions: np.ndarray, velocities: np.ndarray = None,
                    targets: np.ndarray = None, armed: np.ndarray = None):
        """Drone durumlarını güncelle"""
        self.positions = positions.copy()
        if velocities is not None:
            self.velocities = velocities.copy()
        if targets is not None:
            self.targets = targets.copy()
        if armed is not None:
            self.armed = armed.copy()

    def update_sensor_data(self, sensor_data: dict, estimation_errors: dict = None):
        """Sensör verilerini güncelle"""
        self.sensor_data = sensor_data
        if estimation_errors:
            self.estimation_errors = estimation_errors

        # GPS validity güncelle
        if 'gps_valid' in sensor_data:
            self.gps_valid = sensor_data['gps_valid']

    def get_waypoint(self) -> Optional[np.ndarray]:
        """Mevcut waypoint'i döndür (varsa)"""
        return self.waypoint

    def get_selected_drones(self) -> set:
        """Seçili drone'ları döndür"""
        return self.selected_drones

    def clear_waypoint(self):
        """Waypoint'i temizle"""
        if self.waypoint is not None:
            self.waypoint_history.append(self.waypoint[:2].copy())
            if len(self.waypoint_history) > 10:
                self.waypoint_history.pop(0)
        self.waypoint = None

    def handle_events(self) -> bool:
        """Event'leri işle, waypoint ve seçim yönetimi"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.running = False
                    return False

                # Zoom
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom = max(0.2, self.zoom - 0.1)
                elif event.key == pygame.K_MINUS:
                    self.zoom = min(3.0, self.zoom + 0.1)

                # Pan
                elif event.key == pygame.K_LEFT:
                    self.offset[0] += 50
                elif event.key == pygame.K_RIGHT:
                    self.offset[0] -= 50
                elif event.key == pygame.K_UP:
                    self.offset[1] += 50
                elif event.key == pygame.K_DOWN:
                    self.offset[1] -= 50

                # Reset view
                elif event.key == pygame.K_r:
                    self.zoom = 1.0
                    self.offset = np.array([self.config.map_width / 2, self.config.window_height / 2])

                # Drone selection (1-9 keys)
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    drone_id = event.key - pygame.K_1
                    if drone_id < self.num_drones:
                        self.selected_drone = drone_id
                        # Shift ile çoklu seçim
                        if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                            if drone_id in self.selected_drones:
                                self.selected_drones.discard(drone_id)
                            else:
                                self.selected_drones.add(drone_id)
                        else:
                            self.selected_drones = {drone_id}

                # Select all
                elif event.key == pygame.K_a:
                    self.selected_drones = set(range(self.num_drones))

                # Clear waypoint
                elif event.key == pygame.K_c:
                    self.clear_waypoint()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Sol tık
                    mouse_pos = pygame.mouse.get_pos()

                    # Sadece harita alanında tıklamayı kabul et
                    if mouse_pos[0] < self.config.map_width:
                        world_pos = self.screen_to_world(mouse_pos)

                        # Mevcut ortalama irtifayı kullan
                        avg_altitude = np.mean(self.targets[:, 2]) if np.any(self.targets[:, 2] > 0) else 5.0

                        self.waypoint = np.array([world_pos[0], world_pos[1], avg_altitude])
                        print(f"[NAV] Waypoint set: ({world_pos[0]:.1f}, {world_pos[1]:.1f}, {avg_altitude:.1f})")
                        print(f"[NAV] Selected drones: {sorted(self.selected_drones)}")

                elif event.button == 3:  # Sağ tık - waypoint temizle
                    self.clear_waypoint()

        return True

    def draw(self):
        """Her şeyi çiz"""
        self.screen.fill(self.config.background_color)

        # Harita alanı
        map_rect = pygame.Rect(0, 0, self.config.map_width, self.config.window_height)
        pygame.draw.rect(self.screen, (25, 30, 35), map_rect)

        # Grid çiz
        self._draw_grid()

        # Waypoint history (soluk)
        for wp in self.waypoint_history:
            screen_pos = self.world_to_screen(np.array([wp[0], wp[1], 0]))
            if 0 <= screen_pos[0] < self.config.map_width:
                pygame.draw.circle(self.screen, (60, 60, 80), screen_pos, 8, 1)

        # Mevcut waypoint
        if self.waypoint is not None:
            wp_screen = self.world_to_screen(self.waypoint)
            if 0 <= wp_screen[0] < self.config.map_width:
                # Dış halka (animasyonlu)
                pulse = int(5 * np.sin(time.time() * 5) + 20)
                pygame.draw.circle(self.screen, (255, 200, 100), wp_screen, pulse, 2)
                # İç daire
                pygame.draw.circle(self.screen, (255, 150, 50), wp_screen, 10)
                # X işareti
                pygame.draw.line(self.screen, (50, 50, 50),
                               (wp_screen[0]-5, wp_screen[1]-5),
                               (wp_screen[0]+5, wp_screen[1]+5), 2)
                pygame.draw.line(self.screen, (50, 50, 50),
                               (wp_screen[0]-5, wp_screen[1]+5),
                               (wp_screen[0]+5, wp_screen[1]-5), 2)

                # Waypoint koordinatları
                wp_text = self.fonts['small'].render(
                    f"WP: ({self.waypoint[0]:.1f}, {self.waypoint[1]:.1f})",
                    True, (255, 200, 100)
                )
                self.screen.blit(wp_text, (wp_screen[0] + 15, wp_screen[1] - 10))

        # Hedefleri çiz
        self._draw_targets()

        # Drone'ları çiz
        self._draw_drones()

        # Harita bilgi paneli
        self._draw_map_info()

        # Sensör paneli
        self.sensor_panel.draw(
            self.screen,
            self.sensor_data,
            self.selected_drone,
            self.estimation_errors,
            self.num_drones
        )

        pygame.display.flip()
        self.clock.tick(self.config.fps)

    def _draw_grid(self):
        """Arka plan grid'i çiz"""
        scale = 1.0 / (self.config.meters_per_pixel * self.zoom)
        spacing_pixels = self.config.grid_spacing * scale

        # Dikey çizgiler
        start_x = self.offset[0] % spacing_pixels
        x = start_x
        while x < self.config.map_width:
            pygame.draw.line(self.screen, self.config.grid_color,
                           (int(x), 0), (int(x), self.config.window_height), 1)
            x += spacing_pixels

        # Yatay çizgiler
        start_y = self.offset[1] % spacing_pixels
        y = start_y
        while y < self.config.window_height:
            pygame.draw.line(self.screen, self.config.grid_color,
                           (0, int(y)), (self.config.map_width, int(y)), 1)
            y += spacing_pixels

        # Origin işareti
        origin = self.world_to_screen(np.array([0, 0, 0]))
        if 0 <= origin[0] < self.config.map_width and 0 <= origin[1] < self.config.window_height:
            pygame.draw.circle(self.screen, (100, 100, 100), origin, 5)
            pygame.draw.line(self.screen, (150, 80, 80), origin, (origin[0] + 30, origin[1]), 2)  # X axis
            pygame.draw.line(self.screen, (80, 150, 80), origin, (origin[0], origin[1] - 30), 2)  # Y axis

    def _draw_targets(self):
        """Hedef pozisyonları çiz"""
        for i in range(self.num_drones):
            target = self.targets[i]
            screen_pos = self.world_to_screen(target)

            if not (0 <= screen_pos[0] < self.config.map_width):
                continue

            # Seçili drone'ların hedefleri daha belirgin
            if i in self.selected_drones:
                color = (255, 150, 200)
                radius = self.config.target_radius + 2
            else:
                color = (150, 100, 150)
                radius = self.config.target_radius

            pygame.draw.circle(self.screen, color, screen_pos, radius, 2)

    def _draw_drones(self):
        """Drone'ları çiz"""
        for i in range(self.num_drones):
            pos = self.positions[i]
            vel = self.velocities[i]
            screen_pos = self.world_to_screen(pos)

            if not (0 <= screen_pos[0] < self.config.map_width):
                continue

            # İrtifaya göre renk
            altitude = pos[2]
            color_intensity = min(255, int(100 + altitude * 30))

            # GPS durumuna göre renk
            if not self.gps_valid[i]:
                # GPS yok - kırmızı yanıp sönen
                if int(time.time() * 4) % 2:
                    base_color = (255, 50, 50)
                else:
                    base_color = (150, 30, 30)
            elif i == self.selected_drone:
                # Seçili drone - sarı
                base_color = (255, 255, 100)
            elif i in self.selected_drones:
                # Waypoint grubunda - açık mavi
                base_color = (100, 200, 255)
            else:
                # Normal - yeşil tonları
                base_color = (50, color_intensity, 100)

            # Armed durumu
            if self.armed[i]:
                radius = self.config.drone_radius
            else:
                radius = self.config.drone_radius - 3

            # Ana drone gövdesi
            pygame.draw.circle(self.screen, base_color, screen_pos, radius)

            # Dış çerçeve
            border_color = (255, 255, 255) if i == self.selected_drone else (100, 100, 100)
            pygame.draw.circle(self.screen, border_color, screen_pos, radius, 2)

            # Drone ID
            id_text = self.fonts['small'].render(str(i), True, (0, 0, 0))
            text_rect = id_text.get_rect(center=screen_pos)
            self.screen.blit(id_text, text_rect)

            # Hız vektörü
            if np.linalg.norm(vel[:2]) > 0.1:
                vel_scale = 15
                end_pos = (
                    screen_pos[0] + int(vel[0] * vel_scale),
                    screen_pos[1] - int(vel[1] * vel_scale)
                )
                pygame.draw.line(self.screen, (0, 255, 255), screen_pos, end_pos, 2)

            # Hedefe çizgi (seçili drone için)
            if i in self.selected_drones:
                target_screen = self.world_to_screen(self.targets[i])
                if 0 <= target_screen[0] < self.config.map_width:
                    pygame.draw.line(self.screen, (80, 80, 100),
                                   screen_pos, target_screen, 1)

    def _draw_map_info(self):
        """Harita üzerinde bilgi göster"""
        y = 10

        # Zoom
        zoom_text = self.fonts['normal'].render(f"Zoom: {1/self.zoom:.1f}x", True, (150, 150, 150))
        self.screen.blit(zoom_text, (10, y))
        y += 20

        # Seçili drone sayısı
        sel_text = self.fonts['normal'].render(
            f"Selected: {len(self.selected_drones)} drones",
            True, (100, 200, 255)
        )
        self.screen.blit(sel_text, (10, y))
        y += 20

        # Waypoint durumu
        if self.waypoint is not None:
            wp_text = self.fonts['normal'].render(
                f"Waypoint: ({self.waypoint[0]:.1f}, {self.waypoint[1]:.1f}, {self.waypoint[2]:.1f})",
                True, (255, 200, 100)
            )
        else:
            wp_text = self.fonts['normal'].render("Click map to set waypoint", True, (100, 100, 100))
        self.screen.blit(wp_text, (10, y))
        y += 20

        # GPS durumu özeti
        gps_ok = np.sum(self.gps_valid)
        if gps_ok < self.num_drones:
            gps_text = self.fonts['normal'].render(
                f"GPS: {gps_ok}/{self.num_drones} valid",
                True, (255, 100, 100)
            )
        else:
            gps_text = self.fonts['normal'].render(
                f"GPS: All valid",
                True, (100, 255, 100)
            )
        self.screen.blit(gps_text, (10, y))

    def quit(self):
        """Pygame'i kapat"""
        pygame.quit()


# ============================================================
# STANDALONE DEMO
# ============================================================

def demo_visualizer():
    """Visualizer demo (sensör simülasyonu ile)"""
    print("\n" + "="*60)
    print("ADVANCED VISUALIZER DEMO")
    print("="*60)
    print("Controls:")
    print("  - Click on map to set waypoint")
    print("  - 1-9: Select drone for sensor view")
    print("  - A: Select all drones")
    print("  - +/-: Zoom")
    print("  - Arrow keys: Pan")
    print("  - Q/ESC: Quit")
    print("="*60)

    num_drones = 10
    viz = AdvancedSwarmVisualizer(num_drones)

    # Sensör simülasyonu import et
    try:
        from gpu_sensors import GPUSensorSimulator, SensorConfig
        sensors = GPUSensorSimulator(num_drones, SensorConfig(
            gps_dropout_prob=0.05  # %5 GPS kaybı
        ))
        has_sensors = True
        print("[DEMO] Sensör simülasyonu aktif")
    except ImportError:
        has_sensors = False
        print("[DEMO] Sensör simülasyonu bulunamadı")

    # Başlangıç pozisyonları (grid)
    positions = np.zeros((num_drones, 3), dtype=np.float32)
    for i in range(num_drones):
        row, col = divmod(i, 4)
        positions[i] = [col * 4, row * 4, 0.5]

    velocities = np.zeros((num_drones, 3), dtype=np.float32)
    targets = positions.copy()
    targets[:, 2] = 5.0  # Hedef irtifa
    armed = np.ones(num_drones, dtype=bool)

    dt = 1.0 / 30
    running = True

    while running and viz.running:
        # Event'leri işle
        if not viz.handle_events():
            break

        # Waypoint kontrolü
        waypoint = viz.get_waypoint()
        selected = viz.get_selected_drones()

        if waypoint is not None:
            # Seçili drone'ların hedefini waypoint'e ayarla
            for drone_id in selected:
                targets[drone_id, 0] = waypoint[0]
                targets[drone_id, 1] = waypoint[1]
                targets[drone_id, 2] = waypoint[2]

        # Basit hareket simülasyonu
        for i in range(num_drones):
            direction = targets[i] - positions[i]
            dist = np.linalg.norm(direction)
            if dist > 0.1:
                velocities[i] = direction / dist * min(dist, 2.0)
            else:
                velocities[i] *= 0.9

            positions[i] += velocities[i] * dt
            positions[i, 2] = max(positions[i, 2], 0.1)

        # Sensör güncellemesi
        if has_sensors:
            sensors.update_ground_truth(positions, velocities)
            sensor_data = sensors.get_raw_sensors()
            errors = sensors.get_estimation_error()
            viz.update_sensor_data(sensor_data, errors)

        # Visualizer güncelle
        viz.update_state(positions, velocities, targets, armed)
        viz.draw()

    viz.quit()
    print("[DEMO] Finished")


if __name__ == "__main__":
    demo_visualizer()
