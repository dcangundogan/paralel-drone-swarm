#!/usr/bin/env python3
"""
GPU-Parallelized Drone Swarm Simulation
========================================

Bu dosya TÜM simülasyonu açıklar ve GPU üzerinde paralel çalıştırır.

YAZAR: Claude
TARİH: 2024
GERÇEKÇİLİK: %80

=============================================================================
                              MİMARİ AÇIKLAMASI
=============================================================================

Simülasyon 3 katmandan oluşur:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ KATMAN 1: FİZİK MOTORU                                              │
    │                                                                     │
    │ Ne yapar: Drone'ların fiziksel hareketini simüle eder              │
    │ Input:    Velocity komutları (controller'dan)                       │
    │ Output:   Gerçek pozisyon, hız, oryantasyon                        │
    │ GPU:      N drone için TÜM fizik hesapları paralel                  │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ KATMAN 2: SENSÖR SİMÜLASYONU (opsiyonel)                           │
    │                                                                     │
    │ Ne yapar: Gerçek değerlere gürültü/hata ekler                      │
    │ Input:    Gerçek pozisyon (fizik motorundan)                       │
    │ Output:   Gürültülü sensör ölçümleri                               │
    │ GPU:      N drone için TÜM sensörler paralel                        │
    │                                                                     │
    │ NOT: Bu katman OLMADAN da çalışır!                                 │
    │      Sensörsüz = Controller gerçek pozisyonu görür (ideal)         │
    │      Sensörlü = Controller gürültülü pozisyon görür (gerçekçi)     │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ KATMAN 3: SWARM CONTROLLER                                          │
    │                                                                     │
    │ Ne yapar: Drone'lara ne yapacaklarını söyler                       │
    │ Input:    Pozisyonlar (sensörden veya fizikten) + Hedefler         │
    │ Output:   Velocity komutları                                        │
    │ GPU:      N drone için TÜM kontrol hesapları paralel                │
    │                                                                     │
    │ Algoritmalar:                                                       │
    │   - Collision Avoidance: Çarpışmayı önle                           │
    │   - Target Tracking: Hedefe git                                     │
    │   - Formation Control: Şekil oluştur                               │
    └─────────────────────────────────────────────────────────────────────┘

=============================================================================
                              GPU PARALELİZASYON
=============================================================================

CPU (seri):                          GPU (paralel):

for i in range(N):                   # Tek komutla N drone!
    pos[i] += vel[i] * dt            pos += vel * dt

100 drone = 100 işlem                100 drone = 1 işlem (paralel)
Süre: O(N)                           Süre: O(1)

=============================================================================
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================================
# GPU SETUP
# ============================================================================

try:
    import cupy as cp

    # GPU test
    device = cp.cuda.Device(0)
    device.compute_capability
    test = cp.array([1, 2, 3])
    _ = cp.sum(test)

    # Get GPU name properly
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    gpu_name = props["name"].decode("utf-8")

    GPU_AVAILABLE = True
    xp = cp  # xp = cupy (GPU)
    print("=" * 60)
    print("GPU: CUDA AKTIF")
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {device.mem_info[1] / 1024**3:.1f} GB")
    print("=" * 60)

except Exception as e:
    GPU_AVAILABLE = False
    xp = np  # xp = numpy (CPU)
    print("=" * 60)
    print("GPU: BULUNAMADI - CPU KULLANILIYOR")
    print(f"Sebep: {e}")
    print("=" * 60)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SimulationConfig:
    """
    Simülasyon ayarları.

    Tüm parametreler burada tanımlanır.
    """

    # Drone sayısı
    num_drones: int = 25

    # Fizik
    physics_dt: float = 0.004  # 250 Hz fizik (saniye/adım)
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³

    # Quadcopter özellikleri
    mass: float = 1.5  # kg
    arm_length: float = 0.2  # m
    prop_diameter: float = 0.254  # m (10 inch)
    max_thrust: float = 20.0  # N (toplam 4 motor)
    motor_time_constant: float = 0.05  # s (motor tepki süresi)
    drag_coefficient: float = 0.5  # Hava direnci katsayısı

    # Velocity limitleri
    max_velocity_xy: float = 5.0  # m/s
    max_velocity_z: float = 3.0  # m/s
    max_acceleration: float = 5.0  # m/s²

    # Collision avoidance
    collision_radius: float = 0.8  # m (tehlikeli mesafe)
    avoidance_radius: float = 2.5  # m (kaçınma başlar)
    avoidance_strength: float = 3.0  # Kaçınma gücü

    # Target tracking (PD controller)
    kp: float = 1.5  # Proportional gain
    kd: float = 0.3  # Derivative gain

    # Sensör özellikleri (opsiyonel)
    enable_sensors: bool = True
    gps_noise_std: float = 1.0  # m (GPS gürültüsü)
    gps_update_rate: float = 10.0  # Hz
    imu_noise_std: float = 0.1  # m/s² (IMU gürültüsü)


# ============================================================================
# PHYSICS ENGINE (GPU PARALLEL)
# ============================================================================


class GPUPhysicsEngine:
    """
    GPU üzerinde paralel fizik simülasyonu.

    ==========================================================================
    NE YAPAR?
    ==========================================================================

    Her drone için şu fizik hesaplamalarını yapar:

    1. MOTOR DİNAMİĞİ
       - Komut → Thrust dönüşümü
       - Motor response time (anında tepki vermez)

    2. KUVVETLER
       - Thrust (yukarı itme)
       - Gravity (yerçekimi)
       - Drag (hava direnci)

    3. HAREKET
       - Newton: F = m × a
       - a = F / m
       - v += a × dt
       - pos += v × dt

    ==========================================================================
    GPU PARALELİZASYON
    ==========================================================================

    CPU'da:
        for i in range(num_drones):
            forces[i] = compute_force(drone[i])
            accel[i] = forces[i] / mass
            velocity[i] += accel[i] * dt
            position[i] += velocity[i] * dt

    GPU'da (bu kod):
        forces = compute_forces_all()     # N drone paralel
        accel = forces / mass             # N drone paralel
        velocities += accel * dt          # N drone paralel
        positions += velocities * dt      # N drone paralel

    Fark: GPU tüm drone'ları AYNI ANDA hesaplar!
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.num_drones = config.num_drones

        print(f"\n[PHYSICS] Initializing for {self.num_drones} drones...")

        # ====================================================================
        # STATE ARRAYS (GPU'da tutulur)
        # ====================================================================

        # Pozisyon: Her drone için [x, y, z]
        # Shape: (num_drones, 3)
        self.positions = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # Hız: Her drone için [vx, vy, vz]
        self.velocities = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # Oryantasyon: [roll, pitch, yaw] (radyan)
        self.orientations = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # Motor durumu: 4 motor, 0-1 arası throttle
        self.motor_states = xp.zeros((self.num_drones, 4), dtype=xp.float32)

        # Armed (motorlar aktif mi?)
        self.armed = xp.zeros(self.num_drones, dtype=xp.bool_)

        # ====================================================================
        # INPUT: Controller'dan gelen komutlar
        # ====================================================================
        self.velocity_commands = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        print(f"[PHYSICS] Ready! (GPU: {GPU_AVAILABLE})")

    def reset_positions(self, positions: np.ndarray = None):
        """
        Drone pozisyonlarını sıfırla.

        Args:
            positions: (N, 3) numpy array veya None (grid oluşturur)
        """
        self.velocities[:] = 0
        self.orientations[:] = 0
        self.motor_states[:] = 0

        if positions is not None:
            self.positions = xp.asarray(positions, dtype=xp.float32)
        else:
            # Otomatik grid oluştur
            cols = int(xp.ceil(xp.sqrt(self.num_drones)))
            spacing = 3.0
            for i in range(self.num_drones):
                row, col = divmod(i, cols)
                self.positions[i, 0] = (col - cols / 2) * spacing
                self.positions[i, 1] = (row - cols / 2) * spacing
                self.positions[i, 2] = 0.1

    def arm_all(self):
        """Tüm drone'ları aktifleştir"""
        self.armed[:] = True

    def disarm_all(self):
        """Tüm drone'ları deaktif et"""
        self.armed[:] = False
        self.motor_states[:] = 0

    def set_velocity_commands(self, commands):
        """
        Controller'dan velocity komutlarını al.

        Args:
            commands: (N, 3) array - [vx, vy, vz] for each drone
        """
        self.velocity_commands = xp.asarray(commands, dtype=xp.float32)

    def step(self):
        """
        ====================================================================
        BİR FİZİK ADIMI (GPU PARALLEL)
        ====================================================================

        Bu fonksiyon her çağrıldığında:
        1. Velocity komutları → Motor thrust'a çevrilir
        2. Kuvvetler hesaplanır (thrust + gravity + drag)
        3. İvme hesaplanır (F = ma → a = F/m)
        4. Hız güncellenir (v += a × dt)
        5. Pozisyon güncellenir (pos += v × dt)

        TÜM BUNLAR N DRONE İÇİN PARALEL YAPILIR!
        """
        dt = self.cfg.physics_dt

        # ================================================================
        # ADIM 1: Velocity Command → Motor Thrust
        # ================================================================
        # Basitleştirilmiş: İstenen hız → gerekli thrust

        vel_error = self.velocity_commands - self.velocities

        # İstenen ivme (P controller)
        desired_accel = vel_error * 2.0  # kp = 2.0

        # İstenen thrust (Newton: F = m × a)
        # Z ekseni için gravity kompanzasyonu ekle
        desired_force = self.cfg.mass * desired_accel
        desired_force[:, 2] += self.cfg.mass * self.cfg.gravity  # Gravity comp

        # Thrust limiti
        max_thrust = self.cfg.max_thrust
        thrust_magnitude = xp.linalg.norm(desired_force, axis=1, keepdims=True)
        thrust_scale = xp.minimum(1.0, max_thrust / (thrust_magnitude + 0.001))

        target_thrust = desired_force * thrust_scale

        # Armed değilse thrust = 0
        target_thrust = xp.where(
            self.armed[:, None], target_thrust, xp.zeros_like(target_thrust)
        )

        # ================================================================
        # ADIM 2: Motor Dynamics (First-order response)
        # ================================================================
        # Motorlar anında tepki vermez!
        # Gerçek thrust = target'a doğru yavaşça değişir

        # current_thrust değişkenini tanımla (basitleştirilmiş)
        # Motor state → thrust dönüşümü
        tau = self.cfg.motor_time_constant
        alpha = dt / (tau + dt)  # First-order filter katsayısı

        # Mevcut thrust (motor_states'ten)
        current_thrust = (
            xp.sum(self.motor_states, axis=1, keepdims=True) * max_thrust / 4
        )
        current_thrust = xp.tile(current_thrust, (1, 3))
        current_thrust[:, :2] = 0  # Sadece Z thrust var şimdilik

        # Thrust güncelle (yavaşça hedefe yaklaş)
        actual_thrust = (1 - alpha) * current_thrust[:, 2:3] + alpha * target_thrust[
            :, 2:3
        ]

        # Motor states güncelle (normalized)
        self.motor_states[:, :] = (actual_thrust / max_thrust * 4).repeat(4, axis=1)
        self.motor_states = xp.clip(self.motor_states, 0, 1)

        # ================================================================
        # ADIM 3: Kuvvetleri Hesapla (GPU Parallel)
        # ================================================================

        # --- Thrust Force ---
        # Basitleştirilmiş: Thrust yukarı yönlü, tilt ile yön değişir
        thrust_force = xp.zeros_like(self.positions)

        # Roll/Pitch'e göre thrust yönünü ayarla
        roll = self.orientations[:, 0]
        pitch = self.orientations[:, 1]

        thrust_mag = xp.sum(self.motor_states, axis=1) * max_thrust / 4
        thrust_force[:, 0] = thrust_mag * xp.sin(pitch)
        thrust_force[:, 1] = -thrust_mag * xp.sin(roll)
        thrust_force[:, 2] = thrust_mag * xp.cos(roll) * xp.cos(pitch)

        # --- Gravity Force ---
        gravity_force = xp.zeros_like(self.positions)
        gravity_force[:, 2] = -self.cfg.mass * self.cfg.gravity

        # --- Drag Force ---
        # F_drag = -0.5 × ρ × Cd × A × v²
        # Basitleştirilmiş: F_drag = -k × v × |v|
        drag_coeff = self.cfg.drag_coefficient
        drag_force = -drag_coeff * self.velocities * xp.abs(self.velocities)

        # --- Toplam Kuvvet ---
        total_force = thrust_force + gravity_force + drag_force

        # ================================================================
        # ADIM 4: İvme Hesapla (Newton's 2nd Law)
        # ================================================================
        # F = m × a  →  a = F / m

        acceleration = total_force / self.cfg.mass

        # ================================================================
        # ADIM 5: Hız ve Pozisyon Güncelle (Integration)
        # ================================================================
        # Euler integration: v_new = v_old + a × dt
        #                    pos_new = pos_old + v × dt

        self.velocities += acceleration * dt

        # Velocity limitleri uygula
        xy_speed = xp.linalg.norm(self.velocities[:, :2], axis=1, keepdims=True)
        xy_scale = xp.minimum(1.0, self.cfg.max_velocity_xy / (xy_speed + 0.001))
        self.velocities[:, :2] *= xy_scale
        self.velocities[:, 2] = xp.clip(
            self.velocities[:, 2], -self.cfg.max_velocity_z, self.cfg.max_velocity_z
        )

        # Pozisyon güncelle
        self.positions += self.velocities * dt

        # ================================================================
        # ADIM 6: Zemin Çarpışması
        # ================================================================
        on_ground = self.positions[:, 2] < 0
        self.positions[:, 2] = xp.where(on_ground, 0.0, self.positions[:, 2])
        self.velocities[:, 2] = xp.where(
            on_ground & (self.velocities[:, 2] < 0), 0.0, self.velocities[:, 2]
        )

        # ================================================================
        # ADIM 7: Attitude güncelle (basitleştirilmiş)
        # ================================================================
        # Hedef roll/pitch hesapla (velocity direction'dan)
        target_pitch = xp.arctan2(self.velocity_commands[:, 0], self.cfg.gravity)
        target_roll = xp.arctan2(-self.velocity_commands[:, 1], self.cfg.gravity)

        # Limit
        max_tilt = 0.5  # ~30 derece
        target_pitch = xp.clip(target_pitch, -max_tilt, max_tilt)
        target_roll = xp.clip(target_roll, -max_tilt, max_tilt)

        # Yavaşça hedefe yaklaş
        self.orientations[:, 0] = 0.9 * self.orientations[:, 0] + 0.1 * target_roll
        self.orientations[:, 1] = 0.9 * self.orientations[:, 1] + 0.1 * target_pitch

    def get_state(self) -> dict:
        """
        Tüm state'i CPU'ya kopyala ve döndür.
        """
        if GPU_AVAILABLE:
            return {
                "positions": cp.asnumpy(self.positions),
                "velocities": cp.asnumpy(self.velocities),
                "orientations": cp.asnumpy(self.orientations),
                "armed": cp.asnumpy(self.armed),
            }
        return {
            "positions": self.positions.copy(),
            "velocities": self.velocities.copy(),
            "orientations": self.orientations.copy(),
            "armed": self.armed.copy(),
        }


# ============================================================================
# SENSOR SIMULATION (GPU PARALLEL) - OPSİYONEL
# ============================================================================


class GPUSensorSimulator:
    """
    GPU üzerinde paralel sensör simülasyonu.

    ==========================================================================
    BU KATMAN NE YAPAR?
    ==========================================================================

    Fizik motorundan gelen GERÇEK değerlere gürültü ekler.

    Neden? Gerçek drone'larda sensörler mükemmel değil:
    - GPS: ±1-3 metre hata, bazen sinyal kaybı
    - IMU: Gürültü, bias (kayma), sıcaklık etkisi
    - Barometer: Hava basıncı değişimleri

    BU KATMAN OLMADAN simülasyon da çalışır!
    Farkı: Controller "mükemmel" pozisyon görür (gerçekçi değil)

    ==========================================================================
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.num_drones = config.num_drones

        print(f"\n[SENSORS] Initializing for {self.num_drones} drones...")

        # Sensör çıktıları
        self.gps_positions = xp.zeros((self.num_drones, 3), dtype=xp.float32)
        self.gps_valid = xp.ones(self.num_drones, dtype=xp.bool_)
        self.imu_accel = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # Kalman filter state
        self.estimated_positions = xp.zeros((self.num_drones, 3), dtype=xp.float32)
        self.estimated_velocities = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # Timing
        self.last_gps_update = 0.0
        self.gps_update_interval = 1.0 / config.gps_update_rate

        print(f"[SENSORS] GPS noise: ±{config.gps_noise_std}m")
        print(f"[SENSORS] Ready!")

    def update(
        self,
        true_positions: xp.ndarray,
        true_velocities: xp.ndarray,
        dt: float,
        current_time: float,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        """
        Sensörleri güncelle ve tahmin döndür.

        Args:
            true_positions: Fizik motorundan gelen GERÇEK pozisyonlar
            true_velocities: Fizik motorundan gelen GERÇEK hızlar
            dt: Zaman adımı
            current_time: Şu anki simülasyon zamanı

        Returns:
            estimated_positions: Gürültülü/tahmin edilen pozisyonlar
            estimated_velocities: Gürültülü/tahmin edilen hızlar
        """

        # ================================================================
        # GPS SİMÜLASYONU (düşük frekanslı, gürültülü)
        # ================================================================
        if current_time - self.last_gps_update >= self.gps_update_interval:
            self.last_gps_update = current_time

            # GPS gürültüsü ekle (Gaussian noise)
            # GPU'da: xp.random.normal N drone için paralel çalışır!
            gps_noise = xp.random.normal(
                0, self.cfg.gps_noise_std, (self.num_drones, 3)
            ).astype(xp.float32)

            self.gps_positions = true_positions + gps_noise

            # Rastgele GPS dropout (%2 şans)
            dropout_chance = xp.random.random(self.num_drones)
            self.gps_valid = dropout_chance > 0.02

        # ================================================================
        # IMU SİMÜLASYONU (yüksek frekanslı, gürültülü)
        # ================================================================
        imu_noise = xp.random.normal(
            0, self.cfg.imu_noise_std, (self.num_drones, 3)
        ).astype(xp.float32)

        # IMU ivme ölçer (gravity + hareket)
        self.imu_accel = true_velocities / max(dt, 0.001) + imu_noise
        self.imu_accel[:, 2] += self.cfg.gravity  # Gravity eklenir

        # ================================================================
        # BASİT KALMAN FİLTER (sensör füzyonu)
        # ================================================================
        # Gerçek bir Kalman filter karmaşıktır, bu basitleştirilmiş versiyon

        # GPS geçerliyse pozisyonu güncelle
        alpha = 0.3  # Fusion weight
        self.estimated_positions = xp.where(
            self.gps_valid[:, None],
            (1 - alpha) * self.estimated_positions + alpha * self.gps_positions,
            self.estimated_positions + self.estimated_velocities * dt,  # Dead reckoning
        )

        # Hız tahmini (pozisyon değişiminden)
        self.estimated_velocities = (
            1 - alpha
        ) * self.estimated_velocities + alpha * true_velocities

        return self.estimated_positions.copy(), self.estimated_velocities.copy()

    def get_raw_data(self) -> dict:
        """Ham sensör verilerini döndür (debug için)"""
        if GPU_AVAILABLE:
            return {
                "gps_positions": cp.asnumpy(self.gps_positions),
                "gps_valid": cp.asnumpy(self.gps_valid),
                "imu_accel": cp.asnumpy(self.imu_accel),
            }
        return {
            "gps_positions": self.gps_positions.copy(),
            "gps_valid": self.gps_valid.copy(),
            "imu_accel": self.imu_accel.copy(),
        }


# ============================================================================
# SWARM CONTROLLER (GPU PARALLEL)
# ============================================================================


class GPUSwarmController:
    """
    GPU üzerinde paralel swarm controller.

    ==========================================================================
    BU KATMAN NE YAPAR?
    ==========================================================================

    Drone'ların nereye gideceğini hesaplar:

    1. COLLISION AVOIDANCE (Çarpışma Önleme)
       - Her drone diğer tüm drone'lara bakar
       - Yakınsa: uzaklaş!
       - GPU'da O(N²) hesaplama paralel yapılır

    2. TARGET TRACKING (Hedefe Gitme)
       - Hedef pozisyon verilir
       - PD controller ile hedefe doğru hız hesapla

    3. VELOCITY BLENDING (Hız Birleştirme)
       - Collision avoidance + target tracking birleştirilir
       - Öncelik: Önce güvenlik (çarpışma), sonra hedef

    ==========================================================================
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.num_drones = config.num_drones

        print(f"\n[CONTROLLER] Initializing for {self.num_drones} drones...")

        # Hedef pozisyonlar
        self.targets = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        # PD controller için önceki hata
        self.prev_error = xp.zeros((self.num_drones, 3), dtype=xp.float32)

        print(f"[CONTROLLER] Ready!")

    def set_targets(self, targets: np.ndarray):
        """Hedef pozisyonları ayarla"""
        self.targets = xp.asarray(targets, dtype=xp.float32)

    def compute_velocities(
        self, positions: xp.ndarray, velocities: xp.ndarray, dt: float
    ) -> xp.ndarray:
        """
        ====================================================================
        ANA KONTROL HESAPLAMASI (GPU PARALLEL)
        ====================================================================

        Input:
            positions: (N, 3) - Drone pozisyonları (sensörden veya fizikten)
            velocities: (N, 3) - Drone hızları
            dt: Zaman adımı

        Output:
            velocity_commands: (N, 3) - Her drone için hedef hız
        """

        # ================================================================
        # 1. COLLISION AVOIDANCE (O(N²) ama GPU'da paralel!)
        # ================================================================
        """
        Mantık:
        - Her drone çifti için mesafe hesapla
        - Mesafe < avoidance_radius ise: uzaklaşma vektörü ekle
        - Mesafe < collision_radius ise: GÜÇLÜ uzaklaşma

        GPU Optimizasyonu:
        - positions[:, None, :] - positions[None, :, :]
          Bu tek satır N×N mesafe matrisini paralel hesaplar!
        """

        # Tüm drone çiftleri arasındaki fark vektörleri
        # Shape: (N, N, 3) - diff[i, j] = pos[i] - pos[j]
        diff = positions[:, None, :] - positions[None, :, :]

        # Mesafeler
        # Shape: (N, N) - dist[i, j] = ||pos[i] - pos[j]||
        dist = xp.linalg.norm(diff, axis=2)
        dist = xp.maximum(dist, 0.001)  # Division by zero önle

        # Normalized yön vektörleri (uzaklaşma yönü)
        # Shape: (N, N, 3)
        direction = diff / dist[:, :, None]

        # Hangi çiftler avoidance radius içinde?
        # Shape: (N, N) boolean
        mask = (dist < self.cfg.avoidance_radius) & (dist > 0.001)

        # Avoidance gücü (mesafeye göre)
        # Yakınsa güçlü, uzaksa zayıf
        strength = xp.where(
            dist < self.cfg.collision_radius,
            xp.ones_like(dist) * self.cfg.avoidance_strength,  # Tehlike!
            (self.cfg.avoidance_radius - dist)
            / (self.cfg.avoidance_radius - self.cfg.collision_radius),
        )
        strength = xp.maximum(strength, 0)

        # Toplam avoidance vektörü
        # Her drone için: diğer tüm drone'lardan uzaklaşma toplamı
        separation = xp.sum(direction * strength[:, :, None] * mask[:, :, None], axis=1)

        # ================================================================
        # 2. TARGET TRACKING (PD Controller)
        # ================================================================
        """
        PD Controller:
        - P (Proportional): Hataya orantılı düzeltme
        - D (Derivative): Hata değişimine orantılı düzeltme (damping)

        error = target - position
        velocity = Kp × error + Kd × (error - prev_error) / dt
        """

        error = self.targets - positions
        d_error = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error.copy()

        tracking = error * self.cfg.kp + d_error * self.cfg.kd

        # ================================================================
        # 3. VELOCITY BLENDING
        # ================================================================
        """
        Öncelik sırası:
        1. Collision avoidance (güvenlik önce!)
        2. Target tracking

        Ağırlıklı toplam:
        velocity = w1 × separation + w2 × tracking
        """

        velocity_commands = self.cfg.avoidance_strength * separation + 1.0 * tracking

        # ================================================================
        # 4. VELOCITY LIMITING
        # ================================================================

        # XY hız limiti
        xy_speed = xp.linalg.norm(velocity_commands[:, :2], axis=1, keepdims=True)
        xy_scale = xp.minimum(1.0, self.cfg.max_velocity_xy / (xy_speed + 0.001))
        velocity_commands[:, :2] *= xy_scale

        # Z hız limiti
        velocity_commands[:, 2] = xp.clip(
            velocity_commands[:, 2], -self.cfg.max_velocity_z, self.cfg.max_velocity_z
        )

        return velocity_commands

    # ====================================================================
    # FORMATION COMMANDS
    # ====================================================================

    def formation_grid(
        self,
        spacing: float = 3.0,
        altitude: float = 5.0,
        center: Tuple[float, float] = (0, 0),
    ):
        """Grid formasyonu hedefi ayarla"""
        cols = int(xp.ceil(xp.sqrt(self.num_drones)))

        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.targets[i, 0] = center[0] + (col - cols / 2) * spacing
            self.targets[i, 1] = center[1] + (row - cols / 2) * spacing
            self.targets[i, 2] = altitude

    def formation_circle(
        self,
        radius: float = 10.0,
        altitude: float = 5.0,
        center: Tuple[float, float] = (0, 0),
    ):
        """Daire formasyonu hedefi ayarla"""
        angles = xp.linspace(0, 2 * xp.pi, self.num_drones, endpoint=False)

        self.targets[:, 0] = center[0] + radius * xp.cos(angles)
        self.targets[:, 1] = center[1] + radius * xp.sin(angles)
        self.targets[:, 2] = altitude

    def set_waypoint(
        self, waypoint: Tuple[float, float, float], drone_ids: list = None
    ):
        """Belirli drone'ları waypoint'e gönder"""
        if drone_ids is None:
            drone_ids = range(self.num_drones)

        for i in drone_ids:
            self.targets[i, 0] = waypoint[0]
            self.targets[i, 1] = waypoint[1]
            self.targets[i, 2] = waypoint[2]


# ============================================================================
# COMPLETE SIMULATION
# ============================================================================


class GPUSwarmSimulation:
    """
    Tüm katmanları birleştiren ana simülasyon sınıfı.

    ==========================================================================
    KULLANIM
    ==========================================================================

    ```python
    # Simülasyon oluştur
    sim = GPUSwarmSimulation(num_drones=25, enable_sensors=True)

    # Başlat
    sim.reset()
    sim.arm_all()
    sim.takeoff(altitude=5.0)

    # Her frame:
    sim.step()
    state = sim.get_state()

    # Formasyon değiştir
    sim.formation_circle(radius=10)

    # Waypoint
    sim.set_waypoint((10, 5, 8))
    ```
    """

    def __init__(self, num_drones: int = 25, enable_sensors: bool = True):
        """
        Args:
            num_drones: Drone sayısı
            enable_sensors: Sensör simülasyonu aktif mi?
                           True = Gürültülü pozisyon (gerçekçi)
                           False = Mükemmel pozisyon (basit)
        """
        print("\n" + "=" * 70)
        print(f"   GPU SWARM SIMULATION - {num_drones} DRONES")
        print("=" * 70)

        self.config = SimulationConfig(
            num_drones=num_drones, enable_sensors=enable_sensors
        )

        # Katmanları oluştur
        self.physics = GPUPhysicsEngine(self.config)
        self.controller = GPUSwarmController(self.config)

        if enable_sensors:
            self.sensors = GPUSensorSimulator(self.config)
        else:
            self.sensors = None
            print("\n[SENSORS] DEVRE DIŞI - Mükemmel pozisyon kullanılacak")

        # Timing
        self.time = 0.0
        self.control_dt = 0.02  # 50 Hz control loop

        print("\n" + "=" * 70)
        print("   INITIALIZATION COMPLETE")
        print("=" * 70)

    def reset(self):
        """Simülasyonu sıfırla"""
        self.physics.reset_positions()
        self.controller.targets[:] = self.physics.positions.copy()
        self.time = 0.0
        print("\n[SIM] Reset complete")

    def arm_all(self):
        """Tüm drone'ları arm et"""
        self.physics.arm_all()
        print("[SIM] All drones ARMED")

    def disarm_all(self):
        """Tüm drone'ları disarm et"""
        self.physics.disarm_all()
        print("[SIM] All drones DISARMED")

    def takeoff(self, altitude: float = 5.0):
        """Kalkış komutu"""
        self.arm_all()
        self.controller.targets[:, 2] = altitude
        print(f"[SIM] Takeoff to {altitude}m")

    def land(self):
        """İniş komutu"""
        self.controller.targets[:, 2] = 0.1
        print("[SIM] Landing...")

    def formation_grid(self, spacing: float = 3.0):
        """Grid formasyonu"""
        alt = float(xp.mean(self.controller.targets[:, 2]))
        self.controller.formation_grid(spacing, altitude=max(alt, 3.0))
        print(f"[SIM] Grid formation (spacing={spacing}m)")

    def formation_circle(self, radius: float = 10.0):
        """Daire formasyonu"""
        alt = float(xp.mean(self.controller.targets[:, 2]))
        self.controller.formation_circle(radius, altitude=max(alt, 3.0))
        print(f"[SIM] Circle formation (radius={radius}m)")

    def set_waypoint(self, x: float, y: float, z: float = None):
        """Waypoint ayarla"""
        if z is None:
            z = float(xp.mean(self.controller.targets[:, 2]))
        self.controller.set_waypoint((x, y, z))
        print(f"[SIM] Waypoint: ({x:.1f}, {y:.1f}, {z:.1f})")

    def step(self):
        """
        ====================================================================
        ANA SİMÜLASYON ADIMI
        ====================================================================

        Bu fonksiyon her frame çağrılır.

        Sıra:
        1. Fizik motorundan gerçek pozisyonları al
        2. (Opsiyonel) Sensörlerden gürültülü pozisyonları al
        3. Controller ile velocity komutları hesapla
        4. Fizik motoruna velocity komutları ver
        5. Fizik motoru bir adım ilerle
        """

        physics_steps = int(self.control_dt / self.config.physics_dt)

        for _ in range(physics_steps):
            # === 1. Mevcut state ===
            positions = self.physics.positions
            velocities = self.physics.velocities

            # === 2. Sensör simülasyonu (opsiyonel) ===
            if self.sensors:
                positions, velocities = self.sensors.update(
                    positions, velocities, self.config.physics_dt, self.time
                )

            # === 3. Controller ===
            velocity_commands = self.controller.compute_velocities(
                positions, velocities, self.config.physics_dt
            )

            # === 4. Velocity komutları gönder ===
            self.physics.set_velocity_commands(velocity_commands)

            # === 5. Fizik adımı ===
            self.physics.step()

            self.time += self.config.physics_dt

    def get_state(self) -> dict:
        """
        Görselleştirme için state döndür.

        Returns:
            dict with:
                - positions: (N, 3) numpy array
                - velocities: (N, 3) numpy array
                - targets: (N, 3) numpy array
                - armed: (N,) boolean array
                - sensor_data: dict (eğer sensors aktifse)
        """
        state = self.physics.get_state()

        # Hedefleri ekle
        if GPU_AVAILABLE:
            state["targets"] = cp.asnumpy(self.controller.targets)
        else:
            state["targets"] = self.controller.targets.copy()

        # Sensör verilerini ekle
        if self.sensors:
            state["sensor_data"] = self.sensors.get_raw_data()

        state["time"] = self.time

        return state


# ============================================================================
# DEMO & TEST
# ============================================================================


def run_benchmark():
    """Performans testi"""
    print("\n" + "=" * 70)
    print("   GPU SWARM BENCHMARK")
    print("=" * 70)

    test_sizes = [10, 25, 50, 100, 200]

    for num_drones in test_sizes:
        sim = GPUSwarmSimulation(num_drones, enable_sensors=True)
        sim.reset()
        sim.takeoff()

        # Warmup
        for _ in range(10):
            sim.step()

        # Benchmark
        steps = 100
        start = time.time()
        for _ in range(steps):
            sim.step()
        elapsed = time.time() - start

        ms_per_step = elapsed / steps * 1000
        fps = steps / elapsed

        print(f"\n{num_drones:4d} drones:")
        print(f"   {ms_per_step:.2f} ms/frame")
        print(f"   {fps:.1f} FPS")
        print(f"   Realtime: {'YES' if fps >= 30 else 'NO'}")


def run_demo():
    """Basit demo"""
    print("\n" + "=" * 70)
    print("   DEMO: 25 DRONES")
    print("=" * 70)

    sim = GPUSwarmSimulation(num_drones=25, enable_sensors=True)
    sim.reset()
    sim.takeoff(5.0)

    print("\nSimulating 2 seconds...")

    for i in range(100):  # 100 frames = 2 seconds
        sim.step()

        if i == 25:
            sim.formation_circle(radius=8)
        elif i == 50:
            sim.formation_grid(spacing=4)
        elif i == 75:
            sim.set_waypoint(10, 10, 7)

        if i % 25 == 0:
            state = sim.get_state()
            avg_alt = np.mean(state["positions"][:, 2])
            avg_speed = np.mean(np.linalg.norm(state["velocities"], axis=1))
            print(f"   t={sim.time:.1f}s: alt={avg_alt:.2f}m, speed={avg_speed:.2f}m/s")

    print("\nDemo complete!")


if __name__ == "__main__":
    run_benchmark()
    print("\n")
    run_demo()
