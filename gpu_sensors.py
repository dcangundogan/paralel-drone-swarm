"""
GPU-Accelerated Realistic Sensor Simulation for Drone Swarm
============================================================

Bu modül gerçek drone sensörlerini GPU üzerinde simüle eder:
- IMU (İvmeölçer + Jiroskop)
- GPS (gürültülü, düşük frekanslı)
- Barometer (irtifa)
- Magnetometer (pusula)
- Sensör Füzyonu (Kalman Filter)

Tüm hesaplamalar N drone için paralel olarak GPU'da yapılır.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

# Use centralized CUDA configuration
from cuda_config import get_array_module, GPU_AVAILABLE

# Get the appropriate array module (cupy or numpy)
cp = get_array_module()

if GPU_AVAILABLE:
    print("[GPU_SENSORS] CUDA GPU kullanılıyor")
else:
    print("[GPU_SENSORS] GPU bulunamadı, CPU kullanılıyor")


@dataclass
class SensorConfig:
    """Sensör parametreleri - gerçek değerlere yakın"""

    # IMU Parametreleri (MPU6050 benzeri)
    imu_rate: float = 400.0                    # Hz - gerçek IMU hızı
    accel_noise_std: float = 0.1               # m/s² - ivmeölçer gürültüsü
    accel_bias: float = 0.02                   # m/s² - sabit sapma
    accel_bias_drift: float = 0.001            # m/s²/s - zamanla değişen sapma
    gyro_noise_std: float = 0.01               # rad/s - jiroskop gürültüsü
    gyro_bias: float = 0.005                   # rad/s - sabit sapma
    gyro_bias_drift: float = 0.0001            # rad/s/s - zamanla değişen sapma

    # GPS Parametreleri (u-blox M8N benzeri)
    gps_rate: float = 10.0                     # Hz - GPS güncelleme hızı
    gps_horizontal_std: float = 1.5            # m - yatay hata
    gps_vertical_std: float = 3.0              # m - dikey hata
    gps_velocity_std: float = 0.1              # m/s - hız hatası
    gps_dropout_prob: float = 0.01             # GPS sinyal kaybı olasılığı

    # Barometer Parametreleri (BMP280 benzeri)
    baro_rate: float = 50.0                    # Hz
    baro_noise_std: float = 0.5                # m - irtifa gürültüsü
    baro_drift_rate: float = 0.1               # m/dakika - sıcaklık kaynaklı drift

    # Magnetometer Parametreleri (HMC5883L benzeri)
    mag_rate: float = 75.0                     # Hz
    mag_noise_std: float = 0.02                # rad - yön gürültüsü
    mag_interference_std: float = 0.1          # rad - manyetik girişim

    # Kalman Filter Parametreleri
    process_noise: float = 0.1                 # Sistem belirsizliği
    measurement_noise_pos: float = 1.0         # Pozisyon ölçüm belirsizliği
    measurement_noise_vel: float = 0.5         # Hız ölçüm belirsizliği


class GPUSensorSimulator:
    """
    GPU üzerinde paralel sensör simülasyonu.

    Her drone için bağımsız sensör gürültüsü ve bias değerleri üretir.
    Kalman Filter ile sensör füzyonu yapar.
    """

    def __init__(self, num_drones: int, config: Optional[SensorConfig] = None):
        self.num_drones = num_drones
        self.config = config or SensorConfig()
        self.xp = cp  # cupy veya numpy

        # Zaman takibi
        self.last_imu_time = 0.0
        self.last_gps_time = 0.0
        self.last_baro_time = 0.0
        self.last_mag_time = 0.0
        self.start_time = time.time()

        # ===== GPU Arrayleri =====

        # Gerçek durum (Gazebo'dan gelen "ground truth")
        self.true_positions = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.true_velocities = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.true_orientations = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)  # roll, pitch, yaw
        self.true_angular_vel = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)

        # Sensör çıktıları (gürültülü)
        self.imu_accel = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.imu_gyro = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.gps_position = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.gps_velocity = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.gps_valid = self.xp.ones(num_drones, dtype=self.xp.bool_)
        self.baro_altitude = self.xp.zeros(num_drones, dtype=self.xp.float32)
        self.mag_heading = self.xp.zeros(num_drones, dtype=self.xp.float32)

        # Sensör bias değerleri (her drone için farklı)
        self.accel_bias = self.xp.random.uniform(
            -self.config.accel_bias, self.config.accel_bias,
            (num_drones, 3)
        ).astype(self.xp.float32)
        self.gyro_bias = self.xp.random.uniform(
            -self.config.gyro_bias, self.config.gyro_bias,
            (num_drones, 3)
        ).astype(self.xp.float32)
        self.baro_bias = self.xp.random.uniform(
            -1.0, 1.0, num_drones
        ).astype(self.xp.float32)

        # ===== Kalman Filter State =====
        # State: [x, y, z, vx, vy, vz]
        self.kf_state = self.xp.zeros((num_drones, 6), dtype=self.xp.float32)
        self.kf_covariance = self.xp.tile(
            self.xp.eye(6, dtype=self.xp.float32) * 10.0,
            (num_drones, 1, 1)
        )  # (N, 6, 6)

        # Füzyon sonucu (controller'ın kullanacağı)
        self.fused_positions = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)
        self.fused_velocities = self.xp.zeros((num_drones, 3), dtype=self.xp.float32)

        print(f"[GPU_SENSORS] {num_drones} drone için sensör simülasyonu hazır")

    def update_ground_truth(self, positions: np.ndarray, velocities: np.ndarray,
                           orientations: np.ndarray = None, angular_vel: np.ndarray = None):
        """Gazebo'dan gelen gerçek değerleri güncelle"""
        self.true_positions = self.xp.asarray(positions, dtype=self.xp.float32)
        self.true_velocities = self.xp.asarray(velocities, dtype=self.xp.float32)

        if orientations is not None:
            self.true_orientations = self.xp.asarray(orientations, dtype=self.xp.float32)
        if angular_vel is not None:
            self.true_angular_vel = self.xp.asarray(angular_vel, dtype=self.xp.float32)

    def simulate_imu(self, dt: float) -> Tuple:
        """
        IMU simülasyonu (400 Hz)

        Returns:
            accel: (N, 3) ivme ölçümleri [m/s²]
            gyro: (N, 3) açısal hız ölçümleri [rad/s]
        """
        cfg = self.config

        # İvme: gerçek ivme + yerçekimi + gürültü + bias
        # Basitleştirilmiş: hız değişiminden ivme hesapla
        gravity = self.xp.array([0, 0, 9.81], dtype=self.xp.float32)

        # Gürültü ekle
        accel_noise = self.xp.random.normal(
            0, cfg.accel_noise_std, (self.num_drones, 3)
        ).astype(self.xp.float32)

        # Bias drift
        self.accel_bias += self.xp.random.normal(
            0, cfg.accel_bias_drift * dt, (self.num_drones, 3)
        ).astype(self.xp.float32)

        # Toplam ivme ölçümü
        self.imu_accel = (
            self.true_velocities / max(dt, 0.001) +  # Basit ivme tahmini
            gravity +
            self.accel_bias +
            accel_noise
        )

        # Jiroskop
        gyro_noise = self.xp.random.normal(
            0, cfg.gyro_noise_std, (self.num_drones, 3)
        ).astype(self.xp.float32)

        self.gyro_bias += self.xp.random.normal(
            0, cfg.gyro_bias_drift * dt, (self.num_drones, 3)
        ).astype(self.xp.float32)

        self.imu_gyro = self.true_angular_vel + self.gyro_bias + gyro_noise

        return self.imu_accel, self.imu_gyro

    def simulate_gps(self) -> Tuple:
        """
        GPS simülasyonu (10 Hz, gürültülü, bazen sinyal kaybı)

        Returns:
            position: (N, 3) GPS pozisyonu [m]
            velocity: (N, 3) GPS hızı [m/s]
            valid: (N,) GPS geçerlilik durumu
        """
        cfg = self.config

        # Pozisyon gürültüsü (yatay ve dikey farklı)
        pos_noise = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        pos_noise[:, 0:2] = self.xp.random.normal(
            0, cfg.gps_horizontal_std, (self.num_drones, 2)
        )
        pos_noise[:, 2] = self.xp.random.normal(
            0, cfg.gps_vertical_std, self.num_drones
        )

        self.gps_position = self.true_positions + pos_noise

        # Hız gürültüsü
        vel_noise = self.xp.random.normal(
            0, cfg.gps_velocity_std, (self.num_drones, 3)
        ).astype(self.xp.float32)
        self.gps_velocity = self.true_velocities + vel_noise

        # GPS dropout (rastgele sinyal kaybı)
        self.gps_valid = self.xp.random.random(self.num_drones) > cfg.gps_dropout_prob

        return self.gps_position, self.gps_velocity, self.gps_valid

    def simulate_barometer(self, dt: float) -> np.ndarray:
        """
        Barometer simülasyonu (50 Hz)

        Returns:
            altitude: (N,) irtifa ölçümü [m]
        """
        cfg = self.config

        # Gürültü
        noise = self.xp.random.normal(
            0, cfg.baro_noise_std, self.num_drones
        ).astype(self.xp.float32)

        # Sıcaklık kaynaklı drift
        elapsed = time.time() - self.start_time
        drift = self.baro_bias * (cfg.baro_drift_rate * elapsed / 60.0)

        self.baro_altitude = self.true_positions[:, 2] + noise + drift

        return self.baro_altitude

    def simulate_magnetometer(self) -> np.ndarray:
        """
        Magnetometer/Pusula simülasyonu (75 Hz)

        Returns:
            heading: (N,) yön açısı [rad]
        """
        cfg = self.config

        # Temel gürültü
        noise = self.xp.random.normal(
            0, cfg.mag_noise_std, self.num_drones
        ).astype(self.xp.float32)

        # Manyetik girişim (motorlardan)
        # Motor hızına bağlı olabilir, şimdilik rastgele
        interference = self.xp.random.normal(
            0, cfg.mag_interference_std, self.num_drones
        ).astype(self.xp.float32)

        self.mag_heading = self.true_orientations[:, 2] + noise + interference

        return self.mag_heading

    def kalman_predict(self, dt: float):
        """
        Kalman Filter tahmin adımı (GPU paralel)

        State transition: x_new = F * x + B * u
        """
        # State transition matrix
        F = self.xp.eye(6, dtype=self.xp.float32)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Process noise
        Q = self.xp.eye(6, dtype=self.xp.float32) * self.config.process_noise * dt

        # Tahmin: her drone için paralel
        # x_pred = F @ x
        self.kf_state = self.xp.einsum('ij,nj->ni', F, self.kf_state)

        # Kovaryans güncelleme: P = F @ P @ F.T + Q
        self.kf_covariance = (
            self.xp.einsum('ij,njk,lk->nil', F, self.kf_covariance, F) + Q
        )

    def kalman_update_gps(self):
        """
        Kalman Filter güncelleme adımı - GPS ölçümü
        """
        # Measurement matrix (pozisyon ve hız ölçüyoruz)
        H = self.xp.eye(6, dtype=self.xp.float32)

        # Measurement noise
        R = self.xp.diag(self.xp.array([
            self.config.gps_horizontal_std**2,
            self.config.gps_horizontal_std**2,
            self.config.gps_vertical_std**2,
            self.config.gps_velocity_std**2,
            self.config.gps_velocity_std**2,
            self.config.gps_velocity_std**2
        ], dtype=self.xp.float32))

        # Ölçüm vektörü
        z = self.xp.concatenate([self.gps_position, self.gps_velocity], axis=1)

        # Innovation: y = z - H @ x
        y = z - self.kf_state

        # Innovation covariance: S = H @ P @ H.T + R
        S = self.kf_covariance + R

        # Kalman gain: K = P @ H.T @ inv(S)
        # Basitleştirilmiş: her drone için ayrı hesapla
        for i in range(self.num_drones):
            if self.gps_valid[i]:  # Sadece GPS geçerliyse güncelle
                K = self.xp.linalg.solve(S[i].T, self.kf_covariance[i].T).T

                # State güncelleme: x = x + K @ y
                self.kf_state[i] += K @ y[i]

                # Kovaryans güncelleme: P = (I - K @ H) @ P
                I_KH = self.xp.eye(6, dtype=self.xp.float32) - K @ H
                self.kf_covariance[i] = I_KH @ self.kf_covariance[i]

    def kalman_update_baro(self):
        """Barometer ölçümü ile sadece Z güncelle"""
        # Sadece z pozisyonunu güncelle
        H = self.xp.zeros((1, 6), dtype=self.xp.float32)
        H[0, 2] = 1.0  # z pozisyonu

        R = self.xp.array([[self.config.baro_noise_std**2]], dtype=self.xp.float32)

        for i in range(self.num_drones):
            z = self.baro_altitude[i:i+1]
            y = z - self.kf_state[i, 2:3]

            S = H @ self.kf_covariance[i] @ H.T + R
            K = self.kf_covariance[i] @ H.T @ self.xp.linalg.inv(S)

            self.kf_state[i] += (K @ y).flatten()
            self.kf_covariance[i] = (self.xp.eye(6) - K @ H) @ self.kf_covariance[i]

    def get_fused_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sensör füzyonu sonucunu döndür

        Returns:
            positions: (N, 3) füzyon edilmiş pozisyonlar
            velocities: (N, 3) füzyon edilmiş hızlar
        """
        self.fused_positions = self.kf_state[:, 0:3]
        self.fused_velocities = self.kf_state[:, 3:6]

        # GPU'dan CPU'ya transfer (gerekirse)
        if GPU_AVAILABLE:
            import cupy
            return cupy.asnumpy(self.fused_positions), cupy.asnumpy(self.fused_velocities)
        return self.fused_positions, self.fused_velocities

    def update(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tüm sensörleri güncelle ve füzyon yap

        Args:
            dt: Zaman adımı (saniye)

        Returns:
            positions: Füzyon edilmiş pozisyonlar
            velocities: Füzyon edilmiş hızlar
        """
        current_time = time.time()

        # IMU (yüksek frekanslı)
        if current_time - self.last_imu_time >= 1.0 / self.config.imu_rate:
            self.simulate_imu(dt)
            self.last_imu_time = current_time

        # Kalman tahmin (her adımda)
        self.kalman_predict(dt)

        # GPS (düşük frekanslı)
        if current_time - self.last_gps_time >= 1.0 / self.config.gps_rate:
            self.simulate_gps()
            self.kalman_update_gps()
            self.last_gps_time = current_time

        # Barometer
        if current_time - self.last_baro_time >= 1.0 / self.config.baro_rate:
            self.simulate_barometer(dt)
            self.kalman_update_baro()
            self.last_baro_time = current_time

        # Magnetometer
        if current_time - self.last_mag_time >= 1.0 / self.config.mag_rate:
            self.simulate_magnetometer()
            self.last_mag_time = current_time

        return self.get_fused_state()

    def get_raw_sensors(self) -> dict:
        """Ham sensör verilerini döndür (debug için)"""
        if GPU_AVAILABLE:
            import cupy
            return {
                'imu_accel': cupy.asnumpy(self.imu_accel),
                'imu_gyro': cupy.asnumpy(self.imu_gyro),
                'gps_position': cupy.asnumpy(self.gps_position),
                'gps_velocity': cupy.asnumpy(self.gps_velocity),
                'gps_valid': cupy.asnumpy(self.gps_valid),
                'baro_altitude': cupy.asnumpy(self.baro_altitude),
                'mag_heading': cupy.asnumpy(self.mag_heading),
            }
        return {
            'imu_accel': self.imu_accel,
            'imu_gyro': self.imu_gyro,
            'gps_position': self.gps_position,
            'gps_velocity': self.gps_velocity,
            'gps_valid': self.gps_valid,
            'baro_altitude': self.baro_altitude,
            'mag_heading': self.mag_heading,
        }

    def get_estimation_error(self) -> dict:
        """Tahmin hatasını hesapla (ground truth vs füzyon)"""
        pos_error = self.fused_positions - self.true_positions
        vel_error = self.fused_velocities - self.true_velocities

        if GPU_AVAILABLE:
            import cupy
            pos_error = cupy.asnumpy(pos_error)
            vel_error = cupy.asnumpy(vel_error)

        return {
            'position_rmse': np.sqrt(np.mean(pos_error**2, axis=0)),
            'velocity_rmse': np.sqrt(np.mean(vel_error**2, axis=0)),
            'position_max_error': np.max(np.abs(pos_error), axis=0),
            'velocity_max_error': np.max(np.abs(vel_error), axis=0),
        }


# ============================================================
# DEMO / TEST
# ============================================================

def demo_sensor_simulation():
    """Sensör simülasyonu demo"""
    print("\n" + "="*60)
    print("GPU SENSOR SIMULATION DEMO")
    print("="*60)

    num_drones = 10
    sensors = GPUSensorSimulator(num_drones)

    # Sahte ground truth oluştur
    positions = np.random.uniform(-10, 10, (num_drones, 3)).astype(np.float32)
    positions[:, 2] = np.abs(positions[:, 2]) + 2  # Pozitif irtifa
    velocities = np.random.uniform(-2, 2, (num_drones, 3)).astype(np.float32)
    orientations = np.random.uniform(-0.5, 0.5, (num_drones, 3)).astype(np.float32)

    sensors.update_ground_truth(positions, velocities, orientations)

    print(f"\n{num_drones} drone için sensör simülasyonu başlatıldı")
    print(f"GPU Kullanımı: {GPU_AVAILABLE}")

    # Birkaç adım simüle et
    dt = 0.02  # 50 Hz

    print("\n--- 100 adım simülasyon ---")
    start = time.time()

    for i in range(100):
        # Ground truth güncelle (hareket simülasyonu)
        positions += velocities * dt
        positions[:, 2] = np.maximum(positions[:, 2], 0.1)
        sensors.update_ground_truth(positions, velocities, orientations)

        # Sensörleri güncelle
        fused_pos, fused_vel = sensors.update(dt)

    elapsed = time.time() - start
    print(f"100 adım süresi: {elapsed*1000:.2f} ms")
    print(f"Adım başına: {elapsed*10:.2f} ms")

    # Hata analizi
    errors = sensors.get_estimation_error()
    print("\n--- Tahmin Hataları (RMSE) ---")
    print(f"Pozisyon X: {errors['position_rmse'][0]:.3f} m")
    print(f"Pozisyon Y: {errors['position_rmse'][1]:.3f} m")
    print(f"Pozisyon Z: {errors['position_rmse'][2]:.3f} m")
    print(f"Hız X: {errors['velocity_rmse'][0]:.3f} m/s")
    print(f"Hız Y: {errors['velocity_rmse'][1]:.3f} m/s")
    print(f"Hız Z: {errors['velocity_rmse'][2]:.3f} m/s")

    # Ham sensör verileri
    raw = sensors.get_raw_sensors()
    print("\n--- Ham Sensör Verileri (Drone 0) ---")
    print(f"IMU Accel: {raw['imu_accel'][0]}")
    print(f"IMU Gyro: {raw['imu_gyro'][0]}")
    print(f"GPS Pos: {raw['gps_position'][0]}")
    print(f"GPS Valid: {raw['gps_valid'][0]}")
    print(f"Baro Alt: {raw['baro_altitude'][0]:.2f} m")
    print(f"Mag Heading: {np.degrees(raw['mag_heading'][0]):.1f}°")

    print("\n--- Ground Truth vs Fused (Drone 0) ---")
    print(f"True Pos: {positions[0]}")
    print(f"Fused Pos: {fused_pos[0]}")
    print(f"Fark: {np.abs(positions[0] - fused_pos[0])}")


if __name__ == "__main__":
    demo_sensor_simulation()
