# CEM - Sensör ve İletişim Görevleri

## Modül: SENSORS & COMMUNICATION

**Sorumluluk Alanı:** Sensör simülasyonu ve iletişim

**Çalışılacak Dosyalar:**
- `gpu_sensors.py`
- `gpu_swarm_simulation.py` (GPUSensorSimulator sınıfı)
- Yeni: `communication_sim.py`

---

## Görev Listesi

### B1: Extended Kalman Filter [YÜKSEK ÖNCELİK]
**Süre:** 4 gün
**Zorluk:** ⭐⭐⭐⭐

**Açıklama:**
Mevcut basit sensör füzyonunu gerçek EKF ile değiştir.

**Mevcut kod (basit):**
```python
# Çok basit füzyon
estimate = 0.7 * gps + 0.3 * previous
```

**Hedef kod:**
```python
class ExtendedKalmanFilter:
    """
    6-state Extended Kalman Filter.

    State: [x, y, z, vx, vy, vz]

    Measurements:
    - GPS: [x, y, z] at 10 Hz
    - Barometer: [z] at 50 Hz
    - IMU: [ax, ay, az] at 400 Hz (for prediction)
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones

        # State vector: (N, 6) - [x, y, z, vx, vy, vz]
        self.state = np.zeros((num_drones, 6), dtype=np.float32)

        # Covariance matrix: (N, 6, 6)
        self.P = np.tile(np.eye(6) * 10.0, (num_drones, 1, 1)).astype(np.float32)

        # Process noise
        self.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]).astype(np.float32)

        # Measurement noise (GPS)
        self.R_gps = np.diag([1.5**2, 1.5**2, 3.0**2]).astype(np.float32)

        # Measurement noise (Barometer)
        self.R_baro = np.array([[0.5**2]]).astype(np.float32)

    def predict(self, dt: float, imu_accel: np.ndarray = None):
        """
        Prediction step.

        State transition: x_new = F @ x + B @ u
        """
        # State transition matrix
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt

        # Predict state
        for i in range(self.num_drones):
            self.state[i] = F @ self.state[i]

            # If IMU data available, use for velocity prediction
            if imu_accel is not None:
                self.state[i, 3:6] += imu_accel[i] * dt

            # Predict covariance
            self.P[i] = F @ self.P[i] @ F.T + self.Q

    def update_gps(self, gps_measurement: np.ndarray, gps_valid: np.ndarray):
        """
        GPS measurement update.

        Only update if GPS is valid.
        """
        # Measurement matrix: H @ state = [x, y, z]
        H = np.zeros((3, 6), dtype=np.float32)
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z

        for i in range(self.num_drones):
            if not gps_valid[i]:
                continue  # Skip if GPS invalid

            # Innovation
            z = gps_measurement[i]
            y = z - H @ self.state[i]

            # Innovation covariance
            S = H @ self.P[i] @ H.T + self.R_gps

            # Kalman gain
            K = self.P[i] @ H.T @ np.linalg.inv(S)

            # Update state
            self.state[i] = self.state[i] + K @ y

            # Update covariance
            I_KH = np.eye(6) - K @ H
            self.P[i] = I_KH @ self.P[i]

    def update_barometer(self, baro_altitude: np.ndarray):
        """
        Barometer measurement update (only Z).
        """
        H = np.zeros((1, 6), dtype=np.float32)
        H[0, 2] = 1  # z

        for i in range(self.num_drones):
            z = np.array([baro_altitude[i]])
            y = z - H @ self.state[i]

            S = H @ self.P[i] @ H.T + self.R_baro
            K = self.P[i] @ H.T @ np.linalg.inv(S)

            self.state[i] = self.state[i] + K.flatten() * y[0]
            self.P[i] = (np.eye(6) - np.outer(K.flatten(), H)) @ self.P[i]

    def get_position(self) -> np.ndarray:
        return self.state[:, 0:3]

    def get_velocity(self) -> np.ndarray:
        return self.state[:, 3:6]

    def get_uncertainty(self) -> np.ndarray:
        """Position uncertainty (standard deviation)."""
        return np.sqrt(np.array([
            [self.P[i, 0, 0], self.P[i, 1, 1], self.P[i, 2, 2]]
            for i in range(self.num_drones)
        ]))
```

**GPU versiyonu için:**
- `np` yerine `xp` (cupy) kullan
- Matrix işlemlerini `xp.einsum` ile paralelize et

---

### B2: IMU Bias ve Drift [YÜKSEK ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Açıklama:**
Gerçek IMU'lar zamanla kayar (drift). Bunu simüle et.

```python
class RealisticIMU:
    """
    Gerçekçi IMU simülasyonu.

    Özellikler:
    - Gaussian noise
    - Bias (sabit offset)
    - Bias drift (zamanla değişen offset)
    - Scale factor error
    - Temperature sensitivity
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones

        # Noise parameters (MPU6050 benzeri)
        self.accel_noise_density = 0.0004  # g/√Hz
        self.gyro_noise_density = 0.005    # °/s/√Hz

        # Initial bias (her drone için farklı)
        self.accel_bias = np.random.uniform(-0.02, 0.02, (num_drones, 3))  # g
        self.gyro_bias = np.random.uniform(-0.5, 0.5, (num_drones, 3))     # °/s

        # Bias instability (random walk)
        self.accel_bias_instability = 0.0001  # g/√s
        self.gyro_bias_instability = 0.01     # °/s/√s

        # Scale factor error
        self.accel_scale_error = np.random.uniform(0.98, 1.02, (num_drones, 3))
        self.gyro_scale_error = np.random.uniform(0.98, 1.02, (num_drones, 3))

    def update_bias_drift(self, dt: float):
        """Bias'ı zamanla değiştir (random walk)."""
        self.accel_bias += np.random.normal(
            0, self.accel_bias_instability * np.sqrt(dt),
            (self.num_drones, 3)
        )
        self.gyro_bias += np.random.normal(
            0, self.gyro_bias_instability * np.sqrt(dt),
            (self.num_drones, 3)
        )

    def measure_acceleration(self, true_accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Gerçek ivmeden gürültülü ölçüm üret.
        """
        # Update bias drift
        self.update_bias_drift(dt)

        # Apply scale factor
        measured = true_accel * self.accel_scale_error

        # Add bias
        measured += self.accel_bias

        # Add noise
        noise_std = self.accel_noise_density * np.sqrt(1/dt)  # Convert to discrete
        measured += np.random.normal(0, noise_std, measured.shape)

        return measured

    def measure_angular_velocity(self, true_gyro: np.ndarray, dt: float) -> np.ndarray:
        """
        Gerçek açısal hızdan gürültülü ölçüm üret.
        """
        measured = true_gyro * self.gyro_scale_error
        measured += self.gyro_bias

        noise_std = self.gyro_noise_density * np.sqrt(1/dt)
        measured += np.random.normal(0, noise_std, measured.shape)

        return measured
```

---

### B3: Magnetometer [ORTA ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

```python
class Magnetometer:
    """
    Pusula sensörü simülasyonu.

    Sorunlar:
    - Hard iron distortion (sabit offset)
    - Soft iron distortion (scale/rotation)
    - Motor interference
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones

        # Earth's magnetic field (normalize edilmiş)
        self.earth_field = np.array([0.2, 0, 0.4])  # Türkiye civarı

        # Hard iron bias (her drone için farklı)
        self.hard_iron = np.random.uniform(-0.1, 0.1, (num_drones, 3))

        # Noise
        self.noise_std = 0.02  # Normalized

    def measure(self, orientations: np.ndarray, motor_currents: np.ndarray) -> np.ndarray:
        """
        Heading ölçümü yap.

        Returns:
            heading: (N,) array of heading angles in radians
        """
        headings = np.zeros(self.num_drones)

        for i in range(self.num_drones):
            yaw = orientations[i, 2]

            # Rotate earth field to body frame
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, s], [-s, c]])
            field_body = R @ self.earth_field[:2]

            # Add hard iron
            field_body += self.hard_iron[i, :2]

            # Add motor interference (motors create magnetic field)
            motor_interference = motor_currents[i] * 0.01
            field_body += np.random.uniform(-motor_interference, motor_interference, 2)

            # Add noise
            field_body += np.random.normal(0, self.noise_std, 2)

            # Calculate heading
            headings[i] = np.arctan2(field_body[1], field_body[0])

        return headings
```

---

### B4: İletişim Gecikmesi [YÜKSEK ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Yeni dosya oluştur: `communication_sim.py`**

```python
"""
Communication Simulation
========================

Drone-to-Ground ve Drone-to-Drone iletişim simülasyonu.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time


@dataclass
class Packet:
    """Bir iletişim paketi."""
    source_id: int
    dest_id: int  # -1 = ground station
    data: Any
    send_time: float
    arrival_time: float


class CommunicationSimulator:
    """
    İletişim gecikmesi ve paket kaybı simülasyonu.

    Özellikler:
    - Variable latency (Gaussian distribution)
    - Packet loss (random)
    - Bandwidth limitation
    - Distance-based degradation
    """

    def __init__(self, num_drones: int):
        self.num_drones = num_drones

        # Latency parameters (ms)
        self.latency_mean = 50.0        # Ortalama gecikme
        self.latency_std = 20.0         # Standart sapma
        self.latency_min = 10.0         # Minimum gecikme

        # Packet loss
        self.base_packet_loss = 0.01    # %1 base loss
        self.distance_loss_factor = 0.001  # Loss per meter

        # Bandwidth (packets per second)
        self.max_bandwidth = 100

        # Packet queue
        self.packet_queue: deque[Packet] = deque()

        # Statistics
        self.packets_sent = 0
        self.packets_lost = 0
        self.packets_delivered = 0

    def send_command(self, drone_id: int, command: Dict, current_time: float,
                     drone_position: np.ndarray = None) -> bool:
        """
        Ground station'dan drone'a komut gönder.

        Returns:
            True if packet was queued, False if lost
        """
        # Calculate packet loss probability
        loss_prob = self.base_packet_loss
        if drone_position is not None:
            distance = np.linalg.norm(drone_position)
            loss_prob += distance * self.distance_loss_factor

        # Check if packet is lost
        if np.random.random() < loss_prob:
            self.packets_lost += 1
            return False

        # Calculate arrival time
        latency = max(
            self.latency_min,
            np.random.normal(self.latency_mean, self.latency_std)
        ) / 1000.0  # Convert to seconds

        arrival_time = current_time + latency

        # Create and queue packet
        packet = Packet(
            source_id=-1,  # Ground station
            dest_id=drone_id,
            data=command,
            send_time=current_time,
            arrival_time=arrival_time
        )

        self.packet_queue.append(packet)
        self.packets_sent += 1

        return True

    def receive_packets(self, current_time: float) -> Dict[int, list]:
        """
        Zamanı gelen paketleri al.

        Returns:
            Dict mapping drone_id -> list of received commands
        """
        received = {i: [] for i in range(self.num_drones)}

        # Check all packets
        while self.packet_queue and self.packet_queue[0].arrival_time <= current_time:
            packet = self.packet_queue.popleft()

            if packet.dest_id >= 0:  # Drone'a giden paket
                received[packet.dest_id].append(packet.data)
                self.packets_delivered += 1

        return received

    def get_statistics(self) -> Dict:
        """İletişim istatistiklerini döndür."""
        return {
            'packets_sent': self.packets_sent,
            'packets_lost': self.packets_lost,
            'packets_delivered': self.packets_delivered,
            'loss_rate': self.packets_lost / max(self.packets_sent, 1),
            'queue_size': len(self.packet_queue),
        }
```

---

### B5: Paket Kaybı [ORTA ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

B4'e dahil edildi.

---

### B6: Sensör Görselleştirmesi [DÜŞÜK ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

Ali ile koordineli çalış. Sensör verilerinin grafiklerini ekle:
- GPS pozisyon vs gerçek pozisyon
- Kalman filter uncertainty ellipse
- IMU bias drift grafiği

---

## Test Prosedürü

```python
def test_kalman_filter():
    """EKF doğruluk testi."""
    ekf = ExtendedKalmanFilter(1)

    # Simulate trajectory
    true_positions = []
    estimated_positions = []

    for t in range(1000):
        # True position (circle)
        true_pos = np.array([np.sin(t/100), np.cos(t/100), 5.0])

        # Noisy GPS
        gps = true_pos + np.random.normal(0, 1.5, 3)

        # EKF update
        ekf.predict(0.02)
        ekf.update_gps(gps.reshape(1, 3), np.array([True]))

        true_positions.append(true_pos)
        estimated_positions.append(ekf.get_position()[0])

    # Calculate RMSE
    rmse = np.sqrt(np.mean((np.array(true_positions) - np.array(estimated_positions))**2))
    assert rmse < 1.0, f"EKF RMSE too high: {rmse}"
    print(f"EKF RMSE: {rmse:.3f} m")
```

---

## Sorular?

Can'a (takım liderine) sor.
