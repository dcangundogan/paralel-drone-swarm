<<<<<<< HEAD
"""
GPU Controller with Realistic Sensor Integration
=================================================

Bu dosya gpu_controller.py'nin sensör simülasyonu eklenmiş versiyonudur.
Artık "mükemmel" pozisyon yerine gürültülü sensör verisi kullanıyor.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

# Sensör modülünü import et
from gpu_sensors import GPUSensorSimulator, SensorConfig

# GPU desteği
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as cp
    GPU_AVAILABLE = False


@dataclass
class SwarmConfigWithSensors:
    """Sensör destekli swarm konfigürasyonu"""
    num_drones: int = 25
    control_rate: float = 50.0

    # Velocity limits
    max_velocity_xy: float = 3.0
    max_velocity_z: float = 2.0
    max_acceleration: float = 2.0

    # Collision avoidance
    collision_radius: float = 0.8
    avoidance_radius: float = 2.5
    avoidance_strength: float = 3.0

    # Swarm behavior
    cohesion_strength: float = 0.3
    alignment_strength: float = 0.2
    target_gain_p: float = 1.5
    target_gain_d: float = 0.3

    # Altitude
    default_altitude: float = 5.0

    # SENSOR OPTIONS
    use_realistic_sensors: bool = True   # True = gürültülü sensör, False = mükemmel
    sensor_config: Optional[SensorConfig] = None


class GPUSwarmControllerWithSensors:
    """
    Gerçekçi sensör simülasyonu ile GPU-accelerated swarm controller.

    Fark: Pozisyon/hız bilgisi artık direkt Gazebo'dan değil,
    gürültülü sensörlerden ve Kalman Filter füzyonundan geliyor.
    """

    def __init__(self, config: Optional[SwarmConfigWithSensors] = None):
        self.config = config or SwarmConfigWithSensors()
        self.xp = cp
        self.num_drones = self.config.num_drones

        # State arrays (GPU)
        self.positions = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.velocities = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.targets = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.prev_error = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)

        # Drone states
        self.armed = self.xp.zeros(self.num_drones, dtype=self.xp.bool_)

        # === SENSÖR SİMÜLATÖRÜ ===
        self.use_sensors = self.config.use_realistic_sensors
        if self.use_sensors:
            sensor_cfg = self.config.sensor_config or SensorConfig()
            self.sensors = GPUSensorSimulator(self.num_drones, sensor_cfg)
            print(f"[CONTROLLER] Gerçekçi sensör simülasyonu AKTİF")
        else:
            self.sensors = None
            print(f"[CONTROLLER] Mükemmel sensör modu (sensör yok)")

        # Ground truth (Gazebo'dan gelen gerçek değerler)
        self.ground_truth_positions = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.ground_truth_velocities = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)

        # Timing
        self.last_update = time.time()

        print(f"[CONTROLLER] {self.num_drones} drone controller hazır (GPU: {GPU_AVAILABLE})")

    def update_ground_truth(self, positions: np.ndarray, velocities: np.ndarray = None,
                           orientations: np.ndarray = None):
        """
        Gazebo'dan gelen GERÇEK pozisyonları al.
        Bunlar sensörlere input olarak verilecek.
        """
        self.ground_truth_positions = self.xp.asarray(positions, dtype=self.xp.float32)

        if velocities is not None:
            self.ground_truth_velocities = self.xp.asarray(velocities, dtype=self.xp.float32)

        # Sensör simülasyonunu güncelle
        if self.use_sensors:
            self.sensors.update_ground_truth(
                positions,
                velocities if velocities is not None else np.zeros_like(positions),
                orientations
            )

    def get_sensor_estimates(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sensör füzyonundan tahmin edilen pozisyon ve hızları al.

        Eğer sensör modu kapalıysa, ground truth döner.
        """
        if self.use_sensors:
            return self.sensors.update(dt)
        else:
            # Sensör yok, direkt ground truth kullan
            if GPU_AVAILABLE:
                return cp.asnumpy(self.ground_truth_positions), cp.asnumpy(self.ground_truth_velocities)
            return self.ground_truth_positions.copy(), self.ground_truth_velocities.copy()

    def compute_control(self, dt: float = None) -> np.ndarray:
        """
        Ana kontrol hesaplaması.

        Şimdi sensör füzyonundan gelen tahminleri kullanıyor.
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update
        self.last_update = current_time
        dt = max(dt, 0.001)

        # === SENSÖR VERİSİ AL ===
        estimated_pos, estimated_vel = self.get_sensor_estimates(dt)
        self.positions = self.xp.asarray(estimated_pos, dtype=self.xp.float32)
        self.velocities = self.xp.asarray(estimated_vel, dtype=self.xp.float32)

        # === KONTROL HESAPLAMALARI (GPU) ===

        # 1. Collision avoidance
        separation = self._compute_separation()

        # 2. Cohesion
        cohesion = self._compute_cohesion()

        # 3. Alignment
        alignment = self._compute_alignment()

        # 4. Target tracking (PD)
        tracking = self._compute_target_tracking(dt)

        # 5. Blend velocities
        cfg = self.config
        command = (
            cfg.avoidance_strength * separation +
            cfg.target_gain_p * tracking +
            cfg.cohesion_strength * cohesion +
            cfg.alignment_strength * alignment
        )

        # 6. Apply limits
        command = self._apply_velocity_limits(command, dt)

        # 7. Disarm edilmişleri sıfırla
        command = self.xp.where(
            self.armed[:, None],
            command,
            self.xp.zeros_like(command)
        )

        if GPU_AVAILABLE:
            return cp.asnumpy(command)
        return command

    def _compute_separation(self):
        """Reynolds separation (collision avoidance)"""
        cfg = self.config

        # Pairwise distances
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = self.xp.linalg.norm(diff, axis=2)
        dist = self.xp.maximum(dist, 0.001)

        # Mask for neighbors in avoidance range
        mask = (dist < cfg.avoidance_radius) & (dist > 0.001)

        # Direction away from neighbors
        direction = diff / dist[:, :, None]

        # Strength based on distance
        strength = self.xp.where(
            dist < cfg.collision_radius,
            self.xp.ones_like(dist) * cfg.avoidance_strength,
            (cfg.avoidance_radius - dist) / (cfg.avoidance_radius - cfg.collision_radius)
        )
        strength = self.xp.maximum(strength, 0)

        # Apply mask and sum
        separation = self.xp.sum(direction * strength[:, :, None] * mask[:, :, None], axis=1)

        return separation

    def _compute_cohesion(self):
        """Swarm center attraction"""
        center = self.xp.mean(self.positions, axis=0)
        direction = center - self.positions
        dist = self.xp.linalg.norm(direction, axis=1, keepdims=True)
        dist = self.xp.maximum(dist, 0.001)
        return direction / dist * self.xp.minimum(dist, 1.0)

    def _compute_alignment(self):
        """Velocity consensus"""
        avg_velocity = self.xp.mean(self.velocities, axis=0)
        return avg_velocity - self.velocities

    def _compute_target_tracking(self, dt: float):
        """PD controller for target tracking"""
        error = self.targets - self.positions
        d_error = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error.copy()

        return error * self.config.target_gain_p + d_error * self.config.target_gain_d

    def _apply_velocity_limits(self, velocities, dt: float):
        """Enforce velocity and acceleration limits"""
        cfg = self.config

        # Acceleration limit
        vel_change = velocities - self.velocities
        accel = self.xp.linalg.norm(vel_change, axis=1, keepdims=True)
        max_change = cfg.max_acceleration * dt
        scale = self.xp.minimum(1.0, max_change / self.xp.maximum(accel, 0.001))
        velocities = self.velocities + vel_change * scale

        # XY velocity limit
        xy_speed = self.xp.linalg.norm(velocities[:, :2], axis=1, keepdims=True)
        xy_scale = self.xp.minimum(1.0, cfg.max_velocity_xy / self.xp.maximum(xy_speed, 0.001))
        velocities[:, :2] *= xy_scale

        # Z velocity limit
        velocities[:, 2] = self.xp.clip(velocities[:, 2], -cfg.max_velocity_z, cfg.max_velocity_z)

        return velocities

    # === HIGH LEVEL COMMANDS ===

    def arm_all(self):
        self.armed[:] = True

    def disarm_all(self):
        self.armed[:] = False

    def takeoff(self, altitude: float = None):
        alt = altitude or self.config.default_altitude
        self.arm_all()
        self.targets[:, 2] = alt
        print(f"[CONTROLLER] Takeoff to {alt}m")

    def land(self):
        self.targets[:, 2] = 0.0
        print("[CONTROLLER] Landing")

    def set_formation_circle(self, radius: float = 10.0, center: tuple = (0, 0)):
        angles = self.xp.linspace(0, 2 * self.xp.pi, self.num_drones, endpoint=False)
        self.targets[:, 0] = center[0] + radius * self.xp.cos(angles)
        self.targets[:, 1] = center[1] + radius * self.xp.sin(angles)

    def set_formation_grid(self, spacing: float = 3.0):
        cols = int(self.xp.ceil(self.xp.sqrt(self.num_drones)))
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.targets[i, 0] = col * spacing
            self.targets[i, 1] = row * spacing

    # === SENSOR DIAGNOSTICS ===

    def get_sensor_errors(self) -> dict:
        """Sensör tahmin hatalarını döndür"""
        if self.use_sensors:
            return self.sensors.get_estimation_error()
        return {'message': 'Sensör simülasyonu kapalı'}

    def get_raw_sensor_data(self) -> dict:
        """Ham sensör verilerini döndür"""
        if self.use_sensors:
            return self.sensors.get_raw_sensors()
        return {'message': 'Sensör simülasyonu kapalı'}

    def get_sensor_vs_truth(self, drone_id: int = 0) -> dict:
        """Belirli bir drone için sensör vs gerçek karşılaştırması"""
        if GPU_AVAILABLE:
            true_pos = cp.asnumpy(self.ground_truth_positions[drone_id])
            est_pos = cp.asnumpy(self.positions[drone_id])
        else:
            true_pos = self.ground_truth_positions[drone_id]
            est_pos = self.positions[drone_id]

        return {
            'drone_id': drone_id,
            'true_position': true_pos,
            'estimated_position': est_pos,
            'error': np.abs(true_pos - est_pos),
            'error_magnitude': np.linalg.norm(true_pos - est_pos)
        }


# ============================================================
# DEMO
# ============================================================

def demo_controller_with_sensors():
    """Sensörlü controller demo"""
    print("\n" + "="*60)
    print("CONTROLLER WITH SENSORS DEMO")
    print("="*60)

    # Sensörlü config
    config = SwarmConfigWithSensors(
        num_drones=10,
        use_realistic_sensors=True,
        sensor_config=SensorConfig(
            gps_horizontal_std=2.0,  # 2m GPS hatası
            gps_dropout_prob=0.05,   # %5 sinyal kaybı
        )
    )

    controller = GPUSwarmControllerWithSensors(config)

    # Başlangıç pozisyonları
    positions = np.zeros((10, 3), dtype=np.float32)
    for i in range(10):
        positions[i] = [i * 3, 0, 0.5]  # Grid düzeninde

    velocities = np.zeros((10, 3), dtype=np.float32)

    # Takeoff
    controller.update_ground_truth(positions, velocities)
    controller.takeoff(5.0)

    print("\n--- 100 adım simülasyon ---")
    dt = 0.02

    for step in range(100):
        # Simüle hareket
        cmd = controller.compute_control(dt)
        velocities = cmd * 0.5  # Basit dinamik
        positions += velocities * dt
        positions[:, 2] = np.maximum(positions[:, 2], 0.1)

        # Ground truth güncelle
        controller.update_ground_truth(positions, velocities)

        if step % 25 == 0:
            errors = controller.get_sensor_errors()
            comparison = controller.get_sensor_vs_truth(0)
            print(f"\nAdım {step}:")
            print(f"  Drone 0 gerçek poz: {comparison['true_position']}")
            print(f"  Drone 0 tahmin poz: {comparison['estimated_position']}")
            print(f"  Hata: {comparison['error_magnitude']:.3f} m")

    print("\n--- Final Sensör Hataları ---")
    errors = controller.get_sensor_errors()
    print(f"Position RMSE: {errors['position_rmse']}")
    print(f"Velocity RMSE: {errors['velocity_rmse']}")


if __name__ == "__main__":
    demo_controller_with_sensors()
=======
"""
GPU Controller with Realistic Sensor Integration
=================================================

Bu dosya gpu_controller.py'nin sensör simülasyonu eklenmiş versiyonudur.
Artık "mükemmel" pozisyon yerine gürültülü sensör verisi kullanıyor.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple

# Sensör modülünü import et
from gpu_sensors import GPUSensorSimulator, SensorConfig

# GPU desteği
try:
    import cupy as cp
    # Test if CuPy actually works - arange triggers kernel compilation
    _test = cp.arange(1)
    del _test
    GPU_AVAILABLE = True
except (ImportError, Exception):
    import numpy as cp
    GPU_AVAILABLE = False


@dataclass
class SwarmConfigWithSensors:
    """Sensör destekli swarm konfigürasyonu"""
    num_drones: int = 25
    control_rate: float = 50.0

    # Velocity limits
    max_velocity_xy: float = 3.0
    max_velocity_z: float = 2.0
    max_acceleration: float = 2.0

    # Collision avoidance
    collision_radius: float = 0.8
    avoidance_radius: float = 2.5
    avoidance_strength: float = 3.0

    # Swarm behavior
    cohesion_strength: float = 0.3
    alignment_strength: float = 0.2
    target_gain_p: float = 1.5
    target_gain_d: float = 0.3

    # Altitude
    default_altitude: float = 5.0

    # SENSOR OPTIONS
    use_realistic_sensors: bool = True   # True = gürültülü sensör, False = mükemmel
    sensor_config: Optional[SensorConfig] = None


class GPUSwarmControllerWithSensors:
    """
    Gerçekçi sensör simülasyonu ile GPU-accelerated swarm controller.

    Fark: Pozisyon/hız bilgisi artık direkt Gazebo'dan değil,
    gürültülü sensörlerden ve Kalman Filter füzyonundan geliyor.
    """

    def __init__(self, config: Optional[SwarmConfigWithSensors] = None):
        self.config = config or SwarmConfigWithSensors()
        self.xp = cp
        self.num_drones = self.config.num_drones

        # State arrays (GPU)
        self.positions = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.velocities = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.targets = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.prev_error = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)

        # Drone states
        self.armed = self.xp.zeros(self.num_drones, dtype=self.xp.bool_)

        # === SENSÖR SİMÜLATÖRÜ ===
        self.use_sensors = self.config.use_realistic_sensors
        if self.use_sensors:
            sensor_cfg = self.config.sensor_config or SensorConfig()
            self.sensors = GPUSensorSimulator(self.num_drones, sensor_cfg)
            print(f"[CONTROLLER] Gerçekçi sensör simülasyonu AKTİF")
        else:
            self.sensors = None
            print(f"[CONTROLLER] Mükemmel sensör modu (sensör yok)")

        # Ground truth (Gazebo'dan gelen gerçek değerler)
        self.ground_truth_positions = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)
        self.ground_truth_velocities = self.xp.zeros((self.num_drones, 3), dtype=self.xp.float32)

        # Timing
        self.last_update = time.time()

        print(f"[CONTROLLER] {self.num_drones} drone controller hazır (GPU: {GPU_AVAILABLE})")

    def update_ground_truth(self, positions: np.ndarray, velocities: np.ndarray = None,
                           orientations: np.ndarray = None):
        """
        Gazebo'dan gelen GERÇEK pozisyonları al.
        Bunlar sensörlere input olarak verilecek.
        """
        self.ground_truth_positions = self.xp.asarray(positions, dtype=self.xp.float32)

        if velocities is not None:
            self.ground_truth_velocities = self.xp.asarray(velocities, dtype=self.xp.float32)

        # Sensör simülasyonunu güncelle
        if self.use_sensors:
            self.sensors.update_ground_truth(
                positions,
                velocities if velocities is not None else np.zeros_like(positions),
                orientations
            )

    def get_sensor_estimates(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sensör füzyonundan tahmin edilen pozisyon ve hızları al.

        Eğer sensör modu kapalıysa, ground truth döner.
        """
        if self.use_sensors:
            return self.sensors.update(dt)
        else:
            # Sensör yok, direkt ground truth kullan
            if GPU_AVAILABLE:
                return cp.asnumpy(self.ground_truth_positions), cp.asnumpy(self.ground_truth_velocities)
            return self.ground_truth_positions.copy(), self.ground_truth_velocities.copy()

    def compute_control(self, dt: float = None) -> np.ndarray:
        """
        Ana kontrol hesaplaması.

        Şimdi sensör füzyonundan gelen tahminleri kullanıyor.
        """
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_update
        self.last_update = current_time
        dt = max(dt, 0.001)

        # === SENSÖR VERİSİ AL ===
        estimated_pos, estimated_vel = self.get_sensor_estimates(dt)
        self.positions = self.xp.asarray(estimated_pos, dtype=self.xp.float32)
        self.velocities = self.xp.asarray(estimated_vel, dtype=self.xp.float32)

        # === KONTROL HESAPLAMALARI (GPU) ===

        # 1. Collision avoidance
        separation = self._compute_separation()

        # 2. Cohesion
        cohesion = self._compute_cohesion()

        # 3. Alignment
        alignment = self._compute_alignment()

        # 4. Target tracking (PD)
        tracking = self._compute_target_tracking(dt)

        # 5. Blend velocities
        cfg = self.config
        command = (
            cfg.avoidance_strength * separation +
            cfg.target_gain_p * tracking +
            cfg.cohesion_strength * cohesion +
            cfg.alignment_strength * alignment
        )

        # 6. Apply limits
        command = self._apply_velocity_limits(command, dt)

        # 7. Disarm edilmişleri sıfırla
        command = self.xp.where(
            self.armed[:, None],
            command,
            self.xp.zeros_like(command)
        )

        if GPU_AVAILABLE:
            return cp.asnumpy(command)
        return command

    def _compute_separation(self):
        """Reynolds separation (collision avoidance)"""
        cfg = self.config

        # Pairwise distances
        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = self.xp.linalg.norm(diff, axis=2)
        dist = self.xp.maximum(dist, 0.001)

        # Mask for neighbors in avoidance range
        mask = (dist < cfg.avoidance_radius) & (dist > 0.001)

        # Direction away from neighbors
        direction = diff / dist[:, :, None]

        # Strength based on distance
        strength = self.xp.where(
            dist < cfg.collision_radius,
            self.xp.ones_like(dist) * cfg.avoidance_strength,
            (cfg.avoidance_radius - dist) / (cfg.avoidance_radius - cfg.collision_radius)
        )
        strength = self.xp.maximum(strength, 0)

        # Apply mask and sum
        separation = self.xp.sum(direction * strength[:, :, None] * mask[:, :, None], axis=1)

        return separation

    def _compute_cohesion(self):
        """Swarm center attraction"""
        center = self.xp.mean(self.positions, axis=0)
        direction = center - self.positions
        dist = self.xp.linalg.norm(direction, axis=1, keepdims=True)
        dist = self.xp.maximum(dist, 0.001)
        return direction / dist * self.xp.minimum(dist, 1.0)

    def _compute_alignment(self):
        """Velocity consensus"""
        avg_velocity = self.xp.mean(self.velocities, axis=0)
        return avg_velocity - self.velocities

    def _compute_target_tracking(self, dt: float):
        """PD controller for target tracking"""
        error = self.targets - self.positions
        d_error = (error - self.prev_error) / max(dt, 0.001)
        self.prev_error = error.copy()

        return error * self.config.target_gain_p + d_error * self.config.target_gain_d

    def _apply_velocity_limits(self, velocities, dt: float):
        """Enforce velocity and acceleration limits"""
        cfg = self.config

        # Acceleration limit
        vel_change = velocities - self.velocities
        accel = self.xp.linalg.norm(vel_change, axis=1, keepdims=True)
        max_change = cfg.max_acceleration * dt
        scale = self.xp.minimum(1.0, max_change / self.xp.maximum(accel, 0.001))
        velocities = self.velocities + vel_change * scale

        # XY velocity limit
        xy_speed = self.xp.linalg.norm(velocities[:, :2], axis=1, keepdims=True)
        xy_scale = self.xp.minimum(1.0, cfg.max_velocity_xy / self.xp.maximum(xy_speed, 0.001))
        velocities[:, :2] *= xy_scale

        # Z velocity limit
        velocities[:, 2] = self.xp.clip(velocities[:, 2], -cfg.max_velocity_z, cfg.max_velocity_z)

        return velocities

    # === HIGH LEVEL COMMANDS ===

    def arm_all(self):
        self.armed[:] = True

    def disarm_all(self):
        self.armed[:] = False

    def takeoff(self, altitude: float = None):
        alt = altitude or self.config.default_altitude
        self.arm_all()
        self.targets[:, 2] = alt
        print(f"[CONTROLLER] Takeoff to {alt}m")

    def land(self):
        self.targets[:, 2] = 0.0
        print("[CONTROLLER] Landing")

    def set_formation_circle(self, radius: float = 10.0, center: tuple = (0, 0)):
        angles = self.xp.linspace(0, 2 * self.xp.pi, self.num_drones, endpoint=False)
        self.targets[:, 0] = center[0] + radius * self.xp.cos(angles)
        self.targets[:, 1] = center[1] + radius * self.xp.sin(angles)

    def set_formation_grid(self, spacing: float = 3.0):
        cols = int(self.xp.ceil(self.xp.sqrt(self.num_drones)))
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            self.targets[i, 0] = col * spacing
            self.targets[i, 1] = row * spacing

    # === SENSOR DIAGNOSTICS ===

    def get_sensor_errors(self) -> dict:
        """Sensör tahmin hatalarını döndür"""
        if self.use_sensors:
            return self.sensors.get_estimation_error()
        return {'message': 'Sensör simülasyonu kapalı'}

    def get_raw_sensor_data(self) -> dict:
        """Ham sensör verilerini döndür"""
        if self.use_sensors:
            return self.sensors.get_raw_sensors()
        return {'message': 'Sensör simülasyonu kapalı'}

    def get_sensor_vs_truth(self, drone_id: int = 0) -> dict:
        """Belirli bir drone için sensör vs gerçek karşılaştırması"""
        if GPU_AVAILABLE:
            true_pos = cp.asnumpy(self.ground_truth_positions[drone_id])
            est_pos = cp.asnumpy(self.positions[drone_id])
        else:
            true_pos = self.ground_truth_positions[drone_id]
            est_pos = self.positions[drone_id]

        return {
            'drone_id': drone_id,
            'true_position': true_pos,
            'estimated_position': est_pos,
            'error': np.abs(true_pos - est_pos),
            'error_magnitude': np.linalg.norm(true_pos - est_pos)
        }


# ============================================================
# DEMO
# ============================================================

def demo_controller_with_sensors():
    """Sensörlü controller demo"""
    print("\n" + "="*60)
    print("CONTROLLER WITH SENSORS DEMO")
    print("="*60)

    # Sensörlü config
    config = SwarmConfigWithSensors(
        num_drones=10,
        use_realistic_sensors=True,
        sensor_config=SensorConfig(
            gps_horizontal_std=2.0,  # 2m GPS hatası
            gps_dropout_prob=0.05,   # %5 sinyal kaybı
        )
    )

    controller = GPUSwarmControllerWithSensors(config)

    # Başlangıç pozisyonları
    positions = np.zeros((10, 3), dtype=np.float32)
    for i in range(10):
        positions[i] = [i * 3, 0, 0.5]  # Grid düzeninde

    velocities = np.zeros((10, 3), dtype=np.float32)

    # Takeoff
    controller.update_ground_truth(positions, velocities)
    controller.takeoff(5.0)

    print("\n--- 100 adım simülasyon ---")
    dt = 0.02

    for step in range(100):
        # Simüle hareket
        cmd = controller.compute_control(dt)
        velocities = cmd * 0.5  # Basit dinamik
        positions += velocities * dt
        positions[:, 2] = np.maximum(positions[:, 2], 0.1)

        # Ground truth güncelle
        controller.update_ground_truth(positions, velocities)

        if step % 25 == 0:
            errors = controller.get_sensor_errors()
            comparison = controller.get_sensor_vs_truth(0)
            print(f"\nAdım {step}:")
            print(f"  Drone 0 gerçek poz: {comparison['true_position']}")
            print(f"  Drone 0 tahmin poz: {comparison['estimated_position']}")
            print(f"  Hata: {comparison['error_magnitude']:.3f} m")

    print("\n--- Final Sensör Hataları ---")
    errors = controller.get_sensor_errors()
    print(f"Position RMSE: {errors['position_rmse']}")
    print(f"Velocity RMSE: {errors['velocity_rmse']}")


if __name__ == "__main__":
    demo_controller_with_sensors()
>>>>>>> 08031cee7f16bb92e769bcc3e346d79078f6f8a2
