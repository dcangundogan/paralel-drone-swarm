# ALP EREN - Fizik Motoru Görevleri

## Modül: PHYSICS ENGINE

**Sorumluluk Alanı:** Drone fizik simülasyonu

**Çalışılacak Dosyalar:**
- `lightweight_physics.py`
- `realistic_physics.py`
- `gpu_swarm_simulation.py` (GPUPhysicsEngine sınıfı)

---

## Görev Listesi

### A1: Blade Element Theory [YÜKSEK ÖNCELİK]
**Süre:** 3 gün
**Zorluk:** ⭐⭐⭐

**Açıklama:**
Pervane thrust hesabını daha gerçekçi yapmak için Blade Element Theory kullan.

**Mevcut kod:**
```python
# lightweight_physics.py, satır ~200
thrust = motor_command * max_thrust  # Çok basit!
```

**Hedef kod:**
```python
def compute_blade_element_thrust(self, rpm, velocity, air_density):
    """
    Blade Element Theory ile thrust hesapla.

    Pervaneyi küçük parçalara (element) böl.
    Her element için lift ve drag hesapla.
    Toplamı al.
    """
    D = self.prop_diameter
    R = D / 2
    num_elements = 10
    dr = R / num_elements

    total_thrust = 0
    total_torque = 0

    for i in range(num_elements):
        r = (i + 0.5) * dr  # Element merkezi

        # Local velocity at this element
        omega = rpm * 2 * np.pi / 60  # rad/s
        v_tangential = omega * r
        v_axial = velocity[2]  # Dikey hız

        v_local = np.sqrt(v_tangential**2 + v_axial**2)

        # Angle of attack
        phi = np.arctan2(v_axial, v_tangential)
        alpha = self.blade_pitch - phi

        # Lift and drag coefficients (basitleştirilmiş)
        Cl = 2 * np.pi * alpha  # Thin airfoil theory
        Cd = 0.01 + 0.05 * alpha**2

        # Forces
        chord = 0.02  # Blade genişliği
        dL = 0.5 * air_density * v_local**2 * Cl * chord * dr
        dD = 0.5 * air_density * v_local**2 * Cd * chord * dr

        # Thrust and torque contributions
        dT = dL * np.cos(phi) - dD * np.sin(phi)
        dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * r

        total_thrust += dT
        total_torque += dQ

    # 2 blade per propeller
    return total_thrust * 2, total_torque * 2
```

**Test:**
- Hover'da thrust ≈ weight/4 per motor olmalı
- RPM arttıkça thrust artmalı (karesel)

---

### A2: Ground Effect [ORTA ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Açıklama:**
Drone zemine yakınken ekstra lift üretir. Bunu simüle et.

**Formül:**
```
thrust_multiplier = 1 / (1 - (R/(4*h))^2)

R = rotor radius
h = height above ground
```

**Kod:**
```python
def apply_ground_effect(self, thrust, height):
    """
    Zemine yakınken thrust artar.
    """
    R = self.prop_diameter / 2

    if height < 2.0:  # Sadece 2m altında etkili
        # Bölme hatası önle
        h = max(height, 0.1)

        multiplier = 1.0 / (1.0 - (R / (4 * h))**2)
        multiplier = min(multiplier, 1.5)  # Max %50 artış

        return thrust * multiplier

    return thrust
```

---

### A3: Prop Wash Etkileşimi [YÜKSEK ÖNCELİK]
**Süre:** 3 gün
**Zorluk:** ⭐⭐⭐

**Açıklama:**
Bir drone başka bir drone'un üstündeyken, alttaki thrust kaybeder.

**Görsel:**
```
Drone A (üstte)
    ↓↓↓↓↓  Downwash
    ↓↓↓↓↓
═══════════
    ↓↓↓↓↓
Drone B (altta) → %30 thrust kaybı!
```

**Kod:**
```python
def compute_prop_wash_interference(self, positions, thrusts):
    """
    Prop wash etkileşimini hesapla.

    Returns:
        thrust_loss: Her drone için thrust kaybı faktörü (0-0.5)
    """
    N = len(positions)
    thrust_loss = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            # i, j'nin altında mı?
            dz = positions[i, 2] - positions[j, 2]

            if dz < 0:  # i, j'nin altında
                horizontal_dist = np.sqrt(
                    (positions[i, 0] - positions[j, 0])**2 +
                    (positions[i, 1] - positions[j, 1])**2
                )

                # Downwash cone içinde mi?
                cone_radius = abs(dz) * 0.5  # 30 derece cone

                if horizontal_dist < cone_radius + self.prop_diameter:
                    # Thrust kaybı hesapla
                    vertical_factor = np.exp(dz / 2.0)  # dz negatif
                    horizontal_factor = 1 - horizontal_dist / (cone_radius + self.prop_diameter)

                    loss = 0.3 * vertical_factor * horizontal_factor * thrusts[j] / thrusts[i]
                    thrust_loss[i] += loss

    return np.clip(thrust_loss, 0, 0.5)
```

---

### A4: Rüzgar Modeli [ORTA ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Dryden Turbulence modeli kullan.**

```python
def generate_wind(self, dt):
    """
    Dryden türbülans modeli ile rüzgar üret.
    """
    # Base wind (sabit)
    base_wind = np.array([self.wind_speed, 0, 0])

    # Turbulence (rastgele)
    # Low-pass filtered white noise
    alpha = dt / (self.turbulence_time_constant + dt)

    noise = np.random.normal(0, self.turbulence_intensity, 3)
    self.turbulence_state = (1 - alpha) * self.turbulence_state + alpha * noise

    return base_wind + self.turbulence_state
```

---

### A5: Batarya Simülasyonu [DÜŞÜK ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

```python
def update_battery(self, current_draw, dt):
    """
    Batarya durumunu güncelle.

    Voltage sag: Yük altında voltaj düşer
    """
    # Kapasite azalt
    self.battery_mah -= current_draw * dt * 1000 / 3600

    # State of charge
    self.soc = self.battery_mah / self.battery_capacity

    # Voltage = Open circuit voltage - I*R
    ocv = self.voltage_full * self.soc + self.voltage_empty * (1 - self.soc)
    self.voltage = ocv - current_draw * self.internal_resistance

    # Motor max RPM voltaja bağlı
    self.max_rpm = self.motor_kv * self.voltage
```

---

### A6: Motor Arızası [DÜŞÜK ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

```python
def simulate_motor_failure(self, motor_id, failure_type='complete'):
    """
    Motor arızası simüle et.

    failure_type: 'complete' (tamamen durur) veya 'partial' (%50 güç)
    """
    if failure_type == 'complete':
        self.motor_efficiency[motor_id] = 0.0
    elif failure_type == 'partial':
        self.motor_efficiency[motor_id] = 0.5
```

---

## Test Prosedürü

Her görev tamamlandığında:

1. **Unit test yaz:**
```python
def test_blade_element():
    physics = LightweightPhysicsEngine(1)
    physics.motor_rpm[0] = 5000
    thrust = physics.compute_blade_element_thrust(5000, [0,0,0], 1.225)
    assert 2.0 < thrust < 6.0, "Hover thrust aralık dışı"
```

2. **Benchmark çalıştır:**
```bash
python lightweight_physics.py
# 100 drone < 5ms olmalı
```

3. **PR oluştur:**
```
Branch: feature/physics-blade-element
Commit: [PHYSICS] Blade element theory implemented
```

---

## Sorular?

Can'a (takım liderine) sor.
