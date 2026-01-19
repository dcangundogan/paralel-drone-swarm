# GPU-Accelerated Drone Swarm Simulation

## Proje Dokümantasyonu ve Görev Dağılımı

---

# BÖLÜM 1: PROJENİN GENEL YAPISI

## 1.1 Proje Nedir?

Bu proje, **GPU üzerinde paralel çalışan bir drone sürüsü simülasyonudur**. 25-1000 arası drone'u gerçek zamanlı olarak simüle edebilir.

### Temel Özellikler:
- ✅ GPU paralel fizik hesaplamaları
- ✅ Gerçekçi sensör simülasyonu (GPS, IMU, Barometer)
- ✅ Collision avoidance (çarpışma önleme)
- ✅ Formasyon kontrolü (grid, circle, v-formation)
- ✅ Tıkla-git navigasyonu
- ✅ 2D görselleştirme + sensör paneli

### Mevcut Durum: **%80 Tamamlandı**

---

## 1.2 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DRONE SWARM SİMÜLASYONU                           │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        KULLANICI ARAYÜZÜ                             │   │
│  │                                                                      │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    │   │
│  │   │  Visualizer │    │   Sensor    │    │   Control Panel     │    │   │
│  │   │    (2D)     │    │   Panel     │    │   (Keyboard/Mouse)  │    │   │
│  │   └──────┬──────┘    └──────┬──────┘    └──────────┬──────────┘    │   │
│  │          │                  │                      │               │   │
│  └──────────┼──────────────────┼──────────────────────┼───────────────┘   │
│             │                  │                      │                    │
│             ▼                  ▼                      ▼                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      SIMULATION MANAGER                              │   │
│  │                   (Ana koordinasyon katmanı)                         │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                          │
│         ┌───────────────────────┼───────────────────────┐                  │
│         │                       │                       │                  │
│         ▼                       ▼                       ▼                  │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐            │
│  │   PHYSICS   │        │  SENSORS    │        │ CONTROLLER  │            │
│  │   ENGINE    │◄──────►│  SIMULATOR  │◄──────►│   (SWARM)   │            │
│  │             │        │             │        │             │            │
│  │ • Thrust    │        │ • GPS       │        │ • Collision │            │
│  │ • Gravity   │        │ • IMU       │        │ • Tracking  │            │
│  │ • Drag      │        │ • Baro      │        │ • Formation │            │
│  │ • Collision │        │ • Kalman    │        │ • Waypoint  │            │
│  └─────────────┘        └─────────────┘        └─────────────┘            │
│         │                       │                       │                  │
│         └───────────────────────┴───────────────────────┘                  │
│                                 │                                          │
│                                 ▼                                          │
│                    ┌───────────────────────┐                               │
│                    │      GPU (CUDA)       │                               │
│                    │  Paralel Hesaplama    │                               │
│                    └───────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Veri Akışı

```
Her frame (saniyede 50 kez):

    ┌──────────────────────────────────────────────────────────────────┐
    │ 1. PHYSICS ENGINE                                                │
    │    Input:  Velocity komutları                                    │
    │    Output: Gerçek pozisyon, hız, oryantasyon                    │
    │    GPU:    N drone paralel hesaplanır                           │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ 2. SENSOR SIMULATOR (opsiyonel)                                  │
    │    Input:  Gerçek pozisyon (ground truth)                       │
    │    Output: Gürültülü sensör ölçümleri                           │
    │    GPU:    N drone için paralel gürültü eklenir                 │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │ 3. SWARM CONTROLLER                                              │
    │    Input:  Pozisyonlar + Hedefler                               │
    │    Output: Velocity komutları                                    │
    │    GPU:    O(N²) collision check paralel                        │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 └──► Tekrar Physics Engine'e
```

---

# BÖLÜM 2: DOSYA YAPISI VE AÇIKLAMALARI

## 2.1 Mevcut Dosyalar

```
deneme7/
│
├── 📁 CORE (Çekirdek Sistem)
│   ├── gpu_swarm_simulation.py     # Ana simülasyon (Physics + Sensors + Controller)
│   ├── gpu_controller.py           # Eski controller (yedek)
│   └── gpu_sensors.py              # Sensör simülasyonu
│
├── 📁 PHYSICS (Fizik Motorları)
│   ├── lightweight_physics.py      # Hafif fizik (%60 gerçekçilik)
│   └── realistic_physics.py        # Gerçekçi fizik (%80 gerçekçilik)
│
├── 📁 VISUALIZATION (Görselleştirme)
│   ├── visualizer_2d.py            # Basit 2D görselleştirme
│   └── visualizer_with_sensors.py  # Sensör panelli görselleştirme
│
├── 📁 RUNNERS (Çalıştırıcılar)
│   ├── run_gpu_simulation.py       # Ana çalıştırıcı
│   ├── run_with_sensors.py         # Sensörlü çalıştırıcı
│   ├── run_large_swarm.py          # 100+ drone çalıştırıcı
│   └── swarm_manager.py            # Eski manager
│
├── 📁 MODELS (Gazebo Modelleri)
│   ├── model.sdf                   # Drone SDF modeli
│   └── swarm_world.sdf             # Dünya dosyası
│
├── 📁 SCRIPTS (Batch Dosyaları)
│   ├── test_25_drones.bat          # 25 drone testi
│   ├── run_100_drones.bat          # 100 drone testi
│   └── run_500_drones.bat          # 500 drone testi
│
└── 📁 DOCS (Dokümantasyon)
    ├── DOCUMENTATION.md            # İngilizce dokümantasyon
    └── PROJE_DOKUMANTASYONU.md     # Bu dosya
```

---

## 2.2 Her Dosyanın Detaylı Açıklaması

### gpu_swarm_simulation.py (ANA DOSYA)

**Ne yapar:** Tüm simülasyonu yönetir.

**İçindeki sınıflar:**

| Sınıf | Görevi | Satır Sayısı |
|-------|--------|--------------|
| `SimulationConfig` | Tüm ayarları tutar | ~50 |
| `GPUPhysicsEngine` | Fizik hesaplamaları | ~200 |
| `GPUSensorSimulator` | Sensör gürültüsü | ~100 |
| `GPUSwarmController` | Sürü kontrolü | ~150 |
| `GPUSwarmSimulation` | Hepsini birleştirir | ~100 |

**Örnek kullanım:**
```python
sim = GPUSwarmSimulation(num_drones=25, enable_sensors=True)
sim.reset()
sim.takeoff(5.0)

while running:
    sim.step()
    state = sim.get_state()
```

---

### visualizer_with_sensors.py

**Ne yapar:** 2D görselleştirme + sensör paneli gösterir.

**İçindeki sınıflar:**

| Sınıf | Görevi |
|-------|--------|
| `VisualizerConfig` | Pencere ayarları |
| `SensorPanelRenderer` | Sensör verilerini çizer |
| `AdvancedSwarmVisualizer` | Ana görselleştirici |

**Özellikler:**
- Haritada tıklayınca waypoint oluşturma
- Drone seçimi (1-9 tuşları)
- Sensör verileri paneli
- GPS durumu göstergesi
- Zoom/Pan kontrolü

---

# BÖLÜM 3: ALGORİTMALARIN AÇIKLAMASI

## 3.1 Collision Avoidance (Çarpışma Önleme)

```python
# Pseudocode açıklama:

for her_drone_i:
    for her_drone_j:
        if i != j:
            mesafe = ||pozisyon_i - pozisyon_j||

            if mesafe < tehlike_mesafesi:  # 2.5 metre
                yön = (pozisyon_i - pozisyon_j) / mesafe  # Uzaklaşma yönü

                if mesafe < kritik_mesafe:  # 0.8 metre
                    güç = MAKSIMUM  # Acil kaçış!
                else:
                    güç = (tehlike - mesafe) / (tehlike - kritik)  # Mesafeye göre

                kaçış_vektörü += yön * güç

# GPU'da bu O(N²) işlem paralel yapılır!
```

**Görsel açıklama:**
```
     Tehlike bölgesi (2.5m)
         ╱         ╲
        ╱           ╲
       ╱  Kritik     ╲
      ╱   (0.8m)      ╲
     ╱    ┌───┐        ╲
    │     │ ● │ Drone   │
     ╲    └───┘        ╱
      ╲              ╱
       ╲           ╱
        ╲         ╱

Başka drone bu alana girerse → Uzaklaşma kuvveti oluşur
```

---

## 3.2 PD Controller (Hedef Takibi)

```python
# Her frame:
hata = hedef_pozisyon - mevcut_pozisyon
hata_değişimi = (hata - önceki_hata) / dt

velocity_komutu = Kp * hata + Kd * hata_değişimi

# Kp = 1.5 (Proportional gain) → Hedefe ne kadar hızlı git
# Kd = 0.3 (Derivative gain)   → Salınımı azalt (damping)
```

**Görsel:**
```
        Hedef ●─────────────────────┐
              │                     │
              │    hata             │
              │◄──────────────►     │
              │                     │
        Drone ●                     │
              │                     │
              └─────────────────────┘

velocity = Kp × hata + Kd × (hata değişimi)
```

---

## 3.3 Motor Dinamiği

```python
# Motorlar anında tepki VERMEZ!
# First-order system ile modellenir:

tau = 0.05  # 50ms motor tepki süresi

# Her adımda:
alpha = dt / (tau + dt)
gerçek_thrust = (1 - alpha) * gerçek_thrust + alpha * hedef_thrust

# Bu, motor komutunun yavaşça hedefe ulaşmasını sağlar
```

**Grafik:**
```
Thrust
  │
  │         ┌─────────────────── Hedef
  │        ╱
  │       ╱
  │      ╱
  │     ╱   ← Gerçek (yavaşça yaklaşır)
  │    ╱
  │   ╱
  │──╱
  └─────────────────────────────► Zaman
     0    50ms   100ms
```

---

## 3.4 Sensör Simülasyonu

### GPS Simülasyonu
```python
# Gerçek pozisyon: [10.0, 5.0, 8.0]

gürültü = random.normal(0, 1.5)  # ±1.5 metre std sapma
gps_ölçümü = gerçek_pozisyon + gürültü

# Sonuç: [10.3, 4.7, 8.2] (her seferinde farklı!)

# Bazen sinyal kaybı:
if random() < 0.02:  # %2 şans
    gps_valid = False
```

### Kalman Filter (Basitleştirilmiş)
```python
# Tahmin = GPS + IMU füzyonu

if gps_valid:
    # GPS güvenilir, ona ağırlık ver
    tahmin = 0.7 * gps_ölçümü + 0.3 * önceki_tahmin
else:
    # GPS yok, sadece IMU ile devam et (dead reckoning)
    tahmin = önceki_tahmin + hız * dt
```

---

# BÖLÜM 4: MODÜLER YAPI VE GÖREV DAĞILIMI

## 4.1 Modül Haritası

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   MODÜL A: FİZİK MOTORU                    Sorumlu: ALP EREN               │
│   ─────────────────────                                                     │
│   • Motor dinamiği                                                          │
│   • Aerodinamik (thrust, drag)                                             │
│   • Ground effect                                                           │
│   • Prop wash etkileşimi                                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MODÜL B: SENSÖR ve İLETİŞİM              Sorumlu: CEM                    │
│   ───────────────────────────                                               │
│   • GPS, IMU, Barometer simülasyonu                                        │
│   • Kalman Filter (sensör füzyonu)                                         │
│   • İletişim gecikmesi simülasyonu                                         │
│   • Paket kaybı simülasyonu                                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MODÜL C: GÖRSELLEŞTİRME ve UI            Sorumlu: ALİ                    │
│   ─────────────────────────────                                             │
│   • 2D Visualizer iyileştirmeleri                                          │
│   • 3D Visualizer (opsiyonel)                                              │
│   • Kontrol paneli                                                          │
│   • Performans grafikleri                                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MODÜL D: KONTROL ve MİSYON               Sorumlu: CAN (Takım Lideri)     │
│   ──────────────────────────                                                │
│   • Swarm controller iyileştirmeleri                                       │
│   • Mission planning sistemi                                                │
│   • Entegrasyon ve test                                                     │
│   • Dokümantasyon                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Detaylı Görev Listesi

### 🔵 ALP EREN - Fizik Motoru (%80 → %100)

**Mevcut Durum:** Basit fizik var, bazı efektler eksik.

| # | Görev | Öncelik | Zorluk | Süre |
|---|-------|---------|--------|------|
| A1 | Blade Element Theory implementasyonu | Yüksek | Zor | 3 gün |
| A2 | Ground Effect iyileştirmesi | Orta | Orta | 2 gün |
| A3 | Prop Wash etkileşimi (drone'lar arası) | Yüksek | Zor | 3 gün |
| A4 | Rüzgar modeli (Dryden turbulence) | Orta | Orta | 2 gün |
| A5 | Batarya simülasyonu | Düşük | Kolay | 1 gün |
| A6 | Motor arızası simülasyonu | Düşük | Kolay | 1 gün |

**Detaylı açıklamalar:**

**A1 - Blade Element Theory:**
```python
# Mevcut (basit):
thrust = motor_command * max_thrust

# Hedef (gerçekçi):
# Pervane blade'lerini küçük parçalara böl
# Her parça için: lift, drag hesapla
# Toplam thrust = sum(blade_forces)

def blade_element_thrust(rpm, velocity, air_density):
    # Her blade elementi için:
    for r in blade_radius_segments:
        local_velocity = compute_local_velocity(r, rpm, velocity)
        angle_of_attack = compute_aoa(blade_pitch, local_velocity)
        lift = 0.5 * air_density * local_velocity**2 * Cl(aoa) * chord * dr
        drag = 0.5 * air_density * local_velocity**2 * Cd(aoa) * chord * dr
    return total_thrust, total_torque
```

**A3 - Prop Wash:**
```
Drone A (üstte)
    ↓↓↓↓↓
    ↓↓↓↓↓  ← Downwash (aşağı rüzgar)
═══════════
    ↓↓↓↓↓
Drone B    ← Thrust kaybı yaşar!

# Hesaplama:
downwash_velocity = thrust_A / (2 * air_density * rotor_area)
thrust_loss_B = f(vertical_distance, horizontal_offset)
```

---

### 🟢 CEM - Sensör ve İletişim (%80 → %100)

**Mevcut Durum:** Basit GPS/IMU gürültüsü var, Kalman filter basit.

| # | Görev | Öncelik | Zorluk | Süre |
|---|-------|---------|--------|------|
| B1 | Gerçek Kalman Filter implementasyonu | Yüksek | Zor | 4 gün |
| B2 | IMU bias ve drift modeli | Yüksek | Orta | 2 gün |
| B3 | Magnetometer simülasyonu | Orta | Kolay | 1 gün |
| B4 | İletişim gecikmesi simülasyonu | Yüksek | Orta | 2 gün |
| B5 | Paket kaybı simülasyonu | Orta | Kolay | 1 gün |
| B6 | Sensör füzyon görselleştirmesi | Düşük | Kolay | 1 gün |

**Detaylı açıklamalar:**

**B1 - Kalman Filter:**
```python
# Mevcut (basit):
estimate = 0.7 * gps + 0.3 * previous

# Hedef (gerçek Extended Kalman Filter):
class ExtendedKalmanFilter:
    def __init__(self):
        self.state = [x, y, z, vx, vy, vz]  # 6 state
        self.P = covariance_matrix  # 6x6

    def predict(self, dt):
        # State transition
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

    def update_gps(self, gps_measurement):
        # Measurement update
        H = measurement_matrix
        K = self.P @ H.T @ inv(H @ self.P @ H.T + R)
        self.state = self.state + K @ (gps - H @ self.state)
        self.P = (I - K @ H) @ self.P
```

**B4 - İletişim Gecikmesi:**
```python
# Gerçek sistemlerde komutlar anında ulaşmaz!

class CommunicationSimulator:
    def __init__(self):
        self.latency_mean = 0.05      # 50ms ortalama gecikme
        self.latency_std = 0.02       # ±20ms değişkenlik
        self.packet_loss_rate = 0.01  # %1 paket kaybı

    def send_command(self, command):
        if random() < self.packet_loss_rate:
            return None  # Paket kayboldu!

        delay = random.normal(self.latency_mean, self.latency_std)
        schedule_delivery(command, delay)
```

---

### 🟡 ALİ - Görselleştirme ve UI (%80 → %100)

**Mevcut Durum:** 2D visualizer var, sensör paneli var.

| # | Görev | Öncelik | Zorluk | Süre |
|---|-------|---------|--------|------|
| C1 | Performans grafikleri (FPS, CPU, GPU) | Yüksek | Orta | 2 gün |
| C2 | Drone bilgi popup'ı (hover ile) | Orta | Kolay | 1 gün |
| C3 | Formasyon önizlemesi | Orta | Orta | 2 gün |
| C4 | Kayıt/Replay sistemi | Yüksek | Zor | 3 gün |
| C5 | 3D Visualizer (PyOpenGL) | Düşük | Çok Zor | 5 gün |
| C6 | Ayarlar paneli (runtime config) | Orta | Orta | 2 gün |

**Detaylı açıklamalar:**

**C1 - Performans Grafikleri:**
```python
# Ekranda gösterilecek:
┌──────────────────────────────────┐
│ FPS: 58  │ CPU: 23%  │ GPU: 45% │
├──────────────────────────────────┤
│ Physics: 2.1ms │ Control: 1.3ms │
│ Render:  5.2ms │ Total:   8.6ms │
└──────────────────────────────────┘

# Mini grafik (son 60 frame):
FPS ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█
```

**C4 - Kayıt/Replay:**
```python
class SimulationRecorder:
    def __init__(self):
        self.frames = []

    def record_frame(self, state):
        self.frames.append({
            'time': state['time'],
            'positions': state['positions'].copy(),
            'velocities': state['velocities'].copy(),
            'targets': state['targets'].copy(),
        })

    def save(self, filename):
        np.savez_compressed(filename, frames=self.frames)

    def load(self, filename):
        self.frames = np.load(filename)['frames']

    def replay(self, frame_index):
        return self.frames[frame_index]
```

---

### 🔴 CAN (Takım Lideri) - Kontrol ve Misyon (%80 → %100)

**Mevcut Durum:** Temel kontrol var, mission planning yok.

| # | Görev | Öncelik | Zorluk | Süre |
|---|-------|---------|--------|------|
| D1 | 3D Collision Avoidance | Yüksek | Orta | 2 gün |
| D2 | Predictive collision (tahminli) | Yüksek | Zor | 3 gün |
| D3 | Mission Planning sistemi | Yüksek | Zor | 4 gün |
| D4 | Waypoint queue sistemi | Orta | Orta | 2 gün |
| D5 | Entegrasyon testleri | Yüksek | Orta | 2 gün |
| D6 | Final dokümantasyon | Yüksek | Kolay | 2 gün |

**Detaylı açıklamalar:**

**D2 - Predictive Collision:**
```python
# Mevcut: Sadece şu anki pozisyonlara bakıyor
# Hedef: Gelecekteki pozisyonları tahmin et

def predict_collision(pos_i, vel_i, pos_j, vel_j, lookahead=2.0):
    """
    2 saniye sonraki pozisyonları tahmin et
    ve çarpışma olup olmayacağını kontrol et
    """
    for t in np.linspace(0, lookahead, 20):
        future_pos_i = pos_i + vel_i * t
        future_pos_j = pos_j + vel_j * t

        if distance(future_pos_i, future_pos_j) < collision_radius:
            return True, t  # Çarpışma olacak!

    return False, None
```

**D3 - Mission Planning:**
```python
class MissionPlanner:
    """
    Çoklu waypoint ve görev yönetimi
    """
    def __init__(self):
        self.missions = {}  # drone_id -> mission

    def add_mission(self, drone_id, waypoints, actions):
        """
        Örnek:
        waypoints = [(0,0,5), (10,0,5), (10,10,5), (0,10,5)]
        actions = ['takeoff', 'photo', 'photo', 'land']
        """
        self.missions[drone_id] = {
            'waypoints': waypoints,
            'actions': actions,
            'current_index': 0,
            'status': 'pending'
        }

    def update(self, drone_id, current_position):
        mission = self.missions[drone_id]
        target = mission['waypoints'][mission['current_index']]

        if distance(current_position, target) < 0.5:
            # Waypoint'e ulaştı
            action = mission['actions'][mission['current_index']]
            self.execute_action(drone_id, action)
            mission['current_index'] += 1
```

---

# BÖLÜM 5: ZAMAN ÇİZELGESİ

## 5.1 Sprint Planı (4 Hafta)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HAFTA 1                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ALP EREN  │ A1: Blade Element Theory başla                                 │
│ CEM       │ B1: Kalman Filter araştırma + başla                            │
│ ALİ       │ C1: Performans grafikleri                                       │
│ CAN       │ D1: 3D Collision Avoidance                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                           HAFTA 2                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ALP EREN  │ A1: Tamamla, A3: Prop Wash başla                               │
│ CEM       │ B1: Tamamla, B2: IMU bias modeli                               │
│ ALİ       │ C2: Drone popup, C3: Formasyon önizleme                        │
│ CAN       │ D2: Predictive collision                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                           HAFTA 3                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ALP EREN  │ A3: Tamamla, A2: Ground effect                                 │
│ CEM       │ B4: İletişim gecikmesi, B5: Paket kaybı                        │
│ ALİ       │ C4: Kayıt/Replay sistemi                                        │
│ CAN       │ D3: Mission Planning sistemi                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                           HAFTA 4                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ALP EREN  │ A4: Rüzgar modeli, A5: Batarya                                 │
│ CEM       │ B3: Magnetometer, B6: Görselleştirme                           │
│ ALİ       │ C6: Ayarlar paneli                                              │
│ CAN       │ D4: Waypoint queue, D5: Testler, D6: Dokümantasyon             │
│           │ ENTEGRASYON ve FINAL TEST                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Milestone'lar

| Hafta | Hedef | Tamamlanma % |
|-------|-------|--------------|
| Başlangıç | Mevcut durum | 80% |
| Hafta 1 sonu | Temel iyileştirmeler | 85% |
| Hafta 2 sonu | Orta seviye özellikler | 90% |
| Hafta 3 sonu | İleri özellikler | 95% |
| Hafta 4 sonu | Final + Dokümantasyon | 100% |

---

# BÖLÜM 6: KOD STANDARTLARI

## 6.1 Dosya Yapısı

Her yeni dosya şu şablonu takip etmeli:

```python
#!/usr/bin/env python3
"""
Dosya Adı: module_name.py
Açıklama: Bu modül ne yapar
Yazar: İsim
Tarih: 2024-XX-XX
"""

import numpy as np
# ... diğer importlar

# GPU kontrolü
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp
except ImportError:
    GPU_AVAILABLE = False
    xp = np


class MyClass:
    """
    Sınıf açıklaması.

    Attributes:
        attr1: Açıklama
        attr2: Açıklama

    Example:
        >>> obj = MyClass()
        >>> obj.method()
    """

    def __init__(self, param1: int, param2: float = 1.0):
        """
        Args:
            param1: Parametre açıklaması
            param2: Parametre açıklaması (varsayılan: 1.0)
        """
        self.param1 = param1
        self.param2 = param2

    def method(self, arg: np.ndarray) -> np.ndarray:
        """
        Metod açıklaması.

        Args:
            arg: Girdi açıklaması

        Returns:
            Çıktı açıklaması
        """
        return arg * 2
```

---

## 6.2 Git Commit Mesajları

```
Format: [MODÜL] Kısa açıklama

Örnekler:
[PHYSICS] Blade element theory eklendi
[SENSORS] Kalman filter güncellendi
[VIZ] FPS grafiği eklendi
[CONTROL] Mission planner başlatıldı
[DOCS] README güncellendi
[FIX] Collision detection bug düzeltildi
[TEST] Unit testler eklendi
```

---

## 6.3 Branch Yapısı

```
main
  │
  ├── develop
  │     │
  │     ├── feature/physics-blade-element    (Alp Eren)
  │     ├── feature/sensors-kalman           (Cem)
  │     ├── feature/viz-performance          (Ali)
  │     └── feature/control-mission          (Can)
  │
  └── release/v1.0
```

---

# BÖLÜM 7: TEST VE KALİTE

## 7.1 Test Senaryoları

| # | Senaryo | Beklenen Sonuç |
|---|---------|----------------|
| T1 | 25 drone takeoff | Tümü 5m'ye yükselir |
| T2 | Circle formation | Daire oluşur, çarpışma yok |
| T3 | Waypoint click | Tüm drone'lar hedefe gider |
| T4 | GPS dropout | Drone yoluna devam eder (dead reckoning) |
| T5 | 100 drone stress test | 30+ FPS korunur |
| T6 | Collision test | İki drone yaklaşınca kaçınır |

## 7.2 Performans Hedefleri

| Metrik | Hedef |
|--------|-------|
| 25 drone FPS | > 60 |
| 100 drone FPS | > 30 |
| Physics step | < 5ms |
| GPU memory | < 500MB |

---

# BÖLÜM 8: İLETİŞİM

## 8.1 Haftalık Toplantılar

- **Gün:** Her Pazartesi
- **Format:** 15 dakika stand-up
- **Gündem:**
  1. Geçen hafta ne yapıldı?
  2. Bu hafta ne yapılacak?
  3. Engeller var mı?

## 8.2 Kod Review

- Her PR en az 1 kişi tarafından review edilmeli
- Takım lideri (Can) final approval verir

---

# BÖLÜM 9: KAYNAKLAR

## 9.1 Faydalı Linkler

- CuPy Dokümantasyonu: https://docs.cupy.dev/
- Pygame Dokümantasyonu: https://www.pygame.org/docs/
- Kalman Filter Tutorial: https://www.kalmanfilter.net/
- Quadcopter Dynamics: https://scholarsarchive.byu.edu/facpub/2324/

## 9.2 Referans Makaleler

1. Reynolds, C. (1987). "Flocks, Herds, and Schools: A Distributed Behavioral Model"
2. Beard & McLain. "Small Unmanned Aircraft: Theory and Practice"

---

# SON SÖZ

Bu proje %30 tamamlanmış durumda. Yukarıdaki görevler tamamlandığında %100 olacak.

**Her takım üyesinin sorumluluğu:**
- Kendi modülünü zamanında tamamlamak
- Kod standartlarına uymak
- Haftalık toplantılara katılmak
- Sorun olduğunda hemen bildirmek

**Takım Lideri (Can) sorumluluğu:**
- Entegrasyonu yönetmek
- Engelleri kaldırmak
- Final testi koordine etmek


