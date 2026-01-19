# ALİ - Görselleştirme ve UI Görevleri

## Modül: VISUALIZATION & USER INTERFACE

**Sorumluluk Alanı:** Kullanıcı arayüzü ve görselleştirme

**Çalışılacak Dosyalar:**
- `visualizer_with_sensors.py`
- `visualizer_2d.py`
- Yeni: `performance_monitor.py`
- Yeni: `replay_system.py`

---

## Görev Listesi

### C1: Performans Grafikleri [YÜKSEK ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Açıklama:**
Ekranda FPS, CPU, GPU kullanımı ve timing bilgilerini göster.

**Hedef görünüm:**
```
┌──────────────────────────────────────────────────┐
│ FPS: 58 │ CPU: 23% │ GPU: 45% │ RAM: 1.2GB      │
├──────────────────────────────────────────────────┤
│ Timing (ms):                                     │
│   Physics:  2.1 ████████                         │
│   Control:  1.3 █████                            │
│   Sensors:  0.8 ███                              │
│   Render:   5.2 ████████████████████             │
│   Total:    9.4                                  │
├──────────────────────────────────────────────────┤
│ FPS History (60 frames):                         │
│ ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃▄▅▆▇█▇▆▅▄▃▂▁    │
└──────────────────────────────────────────────────┘
```

**Kod:**
```python
import psutil
import time
from collections import deque

class PerformanceMonitor:
    """Performans izleme ve görselleştirme."""

    def __init__(self, history_size: int = 60):
        self.history_size = history_size

        # Timing history
        self.fps_history = deque(maxlen=history_size)
        self.physics_time = deque(maxlen=history_size)
        self.control_time = deque(maxlen=history_size)
        self.render_time = deque(maxlen=history_size)

        # Current frame timing
        self.frame_start = 0
        self.section_start = 0
        self.current_times = {}

        # System stats
        self.cpu_percent = 0
        self.ram_used = 0
        self.gpu_percent = 0
        self.gpu_memory = 0

        self.last_system_update = 0

    def start_frame(self):
        """Frame başlangıcını işaretle."""
        self.frame_start = time.perf_counter()
        self.current_times = {}

    def start_section(self, name: str):
        """Bir bölümün başlangıcını işaretle."""
        self.section_start = time.perf_counter()

    def end_section(self, name: str):
        """Bir bölümün bitişini işaretle."""
        elapsed = (time.perf_counter() - self.section_start) * 1000  # ms
        self.current_times[name] = elapsed

    def end_frame(self):
        """Frame bitişini işaretle ve FPS hesapla."""
        frame_time = (time.perf_counter() - self.frame_start) * 1000
        fps = 1000 / max(frame_time, 0.1)

        self.fps_history.append(fps)

        if 'physics' in self.current_times:
            self.physics_time.append(self.current_times['physics'])
        if 'control' in self.current_times:
            self.control_time.append(self.current_times['control'])
        if 'render' in self.current_times:
            self.render_time.append(self.current_times['render'])

    def update_system_stats(self):
        """Sistem istatistiklerini güncelle (1 saniyede bir)."""
        now = time.time()
        if now - self.last_system_update < 1.0:
            return

        self.last_system_update = now

        # CPU
        self.cpu_percent = psutil.cpu_percent()

        # RAM
        mem = psutil.virtual_memory()
        self.ram_used = mem.used / (1024**3)  # GB

        # GPU (cupy varsa)
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            self.gpu_memory = mempool.used_bytes() / (1024**2)  # MB

            # GPU utilization (nvidia-smi gerektirir)
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            self.gpu_percent = int(result.stdout.strip())
        except:
            self.gpu_percent = 0
            self.gpu_memory = 0

    def get_stats(self) -> dict:
        """İstatistikleri döndür."""
        return {
            'fps': self.fps_history[-1] if self.fps_history else 0,
            'fps_avg': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
            'fps_history': list(self.fps_history),
            'physics_ms': self.physics_time[-1] if self.physics_time else 0,
            'control_ms': self.control_time[-1] if self.control_time else 0,
            'render_ms': self.render_time[-1] if self.render_time else 0,
            'cpu_percent': self.cpu_percent,
            'ram_gb': self.ram_used,
            'gpu_percent': self.gpu_percent,
            'gpu_mb': self.gpu_memory,
        }

    def draw(self, surface, x: int, y: int, font):
        """Pygame surface'e çiz."""
        stats = self.get_stats()

        # Background
        panel_rect = pygame.Rect(x, y, 300, 150)
        pygame.draw.rect(surface, (30, 30, 40), panel_rect)
        pygame.draw.rect(surface, (60, 60, 70), panel_rect, 1)

        # FPS line
        fps_color = (100, 255, 100) if stats['fps'] >= 30 else (255, 100, 100)
        text = font.render(
            f"FPS: {stats['fps']:.0f} | CPU: {stats['cpu_percent']:.0f}% | GPU: {stats['gpu_percent']}%",
            True, fps_color
        )
        surface.blit(text, (x + 10, y + 10))

        # Timing bars
        y_offset = 35
        max_width = 200
        max_time = 20  # 20ms = full bar

        for name, time_ms, color in [
            ('Physics', stats['physics_ms'], (100, 150, 255)),
            ('Control', stats['control_ms'], (150, 255, 100)),
            ('Render', stats['render_ms'], (255, 200, 100)),
        ]:
            bar_width = min(max_width, int(time_ms / max_time * max_width))

            # Label
            label = font.render(f"{name}: {time_ms:.1f}ms", True, (200, 200, 200))
            surface.blit(label, (x + 10, y + y_offset))

            # Bar
            pygame.draw.rect(surface, color, (x + 120, y + y_offset + 2, bar_width, 12))
            pygame.draw.rect(surface, (80, 80, 80), (x + 120, y + y_offset + 2, max_width, 12), 1)

            y_offset += 20

        # FPS mini graph
        y_offset += 10
        graph_text = font.render("FPS History:", True, (150, 150, 150))
        surface.blit(graph_text, (x + 10, y + y_offset))

        y_offset += 18
        if stats['fps_history']:
            graph_height = 30
            graph_width = 280

            # Normalize FPS values
            fps_values = stats['fps_history']
            max_fps = max(fps_values) if fps_values else 60
            min_fps = min(fps_values) if fps_values else 0

            points = []
            for i, fps in enumerate(fps_values):
                px = x + 10 + int(i / len(fps_values) * graph_width)
                normalized = (fps - min_fps) / max(max_fps - min_fps, 1)
                py = y + y_offset + graph_height - int(normalized * graph_height)
                points.append((px, py))

            if len(points) > 1:
                pygame.draw.lines(surface, (100, 200, 100), False, points, 2)
```

---

### C2: Drone Bilgi Popup [ORTA ÖNCELİK]
**Süre:** 1 gün
**Zorluk:** ⭐

**Açıklama:**
Mouse bir drone'un üzerine geldiğinde detaylı bilgi göster.

```python
def draw_drone_tooltip(self, surface, drone_id: int, mouse_pos: tuple):
    """Drone üzerinde hover olunca bilgi göster."""

    drone = self.get_drone_data(drone_id)

    # Tooltip box
    box_width = 180
    box_height = 120
    x = mouse_pos[0] + 15
    y = mouse_pos[1]

    # Ekrandan taşmasın
    if x + box_width > self.screen_width:
        x = mouse_pos[0] - box_width - 15
    if y + box_height > self.screen_height:
        y = self.screen_height - box_height

    # Background
    pygame.draw.rect(surface, (40, 40, 50, 230), (x, y, box_width, box_height))
    pygame.draw.rect(surface, (100, 100, 120), (x, y, box_width, box_height), 1)

    # Content
    lines = [
        f"Drone #{drone_id}",
        f"─────────────",
        f"Pos: ({drone['x']:.1f}, {drone['y']:.1f}, {drone['z']:.1f})",
        f"Vel: {drone['speed']:.2f} m/s",
        f"Alt: {drone['altitude']:.1f} m",
        f"Bat: {drone['battery']:.0f}%",
        f"GPS: {'OK' if drone['gps_valid'] else 'LOST'}",
    ]

    for i, line in enumerate(lines):
        color = (255, 255, 100) if i == 0 else (200, 200, 200)
        text = self.font_small.render(line, True, color)
        surface.blit(text, (x + 10, y + 8 + i * 15))
```

---

### C3: Formasyon Önizlemesi [ORTA ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

**Açıklama:**
Formasyon değiştirmeden önce önizleme göster.

```python
class FormationPreview:
    """Formasyon önizlemesi."""

    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.preview_positions = None
        self.preview_type = None
        self.visible = False

    def show_grid_preview(self, center: tuple, spacing: float, altitude: float):
        """Grid formasyonu önizlemesi."""
        self.preview_type = 'grid'
        self.preview_positions = self._calculate_grid(center, spacing, altitude)
        self.visible = True

    def show_circle_preview(self, center: tuple, radius: float, altitude: float):
        """Circle formasyonu önizlemesi."""
        self.preview_type = 'circle'
        self.preview_positions = self._calculate_circle(center, radius, altitude)
        self.visible = True

    def _calculate_grid(self, center, spacing, altitude):
        cols = int(np.ceil(np.sqrt(self.num_drones)))
        positions = []
        for i in range(self.num_drones):
            row, col = divmod(i, cols)
            x = center[0] + (col - cols/2) * spacing
            y = center[1] + (row - cols/2) * spacing
            positions.append([x, y, altitude])
        return np.array(positions)

    def _calculate_circle(self, center, radius, altitude):
        angles = np.linspace(0, 2*np.pi, self.num_drones, endpoint=False)
        positions = []
        for angle in angles:
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            positions.append([x, y, altitude])
        return np.array(positions)

    def draw(self, surface, world_to_screen_func):
        """Önizlemeyi çiz (yarı saydam)."""
        if not self.visible or self.preview_positions is None:
            return

        for pos in self.preview_positions:
            screen_pos = world_to_screen_func(pos)

            # Yarı saydam daire
            s = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 200, 255, 100), (15, 15), 12)
            surface.blit(s, (screen_pos[0] - 15, screen_pos[1] - 15))

            # Çerçeve
            pygame.draw.circle(surface, (100, 200, 255), screen_pos, 12, 2)

    def hide(self):
        self.visible = False

    def apply(self):
        """Önizlemeyi uygula ve gizle."""
        positions = self.preview_positions
        self.hide()
        return positions
```

---

### C4: Kayıt/Replay Sistemi [YÜKSEK ÖNCELİK]
**Süre:** 3 gün
**Zorluk:** ⭐⭐⭐

**Yeni dosya: `replay_system.py`**

```python
"""
Simulation Recording and Replay System
======================================

Simülasyonu kaydet ve tekrar oynat.
"""

import numpy as np
import json
import gzip
from dataclasses import dataclass, asdict
from typing import List, Optional
import time


@dataclass
class FrameData:
    """Bir frame'in verisi."""
    time: float
    positions: np.ndarray
    velocities: np.ndarray
    targets: np.ndarray
    orientations: np.ndarray
    armed: np.ndarray
    sensor_data: Optional[dict] = None


class SimulationRecorder:
    """Simülasyon kaydedici."""

    def __init__(self, num_drones: int):
        self.num_drones = num_drones
        self.frames: List[FrameData] = []
        self.recording = False
        self.start_time = 0
        self.metadata = {}

    def start_recording(self, metadata: dict = None):
        """Kayda başla."""
        self.frames = []
        self.recording = True
        self.start_time = time.time()
        self.metadata = metadata or {}
        self.metadata['start_time'] = self.start_time
        self.metadata['num_drones'] = self.num_drones
        print("[RECORDER] Recording started")

    def stop_recording(self):
        """Kaydı durdur."""
        self.recording = False
        self.metadata['end_time'] = time.time()
        self.metadata['duration'] = self.metadata['end_time'] - self.metadata['start_time']
        self.metadata['frame_count'] = len(self.frames)
        print(f"[RECORDER] Recording stopped: {len(self.frames)} frames")

    def record_frame(self, state: dict):
        """Bir frame kaydet."""
        if not self.recording:
            return

        frame = FrameData(
            time=state.get('time', time.time() - self.start_time),
            positions=state['positions'].copy(),
            velocities=state['velocities'].copy(),
            targets=state['targets'].copy(),
            orientations=state.get('orientations', np.zeros_like(state['positions'])).copy(),
            armed=state['armed'].copy(),
            sensor_data=state.get('sensor_data', None),
        )

        self.frames.append(frame)

    def save(self, filename: str):
        """Kaydı dosyaya yaz."""
        data = {
            'metadata': self.metadata,
            'frames': []
        }

        for frame in self.frames:
            frame_dict = {
                'time': frame.time,
                'positions': frame.positions.tolist(),
                'velocities': frame.velocities.tolist(),
                'targets': frame.targets.tolist(),
                'orientations': frame.orientations.tolist(),
                'armed': frame.armed.tolist(),
            }
            data['frames'].append(frame_dict)

        # Compress and save
        with gzip.open(filename, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        print(f"[RECORDER] Saved to {filename}")

    def load(self, filename: str):
        """Kaydı dosyadan oku."""
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        self.metadata = data['metadata']
        self.num_drones = self.metadata['num_drones']
        self.frames = []

        for frame_dict in data['frames']:
            frame = FrameData(
                time=frame_dict['time'],
                positions=np.array(frame_dict['positions']),
                velocities=np.array(frame_dict['velocities']),
                targets=np.array(frame_dict['targets']),
                orientations=np.array(frame_dict['orientations']),
                armed=np.array(frame_dict['armed']),
            )
            self.frames.append(frame)

        print(f"[RECORDER] Loaded {len(self.frames)} frames from {filename}")


class SimulationPlayer:
    """Simülasyon oynatıcı."""

    def __init__(self, recorder: SimulationRecorder):
        self.recorder = recorder
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self.play_start_time = 0
        self.play_start_frame = 0

    def play(self):
        """Oynatmaya başla."""
        self.playing = True
        self.play_start_time = time.time()
        self.play_start_frame = self.current_frame
        print("[PLAYER] Playing")

    def pause(self):
        """Duraklat."""
        self.playing = False
        print("[PLAYER] Paused")

    def stop(self):
        """Durdur ve başa sar."""
        self.playing = False
        self.current_frame = 0
        print("[PLAYER] Stopped")

    def set_speed(self, speed: float):
        """Oynatma hızını ayarla."""
        self.playback_speed = speed

    def seek(self, frame: int):
        """Belirli bir frame'e git."""
        self.current_frame = max(0, min(frame, len(self.recorder.frames) - 1))

    def seek_time(self, t: float):
        """Belirli bir zamana git."""
        for i, frame in enumerate(self.recorder.frames):
            if frame.time >= t:
                self.current_frame = i
                return
        self.current_frame = len(self.recorder.frames) - 1

    def update(self) -> Optional[FrameData]:
        """
        Oynatmayı güncelle ve mevcut frame'i döndür.
        """
        if not self.playing:
            if self.recorder.frames:
                return self.recorder.frames[self.current_frame]
            return None

        # Calculate which frame we should be at
        elapsed = (time.time() - self.play_start_time) * self.playback_speed
        start_time = self.recorder.frames[self.play_start_frame].time

        # Find frame by time
        target_time = start_time + elapsed
        for i in range(self.play_start_frame, len(self.recorder.frames)):
            if self.recorder.frames[i].time >= target_time:
                self.current_frame = i
                return self.recorder.frames[i]

        # Reached end
        self.current_frame = len(self.recorder.frames) - 1
        self.playing = False
        return self.recorder.frames[self.current_frame]

    def get_progress(self) -> float:
        """Oynatma ilerlemesini döndür (0-1)."""
        if not self.recorder.frames:
            return 0
        return self.current_frame / (len(self.recorder.frames) - 1)
```

---

### C5: 3D Visualizer [DÜŞÜK ÖNCELİK]
**Süre:** 5 gün
**Zorluk:** ⭐⭐⭐⭐⭐

**Not:** Bu görev opsiyonel. Önce diğerlerini tamamla.

PyOpenGL veya Panda3D kullanılabilir.

---

### C6: Ayarlar Paneli [ORTA ÖNCELİK]
**Süre:** 2 gün
**Zorluk:** ⭐⭐

Runtime'da ayarları değiştirme paneli.

```python
class SettingsPanel:
    """Runtime ayarlar paneli."""

    def __init__(self, config):
        self.config = config
        self.visible = False
        self.sliders = []

        # Ayarlanabilir değerler
        self.settings = {
            'collision_radius': {'min': 0.5, 'max': 3.0, 'value': config.collision_radius},
            'avoidance_strength': {'min': 1.0, 'max': 5.0, 'value': config.avoidance_strength},
            'max_velocity': {'min': 1.0, 'max': 10.0, 'value': config.max_velocity_xy},
            'target_gain_p': {'min': 0.5, 'max': 3.0, 'value': config.kp},
        }

    def toggle(self):
        self.visible = not self.visible

    def draw(self, surface):
        if not self.visible:
            return

        # Panel background
        pygame.draw.rect(surface, (40, 40, 50), (50, 50, 300, 250))
        pygame.draw.rect(surface, (100, 100, 120), (50, 50, 300, 250), 2)

        # Title
        # ... draw sliders ...

    def handle_event(self, event):
        # Handle slider drag
        pass

    def apply(self):
        """Ayarları uygula."""
        self.config.collision_radius = self.settings['collision_radius']['value']
        # ... etc
```

---

## Test

Her görev için görsel test yap:
1. Ekran görüntüsü al
2. Takımla paylaş
3. Feedback al

---

## Sorular?

Can'a (takım liderine) sor.
