# generate_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
n = 200
distance_m = np.linspace(0, 1000, n)
time0 = datetime.now()

gauge_mm = 1435 + np.random.normal(0, 1.0, n)
cant_mm = np.random.normal(0, 0.5, n)
level_variation = np.random.normal(0, 0.8, n)
vibration_g = np.abs(np.random.normal(0.02, 0.01, n))
humidity_pct = np.clip(np.random.normal(12, 6, n), 2, 95)
ballast_density = np.clip(np.random.normal(1.8, 0.15, n), 1.2, 2.4)
temperature_c = np.random.normal(26, 4, n)
noise_db = np.clip(np.random.normal(60, 4, n), 40, 90)

anomaly_idx = [30, 75, 120, 160]
for idx in anomaly_idx:
    level_variation[idx:idx+5] += np.linspace(5, 2, 5)
    vibration_g[idx:idx+5] += np.linspace(0.1, 0.02, 5)
    ballast_density[idx:idx+5] -= np.linspace(0.5, 0.1, 5)
    humidity_pct[idx:idx+5] += np.linspace(20, 5, 5)
    noise_db[idx:idx+5] += np.linspace(10, 3, 5)

track_health = 100 - (np.clip(level_variation,0,50) * 3) - (vibration_g*200) - (np.clip(1.9-ballast_density, 0, 1)*30)
track_health = np.clip(track_health, 0, 100).round(1)
requires_maintenance = (track_health < 60).astype(int)
timestamps = [time0 + timedelta(minutes=int(i*5)) for i in range(n)]

df = pd.DataFrame({
    "segment_id": np.arange(n),
    "distance_m": distance_m,
    "timestamp": timestamps,
    "gauge_mm": gauge_mm.round(2),
    "cant_mm": cant_mm.round(2),
    "level_variation_mm": level_variation.round(2),
    "vibration_g": vibration_g.round(4),
    "humidity_pct": humidity_pct.round(1),
    "ballast_density_t/m3": ballast_density.round(3),
    "temperature_c": temperature_c.round(1),
    "noise_db": noise_db.round(1),
    "track_health_index": track_health,
    "requires_maintenance": requires_maintenance
})

df.to_csv("track_data.csv", index=False)
print("Archivo generado: track_data.csv (n =", n, "segmentos)")