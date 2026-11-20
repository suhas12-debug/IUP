
### 3D AP_R40: Easy / Moderate / Hard   (delta = Moderate vs LiDAR-only baseline at the same beam count)

| Condition | Car | Pedestrian | Cyclist |
|---|---|---|---|
| LiDAR-only baseline (fine-tuned), 64-beam | 86.9 / 75.7 / 72.7 | 54.1 / 49.0 / 44.1 | 81.5 / 62.1 / 58.0 |
| LiDAR-only baseline (fine-tuned), 32-beam | 84.1 / 70.2 / 67.1 | 50.6 / 46.0 / 41.6 | 74.9 / 52.3 / 49.2 |
| LiDAR-only baseline (fine-tuned), 16-beam | 78.3 / 59.8 / 55.5 | 44.1 / 39.7 / 35.3 | 59.8 / 37.7 / 35.4 |
| Fusion, both sensors, 64-beam | 87.2 / 75.7 / 73.0  (-0.1) | 61.6 / 57.2 / 52.5  (+8.1) | 82.2 / 60.1 / 55.9  (-2.0) |
| Fusion, both sensors, 32-beam | 84.4 / 70.5 / 67.6  (+0.3) | 60.3 / 55.9 / 51.2  (+9.9) | 73.0 / 50.2 / 46.6  (-2.0) |
| Fusion, both sensors, 16-beam | 78.3 / 59.2 / 54.6  (-0.6) | 51.7 / 47.9 / 43.5  (+8.2) | 54.3 / 33.2 / 31.1  (-4.5) |
| Fusion, LiDAR only, 64-beam | 87.2 / 75.7 / 73.0  (-0.1) | 61.6 / 57.2 / 52.5  (+8.1) | 82.2 / 60.1 / 55.9  (-2.0) |
| Fusion, LiDAR only, 32-beam | 84.4 / 70.5 / 67.6  (+0.3) | 60.3 / 55.9 / 51.2  (+9.9) | 73.0 / 50.2 / 46.6  (-2.0) |
| Fusion, LiDAR only, 16-beam | 78.3 / 59.2 / 54.6  (-0.6) | 51.7 / 47.9 / 43.5  (+8.2) | 54.3 / 33.2 / 31.1  (-4.5) |
| Fusion, camera only | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |

### BEV AP_R40: Easy / Moderate / Hard   (delta = Moderate vs LiDAR-only baseline at the same beam count)

| Condition | Car | Pedestrian | Cyclist |
|---|---|---|---|
| LiDAR-only baseline (fine-tuned), 64-beam | 93.4 / 87.7 / 86.4 | 59.5 / 54.4 / 49.9 | 83.3 / 64.7 / 60.5 |
| LiDAR-only baseline (fine-tuned), 32-beam | 90.9 / 84.1 / 81.4 | 58.8 / 53.5 / 49.0 | 75.8 / 56.6 / 53.4 |
| LiDAR-only baseline (fine-tuned), 16-beam | 88.5 / 75.6 / 71.1 | 54.8 / 49.1 / 44.5 | 61.4 / 39.1 / 36.9 |
| Fusion, both sensors, 64-beam | 94.0 / 88.4 / 87.7  (+0.7) | 66.9 / 62.8 / 58.8  (+8.4) | 83.8 / 61.9 / 58.6  (-2.7) |
| Fusion, both sensors, 32-beam | 91.9 / 85.2 / 82.7  (+1.0) | 69.0 / 63.8 / 58.9  (+10.4) | 75.3 / 52.7 / 49.8  (-3.9) |
| Fusion, both sensors, 16-beam | 89.4 / 76.4 / 72.2  (+0.8) | 63.5 / 58.6 / 53.5  (+9.5) | 56.4 / 34.6 / 32.6  (-4.6) |
| Fusion, LiDAR only, 64-beam | 94.0 / 88.4 / 87.7  (+0.7) | 66.9 / 62.8 / 58.8  (+8.4) | 83.8 / 61.9 / 58.6  (-2.7) |
| Fusion, LiDAR only, 32-beam | 91.9 / 85.2 / 82.7  (+1.0) | 69.0 / 63.8 / 58.9  (+10.4) | 75.3 / 52.7 / 49.8  (-3.9) |
| Fusion, LiDAR only, 16-beam | 89.4 / 76.4 / 72.2  (+0.8) | 63.5 / 58.6 / 53.5  (+9.5) | 56.4 / 34.6 / 32.6  (-4.6) |
| Fusion, camera only | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 |
