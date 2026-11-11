# E12 sensitivity - VDN / QMIX with lr 1e-4 and target update 500

| variant | best validation delay per training seed [s] |
|---|---|
| vdn | 72.9, 72.8, 72.6 |
| qmix | 78.6, 75.5, 81.0 |

| scenario | variant | delay, lr 1e-4 [s] | spread over training seeds | delay, main run (lr 5e-4) [s] | best handcrafted [s] |
|---|---|---|---|---|---|
| D_med | vdn | 45.0 ± 0.7 | 1.7 | 51.2 | 34.5 (cocycle) |
| D_med | qmix | 45.6 ± 0.8 | 1.6 | 88.5 | 34.5 (cocycle) |
| D_high | vdn | 114.5 ± 6.7 | 1.7 | 283.4 | 98.1 (actuated) |
| D_high | qmix | 258.8 ± 109.5 | 101.4 | 250.4 | 98.1 (actuated) |
| D_tidal | vdn | 53.2 ± 1.0 | 0.8 | 57.8 | 41.7 (cocycle) |
| D_tidal | qmix | 60.2 ± 1.6 | 2.2 | 177.4 | 41.7 (cocycle) |
