# ABL_ablations — Ablations of CoMP and CoCycle: downstream term, kappa = 0, gamma, offsets, veto, K_c

10 seeds (0–9); delay change vs `actuated`, paired over seeds [95% paired-bootstrap CI], Wilcoxon p.

## grid3x3, local detectors

| demand | controller | delay [s] | queue [veh] | throughput [veh/h] | vs actuated | p (vs actuated) |
|---|---|---|---|---|---|---|
| D_med | actuated | 37.2 ± 0.6 | 1.52 | 5158 | – | – |
| D_med | comp | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-no-downstream | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-kappa0 | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-no-gamma | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-no-veto | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-Kc0 | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-Kc1 | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | comp-Kc10 | 37.2 ± 0.6 | 1.52 | 5158 | -0.0% [-0.0, -0.0] | 1 |
| D_med | cocycle | 34.5 ± 0.6 | 1.41 | 5158 | -7.1% [-7.7, -6.5] | 0.002 |
| D_med | cocycle-no-offsets | 34.4 ± 0.5 | 1.41 | 5156 | -7.4% [-7.8, -6.9] | 0.002 |
| D_med | cocycle-Kc0 | 34.4 ± 0.5 | 1.40 | 5155 | -7.4% [-7.8, -6.9] | 0.002 |
| D_med | cocycle-Kc1 | 34.4 ± 0.6 | 1.41 | 5154 | -7.4% [-8.0, -6.8] | 0.002 |
| D_med | cocycle-Kc10 | 34.5 ± 0.6 | 1.41 | 5158 | -7.1% [-7.7, -6.5] | 0.002 |
| D_med | cocycle-occupancy | 34.5 ± 0.6 | 1.41 | 5157 | -7.1% [-7.5, -6.8] | 0.002 |
| D_med | comp+cycle | 45.8 ± 0.9 | 1.87 | 5158 | +23.2% [+22.4, +24.1] | 0.002 |
| D_high | actuated | 98.1 ± 6.9 | 6.71 | 8521 | – | – |
| D_high | comp | 99.1 ± 7.6 | 6.78 | 8521 | +1.0% [+0.2, +1.6] | 0.037 |
| D_high | comp-no-downstream | 97.8 ± 7.0 | 6.69 | 8525 | -0.3% [-1.0, +0.5] | 0.62 |
| D_high | comp-kappa0 | 99.1 ± 7.6 | 6.78 | 8521 | +1.0% [+0.2, +1.6] | 0.037 |
| D_high | comp-no-gamma | 99.7 ± 7.5 | 6.82 | 8520 | +1.6% [+0.9, +2.2] | 0.0059 |
| D_high | comp-no-veto | 99.1 ± 7.6 | 6.78 | 8521 | +1.0% [+0.2, +1.6] | 0.037 |
| D_high | comp-Kc0 | 99.7 ± 7.5 | 6.82 | 8520 | +1.6% [+0.9, +2.2] | 0.0059 |
| D_high | comp-Kc1 | 98.5 ± 7.4 | 6.74 | 8522 | +0.5% [-0.2, +1.1] | 0.28 |
| D_high | comp-Kc10 | 99.1 ± 7.6 | 6.78 | 8521 | +1.0% [+0.2, +1.6] | 0.037 |
| D_high | cocycle | 98.3 ± 7.4 | 6.71 | 8523 | +0.2% [-0.8, +1.2] | 0.92 |
| D_high | cocycle-no-offsets | 98.7 ± 7.0 | 6.75 | 8522 | +0.7% [-0.3, +1.6] | 0.28 |
| D_high | cocycle-Kc0 | 97.9 ± 6.6 | 6.70 | 8521 | -0.2% [-1.2, +1.1] | 0.7 |
| D_high | cocycle-Kc1 | 98.2 ± 6.0 | 6.72 | 8529 | +0.1% [-1.2, +1.5] | 0.7 |
| D_high | cocycle-Kc10 | 98.3 ± 7.4 | 6.71 | 8523 | +0.2% [-0.8, +1.2] | 0.92 |
| D_high | cocycle-occupancy | 148.9 ± 31.8 | 9.72 | 8201 | +51.8% [+37.3, +72.6] | 0.002 |
| D_high | comp+cycle | 115.8 ± 7.6 | 7.98 | 8553 | +18.1% [+16.6, +19.6] | 0.002 |
