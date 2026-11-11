# E1_baselines — Baseline sanity (Stage 2): fixed vs actuated vs local max-pressure

10 seeds (0–9); delay change vs `fixed` and vs `actuated`, paired over seeds [95% paired-bootstrap CI], Wilcoxon p.

## corridor3, local detectors

| demand | controller | delay [s] | queue [veh] | throughput [veh/h] | vs fixed | p (vs fixed) | vs actuated | p (vs actuated) |
|---|---|---|---|---|---|---|---|---|
| D_low | fixed | 12.5 ± 0.3 | 0.43 | 1457 | – | – | +2.5% [+1.5, +3.4] | 0.0059 |
| D_low | actuated | 12.2 ± 0.3 | 0.42 | 1456 | -2.4% [-3.3, -1.5] | 0.0059 | – | – |
| D_low | maxpressure_local | 12.2 ± 0.3 | 0.42 | 1456 | -2.4% [-3.3, -1.5] | 0.0059 | -0.0% [-0.0, -0.0] | 1 |
| D_med | fixed | 21.8 ± 0.5 | 1.80 | 3485 | – | – | +12.3% [+10.8, +13.7] | 0.002 |
| D_med | actuated | 19.5 ± 0.5 | 1.60 | 3487 | -10.9% [-12.0, -9.8] | 0.002 | – | – |
| D_med | maxpressure_local | 19.5 ± 0.5 | 1.60 | 3487 | -10.9% [-12.0, -9.8] | 0.002 | -0.0% [-0.0, -0.0] | 1 |
| D_high | fixed | 68.0 ± 2.7 | 9.38 | 5761 | – | – | +9.7% [+7.0, +11.8] | 0.0039 |
| D_high | actuated | 62.0 ± 2.1 | 8.51 | 5755 | -8.8% [-10.6, -6.5] | 0.0039 | – | – |
| D_high | maxpressure_local | 62.0 ± 2.1 | 8.51 | 5755 | -8.8% [-10.6, -6.5] | 0.0039 | -0.0% [-0.0, -0.0] | 1 |

## grid2x2, local detectors

| demand | controller | delay [s] | queue [veh] | throughput [veh/h] | vs fixed | p (vs fixed) | vs actuated | p (vs actuated) |
|---|---|---|---|---|---|---|---|---|
| D_low | fixed | 18.3 ± 0.4 | 0.47 | 1458 | – | – | +19.9% [+18.1, +21.7] | 0.002 |
| D_low | actuated | 15.3 ± 0.3 | 0.39 | 1458 | -16.6% [-17.8, -15.3] | 0.002 | – | – |
| D_low | maxpressure_local | 15.3 ± 0.3 | 0.39 | 1458 | -16.6% [-17.8, -15.3] | 0.002 | -0.0% [-0.0, -0.0] | 1 |
| D_med | fixed | 31.5 ± 0.5 | 1.94 | 3493 | – | – | +22.7% [+21.6, +24.0] | 0.002 |
| D_med | actuated | 25.7 ± 0.4 | 1.58 | 3485 | -18.5% [-19.3, -17.8] | 0.002 | – | – |
| D_med | maxpressure_local | 25.7 ± 0.4 | 1.58 | 3485 | -18.5% [-19.3, -17.8] | 0.002 | -0.0% [-0.0, -0.0] | 1 |
| D_high | fixed | 85.1 ± 3.1 | 8.78 | 5786 | – | – | +7.7% [+4.8, +11.1] | 0.002 |
| D_high | actuated | 79.0 ± 4.8 | 8.16 | 5779 | -7.1% [-10.0, -4.6] | 0.002 | – | – |
| D_high | maxpressure_local | 79.0 ± 4.8 | 8.16 | 5779 | -7.1% [-10.0, -4.6] | 0.002 | -0.0% [-0.0, -0.0] | 1 |
