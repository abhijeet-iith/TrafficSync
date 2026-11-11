# SUMO experiments

## S1 - validation: grid 3x3, local detectors, seeds 0-4

Same vehicles (same seed, routes, arrival times) and the same controller code in both simulators. SUMO delay = time loss + insertion delay of vehicles generated after the 600 s warm-up (car-following, accelerations, a 60 m left-turn bay per approach); TrafficSync delay = waiting time in the queue model.

| demand | controller | TrafficSync delay [s] | vs actuated | SUMO delay [s] | vs actuated |
|---|---|---|---|---|---|
| D_low | fixed | 28.2 ± 0.5 | +24.1% [+22.4, +25.9] | 28.4 ± 0.6 | -2.5% [-3.8, -1.2] |
| D_low | actuated | 22.7 ± 0.6 | – | 29.1 ± 0.6 | – |
| D_low | comp | 22.7 ± 0.6 | -0.0% [-0.0, -0.0] | 29.1 ± 0.6 | -0.0% [-0.0, -0.0] |
| D_low | cocycle | 24.1 ± 0.3 | +6.0% [+4.2, +8.5] | 28.3 ± 0.5 | -3.0% [-4.6, -1.3] |
| D_med | fixed | 46.6 ± 0.9 | +24.8% [+24.0, +25.7] | 55.0 ± 1.0 | +7.2% [+5.7, +8.6] |
| D_med | actuated | 37.4 ± 0.8 | – | 51.3 ± 1.7 | – |
| D_med | comp | 37.4 ± 0.8 | -0.0% [-0.0, -0.0] | 51.4 ± 1.7 | +0.2% [-0.0, +0.5] |
| D_med | cocycle | 34.7 ± 1.1 | -7.2% [-8.3, -6.2] | 63.2 ± 3.3 | +23.2% [+21.7, +25.1] |
| D_high | fixed | 113.6 ± 12.9 | +9.1% [+7.8, +10.2] | 448.5 ± 223.1 | +7.8% [-7.0, +22.4] |
| D_high | actuated | 104.2 ± 12.2 | – | 415.9 ± 160.5 | – |
| D_high | comp | 106.0 ± 12.6 | +1.7% [+1.5, +1.9] | 369.9 ± 100.7 | -11.1% [-17.6, -1.2] |
| D_high | cocycle | 104.7 ± 12.4 | +0.5% [-0.5, +1.8] | 509.6 ± 234.1 | +22.5% [+2.9, +46.9] |

## S2 - real city maps, real demand (1 h, 10 min warm-up), SUMO seeds 0-2

| map | controller | delay [s] | vs city program (fixed) | waiting [s] | stops/veh | completed trips | switches / agent / h |
|---|---|---|---|---|---|---|---|
| cologne3 (3 signals) | fixed | 35.5 ± 0.7 | – | 22.3 | 0.99 | 2239 | 0 |
| cologne3 (3 signals) | actuated | 22.8 ± 1.6 | -35.7% [-37.7, -34.4] | 9.2 | 1.13 | 2252 | 250 |
| cologne3 (3 signals) | maxpressure | 34.5 ± 53.3 | -2.8% [-40.0, +65.5] | 14.9 | 1.24 | 2250 | 252 |
| cologne3 (3 signals) | comp | 42.4 ± 49.1 | +19.5% [-39.1, +73.1] | 19.9 | 1.32 | 2251 | 250 |
| cologne3 (3 signals) | cocycle | 25.9 ± 1.3 | -27.0% [-27.8, -26.1] | 10.7 | 1.26 | 2253 | 332 |
| cologne8 (8 signals) | fixed | 51.2 ± 0.8 | – | 31.6 | 1.35 | 1674 | 0 |
| cologne8 (8 signals) | actuated | 25.5 ± 0.4 | -50.2% [-50.6, -49.9] | 8.9 | 1.20 | 1686 | 161 |
| cologne8 (8 signals) | maxpressure | 24.2 ± 0.7 | -52.7% [-53.3, -51.9] | 7.8 | 1.19 | 1687 | 163 |
| cologne8 (8 signals) | comp | 24.8 ± 1.1 | -51.5% [-52.2, -50.8] | 8.2 | 1.21 | 1687 | 163 |
| cologne8 (8 signals) | cocycle | 27.3 ± 0.9 | -46.7% [-47.7, -45.6] | 10.2 | 1.29 | 1686 | 251 |
| ingolstadt7 (7 signals) | fixed | 124.7 ± 28.5 | – | 78.3 | 3.15 | 2327 | 0 |
| ingolstadt7 (7 signals) | actuated | 59.3 ± 4.6 | -52.5% [-55.7, -49.5] | 22.8 | 1.91 | 2471 | 191 |
| ingolstadt7 (7 signals) | maxpressure | 54.8 ± 10.2 | -56.1% [-61.2, -49.4] | 24.8 | 2.25 | 2474 | 199 |
| ingolstadt7 (7 signals) | comp | 64.3 ± 9.5 | -48.5% [-55.3, -41.4] | 28.6 | 2.54 | 2477 | 197 |
| ingolstadt7 (7 signals) | cocycle | 82.8 ± 21.6 | -33.6% [-45.2, -21.2] | 32.9 | 2.35 | 2449 | 272 |
| ingolstadt21 (21 signals) | fixed | 139.0 ± 12.2 | – | 93.4 | 3.19 | 3304 | 0 |
| ingolstadt21 (21 signals) | actuated | 98.6 ± 54.9 | -29.0% [-41.0, -8.3] | 50.0 | 3.55 | 3312 | 189 |
| ingolstadt21 (21 signals) | maxpressure | 83.9 ± 17.4 | -39.6% [-44.2, -32.9] | 38.7 | 3.37 | 3356 | 191 |
| ingolstadt21 (21 signals) | comp | 85.0 ± 11.6 | -38.8% [-43.9, -34.2] | 40.9 | 3.46 | 3347 | 185 |
| ingolstadt21 (21 signals) | cocycle | 86.5 ± 5.5 | -37.8% [-40.7, -35.0] | 40.6 | 3.57 | 3350 | 290 |
| bologna_acosta (7 signals) | fixed | 222.1 ± 12.7 | – | 92.4 | 3.93 | 5988 | 0 |
| bologna_acosta (7 signals) | actuated | 269.0 ± 69.5 | +21.1% [+8.9, +38.0] | 183.3 | 3.86 | 4103 | 65 |
| bologna_acosta (7 signals) | maxpressure | 205.8 ± 16.2 | -7.3% [-10.3, -4.6] | 85.7 | 3.73 | 5999 | 90 |
| bologna_acosta (7 signals) | comp | 261.9 ± 23.1 | +17.9% [+13.3, +20.6] | 113.5 | 4.30 | 5775 | 85 |
| bologna_acosta (7 signals) | cocycle | 267.3 ± 17.8 | +20.4% [+16.5, +26.9] | 145.8 | 4.80 | 5169 | 105 |
| bologna_pasubio (8 signals) | fixed | 204.4 ± 21.9 | – | 139.3 | 4.62 | 6256 | 0 |
| bologna_pasubio (8 signals) | actuated | 129.5 ± 14.9 | -36.7% [-42.1, -31.8] | 87.8 | 3.12 | 6467 | 139 |
| bologna_pasubio (8 signals) | maxpressure | 132.2 ± 7.2 | -35.3% [-37.2, -33.8] | 91.2 | 3.30 | 6393 | 140 |
| bologna_pasubio (8 signals) | comp | 133.3 ± 33.6 | -34.8% [-39.3, -31.9] | 93.7 | 3.10 | 6343 | 124 |
| bologna_pasubio (8 signals) | cocycle | 180.4 ± 22.5 | -11.8% [-12.5, -11.4] | 124.4 | 3.37 | 6128 | 144 |

### Summary: mean delay [s] per city (best agent design in bold)

| city | signals | city program | actuated | max-pressure | CoMP | CoCycle | best agent vs city program |
|---|---|---|---|---|---|---|---|
| cologne3 | 3 | 35.5 | **22.8** | 34.5 | 42.4 | 25.9 | actuated -35.7% [-37.7, -34.4] |
| cologne8 | 8 | 51.2 | 25.5 | **24.2** | 24.8 | 27.3 | max-pressure -52.7% [-53.3, -51.9] |
| ingolstadt7 | 7 | 124.7 | 59.3 | **54.8** | 64.3 | 82.8 | max-pressure -56.1% [-61.2, -49.4] |
| ingolstadt21 | 21 | 139.0 | 98.6 | **83.9** | 85.0 | 86.5 | max-pressure -39.6% [-44.2, -32.9] |
| bologna_acosta | 7 | 222.1 | 269.0 | **205.8** | 261.9 | 267.3 | max-pressure -7.3% [-10.3, -4.6] |
| bologna_pasubio | 8 | 204.4 | **129.5** | 132.2 | 133.3 | 180.4 | actuated -36.7% [-42.1, -31.8] |

## S3 - message loss on Cologne (CoMP and CoCycle)

| controller | drop probability | delay [s] | vs local max-pressure |
|---|---|---|---|
| CoMP | 0.0 | 24.8 ± 1.1 | +2.6% [-0.5, +5.3] |
| CoMP | 0.5 | 24.5 ± 2.1 | +1.1% [-1.7, +3.8] |
| CoMP | 1.0 | 24.2 ± 0.7 | -0.0% [-0.0, -0.0] |
| CoCycle | 0.0 | 27.3 ± 0.9 | +12.8% [+11.4, +13.9] |
| CoCycle | 0.5 | 27.3 ± 1.2 | +12.6% [+10.3, +15.7] |
| CoCycle | 1.0 | 26.5 ± 1.1 | +9.6% [+7.0, +11.6] |
