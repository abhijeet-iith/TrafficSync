# E12 - MARL vs the best handcrafted controller (grid3x3, local detectors)

## Training

| variant | training seed | episodes run | gradient steps | best validation delay [s] | wall [s] |
|---|---|---|---|---|---|
| iql | 0 | 80 | 23001 | 68.5 | 119 |
| iql | 1 | 72 | 20601 | 65.1 | 107 |
| iql | 2 | 72 | 20601 | 67.9 | 114 |
| iql+comm | 0 | 72 | 20601 | 70.7 | 122 |
| iql+comm | 1 | 72 | 20601 | 67.3 | 124 |
| iql+comm | 2 | 72 | 20601 | 69.4 | 119 |
| vdn | 0 | 90 | 26001 | 82.0 | 144 |
| vdn | 1 | 72 | 20601 | 73.7 | 114 |
| vdn | 2 | 72 | 20601 | 75.4 | 107 |
| qmix | 0 | 72 | 20601 | 82.0 | 163 |
| qmix | 1 | 72 | 20601 | 120.3 | 167 |
| qmix | 2 | 72 | 20601 | 141.8 | 164 |

## Held-out test (seeds 0-9, 3600 s, metrics from 600 s)

Change vs the best handcrafted controller of that scenario, paired over test seeds [95% bootstrap CI]; MARL rows = mean of the 3 trained policies per test seed; 'spread' = std of the 3 policies' mean delays.

| scenario | controller | delay [s] | vs best handcrafted | p | spread over training seeds | throughput [veh/h] | max red [s] | gridlock runs |
|---|---|---|---|---|---|---|---|---|
| D_med | fixed | 46.9 ± 0.6 | +35.6% [+34.4, +36.9] | 0.002 | – | 5163 | 27 | 0 |
| D_med | actuated | 37.2 ± 0.6 | +7.6% [+7.0, +8.4] | 0.002 | – | 5158 | 48 | 0 |
| D_med | comp | 37.2 ± 0.6 | +7.6% [+7.0, +8.4] | 0.002 | – | 5158 | 48 | 0 |
| D_med | cocycle **(best handcrafted)** | 34.5 ± 0.6 | – | – | – | 5158 | 48 | 0 |
| D_med | comp+cycle | 45.8 ± 0.9 | +32.6% [+31.4, +33.8] | 0.002 | – | 5158 | 42 | 0 |
| D_med | iql | 40.3 ± 0.7 | +16.7% [+15.9, +17.5] | 0.002 | 1.2 | 5157 | 48 | 0 |
| D_med | iql+comm | 40.5 ± 0.6 | +17.4% [+16.7, +18.0] | 0.002 | 0.3 | 5154 | 48 | 0 |
| D_med | vdn | 51.2 ± 0.7 | +48.3% [+47.0, +49.9] | 0.002 | 8.4 | 5157 | 48 | 0 |
| D_med | qmix | 88.5 ± 1.8 | +156.3% [+152.0, +160.6] | 0.002 | 25.9 | 5151 | 48 | 0 |
| D_high | fixed | 107.7 ± 7.0 | +9.8% [+8.9, +10.8] | 0.002 | – | 8531 | 48 | 0 |
| D_high | actuated **(best handcrafted)** | 98.1 ± 6.9 | – | – | – | 8521 | 48 | 0 |
| D_high | comp | 99.1 ± 7.6 | +1.0% [+0.2, +1.6] | 0.037 | – | 8521 | 48 | 0 |
| D_high | cocycle | 98.3 ± 7.4 | +0.2% [-0.8, +1.2] | 0.92 | – | 8523 | 48 | 0 |
| D_high | comp+cycle | 115.8 ± 7.6 | +18.1% [+16.6, +19.6] | 0.002 | – | 8553 | 48 | 0 |
| D_high | iql | 105.1 ± 6.5 | +7.1% [+6.1, +8.2] | 0.002 | 4.0 | 8520 | 48 | 0 |
| D_high | iql+comm | 110.2 ± 7.5 | +12.3% [+11.7, +13.0] | 0.002 | 6.4 | 8500 | 48 | 0 |
| D_high | vdn | 283.4 ± 98.7 | +188.9% [+114.2, +266.6] | 0.002 | 289.9 | 7230 | 48 | 6 |
| D_high | qmix | 250.4 ± 66.6 | +155.3% [+115.3, +204.2] | 0.002 | 124.2 | 7797 | 48 | 2 |
| D_tidal (unseen) | fixed | 56.2 ± 1.8 | +34.7% [+32.8, +37.2] | 0.002 | – | 6011 | 48 | 0 |
| D_tidal (unseen) | actuated | 45.3 ± 0.8 | +8.6% [+7.3, +9.8] | 0.002 | – | 6005 | 48 | 0 |
| D_tidal (unseen) | comp | 45.2 ± 0.8 | +8.3% [+7.0, +9.6] | 0.002 | – | 6007 | 48 | 0 |
| D_tidal (unseen) | cocycle **(best handcrafted)** | 41.7 ± 0.9 | – | – | – | 6008 | 48 | 0 |
| D_tidal (unseen) | comp+cycle | 59.4 ± 1.5 | +42.4% [+41.0, +43.8] | 0.002 | – | 6025 | 48 | 0 |
| D_tidal (unseen) | iql | 49.4 ± 1.0 | +18.5% [+17.5, +19.5] | 0.002 | 1.9 | 6011 | 48 | 0 |
| D_tidal (unseen) | iql+comm | 48.7 ± 1.1 | +16.8% [+15.8, +17.7] | 0.002 | 1.1 | 6009 | 48 | 0 |
| D_tidal (unseen) | vdn | 57.8 ± 1.0 | +38.5% [+37.0, +40.0] | 0.002 | 4.3 | 6007 | 48 | 0 |
| D_tidal (unseen) | qmix | 177.4 ± 7.8 | +325.1% [+313.9, +336.3] | 0.002 | 59.6 | 5784 | 48 | 0 |
| D_hotspot (unseen) | fixed | 164.2 ± 17.1 | +71.2% [+60.2, +84.8] | 0.002 | – | 4116 | 48 | 0 |
| D_hotspot (unseen) | actuated | 96.3 ± 17.5 | +0.5% [-0.3, +1.5] | 0.38 | – | 4261 | 48 | 0 |
| D_hotspot (unseen) | comp **(best handcrafted)** | 95.9 ± 17.8 | – | – | – | 4258 | 48 | 0 |
| D_hotspot (unseen) | cocycle | 99.6 ± 18.1 | +3.9% [+2.3, +5.5] | 0.0039 | – | 4236 | 48 | 0 |
| D_hotspot (unseen) | comp+cycle | 132.5 ± 19.0 | +38.2% [+33.3, +45.5] | 0.002 | – | 4207 | 48 | 0 |
| D_hotspot (unseen) | iql | 105.4 ± 17.8 | +9.9% [+7.9, +12.9] | 0.002 | 7.3 | 4237 | 48 | 0 |
| D_hotspot (unseen) | iql+comm | 147.3 ± 17.8 | +53.6% [+46.0, +63.5] | 0.002 | 74.8 | 4136 | 48 | 0 |
| D_hotspot (unseen) | vdn | 193.1 ± 21.5 | +101.4% [+88.6, +118.1] | 0.002 | 121.1 | 4018 | 48 | 0 |
| D_hotspot (unseen) | qmix | 303.8 ± 15.1 | +216.8% [+184.2, +258.6] | 0.002 | 104.3 | 3870 | 48 | 0 |
| D_incident (unseen) | fixed | 114.3 ± 79.1 | +165.7% [+76.4, +304.4] | 0.002 | – | 4925 | 27 | 1 |
| D_incident (unseen) | actuated | 46.9 ± 4.1 | +9.1% [+7.6, +10.7] | 0.002 | – | 5158 | 48 | 0 |
| D_incident (unseen) | comp | 46.6 ± 4.0 | +8.4% [+6.4, +10.2] | 0.002 | – | 5157 | 48 | 0 |
| D_incident (unseen) | cocycle **(best handcrafted)** | 43.0 ± 3.8 | – | – | – | 5155 | 48 | 0 |
| D_incident (unseen) | comp+cycle | 61.1 ± 5.2 | +42.1% [+40.0, +44.7] | 0.002 | – | 5156 | 48 | 0 |
| D_incident (unseen) | iql | 50.4 ± 4.5 | +17.3% [+15.5, +19.6] | 0.002 | 3.2 | 5158 | 48 | 0 |
| D_incident (unseen) | iql+comm | 49.5 ± 3.6 | +15.1% [+13.0, +17.2] | 0.002 | 0.5 | 5156 | 48 | 0 |
| D_incident (unseen) | vdn | 67.8 ± 5.4 | +57.6% [+52.3, +62.8] | 0.002 | 20.7 | 5148 | 48 | 0 |
| D_incident (unseen) | qmix | 101.2 ± 7.6 | +135.4% [+129.6, +141.1] | 0.002 | 21.4 | 5148 | 48 | 0 |

Figures: `figures/E12_training_curves.png`, `figures/E12_test_delay.png`
