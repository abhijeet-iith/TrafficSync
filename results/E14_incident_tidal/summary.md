# E14 - incident and tidal stress scenarios (grid3x3, local detectors)

Seeds 0-9 (dmpc+stl: 0-4).  Changes are vs actuated control, paired over seeds [95% bootstrap CI].

## D_incident (central eastbound link at 30% capacity, 1200-2100 s)

| controller | delay [s] | delay vs actuated | recovery [s] | recovery vs actuated | censored | incident-link clearance [s] | peak net queue | peak link queue | spillback | rho(Phi) |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed | 114.3 ± 79.1 | +143.4% [+61.7, +271.6] | 1079 ± 235 | +188.7% [+145.0, +233.4] | 2/10 | 918 ± 206 | 8.33 | 40.0 | 1654 | -14.8 |
| actuated | 46.9 ± 4.1 | – | 374 ± 85 | – | 0/10 | 263 ± 87 | 3.48 | 36.5 | 97 | -6.5 |
| comp | 46.6 ± 4.0 | -0.7% [-1.6, +0.1] | 367 ± 93 | -1.7% [-6.6, +2.6] | 0/10 | 263 ± 84 | 3.39 | 36.7 | 104 | -7.1 |
| cocycle | 43.0 ± 3.8 | -8.4% [-9.7, -7.1] | 339 ± 92 | -9.2% [-22.4, +0.2] | 0/10 | 260 ± 95 | 3.09 | 37.6 | 129 | -8.1 |
| comp+cycle | 61.1 ± 5.2 | +30.2% [+28.6, +32.1] | 577 ± 121 | +54.4% [+38.9, +70.4] | 0/10 | 486 ± 105 | 4.42 | 39.4 | 122 | -10.5 |
| dmpc+stl | 48.2 ± 9.1 | +0.6% [-0.4, +1.5] | 364 ± 173 | -8.9% [-30.8, +4.8] | 0/5 | 275 ± 151 | 3.58 | 37.4 | 145 | -8.4 |

## D_tidal (E-W 0.22 veh/s, N-S 0.06 veh/s)

| controller | delay [s] | delay vs actuated | clearance mean [s] | clearance max [s] | episodes | queue main / cross [veh] | max red [s] | rho_clear | rho_resp |
|---|---|---|---|---|---|---|---|---|---|
| fixed | 56.2 ± 1.8 | +24.1% [+21.6, +26.9] | 18 ± 41 | 18 | 0.1 | 2.88 / 2.48 | 48 | 7.0 | 13.9 |
| actuated | 45.3 ± 0.8 | – | 0 ± 0 | 0 | 0.0 | 2.60 / 1.71 | 48 | 7.3 | 16.2 |
| comp | 45.2 ± 0.8 | -0.3% [-0.6, +0.1] | 0 ± 0 | 0 | 0.0 | 2.59 / 1.71 | 48 | 7.3 | 17.2 |
| cocycle | 41.7 ± 0.9 | -7.9% [-8.9, -6.8] | 28 ± 62 | 28 | 0.1 | 2.36 / 1.61 | 48 | 7.1 | 17.3 |
| comp+cycle | 59.4 ± 1.5 | +31.1% [+29.6, +32.6] | 0 ± 0 | 0 | 0.0 | 3.22 / 2.48 | 48 | 6.4 | 14.4 |
| dmpc+stl | 45.5 ± 1.5 | -0.5% [-1.3, +0.2] | 0 ± 0 | 0 | 0.0 | 2.53 / 1.82 | 48 | 8.0 | 16.0 |

Figures: `figures/E14_incident_queue.png`, `figures/E14_recovery_clearance.png`
