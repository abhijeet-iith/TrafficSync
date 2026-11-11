# E11 - congestion spreading as an epidemic (grid3x3)

lambda_max(A) = 2.828; eigenvector centrality v_max = 0: 0.086, 1: 0.121, 2: 0.086, 3: 0.121, 4: 0.172, 5: 0.121, 6: 0.086, 7: 0.121, 8: 0.086

## Congested = mean load of the incoming links > 0.7 for > 60 s (standard definition)

| demand | controller | onsets | beta [1/s] | beta_0 [1/s] | 1/gamma [s] | R = beta lambda_max / gamma | p(beta>0) | spreading predicted | observed mean frac. | peak frac. | MF-SIS frac. | MF no-contagion frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_high | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_high | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_hotspot | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_hotspot | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_incident | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_incident | fixed | 7 | 3.37e-04 | 1.57e-05 | 2071 | 1.97 | 0.00039 | yes | 0.023 | 0.08 | 0.036 | 0.015 |
| D_low | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_low | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_ramp | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_ramp | fixed | 1 | 1.06e-17 | 3.70e-06 | 74 | 0.00 | 1 | no | 0.000 | 0.01 | 0.000 | 0.000 |
| D_tidal | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_tidal | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| uniform_0.25 | actuated | 99 | 3.33e-06 | 4.37e-04 | 583 | 0.01 | 0.48 | no | 0.164 | 0.41 | 0.172 | 0.172 |
| uniform_0.25 | fixed | 124 | 7.38e-18 | 5.94e-04 | 680 | 0.00 | 0.5 | no | 0.227 | 0.56 | 0.241 | 0.241 |

Centrality vs attributed spreading (all scenarios pooled): Spearman(v_max, onsets attributed) = -0.54; per congested hour = +0.60.

## Congested = load of the most loaded incoming link > 0.7 for > 60 s (spillback-oriented variant)

| demand | controller | onsets | beta [1/s] | beta_0 [1/s] | 1/gamma [s] | R = beta lambda_max / gamma | p(beta>0) | spreading predicted | observed mean frac. | peak frac. | MF-SIS frac. | MF no-contagion frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_high | actuated | 17 | 2.70e-18 | 6.40e-05 | 321 | 0.00 | 1 | no | 0.017 | 0.06 | 0.018 | 0.018 |
| D_high | fixed | 31 | 2.25e-07 | 1.19e-04 | 390 | 0.00 | 0.5 | no | 0.038 | 0.16 | 0.039 | 0.039 |
| D_hotspot | actuated | 12 | 1.42e-19 | 4.96e-05 | 4656 | 0.00 | 0.5 | no | 0.103 | 0.11 | 0.058 | 0.058 |
| D_hotspot | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.111 | 0.11 | 0.000 | 0.000 |
| D_incident | actuated | 10 | 8.20e-05 | 3.23e-05 | 471 | 0.11 | 0.081 | no | 0.017 | 0.11 | 0.014 | 0.013 |
| D_incident | fixed | 32 | 3.62e-04 | 5.28e-05 | 1357 | 1.39 | 1.2e-09 | yes | 0.116 | 0.36 | 0.089 | 0.041 |
| D_low | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_low | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_ramp | actuated | 2 | 6.39e-19 | 7.43e-06 | 438 | 0.00 | 1 | no | 0.003 | 0.01 | 0.003 | 0.003 |
| D_ramp | fixed | 95 | 5.63e-04 | 2.38e-04 | 532 | 0.85 | 9.2e-13 | no | 0.187 | 0.63 | 0.176 | 0.095 |
| D_tidal | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_tidal | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| uniform_0.25 | actuated | 75 | 2.11e-18 | 2.34e-03 | 8500 | 0.00 | 1 | no | 0.881 | 0.98 | 0.823 | 0.823 |
| uniform_0.25 | fixed | 56 | 6.42e-19 | 1.86e-03 | 12628 | 0.00 | 0.5 | no | 0.889 | 0.97 | 0.795 | 0.795 |

Centrality vs attributed spreading (all scenarios pooled): Spearman(v_max, onsets attributed) = +0.14; per congested hour = +0.43.

## Congested = load of the most loaded incoming link > 0.5 for > 60 s (spillback-oriented variant)

| demand | controller | onsets | beta [1/s] | beta_0 [1/s] | 1/gamma [s] | R = beta lambda_max / gamma | p(beta>0) | spreading predicted | observed mean frac. | peak frac. | MF-SIS frac. | MF no-contagion frac. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D_high | actuated | 177 | 3.47e-04 | 6.52e-04 | 290 | 0.28 | 0.00018 | no | 0.181 | 0.49 | 0.176 | 0.146 |
| D_high | fixed | 195 | 2.94e-04 | 7.76e-04 | 317 | 0.26 | 0.0014 | no | 0.219 | 0.52 | 0.210 | 0.181 |
| D_hotspot | actuated | 5 | 7.84e-21 | 2.08e-05 | 7397 | 0.00 | 1 | no | 0.110 | 0.11 | 0.027 | 0.027 |
| D_hotspot | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.111 | 0.11 | 0.000 | 0.000 |
| D_incident | actuated | 14 | 9.77e-05 | 4.28e-05 | 563 | 0.16 | 0.034 | no | 0.029 | 0.16 | 0.022 | 0.019 |
| D_incident | fixed | 36 | 3.23e-04 | 6.41e-05 | 1448 | 1.32 | 1.5e-08 | yes | 0.139 | 0.38 | 0.100 | 0.051 |
| D_low | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_low | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | actuated | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_med | fixed | 0 | 0.00e+00 | 0.00e+00 | inf | nan | nan | no | 0.000 | 0.00 | 0.000 | 0.000 |
| D_ramp | actuated | 66 | 7.34e-04 | 1.64e-04 | 252 | 0.52 | 2.2e-12 | no | 0.061 | 0.34 | 0.062 | 0.036 |
| D_ramp | fixed | 126 | 6.22e-04 | 3.72e-04 | 774 | 1.36 | 4.7e-14 | yes | 0.361 | 0.87 | 0.337 | 0.179 |
| D_tidal | actuated | 2 | 1.44e-18 | 7.41e-06 | 136 | 0.00 | 1 | no | 0.001 | 0.02 | 0.001 | 0.001 |
| D_tidal | fixed | 10 | 1.25e-18 | 3.72e-05 | 132 | 0.00 | 0.5 | no | 0.005 | 0.08 | 0.005 | 0.005 |
| uniform_0.25 | actuated | 40 | 4.70e-18 | 4.33e-03 | 11338 | 0.00 | 0.5 | no | 0.966 | 1.00 | 0.906 | 0.906 |
| uniform_0.25 | fixed | 39 | 1.47e-18 | 2.82e-03 | 10675 | 0.00 | 0.5 | no | 0.949 | 1.00 | 0.857 | 0.857 |

Centrality vs attributed spreading (all scenarios pooled): Spearman(v_max, onsets attributed) = +0.73; per congested hour = +0.72.

## H6 prioritisation (CoMP; priority weight gamma on one intersection; 10 seeds; change vs plain CoMP)

Most central intersection = 4; control = corner 8.

| demand | condition | delay [s] | delay change | p | congested node-s (max load > 0.5) | change | p |
|---|---|---|---|---|---|---|---|
| D_high | comp | 99.1 ± 7.6 | +0.0%  | nan | 5102 | +0.0%  | nan |
| D_high | comp-central-g1.5 | 99.9 ± 7.2 | +0.8% [-0.2, +1.7] | 0.13 | 5416 | +6.1% [+1.0, +12.6] | 0.037 |
| D_high | comp-corner-g1.5 | 100.5 ± 7.8 | +1.4% [+0.2, +2.6] | 0.064 | 5521 | +8.2% [+4.1, +13.2] | 0.014 |
| D_high | comp-central-g0.67 | 98.4 ± 7.2 | -0.7% [-1.3, -0.0] | 0.084 | 4788 | -6.2% [-13.8, -0.8] | 0.11 |
| D_high | comp-corner-g0.67 | 99.3 ± 7.1 | +0.2% [-0.6, +1.1] | 0.85 | 5040 | -1.2% [-9.1, +6.3] | 0.82 |
| uniform_0.25 | comp | 411.6 ± 17.8 | +0.0%  | nan | 26323 | +0.0%  | nan |
| uniform_0.25 | comp-central-g1.5 | 415.3 ± 18.5 | +0.9% [+0.4, +1.5] | 0.014 | 26346 | +0.1% [-0.4, +0.5] | 0.61 |
| uniform_0.25 | comp-corner-g1.5 | 425.3 ± 16.4 | +3.3% [+2.8, +3.9] | 0.002 | 26405 | +0.3% [-0.2, +1.1] | 0.98 |
| uniform_0.25 | comp-central-g0.67 | 407.2 ± 16.9 | -1.1% [-1.4, -0.8] | 0.002 | 26175 | -0.6% [-1.1, -0.1] | 0.049 |
| uniform_0.25 | comp-corner-g0.67 | 408.4 ± 17.8 | -0.8% [-1.1, -0.4] | 0.002 | 26334 | +0.0% [-0.3, +0.4] | 0.77 |

Figures: `figures/F11_epidemic_threshold_mean07.png`, `figures/F11_epidemic_threshold_max07.png`, `figures/F11_epidemic_threshold_max05.png`, `figures/E11_congested_fraction_timeseries.png`, `figures/E11_centrality_vs_spreading.png`
