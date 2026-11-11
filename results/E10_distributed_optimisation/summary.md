# E10 - distributed optimisation of the common cycle

Instances: local Webster cycles from the fixed-time design flows, w_i = Y_i; z* = sum w_i C_i* / sum w_i.  rho = 1, DGD/GT step 0.2, Metropolis weights.

With uniform entry rates the *design* flows are equal on every link, so all C_i* coincide (trivial instance, solved at iteration 1); *measured* flows come from one simulated hour.

| network | demand | flows | z* [s] | C_i* range [s] | lambda2 |
|---|---|---|---|---|---|
| corridor3 | D_med | design | 44.23 | 44.2 – 44.2 | 1.000 |
| corridor3 | D_tidal | design | 49.53 | 42.9 – 52.3 | 1.000 |
| corridor3 | D_hotspot | design | 105.86 | 55.4 – 164.3 | 1.000 |
| corridor3 | D_med | measured | 46.20 | 45.3 – 46.7 | 1.000 |
| corridor3 | D_high | measured | 132.42 | 125.5 – 136.9 | 1.000 |
| corridor3 | D_hotspot | measured | 110.25 | 51.0 – 180.6 | 1.000 |
| grid2x2 | D_med | design | 44.23 | 44.2 – 44.2 | 2.000 |
| grid2x2 | D_tidal | design | 64.51 | 64.5 – 64.5 | 2.000 |
| grid2x2 | D_hotspot | design | 96.22 | 37.6 – 165.8 | 2.000 |
| grid2x2 | D_med | measured | 45.58 | 43.7 – 46.8 | 2.000 |
| grid2x2 | D_high | measured | 131.36 | 124.1 – 140.2 | 2.000 |
| grid2x2 | D_hotspot | measured | 113.36 | 36.0 – 210.4 | 2.000 |
| grid3x3 | D_med | design | 44.23 | 44.2 – 44.2 | 1.000 |
| grid3x3 | D_tidal | design | 66.54 | 52.3 – 74.4 | 1.000 |
| grid3x3 | D_hotspot | design | 70.94 | 37.1 – 166.6 | 1.000 |
| grid3x3 | D_med | measured | 43.93 | 42.5 – 45.3 | 1.000 |
| grid3x3 | D_high | measured | 112.49 | 100.0 – 127.8 | 1.000 |
| grid3x3 | D_hotspot | measured | 61.49 | 35.8 – 133.2 | 1.000 |

## Solvers (grid3x3, non-trivial instances)

| demand | flows | solver | final error after 2000 it. [s] | iters to 1e-3 | iters to 1e-6 |
|---|---|---|---|---|---|
| D_tidal | design | ADMM (central z) | 0.00e+00 | 14 | 28 |
| D_tidal | design | ADMM (decentralised) | 1.28e-13 | 69 | 129 |
| D_tidal | design | Gradient tracking | 4.26e-14 | 48 | 98 |
| D_tidal | design | DGD (alpha=0.2) | 2.39e+00 | – | – |
| D_tidal | design | DGD (alpha=0.02) | 2.78e-01 | – | – |
| D_tidal | design | DGD (alpha=0.5/(k+1)^0.6) | 7.40e-02 | – | – |
| D_tidal | design | ADMM (local z, 1 round) | 4.80e+00 | – | – |
| D_hotspot | design | ADMM (central z) | 4.26e-14 | 25 | 41 |
| D_hotspot | design | ADMM (decentralised) | 4.26e-14 | 101 | 177 |
| D_hotspot | design | Gradient tracking | 7.11e-14 | 94 | 159 |
| D_hotspot | design | DGD (alpha=0.2) | 1.98e+01 | – | – |
| D_hotspot | design | DGD (alpha=0.02) | 2.57e+00 | – | – |
| D_hotspot | design | DGD (alpha=0.5/(k+1)^0.6) | 6.77e-01 | – | – |
| D_hotspot | design | ADMM (local z, 1 round) | 2.45e+01 | – | – |
| D_med | measured | ADMM (central z) | 2.13e-14 | 8 | 25 |
| D_med | measured | ADMM (decentralised) | 9.24e-14 | 53 | 133 |
| D_med | measured | Gradient tracking | 4.26e-14 | 45 | 103 |
| D_med | measured | DGD (alpha=0.2) | 1.67e-01 | – | – |
| D_med | measured | DGD (alpha=0.02) | 2.07e-02 | – | – |
| D_med | measured | DGD (alpha=0.5/(k+1)^0.6) | 5.58e-03 | – | – |
| D_med | measured | ADMM (local z, 1 round) | 4.16e-01 | – | – |
| D_high | measured | ADMM (central z) | 1.42e-14 | 12 | 23 |
| D_high | measured | ADMM (decentralised) | 8.53e-14 | 54 | 102 |
| D_high | measured | Gradient tracking | 5.68e-14 | 80 | 143 |
| D_high | measured | DGD (alpha=0.2) | 5.01e+00 | – | – |
| D_high | measured | DGD (alpha=0.02) | 7.42e-01 | – | – |
| D_high | measured | DGD (alpha=0.5/(k+1)^0.6) | 2.02e-01 | – | – |
| D_high | measured | ADMM (local z, 1 round) | 8.96e+00 | – | – |
| D_hotspot | measured | ADMM (central z) | 1.42e-14 | 25 | 43 |
| D_hotspot | measured | ADMM (decentralised) | 1.63e-13 | 104 | 185 |
| D_hotspot | measured | Gradient tracking | 1.42e-14 | 97 | 166 |
| D_hotspot | measured | DGD (alpha=0.2) | 1.43e+01 | – | – |
| D_hotspot | measured | DGD (alpha=0.02) | 1.84e+00 | – | – |
| D_hotspot | measured | DGD (alpha=0.5/(k+1)^0.6) | 4.84e-01 | – | – |
| D_hotspot | measured | ADMM (local z, 1 round) | 1.80e+01 | – | – |

Central ADMM with the default stop (both residual norms < 1e-3): stopped after 1–26 iterations, |z − z*| ≤ 8.8e-04 s over all 15 non-trivial instances.

## Rate vs lambda_2 (Erdos-Renyi G(16, p))

* gradient tracking (best step): Spearman(lambda2, iterations) = -0.95
* decentralised ADMM, rho tuned per graph: -0.64
* decentralised ADMM, fixed rho = 1: +0.98
* DGD constant-step bias: Spearman(lambda2, bias) = -0.85

## ADMM with local z (course 'hack'), grid3x3 D_hotspot (measured flows)

| rounds K | final error [s] | disagreement [s] |
|---|---|---|
| 1 | 1.80e+01 | 3.47e+01 |
| 2 | 1.14e+01 | 2.27e+01 |
| 3 | 7.27e+00 | 1.40e+01 |
| 5 | 3.91e+00 | 7.30e+00 |
| 10 | 9.36e-01 | 1.74e+00 |
| 20 | 6.40e-02 | 1.19e-01 |
| 40 | 3.21e-04 | 5.96e-04 |
| 80 | 8.08e-09 | 1.50e-08 |

Figures: `figures/F07_optimizer_convergence.png`, `figures/E10_admm_residuals.png`, `figures/E10_rate_vs_lambda2.png`, `figures/E10_local_z_rounds.png`
