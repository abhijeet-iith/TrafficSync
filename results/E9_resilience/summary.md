# E9 - resilience (W-MSR, robustness)

## A. Robustness of communication graphs (F = 1)

| network | graph | edges | min deg | max r | (2,2)-robust (F-total, iff) | 3-robust (F-local, suff.) | exact |
|---|---|---|---|---|---|---|---|
| corridor3 | road | 2 | 1 | 1 | no | no | exact |
| corridor3 | road+greedy(F_total) | 3 | 2 | 2 | yes | no | exact |
| corridor3 | road+greedy(F_local) | 3 | 2 | 2 | yes | no | exact |
| corridor3 | 2-hop | 3 | 2 | 2 | yes | no | exact |
| grid2x2 | road | 4 | 2 | 1 | no | no | exact |
| grid2x2 | road+greedy(F_total) | 5 | 2 | 2 | yes | no | exact |
| grid2x2 | road+greedy(F_local) | 6 | 3 | 2 | yes | no | exact |
| grid2x2 | 2-hop | 6 | 3 | 2 | yes | no | exact |
| grid3x3 | road | 12 | 2 | 1 | no | no | exact |
| grid3x3 | road+greedy(F_total) | 16 | 3 | 2 | yes | no | exact |
| grid3x3 | road+greedy(F_local) | 20 | 4 | 3 | yes | yes | exact |
| grid3x3 | 2-hop | 26 | 5 | 3 | yes | yes | exact |
| grid4x4 | road | 24 | 2 | – | no | no | sampled |
| grid4x4 | road+greedy(F_total) | 34 | 4 | – | yes | no | sampled |
| grid4x4 | road+greedy(F_local) | 49 | 6 | – | yes | no | sampled |
| grid4x4 | 2-hop | 58 | 5 | – | yes | yes | sampled |
| grid5x5 | road | 40 | 2 | – | no | no | sampled |
| grid5x5 | road+greedy(F_total) | 48 | 3 | – | no | no | sampled |
| grid5x5 | road+greedy(F_local) | 69 | 5 | – | yes | yes | sampled |
| grid5x5 | 2-hop | 102 | 5 | – | yes | yes | sampled |

Links to add (F = 1).  For N > 12 the check is sampled: 'search' = no counterexample in the search's own samples, 'independent' = none in 80,000 further samples (a 'no' there refutes the search).

| network | target | method | satisfied (search) | independent check | links added | exhaustive minimum (<= 4 links) |
|---|---|---|---|---|---|---|
| corridor3 | F_total | greedy | yes | yes | 1 (road has 2) |  |
| corridor3 | F_total | khop (k=2) | yes | yes | 1 (road has 2) |  |
| corridor3 | F_local | greedy | no | no | – (road has 2) |  |
| corridor3 | F_local | khop (k=1) | no | no | – (road has 2) |  |
| grid2x2 | F_total | greedy | yes | yes | 1 (road has 4) |  |
| grid2x2 | F_total | khop (k=2) | yes | yes | 2 (road has 4) |  |
| grid2x2 | F_local | greedy | no | no | – (road has 4) |  |
| grid2x2 | F_local | khop (k=1) | no | no | – (road has 4) |  |
| grid3x3 | F_total | greedy | yes | yes | 4 (road has 12) | 3 |
| grid3x3 | F_total | khop (k=2) | yes | yes | 14 (road has 12) |  |
| grid3x3 | F_local | greedy | yes | yes | 8 (road has 12) |  |
| grid3x3 | F_local | khop (k=2) | yes | yes | 14 (road has 12) |  |
| grid4x4 | F_total | greedy | yes | yes | 10 (road has 24) |  |
| grid4x4 | F_total | khop (k=2) | yes | yes | 34 (road has 24) |  |
| grid4x4 | F_local | greedy | yes | no | 25 (road has 24) |  |
| grid4x4 | F_local | khop (k=2) | yes | yes | 34 (road has 24) |  |
| grid5x5 | F_total | greedy | yes | no | 8 (road has 40) |  |
| grid5x5 | F_total | khop (k=2) | yes | yes | 62 (road has 40) |  |
| grid5x5 | F_local | greedy | yes | yes | 29 (road has 40) |  |
| grid5x5 | F_local | khop (k=2) | yes | yes | 62 (road has 40) |  |

## B. Network-level consensus (mean over all F-local placements of one faulty node, 20 initial states)

| network | graph | fault | rule | interval kept | benign agreement | mean abs. bias |
|---|---|---|---|---|---|---|
| corridor3 | road | constant | average | 0% | 100% | 0.624 |
| corridor3 | road | constant | wmsr | 100% | 30% | 0.029 |
| corridor3 | road | noise | average | 0% | 33% | 0.200 |
| corridor3 | road | noise | wmsr | 100% | 67% | 0.064 |
| corridor3 | road | drift | average | 0% | 33% | 2.953 |
| corridor3 | road | drift | wmsr | 100% | 35% | 0.051 |
| corridor3 | augmented F_total | constant | average | 0% | 100% | 0.624 |
| corridor3 | augmented F_total | constant | wmsr | 100% | 100% | 0.086 |
| corridor3 | augmented F_total | noise | average | 0% | 100% | 0.213 |
| corridor3 | augmented F_total | noise | wmsr | 100% | 100% | 0.038 |
| corridor3 | augmented F_total | drift | average | 0% | 100% | 2.970 |
| corridor3 | augmented F_total | drift | wmsr | 100% | 100% | 0.063 |
| corridor3 | augmented F_local | constant | average | 0% | 100% | 0.624 |
| corridor3 | augmented F_local | constant | wmsr | 100% | 100% | 0.086 |
| corridor3 | augmented F_local | noise | average | 0% | 100% | 0.213 |
| corridor3 | augmented F_local | noise | wmsr | 100% | 100% | 0.038 |
| corridor3 | augmented F_local | drift | average | 0% | 100% | 2.970 |
| corridor3 | augmented F_local | drift | wmsr | 100% | 100% | 0.063 |
| grid2x2 | road | constant | average | 0% | 100% | 0.652 |
| grid2x2 | road | constant | wmsr | 100% | 32% | 0.053 |
| grid2x2 | road | noise | average | 0% | 0% | 0.188 |
| grid2x2 | road | noise | wmsr | 100% | 100% | 0.052 |
| grid2x2 | road | drift | average | 0% | 0% | 2.950 |
| grid2x2 | road | drift | wmsr | 100% | 54% | 0.045 |
| grid2x2 | augmented F_total | constant | average | 0% | 100% | 0.652 |
| grid2x2 | augmented F_total | constant | wmsr | 100% | 100% | 0.094 |
| grid2x2 | augmented F_total | noise | average | 0% | 2% | 0.200 |
| grid2x2 | augmented F_total | noise | wmsr | 100% | 100% | 0.043 |
| grid2x2 | augmented F_total | drift | average | 0% | 0% | 2.953 |
| grid2x2 | augmented F_total | drift | wmsr | 100% | 100% | 0.066 |
| grid2x2 | augmented F_local | constant | average | 0% | 100% | 0.652 |
| grid2x2 | augmented F_local | constant | wmsr | 100% | 100% | 0.069 |
| grid2x2 | augmented F_local | noise | average | 0% | 100% | 0.196 |
| grid2x2 | augmented F_local | noise | wmsr | 100% | 100% | 0.049 |
| grid2x2 | augmented F_local | drift | average | 0% | 100% | 2.960 |
| grid2x2 | augmented F_local | drift | wmsr | 100% | 100% | 0.054 |
| grid3x3 | road | constant | average | 0% | 100% | 0.648 |
| grid3x3 | road | constant | wmsr | 100% | 15% | 0.044 |
| grid3x3 | road | noise | average | 0% | 0% | 0.156 |
| grid3x3 | road | noise | wmsr | 100% | 61% | 0.041 |
| grid3x3 | road | drift | average | 0% | 0% | 2.842 |
| grid3x3 | road | drift | wmsr | 100% | 27% | 0.039 |
| grid3x3 | augmented F_total | constant | average | 0% | 100% | 0.648 |
| grid3x3 | augmented F_total | constant | wmsr | 100% | 100% | 0.056 |
| grid3x3 | augmented F_total | noise | average | 0% | 0% | 0.157 |
| grid3x3 | augmented F_total | noise | wmsr | 100% | 100% | 0.028 |
| grid3x3 | augmented F_total | drift | average | 0% | 0% | 2.867 |
| grid3x3 | augmented F_total | drift | wmsr | 100% | 100% | 0.037 |
| grid3x3 | augmented F_local | constant | average | 0% | 100% | 0.648 |
| grid3x3 | augmented F_local | constant | wmsr | 100% | 100% | 0.036 |
| grid3x3 | augmented F_local | noise | average | 0% | 0% | 0.161 |
| grid3x3 | augmented F_local | noise | wmsr | 100% | 100% | 0.019 |
| grid3x3 | augmented F_local | drift | average | 0% | 0% | 2.890 |
| grid3x3 | augmented F_local | drift | wmsr | 100% | 100% | 0.027 |

## C. Closed loop (grid3x3, agent 4 faulty, local detectors)

Change of mean delay vs the fault-free controller with plain averaging on the road graph (positive = worse), paired over seeds, [95% paired-bootstrap CI].

| controller | fault | graph | aggregation | delay [s] | change | p | benign cycle [s] | benign cycle spread [s] |
|---|---|---|---|---|---|---|---|---|
| cocycle | none | augmented | average | 34.6 ± 0.5 | +0.1% [-0.4, +0.5] | 0.742 | 46.8 | 0.0 |
| cocycle | none | augmented | wmsr-dynamic | 34.6 ± 0.6 | +0.2% [-0.6, +0.9] | 0.492 | 41.2 | 0.0 |
| cocycle | none | augmented | wmsr-static | 34.7 ± 0.7 | +0.5% [-0.1, +1.0] | 0.16 | 46.4 | 0.8 |
| cocycle | none | road | average | 34.5 ± 0.6 | -0.0% [-0.0, -0.0] | 1 | 46.8 | 0.4 |
| cocycle | none | road | wmsr-dynamic | 34.6 ± 0.6 | +0.1% [-0.7, +1.0] | 0.846 | 40.0 | 0.0 |
| cocycle | none | road | wmsr-static | 34.6 ± 0.7 | +0.2% [-0.5, +1.1] | 0.432 | 46.6 | 2.4 |
| cocycle | constant | augmented | average | 34.3 ± 0.6 | -0.7% [-1.5, +0.1] | 0.232 | 88.0 | 0.0 |
| cocycle | constant | augmented | wmsr-dynamic | 34.4 ± 0.6 | -0.3% [-0.9, +0.4] | 0.492 | 88.0 | 0.0 |
| cocycle | constant | augmented | wmsr-static | 34.6 ± 0.7 | +0.1% [-0.8, +0.8] | 0.846 | 47.2 | 1.6 |
| cocycle | constant | road | average | 34.3 ± 0.6 | -0.7% [-1.5, +0.1] | 0.232 | 88.0 | 0.0 |
| cocycle | constant | road | wmsr-dynamic | 34.4 ± 0.6 | -0.4% [-1.0, +0.2] | 0.232 | 88.0 | 0.0 |
| cocycle | constant | road | wmsr-static | 34.5 ± 0.5 | -0.2% [-0.7, +0.3] | 0.625 | 47.5 | 3.2 |
| cocycle | noise | augmented | average | 34.5 ± 0.6 | -0.1% [-0.9, +0.7] | 1 | 57.3 | 4.8 |
| cocycle | noise | augmented | wmsr-dynamic | 34.5 ± 0.6 | -0.0% [-0.8, +0.6] | 0.695 | 43.6 | 0.0 |
| cocycle | noise | augmented | wmsr-static | 34.5 ± 0.7 | -0.1% [-0.8, +0.6] | 0.77 | 47.0 | 2.4 |
| cocycle | noise | road | average | 34.5 ± 0.7 | -0.1% [-0.8, +0.6] | 0.922 | 57.3 | 4.8 |
| cocycle | noise | road | wmsr-dynamic | 34.6 ± 0.7 | +0.1% [-0.7, +1.0] | 0.77 | 42.0 | 0.0 |
| cocycle | noise | road | wmsr-static | 34.5 ± 0.6 | -0.1% [-0.9, +0.7] | 0.846 | 47.0 | 4.8 |
| comp | none | road | average | 99.1 ± 7.6 | -0.0% [-0.0, -0.0] | 1 | nan | nan |
| comp | none | road | wmsr-dynamic | 97.7 ± 6.4 | -1.4% [-2.7, -0.2] | 0.0645 | nan | nan |
| comp | none | road | wmsr-static | 99.3 ± 7.4 | +0.2% [-0.5, +0.9] | 0.492 | nan | nan |
| comp | constant | road | average | 117.8 ± 10.5 | +18.8% [+17.3, +20.6] | 0.00195 | nan | nan |
| comp | constant | road | wmsr-dynamic | 118.4 ± 11.3 | +19.5% [+17.2, +21.7] | 0.00195 | nan | nan |
| comp | constant | road | wmsr-static | 99.7 ± 7.3 | +0.6% [-0.6, +1.8] | 0.432 | nan | nan |

Figures: `figures/F08_wmsr_vs_average.png`, `figures/E9_closed_loop_delay.png`
