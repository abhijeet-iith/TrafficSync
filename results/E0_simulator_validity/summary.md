# E0 - simulator validity

## Saturation flow (continuous green after a 400 s red, 200 s measured)

| lanes | s_l [veh/s] | measured | error |
|---|---|---|---|
| 1 | 0.5 | 0.500 | +0.0% |
| 2 | 1.0 | 1.000 | +0.0% |
| 3 | 1.5 | 1.500 | +0.0% |

## Conservation and storage

generated = in network + completed, q >= 0, n <= C checked at 75,600 network-steps (3 networks x 7 scenarios x 3600 s, actuated): **0 violations**.

## Isolated intersection, 60 s cycle (24 s effective green per phase): simulated vs Webster delay

Simulated delay counts every movement (through + turns) per vehicle; corridor1 has 4 approaches at lambda each.

| lambda [veh/s] | x | simulated [s] | Webster [s] | difference |
|---|---|---|---|---|
| 0.04 | 0.20 | 12.5 | 12.3 | +1.2% |
| 0.08 | 0.40 | 14.1 | 14.2 | -0.5% |
| 0.12 | 0.60 | 16.1 | 16.6 | -3.2% |
| 0.16 | 0.80 | 22.6 | 22.3 | +1.0% |

## Determinism

Same seed and config run twice (CoMP, grid3x3, D_med, seed 7): all simulation metrics identical.  Fixed, actuated and CoCycle on seed 7 see identical vehicle generation times and routes (5,145 vehicles).
