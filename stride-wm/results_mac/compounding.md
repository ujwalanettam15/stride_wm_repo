# Stride vs error accumulation

Open-loop rollouts from 1007 held-out start frames (data/cache). ± = s.d. over training seeds.
`fresh` = error of the last call when given the true previous latent; `propagated` = error that comes from the call's input being a prediction; `carried in` = share of the total that is not fresh error; `gain` = share of the incoming error that survives one call, over calls 2..n; `gain per step` = gain ** (1/k).

## Decomposition at depth 8 env steps
| model | calls | total MSE | fresh (last call) | propagated | carried in | gain per call | gain per step |
|---|---|---|---|---|---|---|---|
| chunk k=1 | 8 | 0.213 ± 0.010 | 0.114 ± 0.000 | 0.091 ± 0.010 | 47 ± 3% | 0.33 ± 0.02 | 0.335 ± 0.019 |
| chunk k=2 | 4 | 0.167 ± 0.001 | 0.122 ± 0.001 | 0.040 ± 0.001 | 27 ± 1% | 0.20 ± 0.00 | 0.451 ± 0.003 |
| chunk k=4 | 2 | 0.151 ± 0.002 | 0.133 ± 0.001 | 0.018 ± 0.001 | 11 ± 1% | 0.14 ± 0.01 | 0.607 ± 0.008 |
| chunk k=8 | 1 | 0.147 ± 0.002 | 0.147 ± 0.002 | 0.000 ± 0.000 | 0 ± 0% | — | — |

## Decomposition at depth 40 env steps
| model | calls | total MSE | fresh (last call) | propagated | carried in | gain per call | gain per step |
|---|---|---|---|---|---|---|---|
| chunk k=1 | 40 | 0.501 ± 0.074 | 0.113 ± 0.000 | 0.383 ± 0.071 | 77 ± 4% | 0.66 ± 0.04 | 0.660 ± 0.039 |
| chunk k=2 | 20 | 0.365 ± 0.023 | 0.122 ± 0.000 | 0.239 ± 0.024 | 66 ± 2% | 0.53 ± 0.02 | 0.730 ± 0.012 |
| chunk k=4 | 10 | 0.276 ± 0.016 | 0.134 ± 0.000 | 0.139 ± 0.017 | 51 ± 3% | 0.40 ± 0.03 | 0.795 ± 0.015 |
| chunk k=5 | 8 | 0.265 ± 0.018 | 0.137 ± 0.001 | 0.123 ± 0.016 | 48 ± 4% | 0.37 ± 0.03 | 0.820 ± 0.012 |
| chunk k=8 | 5 | 0.233 ± 0.030 | 0.148 ± 0.002 | 0.084 ± 0.020 | 36 ± 7% | 0.30 ± 0.04 | 0.859 ± 0.014 |
| DINO-WM k=5 | 8 | 0.263 ± 0.018 | 0.139 ± 0.001 | 0.120 ± 0.016 | 47 ± 4% | 0.36 ± 0.03 | 0.817 ± 0.012 |
