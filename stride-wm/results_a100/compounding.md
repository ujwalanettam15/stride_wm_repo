# Stride vs error accumulation

Open-loop rollouts from 2013 held-out start frames (data/cache). ± = s.d. over training seeds.
`fresh` = error of the last call when given the true previous latent; `propagated` = error that comes from the call's input being a prediction; `carried in` = share of the total that is not fresh error; `gain` = share of the incoming error that survives one call, over calls 2..n; `gain per step` = gain ** (1/k).

## Decomposition at depth 8 env steps
| model | calls | total MSE | fresh (last call) | propagated | carried in | gain per call | gain per step |
|---|---|---|---|---|---|---|---|
| chunk k=1 | 8 | 0.219 ± 0.010 | 0.118 ± 0.000 | 0.094 ± 0.008 | 46 ± 2% | 0.34 ± 0.02 | 0.337 ± 0.017 |
| chunk k=2 | 4 | 0.172 ± 0.002 | 0.126 ± 0.000 | 0.042 ± 0.002 | 27 ± 1% | 0.20 ± 0.01 | 0.451 ± 0.007 |
| chunk k=4 | 2 | 0.157 ± 0.003 | 0.138 ± 0.001 | 0.019 ± 0.001 | 12 ± 1% | 0.14 ± 0.01 | 0.609 ± 0.010 |
| chunk k=8 | 1 | 0.152 ± 0.001 | 0.152 ± 0.001 | 0.000 ± 0.000 | 0 ± 0% | — | — |
| chunk k=1-u16 | 8 | 0.218 ± 0.019 | 0.156 ± 0.007 | 0.054 ± 0.008 | 28 ± 3% | 0.19 ± 0.01 | 0.194 ± 0.010 |
| chunk k=2-u8 | 4 | 0.175 ± 0.003 | 0.139 ± 0.001 | 0.035 ± 0.001 | 20 ± 1% | 0.17 ± 0.00 | 0.409 ± 0.005 |
| chunk k=4-u4 | 2 | 0.158 ± 0.001 | 0.144 ± 0.001 | 0.017 ± 0.000 | 9 ± 0% | 0.12 ± 0.00 | 0.590 ± 0.002 |
| chunk k=8-u2 | 1 | 0.153 ± 0.002 | 0.153 ± 0.002 | 0.000 ± 0.000 | 0 ± 0% | — | — |

## Decomposition at depth 40 env steps
| model | calls | total MSE | fresh (last call) | propagated | carried in | gain per call | gain per step |
|---|---|---|---|---|---|---|---|
| chunk k=1 | 40 | 0.487 ± 0.069 | 0.118 ± 0.000 | 0.365 ± 0.068 | 76 ± 4% | 0.65 ± 0.04 | 0.653 ± 0.041 |
| chunk k=2 | 20 | 0.365 ± 0.024 | 0.127 ± 0.000 | 0.235 ± 0.023 | 65 ± 2% | 0.52 ± 0.02 | 0.723 ± 0.016 |
| chunk k=4 | 10 | 0.316 ± 0.039 | 0.139 ± 0.001 | 0.170 ± 0.036 | 56 ± 6% | 0.43 ± 0.05 | 0.808 ± 0.023 |
| chunk k=8 | 5 | 0.245 ± 0.028 | 0.153 ± 0.001 | 0.091 ± 0.022 | 37 ± 7% | 0.31 ± 0.05 | 0.864 ± 0.016 |
| chunk k=1-u16 | 40 | 0.329 ± 0.054 | 0.159 ± 0.005 | 0.159 ± 0.045 | 51 ± 6% | 0.38 ± 0.05 | 0.377 ± 0.046 |
| chunk k=2-u8 | 20 | 0.291 ± 0.010 | 0.141 ± 0.001 | 0.144 ± 0.008 | 52 ± 1% | 0.39 ± 0.01 | 0.625 ± 0.008 |
| chunk k=4-u4 | 10 | 0.262 ± 0.009 | 0.146 ± 0.001 | 0.111 ± 0.008 | 44 ± 2% | 0.32 ± 0.01 | 0.755 ± 0.006 |
| chunk k=8-u2 | 5 | 0.222 ± 0.012 | 0.154 ± 0.002 | 0.068 ± 0.007 | 30 ± 3% | 0.25 ± 0.02 | 0.842 ± 0.007 |

## Equal-span ablation (A1): every stride trained on the same 16-step chained loss
Latent MSE; `default` = single k-step loss, `equal span` = chained loss over the span. A model whose unroll would be 1 is its own default.

| stride | default @8 | equal span @8 | default @16 | equal span @16 | default @24 | equal span @24 | default @40 | equal span @40 |
|---|---|---|---|---|---|---|---|---|
| k=1 | 0.219 ± 0.010 | 0.217 ± 0.019 | 0.308 ± 0.026 | 0.252 ± 0.030 | 0.379 ± 0.041 | 0.280 ± 0.040 | 0.486 ± 0.068 | 0.327 ± 0.054 |
| k=2 | 0.172 ± 0.002 | 0.174 ± 0.003 | 0.225 ± 0.006 | 0.211 ± 0.004 | 0.275 ± 0.012 | 0.241 ± 0.006 | 0.364 ± 0.024 | 0.290 ± 0.010 |
| k=4 | 0.157 ± 0.003 | 0.158 ± 0.001 | 0.197 ± 0.009 | 0.186 ± 0.003 | 0.238 ± 0.018 | 0.212 ± 0.005 | 0.316 ± 0.040 | 0.262 ± 0.009 |
| k=8 | 0.151 ± 0.001 | 0.152 ± 0.002 | 0.174 ± 0.006 | 0.170 ± 0.004 | 0.198 ± 0.012 | 0.188 ± 0.007 | 0.244 ± 0.028 | 0.221 ± 0.012 |
| **spread across strides (max − min)** | 0.068 | 0.065 | 0.134 | 0.082 | 0.181 | 0.091 | 0.242 | 0.106 |

If the spread shrinks toward zero under equal span, the E1 stride ordering was error accumulation; if it stays, stride itself changes what the model learns.
