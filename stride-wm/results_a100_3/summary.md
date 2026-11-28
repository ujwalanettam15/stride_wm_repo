## Training · final k-step MSE (train = mean of last 500 steps; val = held-out episodes)
| model | seed | train | val | copy-last on same val pairs | chunk diversity (RMS NN dist / dim) |
|---|---|---|---|---|---|
| chunk_k1-res | 0 | 0.1404 | 0.1511 | 0.2671 | 0.122 |
| chunk_k1-u16 | 0 | 0.2052 | 0.1546 | 0.2671 | 0.125 |
| chunk_k1-u16 | 1 | 0.1920 | 0.1628 | 0.2671 | 0.122 |
| chunk_k1-u16 | 2 | 0.1827 | 0.1506 | 0.2671 | 0.122 |
| chunk_k1 | 0 | 0.1085 | 0.1171 | 0.2671 | 0.122 |
| chunk_k1 | 1 | 0.1077 | 0.1171 | 0.2671 | 0.124 |
| chunk_k1 | 2 | 0.1078 | 0.1175 | 0.2671 | 0.120 |
| chunk_k2-u8 | 0 | 0.1424 | 0.1401 | 0.4445 | 0.213 |
| chunk_k2-u8 | 1 | 0.1448 | 0.1396 | 0.4445 | 0.209 |
| chunk_k2-u8 | 2 | 0.1410 | 0.1381 | 0.4445 | 0.212 |
| chunk_k2 | 0 | 0.1174 | 0.1256 | 0.4445 | 0.209 |
| chunk_k2 | 1 | 0.1168 | 0.1261 | 0.4445 | 0.209 |
| chunk_k2 | 2 | 0.1168 | 0.1265 | 0.4445 | 0.216 |
| chunk_k4-u4 | 0 | 0.1409 | 0.1436 | 0.7654 | 0.310 |
| chunk_k4-u4 | 1 | 0.1408 | 0.1443 | 0.7654 | 0.302 |
| chunk_k4-u4 | 2 | 0.1413 | 0.1450 | 0.7654 | 0.313 |
| chunk_k4 | 0 | 0.1291 | 0.1374 | 0.7654 | 0.303 |
| chunk_k4 | 1 | 0.1284 | 0.1392 | 0.7654 | 0.308 |
| chunk_k4 | 2 | 0.1281 | 0.1380 | 0.7654 | 0.306 |
| chunk_k5-emb10 | 0 | 0.1368 | 0.1437 | 0.8848 | 0.332 |
| chunk_k5-emb10 | 1 | 0.1356 | 0.1442 | 0.8848 | 0.342 |
| chunk_k5-emb10 | 2 | 0.1353 | 0.1440 | 0.8848 | 0.339 |
| chunk_k5-res | 0 | 0.1989 | 0.2025 | 0.8848 | 0.332 |
| chunk_k5 | 0 | 0.1328 | 0.1427 | 0.8848 | 0.332 |
| chunk_k5 | 1 | 0.1318 | 0.1432 | 0.8848 | 0.342 |
| chunk_k5 | 2 | 0.1321 | 0.1428 | 0.8848 | 0.339 |
| chunk_k8-emb128 | 0 | 0.1388 | 0.1476 | 1.1003 | 0.411 |
| chunk_k8-emb128 | 1 | 0.1378 | 0.1514 | 1.1003 | 0.419 |
| chunk_k8-emb128 | 2 | 0.1384 | 0.1511 | 1.1003 | 0.421 |
| chunk_k8-emb32 | 0 | 0.1448 | 0.1552 | 1.1003 | 0.411 |
| chunk_k8-emb32 | 1 | 0.1434 | 0.1537 | 1.1003 | 0.419 |
| chunk_k8-emb32 | 2 | 0.1423 | 0.1511 | 1.1003 | 0.421 |
| chunk_k8-res | 0 | 0.2151 | 0.2190 | 1.1003 | 0.411 |
| chunk_k8-u2 | 0 | 0.1451 | 0.1538 | 1.1003 | 0.424 |
| chunk_k8-u2 | 1 | 0.1441 | 0.1503 | 1.1003 | 0.423 |
| chunk_k8-u2 | 2 | 0.1462 | 0.1541 | 1.1003 | 0.422 |
| chunk_k8 | 0 | 0.1407 | 0.1508 | 1.1003 | 0.411 |
| chunk_k8 | 1 | 0.1397 | 0.1521 | 1.1003 | 0.419 |
| chunk_k8 | 2 | 0.1413 | 0.1509 | 1.1003 | 0.421 |
| dinowm_k5-e64 | 0 | 0.1328 | 0.1421 | 0.8848 | 0.332 |
| dinowm_k5-e64 | 1 | 0.1319 | 0.1436 | 0.8848 | 0.342 |
| dinowm_k5-e64 | 2 | 0.1314 | 0.1407 | 0.8848 | 0.339 |
| dinowm_k5-res | 0 | 0.2063 | 0.2130 | 0.8848 | 0.332 |
| dinowm_k5 | 0 | 0.1364 | 0.1443 | 0.8848 | 0.332 |
| dinowm_k5 | 1 | 0.1352 | 0.1428 | 0.8848 | 0.342 |
| dinowm_k5 | 2 | 0.1343 | 0.1452 | 0.8848 | 0.339 |

## E1 · open-loop latent MSE by rollout depth (env steps), all phases
± = s.e. over held-out episodes (one seed; episode = unit of replication); † = s.d. over training seeds

| label | 8 | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- | --- |
| chunk k=1 | 0.2194 ± 0.0097† | 0.3085 ± 0.0263† | 0.3788 ± 0.0411† | 0.4354 ± 0.0554† | 0.4858 ± 0.0683† |
| chunk k=2 | 0.1716 ± 0.0023† | 0.2251 ± 0.0063† | 0.2747 ± 0.0116† | 0.3208 ± 0.0181† | 0.3643 ± 0.0238† |
| chunk k=4 | 0.1570 ± 0.0032† | 0.1970 ± 0.0091† | 0.2382 ± 0.0181† | 0.2771 ± 0.0279† | 0.3158 ± 0.0399† |
| chunk k=5 | — | — | — | — | 0.3151 ± 0.0460† |
| chunk k=8 | 0.1510 ± 0.0007† | 0.1740 ± 0.0057† | 0.1976 ± 0.0124† | 0.2209 ± 0.0200† | 0.2441 ± 0.0276† |
| DINO-WM k=5 | — | — | — | — | 0.3158 ± 0.0584† |
| copy_last | 1.1002 ± 0.0194 | 1.3627 ± 0.0237 | 1.4727 ± 0.0269 | 1.5025 ± 0.0291 | 1.4909 ± 0.0304 |

## E1 · probe-decoded error at depth 40 (cube / end-effector position, cm)
| model | calls | latent MSE | cos dist | cube cm | cube deg | EE cm | EE deg | grip |
|---|---|---|---|---|---|---|---|---|
| chunk k=1 | 40 | 0.4858 ± 0.0683† | 0.0566 ± 0.0081† | 6.52 ± 1.07† | 21.3 ± 1.0† | 5.53 ± 0.57† | 41.4 ± 1.5† | 0.071 ± 0.019† |
| chunk k=2 | 20 | 0.3643 ± 0.0238† | 0.0426 ± 0.0028† | 4.55 ± 0.22† | 20.0 ± 0.6† | 4.44 ± 0.26† | 30.1 ± 1.5† | 0.048 ± 0.002† |
| chunk k=4 | 10 | 0.3158 ± 0.0399† | 0.0370 ± 0.0047† | 4.12 ± 0.21† | 17.6 ± 0.5† | 3.81 ± 0.56† | 20.4 ± 1.5† | 0.050 ± 0.004† |
| chunk k=5 | 8 | 0.3151 ± 0.0460† | 0.0370 ± 0.0054† | 3.92 ± 0.25† | 17.8 ± 1.0† | 3.69 ± 0.82† | 20.7 ± 4.4† | 0.052 ± 0.004† |
| chunk k=8 | 5 | 0.2441 ± 0.0276† | 0.0286 ± 0.0033† | 3.46 ± 0.05† | 16.2 ± 0.2† | 2.85 ± 0.48† | 17.0 ± 0.5† | 0.048 ± 0.002† |
| DINO-WM k=5 | 8 | 0.3158 ± 0.0584† | 0.0370 ± 0.0069† | 4.08 ± 0.45† | 18.0 ± 0.3† | 3.94 ± 1.34† | 19.7 ± 3.6† | 0.052 ± 0.004† |
| copy_last | 0 | 1.4909 ± 0.0304 | 0.1682 ± 0.0033 | 7.48 ± 0.51 | 8.7 ± 0.7 | 23.84 ± 0.56 | 42.0 ± 1.2 | 0.148 ± 0.006 |
| probe floor (true latents) | – | – | – | 2.37 | 5.3 | 1.03 | 5.7 | 0.055 |

## E1 · cube position error (cm) by phase of the predicted frame, depth 8
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 3.75 ± 0.41† | 2.69 ± 0.59† | 4.40 ± 0.37† | 3.11 ± 0.69† |
| chunk k=2 | 2.99 ± 0.09† | 2.33 ± 0.29† | 3.39 ± 0.26† | 2.56 ± 0.12† |
| chunk k=4 | 2.92 ± 0.14† | 1.94 ± 0.31† | 3.47 ± 0.10† | 2.38 ± 0.32† |
| chunk k=8 | 2.68 ± 0.04† | 1.88 ± 0.06† | 3.07 ± 0.11† | 2.04 ± 0.26† |
| copy_last | 2.67 ± 0.20 | 4.74 ± 0.60 | 11.12 ± 0.34 | 11.46 ± 0.91 |
| probe_floor | 2.35 ± 0.07 | 2.09 ± 0.14 | 2.49 ± 0.09 | 2.16 ± 0.11 |

## E1 · latent MSE by phase, depth 8  (frames: reach n=1472, grasp n=40, transport n=686, place n=33)
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 0.2223 ± 0.0135† | 0.1805 ± 0.0200† | 0.1957 ± 0.0077† | 0.2015 ± 0.0184† |
| chunk k=2 | 0.1720 ± 0.0031† | 0.1474 ± 0.0036† | 0.1589 ± 0.0012† | 0.1710 ± 0.0044† |
| chunk k=4 | 0.1572 ± 0.0032† | 0.1362 ± 0.0023† | 0.1473 ± 0.0042† | 0.1550 ± 0.0052† |
| chunk k=8 | 0.1497 ± 0.0010† | 0.1356 ± 0.0011† | 0.1442 ± 0.0008† | 0.1508 ± 0.0051† |
| copy_last | 1.0937 ± 0.0202 | 0.8513 ± 0.0387 | 0.9124 ± 0.0253 | 0.7967 ± 0.0485 |

## E1 · cube position error (cm) by phase of the predicted frame, depth 40
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 6.49 ± 1.57† | 5.93 ± 1.47† | 7.10 ± 1.51† | 6.64 ± 1.98† |
| chunk k=2 | 4.52 ± 0.18† | 3.75 ± 0.67† | 4.98 ± 0.33† | 4.17 ± 0.52† |
| chunk k=4 | 3.98 ± 0.30† | 3.79 ± 0.30† | 4.99 ± 0.41† | 3.77 ± 0.47† |
| chunk k=5 | 3.94 ± 0.26† | 3.54 ± 0.64† | 4.09 ± 0.07† | 3.41 ± 0.47† |
| chunk k=8 | 3.43 ± 0.02† | 3.21 ± 0.14† | 3.88 ± 0.26† | 3.16 ± 0.17† |
| copy_last | 4.42 ± 0.29 | 16.92 ± 2.41 | 23.23 ± 1.44 | 27.07 ± 2.77 |
| probe_floor | 2.35 ± 0.07 | 2.09 ± 0.14 | 2.49 ± 0.09 | 2.16 ± 0.11 |

## E1 · latent MSE by phase, depth 40  (frames: reach n=1630, grasp n=11, transport n=251, place n=22)
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 0.4949 ± 0.0604† | 0.4203 ± 0.1104† | 0.4410 ± 0.1333† | 0.4574 ± 0.1624† |
| chunk k=2 | 0.3724 ± 0.0278† | 0.2608 ± 0.0206† | 0.3041 ± 0.0130† | 0.2883 ± 0.0088† |
| chunk k=4 | 0.3160 ± 0.0395† | 0.2832 ± 0.0606† | 0.3033 ± 0.0548† | 0.3018 ± 0.0691† |
| chunk k=5 | 0.3251 ± 0.0510† | 0.2558 ± 0.0373† | 0.2496 ± 0.0271† | 0.2593 ± 0.0132† |
| chunk k=8 | 0.2455 ± 0.0286† | 0.2085 ± 0.0176† | 0.2230 ± 0.0156† | 0.2200 ± 0.0366† |
| copy_last | 1.4919 ± 0.0310 | 1.4554 ± 0.1566 | 1.5568 ± 0.0645 | 1.9541 ± 0.1184 |

## E4 · DINO-WM conditioning vs chunk encoder, k=5 (latent MSE)
| label | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- |
| chunk k=5 | 0.1422 ± 0.0004† | 0.1631 ± 0.0014† | 0.2100 ± 0.0125† | 0.3151 ± 0.0460† |
| DINO-WM k=5 | 0.1432 ± 0.0011† | 0.1635 ± 0.0049† | 0.2109 ± 0.0181† | 0.3158 ± 0.0584† |

## Action sensitivity · latent MSE with true vs shuffled actions (all phases)
If shuffling barely changes the error the model is not using its actions.

| model | depth | true actions | shuffled actions | ratio |
|---|---|---|---|---|
| chunk_k1_s0 | 8 | 0.2293 | 1.1458 | 5.00 |
| chunk_k1_s0 | 40 | 0.5507 | 1.4675 | 2.66 |
| chunk_k2_s0 | 8 | 0.1691 | 1.1403 | 6.74 |
| chunk_k2_s0 | 40 | 0.3402 | 1.4313 | 4.21 |
| chunk_k4_s0 | 8 | 0.1560 | 1.1245 | 7.21 |
| chunk_k4_s0 | 40 | 0.3470 | 1.4451 | 4.16 |
| chunk_k5_s0 | 40 | 0.3485 | 1.4719 | 4.22 |
| chunk_k8_s0 | 8 | 0.1505 | 1.1101 | 7.37 |
| chunk_k8_s0 | 40 | 0.2203 | 1.4156 | 6.43 |
| dinowm_k5_s0 | 40 | 0.3794 | 1.4844 | 3.91 |

## Ablations · latent MSE by rollout depth, variants vs default (all phases)
-res = residual prediction; -uN = N chained steps in the training loss; -eN = DINO-WM embedding width; -embN = chunk-encoder width. Compare each variant to the default row of the same k.

| label | 8 | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- | --- |
| chunk k=1 | 0.2194 ± 0.0097† | 0.3085 ± 0.0263† | 0.3788 ± 0.0411† | 0.4354 ± 0.0554† | 0.4858 ± 0.0683† |
| chunk k=1-res | 0.3575 ± 0.0039 | 0.5192 ± 0.0063 | 0.6969 ± 0.0093 | 0.9059 ± 0.0123 | 1.1576 ± 0.0159 |
| chunk k=1-u16 | 0.2169 ± 0.0185† | 0.2516 ± 0.0296† | 0.2796 ± 0.0397† | 0.3043 ± 0.0478† | 0.3271 ± 0.0543† |
| chunk k=2 | 0.1716 ± 0.0023† | 0.2251 ± 0.0063† | 0.2747 ± 0.0116† | 0.3208 ± 0.0181† | 0.3643 ± 0.0238† |
| chunk k=2-u8 | 0.1742 ± 0.0027† | 0.2109 ± 0.0040† | 0.2413 ± 0.0064† | 0.2671 ± 0.0083† | 0.2899 ± 0.0096† |
| chunk k=4 | 0.1570 ± 0.0032† | 0.1970 ± 0.0091† | 0.2382 ± 0.0181† | 0.2771 ± 0.0279† | 0.3158 ± 0.0399† |
| chunk k=4-u4 | 0.1583 ± 0.0010† | 0.1861 ± 0.0030† | 0.2120 ± 0.0051† | 0.2373 ± 0.0066† | 0.2623 ± 0.0086† |
| chunk k=5 | — | — | — | — | 0.3151 ± 0.0460† |
| DINO-WM k=5 | — | — | — | — | 0.3158 ± 0.0584† |
| DINO-WM k=5-e64 | — | — | — | — | 0.2920 ± 0.0372† |
| chunk k=5-emb10 | — | — | — | — | 0.2754 ± 0.0077† |
| chunk k=5-res | — | — | — | — | 0.4690 ± 0.0103 |
| DINO-WM k=5-res | — | — | — | — | 0.5179 ± 0.0113 |
| chunk k=8 | 0.1510 ± 0.0007† | 0.1740 ± 0.0057† | 0.1976 ± 0.0124† | 0.2209 ± 0.0200† | 0.2441 ± 0.0276† |
| chunk k=8-emb128 | 0.1499 ± 0.0017† | 0.1738 ± 0.0056† | 0.1995 ± 0.0114† | 0.2247 ± 0.0185† | 0.2491 ± 0.0262† |
| chunk k=8-emb32 | 0.1532 ± 0.0018† | 0.1774 ± 0.0024† | 0.2030 ± 0.0030† | 0.2279 ± 0.0029† | 0.2529 ± 0.0019† |
| chunk k=8-res | 0.2188 ± 0.0022 | 0.2533 ± 0.0029 | 0.2872 ± 0.0037 | 0.3196 ± 0.0049 | 0.3529 ± 0.0058 |
| chunk k=8-u2 | 0.1522 ± 0.0018† | 0.1701 ± 0.0042† | 0.1881 ± 0.0072† | 0.2049 ± 0.0096† | 0.2207 ± 0.0117† |

| model | cube cm @40 (Reach) | cube cm @40 (all) |
|---|---|---|
| chunk k=1-res | 8.29 ± 0.39 | 9.29 ± 0.43 |
| chunk k=1-u16 | 5.88 ± 1.80† | 6.00 ± 2.35† |
| chunk k=2-u8 | 3.71 ± 0.10† | 3.79 ± 0.07† |
| chunk k=4-u4 | 3.55 ± 0.06† | 3.59 ± 0.10† |
| chunk k=5-emb10 | 3.92 ± 0.18† | 4.00 ± 0.17† |
| chunk k=5-res | 5.19 ± 0.26 | 5.55 ± 0.23 |
| chunk k=8-emb128 | 3.46 ± 0.32† | 3.52 ± 0.27† |
| chunk k=8-emb32 | 3.54 ± 0.31† | 3.53 ± 0.23† |
| chunk k=8-res | 4.21 ± 0.20 | 4.45 ± 0.17 |
| chunk k=8-u2 | 3.26 ± 0.04† | 3.36 ± 0.03† |
| DINO-WM k=5-e64 | 3.63 ± 0.18† | 3.76 ± 0.12† |
| DINO-WM k=5-res | 5.06 ± 0.30 | 5.47 ± 0.25 |
| chunk k=1 | 6.49 ± 1.57† | 6.52 ± 1.07† |
| chunk k=2 | 4.52 ± 0.18† | 4.55 ± 0.22† |
| chunk k=4 | 3.98 ± 0.30† | 4.12 ± 0.21† |
| chunk k=5 | 3.94 ± 0.26† | 3.92 ± 0.25† |
| chunk k=8 | 3.43 ± 0.02† | 3.46 ± 0.05† |
| DINO-WM k=5 | 4.05 ± 0.57† | 4.08 ± 0.45† |

## E2 · planning success by goal offset (env steps): rate [95% Wilson interval] (pairs)
A rate only means something next to the random-action row and its interval; with n < 20 pairs the intervals overlap for almost any two models.

| model | d=16 | d=24 | d=40 |
|---|---|---|---|
| expert replay (pipeline check) | 93% [79-98] (30) | 97% [83-99] (30) | 87% [70-95] (30) |
| random actions (null planner) | 3% [1-17] (30) | 3% [1-17] (30) | 3% [1-17] (30) |
| chunk k=1 · probe_cube cost | 10% [3-26] (30) | 10% [3-26] (30) | 8% [4-15] (90) |
| chunk k=2 · probe_cube cost | 10% [3-26] (30) | 10% [3-26] (30) | 10% [5-18] (90) |
| chunk k=4 · probe_cube cost | 13% [5-30] (30) | 10% [3-26] (30) | 6% [2-12] (90) |
| chunk k=5 · probe_cube cost | — | — | 7% [2-21] (30) |
| chunk k=8 · probe_cube cost | 10% [3-26] (30) | 13% [5-30] (30) | 10% [5-18] (90) |
| DINO-WM k=5 · probe_cube cost | — | — | 7% [2-21] (30) |
| chunk k=1 | 10% [3-26] (30) | 10% [3-26] (30) | 10% [3-26] (30) |
| chunk k=2 | 13% [5-30] (30) | 7% [2-21] (30) | 10% [3-26] (30) |
| chunk k=4 | 10% [3-26] (30) | 17% [7-34] (30) | 10% [3-26] (30) |
| chunk k=5 | — | — | 3% [1-17] (30) |
| chunk k=8 | 10% [3-26] (30) | 13% [5-30] (30) | 10% [3-26] (30) |
| DINO-WM k=5 | — | — | 13% [5-30] (30) |
| chunk k=8 · probe_cube cost · 300x30 search | — | — | 13% [5-30] (30) |
| chunk k=8 · 300x30 search | — | — | 7% [2-21] (30) |
| chunk k=1 · probe_cube_rand cost | — | — | 11% [6-19] (90) |
| chunk k=2 · probe_cube_rand cost | — | — | 8% [4-15] (90) |
| chunk k=4 · probe_cube_rand cost | — | — | 7% [3-14] (90) |
| chunk k=8 · probe_cube_rand cost | — | — | 10% [5-18] (90) |
| simulator as model (perfect predictions) · true_cube cost | 33% [19-51] (30) | 43% [27-61] (30) | 77% [59-88] (30) |
| simulator as model (perfect predictions) · latent cost | — | — | 20% [6-51] (10) |
| simulator as model (perfect predictions) · probe_cube cost | — | — | 20% [6-51] (10) |
| simulator as model (perfect predictions) · probe_cube_rand cost | — | — | 40% [22-61] (20) |

## E2 · progress toward the goal (cm; start distance minus final / closest distance)
| model | start | final | closest | progress (final) | progress (closest) | pairs |
|---|---|---|---|---|---|---|
| expert replay (pipeline check) | 20.0 | 3.4 | 3.2 | +16.6 | +16.8 | 90 |
| random actions (null planner) | 20.0 | 19.0 | 17.2 | +1.0 | +2.8 | 90 |
| chunk k=1 · probe_cube cost | 21.0 | 17.8 | 16.9 | +3.2 | +4.2 | 150 |
| chunk k=2 · probe_cube cost | 21.0 | 17.6 | 16.9 | +3.5 | +4.1 | 150 |
| chunk k=4 · probe_cube cost | 21.0 | 17.5 | 17.0 | +3.6 | +4.1 | 150 |
| chunk k=5 · probe_cube cost | 22.6 | 18.7 | 18.5 | +3.9 | +4.1 | 30 |
| chunk k=8 · probe_cube cost | 21.0 | 17.1 | 16.7 | +3.9 | +4.4 | 150 |
| DINO-WM k=5 · probe_cube cost | 22.6 | 18.4 | 17.9 | +4.2 | +4.7 | 30 |
| chunk k=1 | 20.0 | 16.8 | 16.3 | +3.2 | +3.7 | 90 |
| chunk k=2 | 20.0 | 16.8 | 16.4 | +3.2 | +3.6 | 90 |
| chunk k=4 | 20.0 | 16.4 | 16.1 | +3.6 | +3.9 | 90 |
| chunk k=5 | 22.6 | 19.8 | 19.3 | +2.8 | +3.3 | 30 |
| chunk k=8 | 20.0 | 16.5 | 15.9 | +3.5 | +4.1 | 90 |
| DINO-WM k=5 | 22.6 | 18.7 | 18.4 | +3.9 | +4.2 | 30 |
| chunk k=8 · probe_cube cost · 300x30 search | 22.6 | 18.7 | 18.2 | +3.9 | +4.4 | 30 |
| chunk k=8 · 300x30 search | 22.6 | 19.0 | 18.5 | +3.6 | +4.1 | 30 |
| chunk k=1 · probe_cube_rand cost | 22.6 | 18.8 | 18.2 | +3.8 | +4.4 | 90 |
| chunk k=2 · probe_cube_rand cost | 22.6 | 19.1 | 18.2 | +3.5 | +4.3 | 90 |
| chunk k=4 · probe_cube_rand cost | 22.6 | 18.8 | 18.2 | +3.8 | +4.4 | 90 |
| chunk k=8 · probe_cube_rand cost | 22.6 | 18.5 | 18.1 | +4.1 | +4.5 | 90 |
| simulator as model (perfect predictions) · true_cube cost | 20.0 | 7.8 | 7.6 | +12.1 | +12.4 | 90 |
| simulator as model (perfect predictions) · latent cost | 19.4 | 14.6 | 14.2 | +4.9 | +5.2 | 10 |
| simulator as model (perfect predictions) · probe_cube cost | 19.4 | 10.8 | 9.5 | +8.6 | +9.9 | 10 |
| simulator as model (perfect predictions) · probe_cube_rand cost | 22.1 | 10.0 | 9.8 | +12.1 | +12.3 | 20 |

## E3 · cost per plan (same CEM budget, horizon and replanning for all)
For the simulator planner, calls = simulator steps per candidate path.

| model | model calls / plan | sec / plan |
|---|---|---|
| chunk k=1 · probe_cube cost | 400 | 4.80 |
| chunk k=2 · probe_cube cost | 200 | 2.96 |
| chunk k=4 · probe_cube cost | 100 | 1.44 |
| chunk k=5 · probe_cube cost | 80 | 0.42 |
| chunk k=8 · probe_cube cost | 50 | 0.69 |
| DINO-WM k=5 · probe_cube cost | 80 | 0.58 |
| chunk k=1 | 400 | 1.81 |
| chunk k=2 | 200 | 0.90 |
| chunk k=4 | 100 | 0.45 |
| chunk k=5 | 80 | 0.41 |
| chunk k=8 | 50 | 0.24 |
| DINO-WM k=5 | 80 | 0.58 |
| chunk k=8 · probe_cube cost · 300x30 search | 150 | 3.61 |
| chunk k=8 · 300x30 search | 150 | 3.61 |
| chunk k=1 · probe_cube_rand cost | 400 | 12.43 |
| chunk k=2 · probe_cube_rand cost | 200 | 5.52 |
| chunk k=4 · probe_cube_rand cost | 100 | 1.68 |
| chunk k=8 · probe_cube_rand cost | 50 | 0.78 |
| simulator as model (perfect predictions) · true_cube cost | 400 | 71.75 |
| simulator as model (perfect predictions) · latent cost | 400 | 106.77 |
| simulator as model (perfect predictions) · probe_cube cost | 400 | 110.42 |
| simulator as model (perfect predictions) · probe_cube_rand cost | 400 | 137.14 |
