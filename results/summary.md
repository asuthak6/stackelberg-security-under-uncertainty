# Results summary

**E1 main grid.** Mean ± 95% t-interval half-width across seeds, 50 seeds x 500 rounds per cell.

| defender | attacker | attacker_payoff | defender_payoff | catch_rate | takedown_rate | attack_share | undefended_attack_share | undefended_catch_rate | perception_gap |
|---|---|---|---|---|---|---|---|---|---|
| sse | random | -0.830 ± 0.076 | +0.034 ± 0.076 | +0.250 ± 0.003 | +0.301 ± 0.004 | +0.985 ± 0.002 | +0.354 ± 0.004 | +0.707 ± 0.006 | +1.216 ± 0.089 |
| sse | naive | +0.414 ± 0.073 | -1.414 ± 0.073 | +0.263 ± 0.007 | +0.435 ± 0.006 | +1.000 ± 0.000 | +0.333 ± 0.009 | +0.794 ± 0.021 | -0.028 ± 0.093 |
| sse | bayesian | +0.283 ± 0.082 | -1.282 ± 0.082 | +0.284 ± 0.005 | +0.389 ± 0.005 | +0.999 ± 0.000 | +0.317 ± 0.006 | +0.898 ± 0.011 | +0.103 ± 0.101 |
| sse | bayesian_uniform | +0.348 ± 0.082 | -1.241 ± 0.088 | +0.295 ± 0.007 | +0.417 ± 0.008 | +0.897 ± 0.011 | +0.319 ± 0.008 | +0.925 ± 0.014 | +0.038 ± 0.098 |
| sse | strategic | +0.375 ± 0.084 | -1.375 ± 0.084 | +0.299 ± 0.005 | +0.396 ± 0.006 | +1.000 ± 0.000 | +0.299 ± 0.005 | +1.000 ± 0.000 | +0.011 ± 0.103 |
| sse | strategic_uniform | +0.372 ± 0.082 | -1.240 ± 0.109 | +0.318 ± 0.007 | +0.397 ± 0.007 | +0.868 ± 0.047 | +0.318 ± 0.007 | +1.000 ± 0.000 | +0.014 ± 0.095 |
| sse | omniscient | +0.386 ± 0.042 | -1.136 ± 0.042 | +0.148 ± 0.004 | +0.575 ± 0.006 | +1.000 ± 0.000 | +0.566 ± 0.006 | +0.261 ± 0.007 | +0.000 ± 0.000 |
| adaptive | random | -0.504 ± 0.073 | -0.292 ± 0.073 | +0.251 ± 0.003 | +0.308 ± 0.004 | +0.985 ± 0.002 | +0.419 ± 0.003 | +0.600 ± 0.006 | +1.647 ± 0.102 |
| adaptive | naive | -1.608 ± 0.086 | +0.764 ± 0.085 | +0.503 ± 0.008 | +0.305 ± 0.007 | +0.844 ± 0.016 | +0.590 ± 0.008 | +0.852 ± 0.007 | +2.751 ± 0.108 |
| adaptive | bayesian | -0.175 ± 0.076 | -0.789 ± 0.078 | +0.336 ± 0.007 | +0.393 ± 0.005 | +0.966 ± 0.005 | +0.445 ± 0.007 | +0.755 ± 0.009 | +1.318 ± 0.094 |
| adaptive | bayesian_uniform | -0.111 ± 0.052 | -0.357 ± 0.062 | +0.357 ± 0.008 | +0.399 ± 0.010 | +0.472 ± 0.016 | +0.462 ± 0.008 | +0.772 ± 0.011 | +1.254 ± 0.095 |
| adaptive | strategic | +0.623 ± 0.081 | -1.622 ± 0.081 | +0.284 ± 0.005 | +0.442 ± 0.006 | +1.000 ± 0.000 | +0.380 ± 0.005 | +0.747 ± 0.008 | +0.521 ± 0.091 |
| adaptive | strategic_uniform | -0.005 ± 0.009 | -0.019 ± 0.012 | +0.427 ± 0.036 | +0.337 ± 0.044 | +0.024 ± 0.006 | +0.500 ± 0.039 | +0.864 ± 0.043 | +1.148 ± 0.076 |
| adaptive | omniscient | +1.143 ± 0.076 | -2.143 ± 0.076 | +0.217 ± 0.005 | +0.485 ± 0.006 | +1.000 ± 0.000 | +0.362 ± 0.006 | +0.598 ± 0.009 | +0.000 ± 0.000 |

**Paired comparisons.** Seed-paired differences (first minus second), mean and 95% t-interval, 50 seeds x 500 rounds per cell.

| comparison | defender | attacker | metric | mean diff | 95% CI | seeds first higher | seeds second higher | ties |
|---|---|---|---|---|---|---|---|---|
| strategic - naive | sse | strategic vs naive | attacker_payoff | -0.039 | [-0.126, +0.048] | 20 | 29 | 1 |
| strategic - naive | adaptive | strategic vs naive | attacker_payoff | +2.231 | [+2.111, +2.350] | 50 | 0 | 0 |
| bayesian - naive | sse | bayesian vs naive | attacker_payoff | -0.131 | [-0.210, -0.051] | 17 | 33 | 0 |
| bayesian - naive | adaptive | bayesian vs naive | attacker_payoff | +1.433 | [+1.315, +1.551] | 50 | 0 | 0 |
| strategic - bayesian | sse | strategic vs bayesian | attacker_payoff | +0.091 | [-0.002, +0.185] | 30 | 20 | 0 |
| strategic - bayesian | adaptive | strategic vs bayesian | attacker_payoff | +0.797 | [+0.700, +0.895] | 50 | 0 | 0 |
| sse - adaptive | sse vs adaptive | random | defender_payoff | +0.326 | [+0.259, +0.394] | 48 | 2 | 0 |
| sse - adaptive | sse vs adaptive | naive | defender_payoff | -2.178 | [-2.288, -2.068] | 0 | 50 | 0 |
| sse - adaptive | sse vs adaptive | bayesian | defender_payoff | -0.493 | [-0.609, -0.377] | 7 | 43 | 0 |
| sse - adaptive | sse vs adaptive | bayesian_uniform | defender_payoff | -0.884 | [-0.997, -0.771] | 1 | 49 | 0 |
| sse - adaptive | sse vs adaptive | strategic | defender_payoff | +0.248 | [+0.142, +0.353] | 38 | 12 | 0 |
| sse - adaptive | sse vs adaptive | strategic_uniform | defender_payoff | -1.221 | [-1.331, -1.112] | 0 | 50 | 0 |
| sse - adaptive | sse vs adaptive | omniscient | defender_payoff | +1.007 | [+0.917, +1.098] | 50 | 0 | 0 |
| attacker - equilibrium value | sse | random | attacker_payoff | -1.215 | [-1.290, -1.139] | 0 | 50 | 0 |
| attacker - equilibrium value | sse | naive | attacker_payoff | +0.030 | [-0.043, +0.103] | 27 | 23 | 0 |
| attacker - equilibrium value | sse | bayesian | attacker_payoff | -0.101 | [-0.183, -0.019] | 21 | 29 | 0 |
| attacker - equilibrium value | sse | bayesian_uniform | attacker_payoff | -0.036 | [-0.118, +0.046] | 21 | 29 | 0 |
| attacker - equilibrium value | sse | strategic | attacker_payoff | -0.010 | [-0.094, +0.075] | 25 | 25 | 0 |
| attacker - equilibrium value | sse | strategic_uniform | attacker_payoff | -0.012 | [-0.095, +0.070] | 19 | 31 | 0 |
| attacker - equilibrium value | sse | omniscient | attacker_payoff | +0.001 | [-0.041, +0.044] | 31 | 19 | 0 |

**E3 value of traps.** Defender payoff (mean ± 95% half-width) by number of traps T, with the SSE equilibrium values re-solved per T, 50 seeds x 500 rounds per cell.

| T | defender | attacker | defender payoff | SSE defender value | SSE attacker value |
|---|---|---|---|---|---|
| 0 | sse | naive | -6.822 ± 0.090 | -6.793 | +5.793 |
| 0 | sse | bayesian | -6.822 ± 0.090 | -6.793 | +5.793 |
| 0 | sse | strategic | -6.822 ± 0.090 | -6.793 | +5.793 |
| 0 | sse | omniscient | -6.841 ± 0.097 | -6.793 | +5.793 |
| 0 | adaptive | naive | -7.923 ± 0.106 | -6.793 | +5.793 |
| 0 | adaptive | bayesian | -7.923 ± 0.106 | -6.793 | +5.793 |
| 0 | adaptive | strategic | -7.923 ± 0.106 | -6.793 | +5.793 |
| 0 | adaptive | omniscient | -7.931 ± 0.107 | -6.793 | +5.793 |
| 1 | sse | naive | -1.414 ± 0.073 | -1.134 | +0.384 |
| 1 | sse | bayesian | -1.282 ± 0.082 | -1.134 | +0.384 |
| 1 | sse | strategic | -1.375 ± 0.084 | -1.134 | +0.384 |
| 1 | sse | omniscient | -1.136 ± 0.042 | -1.134 | +0.384 |
| 1 | adaptive | naive | +0.764 ± 0.085 | -1.134 | +0.384 |
| 1 | adaptive | bayesian | -0.789 ± 0.078 | -1.134 | +0.384 |
| 1 | adaptive | strategic | -1.622 ± 0.081 | -1.134 | +0.384 |
| 1 | adaptive | omniscient | -2.143 ± 0.076 | -1.134 | +0.384 |
| 2 | sse | naive | -0.478 ± 0.086 | +0.000 | +0.000 |
| 2 | sse | bayesian | -0.026 ± 0.084 | +0.000 | +0.000 |
| 2 | sse | strategic | -0.007 ± 0.004 | +0.000 | +0.000 |
| 2 | sse | omniscient | +0.000 ± 0.000 | +0.000 | +0.000 |
| 2 | adaptive | naive | +0.294 ± 0.029 | +0.000 | +0.000 |
| 2 | adaptive | bayesian | +0.555 ± 0.024 | +0.000 | +0.000 |
| 2 | adaptive | strategic | +0.008 ± 0.004 | +0.000 | +0.000 |
| 2 | adaptive | omniscient | +0.000 ± 0.000 | +0.000 | +0.000 |

**E6 memory robustness.** Attacker payoff (mean ± 95% half-width) against the adaptive defender by memory window W, 50 seeds x 500 rounds per cell.

| W | attacker | attacker payoff |
|---|---|---|
| 10 | naive | -0.281 ± 0.068 |
| 10 | bayesian | -0.433 ± 0.079 |
| 10 | strategic | +0.028 ± 0.072 |
| 50 | naive | -1.608 ± 0.086 |
| 50 | bayesian | -0.175 ± 0.076 |
| 50 | strategic | +0.623 ± 0.081 |
| 200 | naive | -0.704 ± 0.083 |
| 200 | bayesian | +0.031 ± 0.066 |
| 200 | strategic | +0.830 ± 0.073 |
