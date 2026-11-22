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

**E7 misspecified strategic attacker.** Strategic attacker whose model of the adaptive defender assumes smoothing rate model_rate (the defender's true rate stays 0.2; floor unchanged). Mean ± 95% half-width; paired columns are seed-paired differences in attacker payoff (strategic minus baseline, E1 runs with the same seeds) with 95% t-interval and seeds strategic higher / baseline higher / tied, 50 seeds x 500 rounds per cell.

| defender | model_rate | attacker payoff | catch rate | attack share | vs naive: diff | 95% CI | won/lost/tied | vs bayesian: diff | 95% CI | won/lost/tied |
|---|---|---|---|---|---|---|---|---|---|---|
| sse | 0.05 | +0.351 ± 0.084 | +0.304 ± 0.006 | +1.000 ± 0.000 | -0.063 | [-0.146, +0.019] | 21/29/0 | +0.067 | [-0.013, +0.148] | 29/21/0 |
| sse | 0.1 | +0.311 ± 0.073 | +0.307 ± 0.005 | +1.000 ± 0.000 | -0.103 | [-0.175, -0.030] | 16/33/1 | +0.028 | [-0.063, +0.119] | 26/23/1 |
| sse | 0.2 (true) | +0.375 ± 0.084 | +0.299 ± 0.005 | +1.000 ± 0.000 | -0.039 | [-0.126, +0.048] | 20/29/1 | +0.091 | [-0.002, +0.185] | 30/20/0 |
| sse | 0.4 | +0.359 ± 0.082 | +0.299 ± 0.006 | +1.000 ± 0.000 | -0.055 | [-0.133, +0.024] | 23/27/0 | +0.076 | [-0.027, +0.180] | 27/23/0 |
| sse | 0.8 | +0.171 ± 0.074 | +0.300 ± 0.004 | +1.000 ± 0.000 | -0.243 | [-0.330, -0.156] | 12/38/0 | -0.112 | [-0.206, -0.018] | 16/34/0 |
| adaptive | 0.05 | +0.468 ± 0.075 | +0.299 ± 0.005 | +1.000 ± 0.000 | +2.076 | [+1.961, +2.191] | 50/0/0 | +0.643 | [+0.542, +0.744] | 48/2/0 |
| adaptive | 0.1 | +0.522 ± 0.069 | +0.292 ± 0.005 | +1.000 ± 0.000 | +2.130 | [+2.030, +2.231] | 50/0/0 | +0.697 | [+0.605, +0.790] | 50/0/0 |
| adaptive | 0.2 (true) | +0.623 ± 0.081 | +0.284 ± 0.005 | +1.000 ± 0.000 | +2.231 | [+2.111, +2.350] | 50/0/0 | +0.797 | [+0.700, +0.895] | 50/0/0 |
| adaptive | 0.4 | +0.638 ± 0.076 | +0.283 ± 0.005 | +1.000 ± 0.000 | +2.246 | [+2.128, +2.364] | 50/0/0 | +0.813 | [+0.703, +0.923] | 49/1/0 |
| adaptive | 0.8 | +0.246 ± 0.079 | +0.304 ± 0.005 | +1.000 ± 0.000 | +1.854 | [+1.739, +1.968] | 50/0/0 | +0.421 | [+0.339, +0.502] | 44/6/0 |

E7 reference values: attacker payoff of the baselines against the same defender (E1).

| defender | naive payoff | bayesian payoff | omniscient payoff |
|---|---|---|---|
| sse | +0.414 ± 0.073 | +0.283 ± 0.082 | +0.386 ± 0.042 |
| adaptive | -1.608 ± 0.086 | -0.175 ± 0.076 | +1.143 ± 0.076 |

**E8 target selection.** Per server: share of all attack units sent to it (unit share) and share of rounds in which it was attacked (attack rate); mean ± 95% half-width across seeds (E1 runs), 50 seeds x 500 rounds per cell. Seeds with no attack at all have no unit share and are left out of that mean. The last row is the attacker's response in the Strong Stackelberg Equilibrium.

| defender | attacker | unit share s1 (v=10) | unit share s2 (v=8) | unit share s3 (v=6) | unit share s4 (v=4) | attack rate s1 (v=10) | attack rate s2 (v=8) | attack rate s3 (v=6) | attack rate s4 (v=4) |
|---|---|---|---|---|---|---|---|---|---|
| sse | random | +0.249 ± 0.004 | +0.248 ± 0.004 | +0.252 ± 0.003 | +0.251 ± 0.004 | +0.498 ± 0.007 | +0.494 ± 0.006 | +0.500 ± 0.006 | +0.497 ± 0.006 |
| sse | naive | +0.344 ± 0.031 | +0.300 ± 0.028 | +0.224 ± 0.024 | +0.132 ± 0.016 | +0.359 ± 0.032 | +0.321 ± 0.029 | +0.253 ± 0.026 | +0.187 ± 0.020 |
| sse | bayesian | +0.456 ± 0.017 | +0.305 ± 0.014 | +0.186 ± 0.014 | +0.053 ± 0.005 | +0.510 ± 0.018 | +0.370 ± 0.015 | +0.251 ± 0.017 | +0.101 ± 0.010 |
| sse | bayesian_uniform | +0.498 ± 0.030 | +0.290 ± 0.024 | +0.158 ± 0.020 | +0.054 ± 0.010 | +0.449 ± 0.027 | +0.264 ± 0.022 | +0.146 ± 0.019 | +0.058 ± 0.011 |
| sse | strategic | +0.502 ± 0.017 | +0.336 ± 0.015 | +0.162 ± 0.011 | +0.000 ± 0.000 | +0.522 ± 0.017 | +0.375 ± 0.015 | +0.194 ± 0.014 | +0.000 ± 0.000 |
| sse | strategic_uniform | +0.608 ± 0.029 | +0.356 ± 0.026 | +0.036 ± 0.014 | +0.000 ± 0.000 | +0.529 ± 0.038 | +0.311 ± 0.027 | +0.031 ± 0.012 | +0.000 ± 0.000 |
| sse | omniscient | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +1.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +0.000 ± 0.000 | +1.000 ± 0.000 |
| adaptive | random | +0.249 ± 0.004 | +0.248 ± 0.004 | +0.252 ± 0.003 | +0.251 ± 0.004 | +0.498 ± 0.007 | +0.494 ± 0.006 | +0.500 ± 0.006 | +0.497 ± 0.006 |
| adaptive | naive | +0.418 ± 0.008 | +0.325 ± 0.009 | +0.165 ± 0.008 | +0.091 ± 0.005 | +0.359 ± 0.011 | +0.281 ± 0.008 | +0.147 ± 0.007 | +0.089 ± 0.005 |
| adaptive | bayesian | +0.402 ± 0.005 | +0.311 ± 0.005 | +0.209 ± 0.006 | +0.078 ± 0.004 | +0.412 ± 0.006 | +0.331 ± 0.006 | +0.235 ± 0.007 | +0.095 ± 0.005 |
| adaptive | bayesian_uniform | +0.410 ± 0.010 | +0.319 ± 0.009 | +0.205 ± 0.010 | +0.066 ± 0.006 | +0.193 ± 0.006 | +0.151 ± 0.007 | +0.098 ± 0.007 | +0.035 ± 0.004 |
| adaptive | strategic | +0.423 ± 0.004 | +0.338 ± 0.004 | +0.226 ± 0.004 | +0.013 ± 0.002 | +0.424 ± 0.004 | +0.341 ± 0.004 | +0.229 ± 0.004 | +0.013 ± 0.003 |
| adaptive | strategic_uniform | +0.680 ± 0.032 | +0.292 ± 0.029 | +0.028 ± 0.010 | +0.000 ± 0.000 | +0.015 ± 0.003 | +0.008 ± 0.003 | +0.001 ± 0.000 | +0.000 ± 0.000 |
| adaptive | omniscient | +0.362 ± 0.000 | +0.305 ± 0.000 | +0.222 ± 0.000 | +0.111 ± 0.000 | +0.362 ± 0.000 | +0.306 ± 0.000 | +0.250 ± 0.000 | +0.138 ± 0.000 |
| (equilibrium) | SSE response (0, 0, 0, 3) | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |

**E9 naive memory diagnostic.** Naive attacker against the adaptive defender by memory W. Belief error: mean over rounds and servers of |naive estimate of P(trap | defense 0) - true P(trap on server | server undefended)| under the defender's current mixed strategy, measured before each round; 'tested' restricts it to cells with at least one remembered attack. Unsafe untested rounds: share of rounds with an untested (so assumed safe) level-0 cell whose true trap probability is above 0.5. Mean ± 95% half-width, 50 seeds x 500 rounds per cell; runs are identical to E6.

| W | belief_error | belief_error_tested | untested_cell_share | unsafe_untested_round_share | undefended_attack_share | undefended_catch_rate | attacker_payoff |
|---|---|---|---|---|---|---|---|
| 10 | +0.386 ± 0.003 | +0.360 ± 0.003 | +0.163 ± 0.003 | +0.321 ± 0.008 | +0.454 ± 0.005 | +0.668 ± 0.008 | -0.281 ± 0.068 |
| 50 | +0.394 ± 0.006 | +0.392 ± 0.006 | +0.028 ± 0.001 | +0.047 ± 0.004 | +0.590 ± 0.008 | +0.852 ± 0.007 | -1.608 ± 0.086 |
| 200 | +0.387 ± 0.012 | +0.385 ± 0.012 | +0.013 ± 0.001 | +0.020 ± 0.003 | +0.573 ± 0.014 | +0.866 ± 0.009 | -0.704 ± 0.083 |


**E10 learned strategic attacker.** strategic_learned (level 2, not given the defender's rule; learns trap probability per server, defended?, and its own attack-pressure bin). Mean ± 95% half-width, 50 seeds x 500 rounds per cell.

| W | defender | attacker_payoff | defender_payoff | catch_rate | attack_share | undefended_attack_share | undefended_catch_rate |
|---|---|---|---|---|---|---|---|
| 50 | sse | +0.133 ± 0.079 | -1.132 ± 0.079 | +0.284 ± 0.005 | +0.999 ± 0.000 | +0.311 ± 0.005 | +0.915 ± 0.009 |
| 50 | adaptive | +0.089 ± 0.086 | -1.084 ± 0.086 | +0.298 ± 0.004 | +0.995 ± 0.001 | +0.411 ± 0.004 | +0.725 ± 0.007 |
| 200 | adaptive | +0.537 ± 0.088 | -1.534 ± 0.088 | +0.267 ± 0.005 | +0.997 ± 0.001 | +0.399 ± 0.005 | +0.670 ± 0.009 |

**E10 paired comparisons.** Seed-paired differences in attacker payoff (strategic_learned minus baseline, same seeds and random streams), mean and 95% t-interval, and seeds strategic_learned higher / baseline higher / tied. W = 50 baselines are the E1 runs; W = 200 baselines are the E6 runs (naive, bayesian, strategic only), 50 seeds x 500 rounds per cell.

| W | defender | comparison | strategic_learned payoff | baseline payoff | mean diff | 95% CI | won/lost/tied |
|---|---|---|---|---|---|---|---|
| 50 | sse | strategic_learned - naive | +0.133 | +0.414 | -0.281 | [-0.359, -0.203] | 10/40/0 |
| 50 | sse | strategic_learned - bayesian | +0.133 | +0.283 | -0.150 | [-0.247, -0.053] | 16/34/0 |
| 50 | sse | strategic_learned - strategic | +0.133 | +0.375 | -0.242 | [-0.348, -0.136] | 13/37/0 |
| 50 | sse | strategic_learned - omniscient | +0.133 | +0.386 | -0.253 | [-0.345, -0.161] | 12/38/0 |
| 50 | adaptive | strategic_learned - naive | +0.089 | -1.608 | +1.697 | [+1.586, +1.808] | 50/0/0 |
| 50 | adaptive | strategic_learned - bayesian | +0.089 | -0.175 | +0.264 | [+0.163, +0.365] | 37/13/0 |
| 50 | adaptive | strategic_learned - strategic | +0.089 | +0.623 | -0.533 | [-0.644, -0.422] | 4/46/0 |
| 50 | adaptive | strategic_learned - omniscient | +0.089 | +1.143 | -1.054 | [-1.155, -0.952] | 0/50/0 |
| 200 | adaptive | strategic_learned - naive | +0.537 | -0.704 | +1.242 | [+1.128, +1.355] | 50/0/0 |
| 200 | adaptive | strategic_learned - bayesian | +0.537 | +0.031 | +0.507 | [+0.395, +0.618] | 45/5/0 |
| 200 | adaptive | strategic_learned - strategic | +0.537 | +0.830 | -0.292 | [-0.400, -0.185] | 13/37/0 |

**E10 learned dependence.** Posterior mean trap probability of undefended cells at the end of each run, averaged over servers with at least one remembered observation in that pressure bin; mean ± 95% half-width across the n runs that had such a cell. The prior mean is T/N. 50 seeds x 500 rounds per cell.

| W | defender | low pressure | medium pressure | high pressure |
|---|---|---|---|---|
| 50 | sse | +0.404 ± 0.017 (n = 50) | +0.500 ± 0.016 (n = 50) | +0.623 ± 0.020 (n = 50) |
| 50 | adaptive | +0.324 ± 0.021 (n = 50) | +0.469 ± 0.019 (n = 50) | +0.575 ± 0.023 (n = 50) |
| 200 | adaptive | +0.377 ± 0.015 (n = 50) | +0.550 ± 0.013 (n = 50) | +0.611 ± 0.016 (n = 50) |


**E10 within-server learned dependence.** For each server, the learned P(trap | undefended) (posterior mean at the end of the run) in one pressure bin minus that in another, using only servers with at least one remembered observation in both bins; averaged over those servers per run, then mean ± 95% half-width across the n runs with at least one such server (mean number of servers per run in brackets). Same seeds and streams as E10; the W = 200 SSE runs exist only for this table. 50 seeds x 500 rounds per cell.

| W | defender | medium - low | high - medium | high - low |
|---|---|---|---|---|
| 50 | sse | +0.046 ± 0.023 (n = 48; 2.0 servers) | +0.076 ± 0.027 (n = 49; 2.0 servers) | +0.082 ± 0.035 (n = 42; 1.2 servers) |
| 50 | adaptive | +0.089 ± 0.033 (n = 49; 2.0 servers) | +0.092 ± 0.029 (n = 50; 2.4 servers) | +0.137 ± 0.053 (n = 42; 1.4 servers) |
| 200 | sse | +0.079 ± 0.020 (n = 50; 3.6 servers) | +0.023 ± 0.020 (n = 50; 2.9 servers) | +0.097 ± 0.033 (n = 50; 2.8 servers) |
| 200 | adaptive | +0.165 ± 0.023 (n = 50; 3.8 servers) | +0.013 ± 0.018 (n = 50; 3.2 servers) | +0.197 ± 0.023 (n = 50; 3.1 servers) |
