# Design: Perception and Reasoning in a Repeated Security Game with Traps

This document is the specification for the project. Code, experiments, and the report follow it.

## 1. Research question

How do an attacker's perception limits and reasoning style affect its success against a defender who protects servers and sets traps, and does better reasoning compensate for limited perception?

| Aspect | Choice |
|---|---|
| Focus of enquiry | Perception and reasoning |
| Domain | Network defence against DDoS-style attacks on servers |
| Formal model | Stackelberg security game with defensive resources and traps (honeypots), played repeatedly |

## 2. Stage game (one round)

### 2.1 Players and actions

- **Servers.** N = 4 servers with values v = (10, 8, 6, 4).
- **Defender (leader).** Commits to a mixed strategy over pure strategies. A pure strategy is a pair:
  - a defense allocation: D = 4 integer units split over the servers (35 allocations), and
  - a trap placement: T = 1 trap on one server (4 placements).

  This gives 35 x 4 = 140 pure strategies. Each round one pure strategy is sampled from the mixed strategy.
- **Attacker (follower).** Chooses an allocation of A = 4 integer attack units over the servers, with hold-back allowed (any units not placed are kept back). This gives 70 allocations, including the all-zero allocation. The attacker commits before seeing the current round's realized defense or traps.

| Strategy space | Definition | Size (defaults) |
|---|---|---|
| Defense allocations | integer d with sum(d) = D | 35 |
| Trap placements | subsets of T servers | 4 |
| Defender pure strategies | defense allocation x trap placement | 140 |
| Attack allocations | integer a with sum(a) <= A | 70 |

### 2.2 Outcome on each attacked server

For each server i with a_i > 0:

1. If server i is trapped, the attacker is **caught** with probability c (c = 1 by default).
2. Otherwise (not trapped, or trapped but not caught), a contest decides the result:
   - **success** (server taken down) with probability a_i / (a_i + d_i + k), with k = 1;
   - **blocked** otherwise.

Servers with a_i = 0 are not attacked and contribute nothing.

| Probability | Trapped server | Untrapped server |
|---|---|---|
| Caught | c | 0 |
| Success | (1 - c) * p_i | p_i |
| Blocked | (1 - c) * (1 - p_i) | 1 - p_i |

where p_i = a_i / (a_i + d_i + k).

### 2.3 Payoffs (general-sum)

Per attacked server i:

| Outcome | Attacker | Defender |
|---|---|---|
| Server taken down | +v_i | -v_i |
| Caught by trap | -C = -6 | +R = +6 |
| Blocked | -b = -1 | +r_b = +1 |
| Effort (always, per attacked server) | -kappa * a_i, kappa = 0.25 | 0 |

The round payoff is the sum over servers. The expected payoff for a pure defender strategy against an attack allocation is computed exactly from the probabilities above; these values form the payoff matrices UA and UD of shape 140 x 70 (rows: defender pure strategies, columns: attack allocations).

### 2.4 Worked example

Default parameters, defense (2, 1, 1, 0), trap on server 4, attack (2, 0, 0, 2).

| Server | Computation | Attacker | Defender |
|---|---|---|---|
| 1 | p = 2 / (2 + 2 + 1) = 0.4; -0.5 + 0.4 * 10 - 0.6 * 1 | 2.9 | -0.4 * 10 + 0.6 * 1 = -3.4 |
| 4 | trapped, c = 1: -0.5 - 6 | -6.5 | +6 |
| Total | | -3.6 | 2.6 |

## 3. Equilibrium (formal solution)

The benchmark solution is the Strong Stackelberg Equilibrium (SSE), computed with the multiple-LP method.

For each attacker allocation j, solve a linear program over the defender's distribution x over the 140 pure strategies:

```
maximize    x . UD[:, j]
subject to  x . UA[:, j] >= x . UA[:, j']   for all j'
            sum(x) = 1
            x >= 0
```

Among the feasible j, keep the one with the highest defender value. The resulting x is the SSE defender strategy and j is the attacker's best response (ties broken in the defender's favour, as in the strong form).

### 3.1 Preliminary solution (default parameters)

| Quantity | Value |
|---|---|
| Defender value | about -1.13 |
| Defender value with no traps (T = 0) | about -6.79 |
| Attacker value | about +0.38 |
| Attacker best response | (0, 0, 0, 3) |
| Trap probability per server | roughly (0.35, 0.28, 0.22, 0.15) |
| Support | 5 of 140 pure strategies |
| P(trap on i given defense 0 on i) | 1 on servers 1-3, about 0.26 on server 4 |
| P(trap on i given defense >= 1 on i) | 0 on every server |

The trapped server is always left undefended, so the equilibrium itself has the lure structure: trap and defense are strongly correlated, and a server that looks soft is often a trap.

The attacker is exactly indifferent between five allocations: all 4 units on any one server, or 3 units on server 4. The SSE value relies on defender-favourable tie-breaking (the attacker plays (0, 0, 0, 3)); any of the other tied responses costs the defender about 0.25 per round with default parameters (defender value about -1.38).

With T = 2 the attacker's best response is to never attack (already at C = 6, and more so with a larger caught penalty), which makes the game degenerate. This is why T = 1 and C = 6 were chosen.

## 4. Repeated game and perception

### 4.1 Round sequence

1. The defender samples a pure strategy from its current mixed strategy.
2. The attacker picks an allocation from its current beliefs.
3. Outcomes are sampled per server.
4. Payoffs are recorded.
5. Both agents observe and update.

Each run uses three independent random streams derived from its seed: defender, attacker, and environment. The environment draws one uniform number per server every round, whether or not the server is attacked, so different attackers face the same environment randomness for the same seed.

### 4.2 Information structure

| Agent | Before acting | After the round |
|---|---|---|
| Attacker | Only its memory of the last W = 50 rounds | Realized defense on every server (surveillance); for each server it attacked, whether that server was trapped (with c = 1, attacking a trapped server always reveals it). It never learns about traps on servers it did not attack. |
| Defender | Its own state | The full attack allocation and the per-server outcomes |

The perception limit under study is therefore on traps: defense is observed after the fact, traps are only observed by attacking them.

## 5. Defenders

### 5.1 SSE defender (control)

Plays the SSE mixed strategy from Section 3 every round. Does not learn.

### 5.2 Adaptive defender (lure rule)

- Keeps an exponentially smoothed attack frequency f_i per server, smoothing rate 0.2, initialized uniform (1/N). After each round, f <- 0.8 f + 0.2 s, where s_i is the share of the round's attack units sent to server i. A round with no attack leaves f unchanged.
- **Trap.** Placed on server i with probability proportional to f_i + 0.05. For T > 1, the T servers are drawn sequentially without replacement with these weights; placement probabilities are computed exactly by enumerating the orderings.
- **Defense.** The D units are placed independently, each on a non-trapped server chosen with probability proportional to v_i (a multinomial). Trapped servers receive no defense and look soft (the lure). With T = 0 every server is eligible.
- It exposes its current mixed strategy exactly as a distribution over the 140 pure strategies (trap distribution times the multinomial distribution over defense allocations given the trap), and samples its pure strategy from that distribution.

## 6. Attackers

### 6.1 Common decision step

Because the equilibrium (Section 3.1) correlates traps with undefended servers, learning attackers estimate trap probability per **cell** (server i, defense level L), not per server. A cell is learned only from remembered rounds in which the attacker attacked server i: in those rounds it knows both the realized defense level on i and whether i was trapped. Learning attackers assume c = 1.

All learning attackers (Naive, Bayesian, Strategic) use the same decision step:

1. Per server, estimate the distribution of defense levels P(d_i = L): counts of each level 0..D over the memory window, with add-one smoothing.
2. Per cell, count attacks_iL (remembered rounds where i was attacked at defense level L) and traps_iL (those where i was trapped), and estimate a trap probability pi_iL (method differs by attacker, below).
3. Compute the expected payoff of every one of the 70 allocations:

   sum over attacked servers i of [ -kappa a_i + sum_L P(d_i = L) ( pi_iL (-C) + (1 - pi_iL) ( p_iL v_i - (1 - p_iL) b ) ) ], with p_iL = a_i / (a_i + L + k).

4. Play the allocation with the highest expected payoff. Ties are broken uniformly at random with the attacker's rng.

### 6.2 Attacker types

| Attacker | Reasoning level | Estimate of pi_iL |
|---|---|---|
| Random | lower bound | None; uniform over the 70 allocations |
| Naive | level 0 | traps_iL / attacks_iL; untested cells count as safe (pi_iL = 0) |
| Bayesian explorer | belief-based | Beta(alpha_0 + traps_iL, beta_0 + attacks_iL - traps_iL) per cell; each round samples pi_iL from the posterior (Thompson sampling), so it retests uncertain cells |
| Strategic | level 2 | Models the defender as adaptive (see below) and averages that prediction with the Bayesian posterior mean of the cell |
| Strategic (learned) | level 2, learned | Not given any defender rule. Beta posterior per cell (server i, defended?, its own attack-pressure bin on i); Thompson sampling (see below) |
| Omniscient | benchmark | Knows the defender's current true mixed strategy and best-responds exactly using the payoff matrix |

**Trap prior.** The number of traps T and servers N are part of the game rules, so the attacker knows the base trap rate T/N. The Bayesian and strategic attackers therefore use an informed prior per cell with mean T/N and strength 4: Beta(alpha_0, beta_0) = Beta(4 T/N, 4 (1 - T/N)), which is Beta(1, 3) with default parameters. If T = 0, every learning attacker sets every pi_iL to 0; if T = N, the Bayesian and strategic attackers set every pi_iL to 1 (the Beta prior is undefined at both extremes). As an ablation, the variants "bayesian_uniform" and "strategic_uniform" use the uninformative prior Beta(1, 1).

**Strategic attacker.** Keeps its own smoothed attack frequency g_i (rate 0.2, initial uniform, same update as the adaptive defender). It predicts the per-server trap probability q_i = min(1, T (g_i + 0.05) / sum_j (g_j + 0.05)). Because it models the lure rule (traps sit on undefended servers), it converts this to cells: predicted pi_i0 = min(1, q_i / P(d_i = 0)) and predicted pi_iL = 0 for L >= 1. Its final pi_iL is the average of the prediction and the Bayesian posterior mean (alpha_0 + traps_iL) / (alpha_0 + beta_0 + attacks_iL).

The smoothing rate and the floor in this model are constructor parameters, `model_rate` (default 0.2) and `model_floor` (default 0.05). They are used only in the attacker's prediction of the defender; the defaults equal the adaptive defender's true constants, so by default the model is correctly specified. E7 varies `model_rate` to test a misspecified model.

**Learned strategic attacker ("strategic_learned").** A level-2 attacker that is not given the defender's rule: it imports nothing from the defender code and assumes no defender rule, and must learn from experience whether and how the defender reacts to its attacks.

- *Own attack pressure.* It keeps an exponentially smoothed share of its own attack units per server, g_i, initial 1/N, updated after each round from its own attack with its own rate 0.3 (`LEARNED_PRESSURE_RATE`; the attacker's own modelling choice, not the defender's rate). A round with no attack leaves g unchanged.
- *Pressure bins.* Low if g_i < 0.15, medium if 0.15 <= g_i <= 0.35, high if g_i > 0.35 (`PRESSURE_LOW`, `PRESSURE_HIGH`). With N = 4 the initial g_i = 0.25 is medium.
- *Cells.* (server i, defended? = defense level >= 1, pressure bin of i at the time the attack was chosen). Each remembered observation stores the bins it was chosen under, so the memory window W works as for the other learners. Counts come from remembered rounds in which i was attacked.
- *Beliefs.* Beta posterior per cell with the same informed prior as the Bayesian attacker, Beta(4 T/N, 4 (1 - T/N)), and the same T = 0 / T = N handling. Requires c = 1.
- *Decision.* Each round it takes each server's current bin, Thompson-samples pi for the (i, undefended) and (i, defended) cells at that bin, sets pi_i0 to the undefended sample and pi_iL (L >= 1) to the defended sample, and applies the common decision step (Section 6.1) with the same defense-level estimate and tie-breaking.

If traps follow its attacks (as with the adaptive defender), its high-pressure undefended cells should learn a high trap probability, so it backs off servers it has been hitting while they look soft. If traps do not depend on its attacks (SSE defender), the bins should look alike.

**Omniscient attacker.** Computes the expected attacker payoff of every allocation under the defender's current mixed strategy. Among allocations within 1e-9 of the best attacker value, it picks the one with the highest defender value; remaining ties are broken at random. This defender-favourable tie-breaking reproduces the Strong Stackelberg Equilibrium against the SSE defender.

### 6.3 Level-k framing

| Level | Agent | Reasoning |
|---|---|---|
| 0 | Naive attacker | Goes where defense looks weak |
| 1 | Adaptive defender | Traps where the attacker goes and keeps that server looking weak |
| 2 | Strategic attacker | Expects the level-1 defender and treats weak-looking, frequently hit servers as likely traps |
| 2 (learned) | Learned strategic attacker | Not told the level-1 rule; learns from experience whether servers it has been hitting hard are trapped more often when they look soft |

## 7. Solver

| Component | Method |
|---|---|
| Strategy enumeration | Integer compositions; trap subsets by itertools.combinations order |
| Payoff matrices | Exact expected payoffs, 140 x 70, cached per configuration |
| SSE | Multiple-LP method (Section 3), one LP per attacker allocation, solved with scipy.optimize.linprog (HiGHS) |
| Attacker best response (belief-based) | Exact expected payoff of all 70 allocations under per-cell trap beliefs (Section 6.1) |
| Omniscient best response | argmax over columns of x . UA for the defender's current mixed strategy x |

## 8. Experiments

Implemented in `security_game/experiments.py` and run with `python -m security_game.experiments`. Defaults: 50 seeds x 500 rounds per cell (`--quick`: 3 seeds x 60 rounds). Seed s is used for every attacker/defender pair (paired design with common random numbers, Section 4.1), so differences between attackers can be compared seed by seed. All intervals are 95% t-intervals across seeds. Outputs go to `results/` (Section 8.5).

### 8.1 Agents and metrics

Attackers: random, naive, bayesian, bayesian_uniform, strategic, strategic_uniform, omniscient (E1 grid); strategic_learned (E10 only). Defenders: sse, adaptive. Per-seed metrics:

| Metric | Definition |
|---|---|
| Attacker payoff | Average attacker payoff per round |
| Defender payoff | Average defender payoff per round |
| Catch rate | Catches per attacked server |
| Takedown rate | Successful takedowns per attacked server |
| Attack share | Share of rounds with any attack |
| Undefended attack share | Share of attacked servers that had 0 realized defense |
| Undefended catch rate | Catches per attack on a 0-defense server (NaN if none; seeds with NaN are left out of the mean) |
| Perception gap | Omniscient attacker payoff minus this attacker's payoff, same defender and same seed |

`summarize()` also reports the defended takedown rate (takedowns per attack on a defended server); it is shown by the game CLI and not used in the experiment tables.

With `per_server=True` (used by the experiments), `summarize()` also reports, for each server i = 1..N, `attack_unit_share_s{i}` (share of all attack units over the run sent to server i; NaN if the attacker never attacked) and `attack_rate_s{i}` (share of rounds in which server i was attacked at all). These are used in E8.

### 8.2 Experiments

| ID | Question | Setup | Output |
|---|---|---|---|
| E1 | Main grid | Every attacker x both defenders; all metrics above | Per-seed and summary CSV; figures: attacker payoff, defender payoff, perception gap (one panel per defender) |
| E2 | Learning over time | naive, bayesian, strategic, omniscient x both defenders (E1 runs). Per round: attacker payoff and a catch indicator (1 if the attacker was caught on any server that round). Trailing 25-round rolling mean per seed, then mean and 95% band across seeds | Summary CSV per round; two figures (payoff, catch), one panel per defender |
| E3 | Value of traps | T in {0, 1, 2}; sse (SSE re-solved for each T) and adaptive; naive, bayesian, strategic, omniscient. Metric: defender payoff (attacker payoff also recorded). The equilibrium defender and attacker values for each T are recorded with the summary | Per-seed and summary CSV; one figure, one panel per defender, with the SSE value per T |
| E4 | Prior sensitivity | bayesian vs bayesian_uniform and strategic vs strategic_uniform, both defenders; attacker payoff and attack share (E1 data) | Figure only |
| E5 | Lure | Undefended attack share and undefended catch rate per attacker and defender (E1 data). Neither defender ever defends a trapped server, so every catch is on an undefended server by construction; the informative quantities are how often each attacker attacks undefended servers and how often that gets it caught | Figure only |
| E6 | Memory robustness | W in {10, 50, 200}; naive, bayesian, strategic against the adaptive defender; attacker payoff | Per-seed and summary CSV; one figure |
| E7 | Misspecified strategic attacker | Strategic attacker with `model_rate` in {0.05, 0.1, 0.2, 0.4, 0.8} (the adaptive defender's true rate stays 0.2; `model_floor` stays 0.05), against both defenders, same seeds as E1. The 0.2 cell is the E1 strategic run. Metrics: attacker payoff (mean, 95% CI) and seed-paired differences in attacker payoff vs the E1 naive and bayesian runs (mean, 95% CI, seeds won) | Per-seed, summary and paired CSV; one figure (attacker payoff vs `model_rate`, with the naive, bayesian and omniscient E1 values against the same defender as reference lines with 95% bands), one panel per defender |
| E8 | Target selection | Every attacker x both defenders (E1 runs). Per server: attack-unit share and attack rate (Section 8.1). Compared with the attacker's SSE equilibrium response (0, 0, 0, 3) | Per-seed and summary CSV; one figure (grouped bars: one group per server, one bar per attacker, attack-unit share; marker for the SSE response), one panel per defender |
| E9 | Naive memory diagnostic | Naive attacker vs the adaptive defender, W in {10, 50, 200} (same runs as E6). Before each round, for each server i: the naive estimate of pi_i0 (0 if the cell is untested) and the true P(trap on i given d_i = 0) under the defender's current mixed strategy. Per seed: mean absolute belief error (all cells, and tested cells only), share of cells untested, share of rounds in which some untested (so assumed safe) level-0 cell has true trap probability above 0.5, undefended attack share, undefended catch rate, attacker payoff | Per-seed and summary CSV; table only |
| E10 | Learned strategic attacker | strategic_learned against both defenders at W = 50, and against the adaptive defender at W = 200; same seeds and random streams as every other run. Seed-paired differences in attacker payoff (mean, 95% CI, seeds won/lost/tied) vs naive, bayesian, strategic and omniscient (E1 runs) at W = 50, and vs naive, bayesian and strategic (E6 runs) at W = 200. Learned-dependence diagnostic: at the end of each run, the posterior mean trap probability of the undefended cells, averaged over servers with at least one remembered observation in that cell, per pressure bin; mean and 95% CI across the runs that have such a cell (their number is reported). Within-server version: for each server, the learned P(trap | undefended) in one bin minus that in another (medium - low, high - medium, high - low), using only servers observed in both bins, averaged over those servers per run, then across runs; both defenders at W = 50 and W = 200 (the W = 200 SSE runs exist only for this table) | Per-seed, summary and paired CSV; two figures: attacker payoff at W = 50 for naive, bayesian, strategic_learned, strategic, omniscient, and learned trap probability by pressure bin, one panel per defender |

### 8.3 Paired comparisons

Seed-paired differences (first minus second) with mean, 95% t-interval, and the number of seeds in which each side is higher, all from E1 data:

| Comparison | Metric | Per |
|---|---|---|
| strategic - naive | Attacker payoff | Defender |
| bayesian - naive | Attacker payoff | Defender |
| strategic - bayesian | Attacker payoff | Defender |
| SSE defender - adaptive defender | Defender payoff | Attacker |
| attacker - SSE equilibrium attacker value | Attacker payoff against the SSE defender | Attacker |

### 8.4 Expected pattern (to be tested, not assumed)

- The adaptive defender catches the naive attacker often.
- The strategic attacker does better than the naive attacker against the adaptive defender.
- Against the SSE defender the strategic attacker may overthink and do worse than the naive attacker.
- No attacker can exceed the SSE attacker value against the SSE defender in expectation, since the SSE defender's strategy is fixed and every round is independent of the history.

### 8.5 Output files

| File | Content |
|---|---|
| `config.json` | Game parameters, seeds, rounds, experiment settings, SSE values |
| `e1_main_grid_seeds.csv`, `e1_main_grid_summary.csv` | E1 per seed; mean and 95% t-interval per defender, attacker, metric |
| `e2_learning_summary.csv` | E2 rolling mean and band per defender, attacker, round |
| `e3_traps_seeds.csv`, `e3_traps_summary.csv` | E3 per seed; summary with equilibrium values per T |
| `e6_memory_seeds.csv`, `e6_memory_summary.csv` | E6 per seed; summary |
| `paired_comparisons.csv` | Section 8.3 |
| `e7_misspecified_seeds.csv`, `e7_misspecified_summary.csv`, `e7_paired_comparisons.csv` | E7 per seed; summary per model rate and defender; seed-paired differences vs naive and bayesian |
| `e8_targets_seeds.csv`, `e8_targets_summary.csv` | E8 per seed; summary per defender, attacker, server metric |
| `e9_naive_memory_seeds.csv`, `e9_naive_memory_summary.csv` | E9 per seed; summary per W |
| `e10_learned_seeds.csv`, `e10_learned_summary.csv`, `e10_paired_comparisons.csv` | E10 per seed (with the learned-dependence values); summary per W and defender; seed-paired differences |
| `e10_within_server_seeds.csv`, `e10_within_server_summary.csv` | E10 within-server differences per seed (with the number of servers contributing); summary per W and defender |
| `summary.md` | Auto-generated tables (E1, paired comparisons, E3, E6, E7, E8, E9, E10) |
| `fig_e1a_attacker_payoff.png`, `fig_e1b_defender_payoff.png`, `fig_e1c_perception_gap.png` | E1 figures |
| `fig_e2_learning_payoff.png`, `fig_e2_learning_catch.png` | E2 figures |
| `fig_e3_traps.png`, `fig_e4_prior.png`, `fig_e5_lure.png`, `fig_e6_memory.png` | E3-E6 figures |
| `fig_e7_misspecified.png`, `fig_e8_targets.png` | E7 and E8 figures (E9 has no figure) |
| `fig_e10_payoff.png`, `fig_e10_learned_dependence.png` | E10 figures |

Figures are PNG at 150 dpi with 95% error bars or bands. Each attacker has one fixed color in every figure; uniform-prior variants are also hatched, and line charts use a distinct marker per attacker.

## 9. Extension (only if time allows)

**Probing.** The attacker may spend attack units to probe servers before attacking, revealing that round's trap on the probed server. Simulated only; not included in the equilibrium solve.

## 10. Parameters

### 10.1 Game parameters (GameConfig)

| Symbol | Name in code | Default | Meaning |
|---|---|---|---|
| v | values | (10, 8, 6, 4) | Server values |
| N | n | 4 | Number of servers (derived from values) |
| D | defense_units | 4 | Defense units per round |
| T | traps | 1 | Traps per round |
| A | attack_units | 4 | Attack units per round (hold-back allowed) |
| k | resilience | 1.0 | Contest constant |
| c | catch_prob | 1.0 | Probability a trapped server catches the attacker |
| C | caught_penalty | 6.0 | Attacker loss when caught |
| R | catch_reward | 6.0 | Defender gain when catching |
| b | block_cost | 1.0 | Attacker loss when blocked |
| r_b | block_reward | 1.0 | Defender gain when blocking |
| kappa | effort_cost | 0.25 | Attacker cost per attack unit |
| W | memory | 50 | Attacker memory window (rounds) |

### 10.2 Agent and experiment parameters

| Parameter | Value |
|---|---|
| Adaptive defender smoothing rate | 0.2 |
| Adaptive defender initial attack frequency | Uniform |
| Trap weight floor (defender and strategic attacker model) | 0.05 (`model_floor` for the attacker) |
| Strategic attacker smoothing rate | 0.2 (`model_rate`; varied in E7) |
| Bayesian / strategic prior on trap rate | Beta(4 T/N, 4 (1 - T/N)) = Beta(1, 3) (informed, default); Beta(1, 1) (uniform ablation) |
| Defense-level smoothing | Add-one over levels 0..D |
| Seeds per cell | 50 (paired across cells) |
| Rounds per run | 500 |
| Confidence intervals | 95% t-intervals across seeds |
| Learning-curve rolling window | 25 rounds |
| Trap counts (E3) | 0, 1, 2 |
| Memory windows (E6, E9) | 10, 50, 200 |
| Assumed defender smoothing rates (E7) | 0.05, 0.1, 0.2, 0.4, 0.8 |
| "Unsafe" trap-probability threshold (E9) | 0.5 |
| Learned strategic attacker: own pressure smoothing rate | 0.3 (attacker's own choice) |
| Learned strategic attacker: pressure bins | low < 0.15 <= medium <= 0.35 < high |
| Memory windows (E10) | 50 (both defenders), 200 (adaptive defender) |

## 11. Limitations

- Trap cells (server, defense level) are learned only from attacked rounds within the memory window, so estimates can be sparse, especially for rarely seen defense levels.
- The adaptive defender and the strategic attacker use fixed heuristic rules, not learned or optimized policies.
- Payoff values are chosen to avoid a degenerate equilibrium and are not calibrated to real data.
