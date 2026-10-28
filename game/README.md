# The Game — Stackelberg Security Game (Layer 1)

This folder is the **formal model** half of the project (the `../live_demo`
folder is the real-system half). It turns the original prototype into a proper
repeated **Stackelberg security game** with a probabilistic contest, noisy
attacker perception, adaptive agents, and a computable equilibrium baseline.

Read **`MODEL.md`** first — it defines the game formally and is your study guide
for the Sprint 3 report. (Per the course AI policy, write the report yourselves;
this code and spec are allowed scaffolding.)

## Files

| File | What it is |
|---|---|
| `MODEL.md` | The formal model: players, actions, payoffs, information, the SSE baseline, and the planned trap extension. |
| `stackelberg_baseline.py` | Computes the exact **Strong Stackelberg Equilibrium** for the single-target coverage game (multiple-LP method). The theoretical anchor. Needs scipy. |
| `security_game.py` | The repeated simulated game: probabilistic contest, noisy observation, distributed/single attackers, adaptive/sse/uniform defenders, learning. Pure stdlib. |
| `experiments.py` | Runs the three controlled experiments and writes figures + CSVs. Needs matplotlib. |

## Install

```bash
pip install scipy matplotlib      # numpy comes with scipy
```

`security_game.py` itself needs only the standard library; scipy is used by the
baseline solver and the `sse` defender, matplotlib only for figures.

## Run

```bash
# 1. The equilibrium baseline — optimal defender coverage for 4 servers:
python stackelberg_baseline.py

# 2. One game (defender={adaptive,sse,uniform}, attacker={distributed,single,random}):
python security_game.py --defender adaptive --attacker distributed --noise 1.0

# 3. All experiments + figures (~1 min):
python experiments.py --seeds 20
```

## What each experiment shows

- **`fig_noise.png` (the headline):** as the attacker's observation noise rises,
  its payoff falls and the defender's rises. This is the project's core finding —
  *imperfect perception degrades the attacker* — and it only appears because the
  defender plays a randomized, concentrated strategy the attacker must read each
  round.
- **`fig_strategies.png`:** the `adaptive` (learning) defender beats the static
  `uniform` one and edges out the Stackelberg-shaped `sse` baseline; the
  distributed attacker outperforms the single-target one here.
- **`fig_learning.png`:** the adaptive defender's payoff improving over rounds as
  it learns where the attacker strikes.

## Representative numbers (20 seeds × 200 rounds; yours will vary slightly)

Noise sweep (adaptive defender vs distributed attacker):

| noise σ | attacker payoff | defender payoff |
|---|---|---|
| 0.0 | +5.83 | −7.53 |
| 1.0 | +5.48 | −7.18 |
| 2.0 | +5.08 | −6.76 |
| 3.0 | +4.83 | −6.47 |

Equilibrium baseline (`stackelberg_baseline.py`, budget 2): optimal coverage
`c ≈ (0.67, 0.59, 0.48, 0.27)`, which equalizes the attacker's expected payoff
across all four servers — the Stackelberg signature.

## How this connects to the rest of the project

- `../security_agents.py` — the **original prototype**. These agents still work;
  here they're generalized into a proper game with an equilibrium baseline.
- `../live_demo/` — the **real system**. Its measured outage can replace the
  contest function as the outcome, turning this model into a closed loop over a
  live service (a Sprint 4 goal).
- **Traps / deception** (`MODEL.md` §10) are specified and are the next build.

## Honesty notes for the report
- Agents are adaptive **heuristics**; only the single-target baseline is solved
  exactly. The full noisy/distributed/repeated game has no clean closed form.
- Belief and threat updates are exponential smoothing, not exact Bayesian.
- All headline numbers are **means over many seeds** — the game is stochastic.
