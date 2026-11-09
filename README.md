# SE 4457: Attacker vs. Defender Security Game

## Overview

This project studies how an attacker's perception limits and reasoning style affect its success against a defender who protects servers and sets traps, and whether better reasoning compensates for limited perception. The domain is network defence against DDoS-style attacks on servers. The formal model is a Stackelberg security game with defensive resources and traps (honeypots), played repeatedly. The equilibrium is computed exactly with linear programming, and simulated attackers with different reasoning styles are compared against an equilibrium defender and an adaptive defender over many seeded runs.

The full specification is in [DESIGN.md](DESIGN.md).

## The game in brief

- **Servers:** 4 servers with values v = (10, 8, 6, 4).
- **Defender (leader):** each round places D = 4 defense units over the servers and T = 1 trap on one server (140 pure strategies). It commits to a mixed strategy over these.
- **Attacker (follower):** each round splits up to A = 4 attack units over the servers and may hold units back (70 allocations). It commits before seeing the round's defense or trap.
- **Outcome on each attacked server:** if trapped, the attacker is caught (probability c = 1). Otherwise the server is taken down with probability a / (a + d + 1), and the attack is blocked otherwise.

| Outcome per attacked server | Attacker | Defender |
|---|---|---|
| Server taken down | +v_i | -v_i |
| Caught by trap | -6 | +6 |
| Blocked | -1 | +1 |
| Effort (always) | -0.25 per attack unit | 0 |

**Observations.** Before acting, the attacker has only its memory of the last 50 rounds. After a round it sees the realized defense on every server, and whether each server it attacked was trapped; it never learns about traps on servers it did not attack. The defender sees the full attack and the outcomes.

## Agents

| Agent | Behaviour |
|---|---|
| SSE defender | Plays the Strong Stackelberg Equilibrium mixed strategy every round; does not learn |
| Adaptive defender | Traps servers in proportion to smoothed recent attack frequency and leaves the trapped server undefended (lure) |
| Random attacker | Uniform over the 70 allocations |
| Naive attacker | Empirical trap rate per (server, defense level) cell; untested cells are assumed safe |
| Bayesian attacker | Beta posterior per cell with an informed prior (mean T/N); Thompson sampling |
| Strategic attacker | Models the adaptive defender's lure rule and averages that prediction with the Bayesian posterior mean |
| Omniscient attacker | Knows the defender's current mixed strategy and best-responds exactly (benchmark) |

The variants `bayesian_uniform` and `strategic_uniform` use a uniform Beta(1, 1) prior instead of the informed prior.

## Repository structure

| Path | Content |
|---|---|
| `DESIGN.md` | Full model, agents, solver, experiments, parameters |
| `security_game/config.py` | `GameConfig`: game parameters with validation |
| `security_game/model.py` | Strategy spaces, outcome probabilities, expected and realized payoffs |
| `security_game/solver.py` | Strong Stackelberg Equilibrium (multiple-LP) and mixed-strategy helpers |
| `security_game/inspect_equilibrium.py` | Command-line summary of the equilibrium |
| `security_game/defenders.py` | SSE and adaptive defenders |
| `security_game/attackers.py` | Random, naive, Bayesian, strategic, omniscient attackers |
| `security_game/game.py` | Repeated game loop, per-run summary metrics, command-line single run |
| `security_game/experiments.py` | Experiments E1-E6, paired comparisons, figures, summary tables |
| `tests/` | pytest suite |
| `results/` | Experiment outputs (CSV, PNG, summary.md) |
| `extras/live_demo/` | Archived earlier prototype (see below) |
| `requirements.txt`, `pyproject.toml` | Dependencies and pytest configuration |

## Setup

Requires Python 3.10 or newer.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

All commands are run from the repository root.

```bash
# Tests
pytest -q

# Equilibrium summary (options: --traps, --caught-penalty, --catch-reward,
# --defense-units, --attack-units, --catch-prob)
python -m security_game.inspect_equilibrium
python -m security_game.inspect_equilibrium --traps 2

# One game with a chosen pair of agents
python -m security_game.game --defender adaptive --attacker strategic --rounds 500 --seed 0

# All experiments (default 50 seeds x 500 rounds, written to results/)
python -m security_game.experiments

# Quick check (3 seeds x 60 rounds)
python -m security_game.experiments --quick --out results_quick
```

Defender names: `sse`, `adaptive`. Attacker names: `random`, `naive`, `bayesian`, `bayesian_uniform`, `strategic`, `strategic_uniform`, `omniscient`.

The full experiment run took about 8 minutes (465 s) on the development laptop (single process). The quick run takes a few seconds.

## Outputs

`python -m security_game.experiments` writes to `results/`:

- `config.json`: parameters, seeds, rounds, and the SSE values used.
- Per-seed and summary CSV files for E1 (main grid), E3 (value of traps), and E6 (memory robustness); a per-round summary for E2 (learning over time); `paired_comparisons.csv`.
- `summary.md`: auto-generated tables of the main numbers.
- Figures `fig_e1a` to `fig_e6` (PNG, 150 dpi) with 95% confidence intervals.

See DESIGN.md, Section 8, for what each experiment and file contains.

## Archived prototype

`extras/live_demo/` holds the earlier prototype, including a real loopback HTTP server used as a DDoS environment. It is not maintained against the current model; see its README for how to run it.
