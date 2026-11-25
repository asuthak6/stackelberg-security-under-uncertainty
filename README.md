# Stackelberg Security Under Uncertainty
 
Does an attacker's limited perception and reasoning style change how well it does against a defender who protects servers and sets honeypot traps? This project models DDoS-style network defense as a **repeated Stackelberg security game**, solves the equilibrium exactly with **linear programming**, and compares seven attacker types against an equilibrium (SSE) defender and an adaptive lure-setting defender over **50 seeds × 500 rounds**.
 
> Originally developed as a team project for **SE4457 (Foundations of Multiagent Systems)**, Western University, Fall 2026, by [Name], [Name], [Name] and Avanish Suthakaran. Forked from [vekaria04/SE4457-Multiagent-Security-Game](https://github.com/vekaria04/SE4457-Multiagent-Security-Game).
>
> **My contributions:** [e.g. SSE solver, Bayesian/strategic attackers, experiment harness E1–E10]
 
---
 
## Key findings
 
- **The equilibrium defender is robust.** Against the SSE defender, every reasoning attacker (naive, Bayesian, strategic, omniscient) lands within ~0.1 of the equilibrium attacker value, so how cleverly the attacker reasons barely matters. Only the random attacker does meaningfully worse.
- **The adaptive defender trades robustness for exploitation.** Its lure strategy crushes naive attackers (defender payoff +0.76, vs −1.41 under SSE) but is exploited by attackers that model it: the strategic attacker earns +0.62 and the omniscient benchmark +1.14.
- **Reasoning only pays against a predictable defender.** Strategic beats naive by **+2.23 [95% CI +2.11, +2.35]** against the adaptive defender (50/50 seeds), but the difference is indistinguishable from zero against SSE.
- **Traps are the biggest lever.** Defender equilibrium value goes from −6.79 with no traps to −1.13 with one trap and 0.00 with two.
- **Learning the defender's rule partly works.** A strategic attacker that doesn't know the lure rule and has to learn it beats naive and Bayesian attackers against the adaptive defender, but still trails one that knows the rule.
### Main results (E1: mean ± 95% CI half-width, 50 seeds × 500 rounds)
 
| Defender | Attacker | Attacker payoff | Defender payoff |
|---|---|---|---|
| SSE | random | −0.830 ± 0.076 | +0.034 ± 0.076 |
| SSE | naive | +0.414 ± 0.073 | −1.414 ± 0.073 |
| SSE | bayesian | +0.283 ± 0.082 | −1.282 ± 0.082 |
| SSE | strategic | +0.375 ± 0.084 | −1.375 ± 0.084 |
| SSE | omniscient | +0.386 ± 0.042 | −1.136 ± 0.042 |
| Adaptive | random | −0.504 ± 0.073 | −0.292 ± 0.073 |
| Adaptive | naive | −1.608 ± 0.086 | +0.764 ± 0.085 |
| Adaptive | bayesian | −0.175 ± 0.076 | −0.789 ± 0.078 |
| Adaptive | strategic | +0.623 ± 0.081 | −1.622 ± 0.081 |
| Adaptive | omniscient | +1.143 ± 0.076 | −2.143 ± 0.076 |
 
Full tables, including uniform-prior variants, trap values, memory robustness and paired comparisons: [`results/summary.md`](results/summary.md)
 
![Attacker vs defender payoffs](results/fig_e1a.png)
 
---
 
## The game in brief
 
- **Servers:** 4, with values (10, 8, 6, 4).
- **Defender (leader):** each round places 4 defense units across servers and 1 trap on one server (140 pure strategies), and commits to a mixed strategy.
- **Attacker (follower):** each round splits up to 4 attack units across servers or holds some back (70 allocations), and commits before seeing that round's defense or trap.
- **Outcomes:** a trapped server always catches the attacker. An untrapped attacked server falls with probability `a / (a + d + 1)`, where `a` and `d` are the attack and defense levels on that server. Otherwise the attack is blocked.
- **Payoffs:**
  | Event | Attacker | Defender |
  |---|---|---|
  | Server taken down | +v | −v |
  | Caught by trap | −6 | +6 |
  | Attack blocked | −1 | +1 |
  | Per attack unit | −0.25 | 0 |
- **Observations:** the attacker remembers only the last 50 rounds. After each round it sees the realized defense on every server and whether each server it attacked was trapped. It never learns about traps on servers it did not attack. The defender sees the full attack and its outcomes.
The equilibrium is computed as a **Strong Stackelberg Equilibrium (SSE)** using the multiple-LP method.
 
---
 
## Agents
 
| Agent | CLI name | Behavior |
|---|---|---|
| SSE defender | `sse` | Plays the SSE mixed strategy every round; does not learn. |
| Adaptive defender | `adaptive` | Traps servers in proportion to smoothed recent attack frequency and leaves the trapped server undefended as a lure. |
| Random attacker | `random` | Chooses uniformly among the 70 allocations. |
| Naive attacker | `naive` | Estimates the trap rate per (server, defense level) cell empirically; untested cells are assumed safe. |
| Bayesian attacker | `bayesian` | Beta posterior per cell with an informed prior (mean T/N); Thompson sampling. |
| Strategic attacker | `strategic` | Models the adaptive defender's lure rule and averages that prediction with the Bayesian posterior mean. |
| Learned strategic attacker | `strategic_learned` | Doesn't know the defender's rule; learns per (server, defended?, own attack-pressure bin) whether traps follow its attacks. |
| Omniscient attacker | `omniscient` | Knows the defender's current mixed strategy and best-responds exactly (benchmark). |
 
`bayesian_uniform` and `strategic_uniform` are variants that use a uniform Beta(1, 1) prior instead of the informed prior.
 
---
 
## Repository structure
 
| Path | Contents |
|---|---|
| `DESIGN.md` | Full model, agents, solver, experiments and parameters |
| `security_game/config.py` | `GameConfig`: game parameters and validation |
| `security_game/model.py` | Strategy spaces, outcome probabilities, expected and realized payoffs |
| `security_game/solver.py` | SSE (multiple-LP) and mixed-strategy helpers |
| `security_game/inspect_equilibrium.py` | Command-line equilibrium summary |
| `security_game/defenders.py` | SSE and adaptive defenders |
| `security_game/attackers.py` | Random, naive, Bayesian, strategic and omniscient attackers |
| `security_game/game.py` | Repeated game loop, summary metrics, single-run CLI |
| `security_game/experiments.py` | Experiments E1–E10, paired comparisons, figures, tables |
| `tests/` | pytest suite |
| `results/` | Experiment outputs (CSV, PNG, `summary.md`) |
| `extras/live_demo/` | Archived prototype (see below) |
| `requirements.txt`, `pyproject.toml` | Dependencies and pytest configuration |
 
---
 
## Setup
 
Requires **Python 3.10+**.
 
```bash
# bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
 
```powershell
# PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
 
---
 
## How to run
 
Run all commands from the repository root.
 
```bash
# Run the tests
pytest -q
 
# Print the equilibrium summary (options for traps, penalties, rewards, units, catch probability)
python -m security_game.inspect_equilibrium
 
# Run a single game
python -m security_game.game --defender adaptive --attacker strategic --rounds 500 --seed 0
 
# Run all experiments (default: 50 seeds x 500 rounds, ~6.5 min single-process)
python -m security_game.experiments
 
# Quick check (3 seeds x 60 rounds, ~15 s)
python -m security_game.experiments --quick --out results_quick
```
 
**Defenders:** `sse`, `adaptive`
**Attackers:** `random`, `naive`, `bayesian`, `bayesian_uniform`, `strategic`, `strategic_uniform`, `strategic_learned`, `omniscient`
 
---
 
## Experiments
 
| ID | Question |
|---|---|
| E1 | Main grid: every attacker vs both defenders |
| E2 | Learning over time (per-round summary) |
| E3 | Value of traps (T = 0, 1, 2) |
| E6 | Robustness to attacker memory length |
| E7 | Misspecified strategic attacker: payoff vs the smoothing rate it assumes for the adaptive defender |
| E8 | Per-server target selection vs the SSE response |
| E9 | Naive attacker memory diagnostic (table only) |
| E10 | Learned strategic attacker, including its learned within-server dependence |
 
Full experimental design is in [`DESIGN.md`](DESIGN.md), Section 8.
 
---
 
## Outputs
 
Written to `results/` (or the `--out` directory):
 
- `config.json`: parameters, seeds, rounds and SSE values
- Per-seed and summary CSVs for E1, E3, E6, E7, E8, E9 and E10, plus a per-round CSV for E2
- Paired-comparison CSVs for E1, E7 and E10, and within-server CSVs for E10
- `summary.md`: auto-generated results tables
- PNG figures (150 dpi, 95% confidence intervals), e.g. `fig_e7_misspecified.png`, `fig_e8_targets.png`, `fig_e10_payoff.png`, `fig_e10_learned_dependence.png`
---
 
## Archived prototype
 
`extras/live_demo/` contains an earlier prototype that ran attacker and defender agents against a real loopback HTTP service under DDoS-style load, before the project moved to the analytical game model. It is kept for reference and is not part of the experiments above.
 
---
 
## License
 
[Add once agreed with the original team, e.g. MIT.]
 
