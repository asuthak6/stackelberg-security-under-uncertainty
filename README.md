# Attacker vs Defender: Adaptive Multiagent Security Game

Small Python prototype for SE 4457 Foundations of Multiagent Systems. This is a prototype rather than a complete Stackelberg equilibrium solver.

## Run it

Requires Python 3.10 or later. No packages to install.

From PowerShell in the extracted folder:

```powershell
python security_agents.py
```

Or experiment:

```powershell
python security_agents.py --rounds 1000 --noise 0.4 --budget 2 --output noisy_results.csv
python security_agents.py --rounds 1000 --noise 0.0 --budget 2 --output clear_results.csv
python security_agents.py --rounds 1000 --noise 0.4 --budget 2 --no-learning --output baseline.csv
```

The script prints a summary and saves per-round data as CSV.

## Where the agents are

* `DefenderAgent`: **perception** of attacker history (`attack_counts`), **reasoning** to rank likely losses, **action** to allocate `budget` protection units, **learning** from observed attacks.
* `AttackerAgent`: **perception** of defender allocations through noisy binary observations, **belief** (`defense_beliefs`) about defended targets, **reasoning** using expected rewards and penalties, **action** to attack one target, **learning** after the outcome. Exploration allows occasional different decisions.
* `simulate`: environment that organizes simultaneous sequential game rounds, tells each agent only its allowed information, resolves blocked or successful attacks, and awards payoffs.

## Assumptions and current limitations

* Four static targets worth 10, 8, 6 and 4 points. One defender protection unit protects one target in a round. A blocked attack gives attacker -5, a successful attack gives the target value.
* The defender chooses protection first; the attacker observes **noisy signals** of the allocation, not the true allocation. The attacker learns the truth only for the target it attacked. Defender observes which target was attacked.
* The agent policies are simple adaptive heuristics, **not** optimal strategies, equilibrium strategies, reinforcement learning, or a mathematically solved Stackelberg game.
* The attacker sensor noise independently flips each target's defended/not defended status with probability `noise`. At 0.5 it is uninformative; values higher than 0.5 are treated as anti-correlated.
* The defender can change its allocations across rounds; the attacker learns protection frequencies for targets it actually probes. These learning rules are intentionally simple.
* Results are stochastic and depend on the seed; one run is not enough to draw research conclusions.

## Suggested lab demo

1. Open `security_agents.py` and point out the **two different agent classes**.
2. Explain that agents have different goals, state, observations and decisions.
3. Run `python security_agents.py` and inspect `results.csv`.
4. Run with different `--noise` values; compare summaries but do not claim a trend from one seed.
5. State that the next step is repeated trials over many seeds, plots, and better decision-making algorithms.

## Next steps

1. Run multiple seeds per noise/budget setting; report means and uncertainty.
2. Add graphs for attack rate and payoffs and compare adaptive agents to fixed policies.
3. Refine belief updates with a Bayesian observation model and test perception assumptions.
4. Design a principled defender strategy and investigate Stackelberg optimization if relevant to project scope.
5. Define evaluation hypotheses and document the experiment methodology.
