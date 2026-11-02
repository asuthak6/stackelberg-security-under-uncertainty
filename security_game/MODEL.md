# Formal Model — One Attacker/Defender Security Game, Two Environments

> **Study guide, not report text.** It documents the model so the team can write
> the Sprint 3 report in your own words (the course forbids AI-written reports;
> code generation is allowed). "§" references point within this file.

This is **one model**. The attacker and defender agents, their payoffs, and the
round structure are defined once. The only thing that changes between the fast
simulation and the live "real DDoS" demo is the **environment** — the component
that answers, each round, *"did this attack cause an outage?"*

- **Formula environment** answers with a probability, `a/(a+d+k)`. Fast and
  analyzable: used to solve for equilibrium and to run thousands of rounds.
- **Live environment** answers by *measuring* a real loopback HTTP server: the
  defense becomes real serving capacity, the attack becomes a real request
  flood, and an outage is observed when a legitimate user can no longer get
  through. This is the TA's "make it real", wired in as the environment.

`validate.py` checks the formula predicts what the real server does, so the fast
backend is a trustworthy surrogate for the real one. **That is the whole design:
not two models, but one game that can run on a fast approximation or on the real
thing.**

```
              ┌─────────────────────────────────────┐
              │  THE GAME  (agents, payoffs, rounds) │   <- defined once
              └───────────────────┬─────────────────┘
                                  │ "did the attack cause an outage?"
                   ┌──────────────┴──────────────┐
                   ▼                              ▼
          Formula environment            Live environment
          a/(a+d+k)  (fast,              real loopback server,
          analyzable, 1000s             measured outage
          of rounds)                    (a few rounds)
                   └────────── validate.py: they agree ──────────┘
```

## 1. Players
Two players over a set of servers: the **defender** (Stackelberg *leader*, commits
first) and the **attacker** (*follower*, observes and responds).

## 2. Environment parameters
- `N` servers; server `i` has value `vᵢ` (default `10, 8, 6, 4`).
- Defender budget `B_d`, attacker budget `B_a` per round.
- Contest resilience `k` (formula backend) — or, equivalently, real server
  capacity (live backend).

## 3. Actions (one round)
- **Defender** commits an allocation `d = (d₁…d_N)`, `Σdᵢ ≤ B_d`, played as a
  *randomized, concentrated* mixed strategy (lumps of budget land on servers in
  proportion to priority weights). This is what makes perception noise matter.
- **Attacker** commits `a = (a₁…a_N)`, `Σaᵢ ≤ B_a`, spread across servers or held
  back.

## 4. Information (the research focus)
The attacker sees the defense with noise: `d̂ᵢ = max(0, dᵢ + εᵢ)`,
`εᵢ ~ Normal(0, σ²)`. Because the defender's realized defense is concentrated and
shifts each round, the attacker cannot average the noise away — it must read this
round's realization, and noise degrades that read.

## 5. Outcome — the environment's job
For each attacked server, the environment returns success (outage) or not:
- **Formula backend:** success with probability `P = aᵢ / (aᵢ + dᵢ + k)`. Replaces
  the old degenerate threshold (one unit of defense beat the strongest attack).
- **Live backend:** the defense sets a real server's capacity, the attack sets a
  real flood rate, and success = the legitimate user's measured availability fell
  below a threshold. Same question, real answer.

## 6. Payoffs (model level — identical for both backends)
Per attacked server `i`, effort coefficient `κ`:

| Outcome | Attacker | Defender |
|---|---|---|
| success | `+vᵢ` | `−vᵢ` |
| blocked | `−c_block` | `+r_block` |
| effort (always) | `−κ·aᵢ` | — |

General-sum (payoffs need not sum to zero); the block penalty is why entering a
server is a real decision, so the attacker **searches** over allocations rather
than greedily adding pressure.

## 7. Timing & adaptation
Each round: defender commits → attacker observes (noisy) and best-responds →
environment resolves → both adapt (attacker updates its belief about where
defense goes; defender raises its priority on servers it saw attacked).

## 8. Solvable baseline (`baseline.py`)
For a theoretical anchor, the single-target, perfect-information version is solved
exactly with the **multiple-LP method**: for each target assume the attacker hits
it, solve a linear program for the coverage that makes that the best response,
keep the best. At the solution the defender equalizes the attacker's expected
payoff across servers — the Stackelberg signature.

## 9. Validating the surrogate (`validate.py`)
For a grid of (defense, attack) levels on a single server, we measure the real
outage probability and fit the formula's one parameter `k` to it. A close match
means the fast formula is a faithful stand-in for the real system, which is what
licenses running the equilibrium and the big experiments on the formula backend.

## 10. Research questions
1. **Perception:** how does observation noise `σ` change attacker/defender
   payoffs? (Main experiment.)
2. **Adaptation:** does a learning defender beat a static one, and how does it
   compare to the Stackelberg-shaped baseline?
3. **Model fidelity:** does the formula predict the real server? (§9.)

## 11. Planned extension — traps / deception (Layer 2)
Split the defender budget into defense and **traps (honeypots)**; add a third
outcome, **caught**, when an attack hits a trap. For traps to matter, catching
must pay the defender more than merely blocking — which is why §6 is general-sum.
Specified, not yet built.

## 12. Honest limitations (state these)
- Agents are adaptive **heuristics**; only the single-target baseline is solved
  exactly.
- Belief/threat updates are exponential smoothing, not exact Bayesian.
- The formula is a **surrogate**; validation shows the trend matches, not a
  perfect fit — the real server is more threshold-like than the smooth formula.
- All numbers are means over many seeds; the game is stochastic.
- Traps (§11) are specified but not implemented.
