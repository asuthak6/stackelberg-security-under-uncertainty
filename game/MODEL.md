# Formal Model — Repeated Stackelberg Security Game with Noisy Perception

> **Use this as a reference / study guide, not as report text to hand in.** It
> documents what the code implements and why, so the team can understand the
> model and write the Sprint 3 report in your own words. (Course policy: the
> report must be your own writing and track your own understanding — code
> generation is allowed, ghost-written reports are not.)

This describes the model implemented in `security_game.py` (the simulated game)
and `stackelberg_baseline.py` (the solvable anchor). Traps/deception (the
"caught" outcome) are specified in §7 as Layer 2 and are **not** in the code yet.

---

## 1. Players and intuition

Two players repeatedly contest a set of servers:

- **Defender** (the Stackelberg **leader**): commits first to how it protects the
  servers.
- **Attacker** (the **follower**): observes the defender's setup — imperfectly —
  and then attacks.

It is a **security game** (defender protects valued assets, attacker tries to
compromise them), played as a **Stackelberg** interaction (leader commits,
follower best-responds), **repeated** over many rounds so both sides adapt, under
**imperfect information** (the attacker's view is noisy).

## 2. Environment

- `N` servers. Server `i` has value `vᵢ > 0`. Default: `v = (10, 8, 6, 4)`.
- **Defender budget** `B_d`: total protection resource per round.
- **Attacker budget** `B_a`: total attack resource per round.
- **Resilience** `k > 0`: base robustness every server has even when undefended.

## 3. Actions (one round)

- **Defender** commits an allocation `d = (d₁,…,d_N)`, `dᵢ ≥ 0`, `Σ dᵢ ≤ B_d`.
  In the code the defender plays a **mixed strategy**: it holds priority weights
  `wᵢ` and realizes `d` by dropping `U` lumps of budget, each on a server drawn
  with probability `∝ wᵢ`. So the realized defense is **concentrated and varies
  round to round** — a sampled realization of the defender's strategy. This is
  what makes perception matter (see §5).
- **Attacker** commits an allocation `a = (a₁,…,a_N)`, `aᵢ ≥ 0`, `Σ aᵢ ≤ B_a`.
  It may **spread** across servers (a distributed attack) and may **hold budget
  back** if nothing is worth attacking.

## 4. Outcome: the probabilistic contest

For each server `i` with `aᵢ > 0`, the attack **succeeds** with probability

```
            aᵢ
P(i) = ───────────────         (Tullock-style contest success function)
        aᵢ + dᵢ + k
```

This replaces the old project's hard threshold (`intensity > capacity`), which
was degenerate (one unit of defense beat the strongest attack, so only
completely undefended servers ever fell). With the ratio form, more attack always
helps, more defense always hurts, and nothing is ever perfectly safe or hopeless.

## 5. Information structure (the research focus)

The attacker does **not** see `d`. It sees a noisy observation

```
d̂ᵢ = max(0, dᵢ + εᵢ),   εᵢ ~ Normal(0, σ²)
```

where `σ` (`noise`) is the perception-noise level — the independent variable in
our main experiment. Because the defender's realized defense is **concentrated
and shifts each round** (§3), the attacker cannot simply average noise away over
time: it must read *this round's* realization to find the soft spots, and noise
degrades that read. The defender sees only the **outcomes** (which servers were
attacked and whether they fell), not the attacker's reasoning.

## 6. Payoffs (general-sum-ready)

Per attacked server `i`, with effort cost coefficient `κ`:

| Outcome | Attacker | Defender |
|---|---|---|
| success (prob `P(i)`)       | `+vᵢ`           | `−vᵢ`           |
| blocked (prob `1 − P(i)`)   | `−c_block`      | `+r_block`      |
| effort (always)             | `−κ·aᵢ`         | —               |

Not attacking a server costs the attacker nothing. The **block penalty**
`c_block` is why starting to attack a server is a real decision with downside —
and why the attacker must *search over allocations* rather than greedily add
pressure (greedy mishandles this entry cost; see `security_game.py`).

Round payoffs are the sums over servers. The design is **general-sum** (payoffs
need not sum to zero); in Layer 1 (no traps) it is effectively near-zero-sum, and
the general-sum structure becomes essential once traps are added (§7).

## 7. Timing and adaptation

Each round `t`:
1. Defender commits `d⁽ᵗ⁾` (randomized realization of its current strategy).
2. Attacker observes `d̂⁽ᵗ⁾` (noisy) and **best-responds** with `a⁽ᵗ⁾`.
3. The contest resolves each server; both receive payoffs.
4. **Both adapt:** the attacker blends its observation into a running belief about
   where defense tends to go; the defender raises its priority weight on servers
   it saw attacked (and decays the rest).

This is the repeated / multiagent-learning layer on top of the one-round
Stackelberg game.

## 8. The solvable baseline (`stackelberg_baseline.py`)

The full game above is studied by simulation. For a **theoretical anchor** we
also solve the simplest honest version exactly:

- One shot, **perfect information**, **single-target** attacker.
- Defender commits a **coverage probability** `cᵢ ∈ [0,1]`, `Σ cᵢ ≤ B`.
- Attacker attacks the one server maximizing its expected payoff
  `EUₐ(i) = (1−cᵢ)·vᵢ + cᵢ·(−penalty)`.
- Payoffs are zero-sum (classic security game).

The **Strong Stackelberg Equilibrium** is found with the **multiple-LP method**:
for each target `t`, solve a linear program for the coverage that maximizes the
defender's payoff subject to "`t` is the attacker's best response" and the budget;
take the best `t`. At the solution the defender equalizes the attacker's expected
payoff across servers (the SSG signature). We use this optimal coverage as the
`sse` defender strategy in the simulation, as a principled baseline to compare the
learning heuristics against.

## 9. Research questions

1. **Perception.** How do attacker success and payoff change as observation noise
   `σ` grows? (Main experiment — noise sweep.)
2. **Adaptation.** Does a learning (`adaptive`) defender outperform a static
   `uniform` one, and how does it compare to the Stackelberg-shaped `sse`
   defender?
3. **Attack structure.** When is a concentrated (single-target) attack better
   than a distributed one, given the per-server block penalty?

## 10. Planned extension — traps / deception (Layer 2)

To realize the "lured into a trap" idea, add a third outcome. The defender splits
its budget into **defense** `dᵢ` and **traps** `tᵢ`. When a server is attacked,
first a **detection** check with probability rising in `tᵢ`:

| Outcome | Attacker | Defender |
|---|---|---|
| **caught** (trap detects) | `−C_catch` (large) | `+G_catch` (large) + learns target |
| success | `+vᵢ` | `−vᵢ` |
| blocked | `−c_block` | `+r_block` |

The key design point — and why §6 is **general-sum** — is that **catching must be
strictly better for the defender than merely blocking** (`G_catch > r_block`),
otherwise traps add nothing. Traps are observed through the **same noisy channel**
as defenses, so a trap can be mistaken for a soft target: that is the lure. This
is a known research direction (deception / honeypots in Stackelberg security
games) and is the natural Sprint-4 extension.

## 11. Honest limitations (state these in the report)

- The agents use **adaptive heuristics**, not an exact equilibrium of the full
  (distributed, noisy, repeated) game — that game has no clean closed form. The
  single-target SSE (§8) is the only part solved exactly.
- The attacker's belief update and the defender's threat update are **heuristic**
  (exponential smoothing), not formal Bayesian inference.
- Results are **stochastic**; all reported numbers are means over many seeds.
- Traps (§7, §10) are specified but not yet implemented.
