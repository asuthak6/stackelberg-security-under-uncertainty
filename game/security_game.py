"""Repeated Stackelberg security game with noisy observation (Layer 1, no traps).

This is the simulated game that the whole project is built around. Compared with
the single-target baseline in stackelberg_baseline.py, it adds the three things
that make it a genuine multiagent system:

  * DISTRIBUTED actions -- the attacker (and defender) spread a divisible budget
    across all servers, not just pick one.
  * NOISY perception  -- each round the attacker sees the defender's allocation
    through noise, so it acts on beliefs, not facts.
  * ADAPTATION        -- both agents update across rounds from what they observe.

Traps/deception (the "caught" outcome) are Layer 2 and are intentionally NOT in
this file yet; the design is in MODEL.md.

ONE ROUND (Stackelberg: leader commits, follower responds)
----------------------------------------------------------
1. Defender commits a defense allocation d = (d_1..d_N), sum(d_i) <= B_d.
2. Attacker observes d with noise -> belief d_hat, and best-responds with an
   attack allocation a = (a_1..a_N), sum(a_i) <= B_a.
3. Each attacked server resolves with a PROBABILISTIC CONTEST:
       P(success on i) = a_i / (a_i + d_i + k)
   (k > 0 is base resilience; it removes the old all-or-nothing degeneracy.)
4. Payoffs (general-sum ready; traps would make catch >> block in Layer 2):
       success : attacker +v_i,          defender -v_i
       blocked : attacker -block_cost,    defender +block_reward
       minus the attacker's effort cost  cost_coef * a_i on every server.
5. Both agents update: the attacker blends this round's noisy view into its
   belief; the defender raises its threat estimate on servers it saw attacked.

Pure standard library.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import random
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_VALUES = (10.0, 8.0, 6.0, 4.0)


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
@dataclass
class GameConfig:
    values: tuple[float, ...] = DEFAULT_VALUES
    defender_budget: float = 6.0
    attacker_budget: float = 6.0
    resilience: float = 1.0      # k in the contest function
    block_cost: float = 3.0      # attacker penalty for a blocked attack
    block_reward: float = 2.0    # defender reward for blocking
    effort_cost: float = 0.15    # attacker cost per unit of attack
    noise: float = 0.3           # std-dev of observation noise (defense units)

    @property
    def n(self) -> int:
        return len(self.values)


def success_prob(attack: float, defense: float, k: float) -> float:
    """Tullock-style contest: more attack helps, more defense + resilience hurts."""
    if attack <= 0:
        return 0.0
    return attack / (attack + defense + k)


def _compositions(total: int, bins: int):
    """Yield every way to split `total` indistinguishable units into `bins`
    ordered non-negative parts (stars and bars). Used to enumerate candidate
    attack allocations."""
    for cuts in itertools.combinations(range(total + bins - 1), bins - 1):
        prev, part = -1, []
        for c in cuts:
            part.append(c - prev - 1)
            prev = c
        part.append(total + bins - 1 - prev - 1)
        yield part


# --------------------------------------------------------------------------- #
# Defender (the leader). Commits an allocation; may adapt across rounds.
# --------------------------------------------------------------------------- #
@dataclass
class Defender:
    cfg: GameConfig
    strategy: str = "adaptive"          # "adaptive" | "sse" | "uniform"
    learning_rate: float = 0.3
    units: int = 6                      # lumps of budget placed per round (mixing)
    threat: list[float] = field(init=False)
    _sse_shape: list[float] | None = field(default=None)

    def __post_init__(self) -> None:
        self.threat = [1.0] * self.cfg.n

    def _weights(self) -> list[float]:
        """The defender's PRIORITY over servers -- its mixed-strategy weights."""
        n = self.cfg.n
        if self.strategy == "uniform":
            return [1.0] * n
        if self.strategy == "sse":
            return list(self._sse_shape or [1.0] * n)
        # "adaptive": value x learned threat.
        return [self.cfg.values[i] * self.threat[i] for i in range(n)]

    def allocate(self, rng: random.Random) -> list[float]:
        """Commit a concentrated, RANDOMIZED defense for this round.

        Rather than smear the budget evenly (which an attacker could just learn),
        the defender samples where to concentrate: it drops `units` lumps of
        budget, each onto a server chosen with probability proportional to its
        priority weights. So each round a different few servers are heavily
        defended -- a realized sample of the defender's mixed strategy. The
        attacker must read THIS round's realization to find the soft spots, which
        is exactly what observation noise interferes with.
        """
        n, B = self.cfg.n, self.cfg.defender_budget
        weights = self._weights()
        alloc = [0.0] * n
        lump = B / self.units
        for _ in range(self.units):
            i = rng.choices(range(n), weights=weights, k=1)[0]
            alloc[i] += lump
        return alloc

    def learn(self, attack_alloc: list[float]) -> None:
        if self.strategy != "adaptive":
            return
        total = sum(attack_alloc) or 1.0
        for i in range(self.cfg.n):
            # Decay all threats, then raise the ones we actually saw attacked
            # (in proportion to how hard they were hit).
            self.threat[i] = (1 - self.learning_rate) * self.threat[i]
            self.threat[i] += self.learning_rate * (attack_alloc[i] / total) * self.cfg.n


# --------------------------------------------------------------------------- #
# Attacker (the follower). Observes defense noisily, best-responds, adapts.
# --------------------------------------------------------------------------- #
@dataclass
class Attacker:
    cfg: GameConfig
    mode: str = "distributed"           # "distributed" | "single" | "random"
    learning_rate: float = 0.5          # how fast beliefs track observations
    grid: int = 8                       # allocation search granularity (units of budget)
    belief: list[float] = field(init=False)

    def __post_init__(self) -> None:
        # Start believing defense is spread evenly.
        self.belief = [self.cfg.defender_budget / self.cfg.n] * self.cfg.n

    def perceive(self, true_defense: list[float], rng: random.Random) -> list[float]:
        """Noisy view of the committed defense (Gaussian noise, clamped >= 0)."""
        return [max(0.0, d + rng.gauss(0.0, self.cfg.noise)) for d in true_defense]

    def _expected_payoff(self, i: int, a_i: float) -> float:
        """Expected payoff for putting a_i on server i, using current beliefs.

        Not attacking a server (a_i = 0) costs nothing. Attacking it risks the
        block penalty, so entering a server is a real decision with downside --
        which is exactly why we SEARCH over allocations rather than greedily
        water-fill (greedy mis-handles that entry cost).
        """
        if a_i <= 0:
            return 0.0
        p = success_prob(a_i, self.belief[i], self.cfg.resilience)
        reward = p * self.cfg.values[i] - (1 - p) * self.cfg.block_cost
        return reward - self.cfg.effort_cost * a_i

    def _total_payoff(self, alloc: list[float]) -> float:
        return sum(self._expected_payoff(i, alloc[i]) for i in range(len(alloc)))

    def choose(self, observed: list[float], rng: random.Random) -> list[float]:
        n, B = self.cfg.n, self.cfg.attacker_budget
        # Fold this observation into the running belief before deciding.
        for i in range(n):
            self.belief[i] = (1 - self.learning_rate) * self.belief[i] \
                + self.learning_rate * observed[i]

        if self.mode == "random":
            cuts = sorted(rng.uniform(0, B) for _ in range(n - 1))
            cuts = [0.0, *cuts, B]
            return [cuts[i + 1] - cuts[i] for i in range(n)]

        if self.mode == "single":
            # All budget on the single most profitable server (or nothing if no
            # server has positive expected payoff).
            best_i, best_v = -1, 0.0
            for i in range(n):
                v = self._expected_payoff(i, B)
                if v > best_v:
                    best_i, best_v = i, v
            alloc = [0.0] * n
            if best_i >= 0:
                alloc[best_i] = B
            return alloc

        # "distributed": search over allocations of the budget across all
        # servers, plus an extra "hold back" bin so the attacker can decline to
        # spend. The budget is discretized into `grid` units; we evaluate every
        # split and keep the expected-best. This is correct in the face of the
        # entry-cost non-convexity that defeats greedy allocation.
        unit = B / self.grid
        best_alloc, best_val = [0.0] * n, 0.0  # holding back entirely -> 0
        for comp in _compositions(self.grid, n + 1):   # last bin = unspent
            alloc = [comp[i] * unit for i in range(n)]
            val = self._total_payoff(alloc)
            if val > best_val:
                best_alloc, best_val = alloc, val
        return best_alloc


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class RoundLog:
    rnd: int
    defense: list[float]
    attack: list[float]
    successes: list[int]
    attacker_payoff: float
    defender_payoff: float
    value_captured: float


def resolve(cfg: GameConfig, defense: list[float], attack: list[float],
            rng: random.Random) -> RoundLog:
    successes, a_pay, d_pay, captured = [], 0.0, 0.0, 0.0
    for i in range(cfg.n):
        a_pay -= cfg.effort_cost * attack[i]
        if attack[i] <= 0:
            successes.append(0)
            continue
        p = success_prob(attack[i], defense[i], cfg.resilience)
        if rng.random() < p:
            successes.append(1)
            a_pay += cfg.values[i]
            d_pay -= cfg.values[i]
            captured += cfg.values[i]
        else:
            successes.append(0)
            a_pay -= cfg.block_cost
            d_pay += cfg.block_reward
    return RoundLog(0, defense, attack, successes, a_pay, d_pay, captured)


def simulate(cfg: GameConfig, rounds: int, defender: Defender, attacker: Attacker,
             seed: int = 0) -> list[RoundLog]:
    rng = random.Random(seed)
    logs: list[RoundLog] = []
    for r in range(1, rounds + 1):
        defense = defender.allocate(rng)           # leader commits (randomized)
        observed = attacker.perceive(defense, rng)  # noisy observation
        attack = attacker.choose(observed, rng)     # follower best-responds
        log = resolve(cfg, defense, attack, rng)
        log.rnd = r
        logs.append(log)
        defender.learn(attack)                      # both adapt
    return logs


def summarize(cfg: GameConfig, logs: list[RoundLog]) -> dict:
    n = len(logs)
    attacked = sum(1 for lg in logs for a in lg.attack if a > 1e-6)
    succ = sum(sum(lg.successes) for lg in logs)
    return {
        "rounds": n,
        "attacker_payoff_avg": sum(lg.attacker_payoff for lg in logs) / n,
        "defender_payoff_avg": sum(lg.defender_payoff for lg in logs) / n,
        "value_captured_avg": sum(lg.value_captured for lg in logs) / n,
        "attacks_launched": attacked,
        "attacks_succeeded": succ,
        "success_rate": (succ / attacked) if attacked else 0.0,
    }


def sse_shape(values, budget, penalty):
    """Fetch the single-target SSE coverage to shape the 'sse' defender.

    Imported lazily so the simulation has no hard dependency on scipy unless the
    sse defender is actually used.
    """
    from stackelberg_baseline import solve_sse
    return list(solve_sse(values, budget, penalty).coverage)


def main() -> None:
    p = argparse.ArgumentParser(description="Repeated Stackelberg security game (Layer 1)")
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--defender", choices=["adaptive", "sse", "uniform"], default="adaptive")
    p.add_argument("--attacker", choices=["distributed", "single", "random"], default="distributed")
    p.add_argument("--defender-budget", type=float, default=6.0)
    p.add_argument("--attacker-budget", type=float, default=6.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="game_results.csv")
    args = p.parse_args()

    cfg = GameConfig(noise=args.noise, defender_budget=args.defender_budget,
                     attacker_budget=args.attacker_budget)
    defender = Defender(cfg, strategy=args.defender)
    if args.defender == "sse":
        defender._sse_shape = sse_shape(cfg.values, min(2.0, cfg.n - 0.5), 1.0)
    attacker = Attacker(cfg, mode=args.attacker)

    logs = simulate(cfg, args.rounds, defender, attacker, args.seed)
    s = summarize(cfg, logs)

    print(f"Repeated Stackelberg security game  (defender={args.defender}, "
          f"attacker={args.attacker}, noise={args.noise})")
    print(f"  servers (value): {', '.join(f'S{i+1}={v:g}' for i, v in enumerate(cfg.values))}")
    print(f"  rounds: {s['rounds']}")
    print(f"  attacker payoff / round : {s['attacker_payoff_avg']:+.2f}")
    print(f"  defender payoff / round : {s['defender_payoff_avg']:+.2f}")
    print(f"  value captured / round  : {s['value_captured_avg']:.2f}")
    print(f"  attack success rate     : {s['success_rate']:.1%} "
          f"({s['attacks_succeeded']}/{s['attacks_launched']})")

    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", *[f"def_S{i+1}" for i in range(cfg.n)],
                    *[f"atk_S{i+1}" for i in range(cfg.n)],
                    *[f"win_S{i+1}" for i in range(cfg.n)],
                    "attacker_payoff", "defender_payoff", "value_captured"])
        for lg in logs:
            w.writerow([lg.rnd,
                        *[f"{x:.3f}" for x in lg.defense],
                        *[f"{x:.3f}" for x in lg.attack],
                        *lg.successes,
                        f"{lg.attacker_payoff:.3f}", f"{lg.defender_payoff:.3f}",
                        f"{lg.value_captured:.3f}"])
    print(f"  saved per-round log -> {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
