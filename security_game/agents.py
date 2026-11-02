"""The two agents. They are environment-agnostic: an agent chooses an allocation
and never learns whether the outcome came from the formula or a real server.

- Defender (leader): commits a randomized, concentrated defense allocation.
- Attacker (follower): observes the defense with noise and best-responds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from config import GameConfig, success_prob, compositions


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
        n = self.cfg.n
        if self.strategy == "uniform":
            return [1.0] * n
        if self.strategy == "sse":
            return list(self._sse_shape or [1.0] * n)
        return [self.cfg.values[i] * self.threat[i] for i in range(n)]

    def allocate(self, rng: random.Random) -> list[float]:
        """Commit a concentrated, randomized defense (a sampled mixed strategy).

        Lumps of budget land on servers chosen in proportion to the priority
        weights, so each round a different few servers are heavily defended. This
        is what makes the attacker's perception noise matter: it has to read THIS
        round's realization to find the soft spots.
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
            self.threat[i] = (1 - self.learning_rate) * self.threat[i]
            self.threat[i] += self.learning_rate * (attack_alloc[i] / total) * self.cfg.n


@dataclass
class Attacker:
    cfg: GameConfig
    mode: str = "distributed"           # "distributed" | "single" | "random"
    learning_rate: float = 0.5
    grid: int = 8                       # allocation search granularity
    belief: list[float] = field(init=False)

    def __post_init__(self) -> None:
        self.belief = [self.cfg.defender_budget / self.cfg.n] * self.cfg.n

    def perceive(self, true_defense: list[float], rng: random.Random) -> list[float]:
        return [max(0.0, d + rng.gauss(0.0, self.cfg.noise)) for d in true_defense]

    def _expected_payoff(self, i: int, a_i: float) -> float:
        """Expected payoff of putting a_i on server i, from the attacker's BELIEF.

        Not attacking (a_i = 0) costs nothing; attacking risks the block penalty,
        so entering a server is a real decision with downside. That is why we
        search over allocations instead of greedily water-filling.
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
        for i in range(n):
            self.belief[i] = (1 - self.learning_rate) * self.belief[i] \
                + self.learning_rate * observed[i]

        if self.mode == "random":
            cuts = sorted(rng.uniform(0, B) for _ in range(n - 1))
            cuts = [0.0, *cuts, B]
            return [cuts[i + 1] - cuts[i] for i in range(n)]

        if self.mode == "single":
            best_i, best_v = -1, 0.0
            for i in range(n):
                v = self._expected_payoff(i, B)
                if v > best_v:
                    best_i, best_v = i, v
            alloc = [0.0] * n
            if best_i >= 0:
                alloc[best_i] = B
            return alloc

        # "distributed": search splits of the budget (plus a hold-back bin).
        unit = B / self.grid
        best_alloc, best_val = [0.0] * n, 0.0
        for comp in compositions(self.grid, n + 1):
            alloc = [comp[i] * unit for i in range(n)]
            val = self._total_payoff(alloc)
            if val > best_val:
                best_alloc, best_val = alloc, val
        return best_alloc
