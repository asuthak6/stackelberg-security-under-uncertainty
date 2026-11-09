"""Shared configuration and the contest helper for the security game.

One model, one config. The same GameConfig drives the fast formula environment
and the real-server environment, so the agents never know which backend they are
playing against.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

DEFAULT_VALUES = (10.0, 8.0, 6.0, 4.0)


@dataclass
class GameConfig:
    values: tuple[float, ...] = DEFAULT_VALUES
    defender_budget: float = 6.0
    attacker_budget: float = 6.0
    resilience: float = 1.0      # k in the contest function (base robustness)
    block_cost: float = 3.0      # attacker penalty for a blocked attack
    block_reward: float = 2.0    # defender reward for blocking
    effort_cost: float = 0.15    # attacker cost per unit of attack
    noise: float = 0.3           # std-dev of the attacker's observation noise

    @property
    def n(self) -> int:
        return len(self.values)


def success_prob(attack: float, defense: float, k: float) -> float:
    """Tullock-style contest success probability.

    This is the model's *prediction* of whether an attack overwhelms a server.
    The live environment measures the same thing on a real server instead; the
    formula is a fast, analyzable surrogate for that measurement (see validate.py).
    """
    if attack <= 0:
        return 0.0
    return attack / (attack + defense + k)


def compositions(total: int, bins: int):
    """Every way to split `total` units into `bins` ordered non-negative parts."""
    for cuts in itertools.combinations(range(total + bins - 1), bins - 1):
        prev, part = -1, []
        for c in cuts:
            part.append(c - prev - 1)
            prev = c
        part.append(total + bins - 1 - prev - 1)
        yield part
