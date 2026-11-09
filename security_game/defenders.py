"""Defender agents. See DESIGN.md, section "Defenders".

Every defender exposes its current mixed strategy over
``defender_pure_strategies(cfg)`` and samples its pure strategy from it.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from typing import Sequence

import numpy as np

from .config import GameConfig
from .model import Outcome, defender_pure_strategies, defense_allocations, trap_placements
from .solver import solve_sse

SMOOTHING = 0.2
TRAP_FLOOR = 0.05


class Defender:
    """Base defender: a mixed strategy over pure strategies, sampled each round."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.pure_strategies = defender_pure_strategies(cfg)

    def mixed_strategy(self) -> np.ndarray:
        raise NotImplementedError

    def act(self, rng: random.Random) -> int:
        """Sample the index of this round's pure strategy."""
        cumulative = np.cumsum(self.mixed_strategy())
        r = rng.random() * cumulative[-1]
        return min(bisect.bisect_right(cumulative, r), len(cumulative) - 1)

    def observe(self, attack: Sequence[int], outcomes: Sequence[Outcome]) -> None:
        pass


class SSEDefender(Defender):
    """Plays the Strong Stackelberg Equilibrium strategy every round."""

    def __init__(self, cfg: GameConfig):
        super().__init__(cfg)
        self._distribution = solve_sse(cfg).distribution

    def mixed_strategy(self) -> np.ndarray:
        return self._distribution


class AdaptiveDefender(Defender):
    """Lure rule: trap where the attacker goes, leave the trapped server undefended."""

    def __init__(self, cfg: GameConfig):
        super().__init__(cfg)
        self.freq = np.full(cfg.n, 1.0 / cfg.n)
        self.placements = trap_placements(cfg)
        # P(defense allocation | trap placement): fixed, independent of freq.
        self._defense_given_trap = np.array(
            [self._defense_distribution(t) for t in self.placements])
        self._cached: np.ndarray | None = None

    def _defense_distribution(self, traps: tuple[int, ...]) -> np.ndarray:
        """Multinomial over defense allocations with units only on non-trapped servers."""
        cfg = self.cfg
        eligible = [i for i in range(cfg.n) if i not in traps] or list(range(cfg.n))
        weights = np.zeros(cfg.n)
        weights[eligible] = [cfg.values[i] for i in eligible]
        if weights.sum() <= 0:
            weights[eligible] = 1.0
        p = weights / weights.sum()
        out = []
        for alloc in defense_allocations(cfg):
            if any(d > 0 and p[i] == 0 for i, d in enumerate(alloc)):
                out.append(0.0)
                continue
            coef = math.factorial(cfg.defense_units)
            for d in alloc:
                coef //= math.factorial(d)
            out.append(coef * math.prod(p[i] ** d for i, d in enumerate(alloc)))
        return np.array(out)

    def trap_distribution(self) -> np.ndarray:
        """Exact probability of each trap placement (sequential sampling without replacement)."""
        w = self.freq + TRAP_FLOOR
        total = w.sum()
        probs = []
        for placement in self.placements:
            p = 0.0
            for order in itertools.permutations(placement):
                q, remaining = 1.0, total
                for i in order:
                    q *= w[i] / remaining
                    remaining -= w[i]
                p += q
            probs.append(p)
        return np.array(probs)

    def mixed_strategy(self) -> np.ndarray:
        if self._cached is None:
            # defender_pure_strategies is product(defense allocations, placements),
            # so index = defense_index * n_placements + placement_index.
            joint = self._defense_given_trap.T * self.trap_distribution()
            self._cached = joint.ravel()
            self._cached.setflags(write=False)
        return self._cached

    def observe(self, attack: Sequence[int], outcomes: Sequence[Outcome]) -> None:
        total = sum(attack)
        if total <= 0:
            return
        share = np.asarray(attack, dtype=float) / total
        self.freq = (1.0 - SMOOTHING) * self.freq + SMOOTHING * share
        self._cached = None
