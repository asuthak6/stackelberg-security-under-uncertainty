"""Attacker agents. See DESIGN.md, section "Attackers".

Learning attackers estimate a trap probability per cell (server i, defense
level L), learned only from remembered rounds in which they attacked server i.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import GameConfig
from .defenders import SMOOTHING, TRAP_FLOOR, Defender
from .model import AttackAlloc, Outcome, attack_allocations, payoff_matrices

TIE_TOL = 1e-9


@dataclass(frozen=True)
class Observation:
    """What the attacker perceives after a round."""

    defense: tuple[int, ...]       # realized defense on every server
    trap_seen: dict[int, bool]     # only for servers attacked this round
    outcomes: tuple[Outcome, ...]
    attack: AttackAlloc


def _pick_tied(values: np.ndarray, rng: random.Random) -> int:
    best = values.max()
    return rng.choice(np.flatnonzero(values >= best - TIE_TOL).tolist())


class Attacker:
    """Base attacker with a bounded memory of observations."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.allocations = attack_allocations(cfg)
        self.memory: deque[Observation] = deque(maxlen=cfg.memory)

    def act(self, rng: random.Random) -> AttackAlloc:
        raise NotImplementedError

    def observe(self, obs: Observation) -> None:
        self.memory.append(obs)


class RandomAttacker(Attacker):
    def act(self, rng: random.Random) -> AttackAlloc:
        return rng.choice(self.allocations)


class LearningAttacker(Attacker):
    """Shared beliefs and decision step; subclasses supply trap_estimates."""

    def __init__(self, cfg: GameConfig):
        if cfg.catch_prob != 1.0:
            raise ValueError("learning attackers assume catch_prob == 1")
        super().__init__(cfg)
        n, A, D = cfg.n, cfg.attack_units, cfg.defense_units
        self._alloc_array = np.array(self.allocations, dtype=int).reshape(-1, n)
        a = np.arange(A + 1)[None, :, None]
        L = np.arange(D + 1)[None, None, :]
        v = np.array(cfg.values)[:, None, None]
        p = a / (a + L + cfg.resilience)
        # Untrapped payoff per (server, attack amount, defense level), shape (n, A+1, D+1).
        self._contest = p * v - (1.0 - p) * cfg.block_cost
        self._effort = cfg.effort_cost * np.arange(A + 1)

    # --- beliefs from memory -------------------------------------------------

    def defense_level_probs(self) -> np.ndarray:
        """P(d_i = L) with add-one smoothing, shape (n, D+1)."""
        counts = np.ones((self.cfg.n, self.cfg.defense_units + 1))
        for obs in self.memory:
            for i, d in enumerate(obs.defense):
                counts[i, d] += 1
        return counts / counts.sum(axis=1, keepdims=True)

    def cell_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """(attacks_iL, traps_iL) over remembered attacked rounds, shape (n, D+1)."""
        attacks = np.zeros((self.cfg.n, self.cfg.defense_units + 1))
        traps = np.zeros_like(attacks)
        for obs in self.memory:
            for i, trapped in obs.trap_seen.items():
                L = obs.defense[i]
                attacks[i, L] += 1
                traps[i, L] += trapped
        return attacks, traps

    def trap_estimates(self, rng: random.Random) -> np.ndarray:
        """pi_iL, shape (n, D+1)."""
        raise NotImplementedError

    # --- decision step -------------------------------------------------------

    def allocation_values(self, level_probs: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """Expected attacker payoff of every allocation under the given beliefs."""
        C = self.cfg.caught_penalty
        per_level = pi[:, None, :] * (-C) + (1.0 - pi[:, None, :]) * self._contest
        server = (level_probs[:, None, :] * per_level).sum(axis=2) - self._effort
        server[:, 0] = 0.0  # unattacked servers contribute nothing
        n = self.cfg.n
        return server[np.arange(n)[None, :], self._alloc_array].sum(axis=1)

    def act(self, rng: random.Random) -> AttackAlloc:
        values = self.allocation_values(self.defense_level_probs(), self.trap_estimates(rng))
        return self.allocations[_pick_tied(values, rng)]


class NaiveAttacker(LearningAttacker):
    """Level 0: empirical trap rate per cell; untested cells are safe."""

    def trap_estimates(self, rng: random.Random | None = None) -> np.ndarray:
        attacks, traps = self.cell_counts()
        return np.divide(traps, attacks, out=np.zeros_like(traps), where=attacks > 0)


PRIORS = ("informed", "uniform")
PRIOR_STRENGTH = 4.0


class BayesianAttacker(LearningAttacker):
    """Beta prior per cell; Thompson-samples pi_iL each round.

    prior="informed": mean T/N, strength 4, i.e. Beta(4T/N, 4(1 - T/N)).
    prior="uniform":  Beta(1, 1).
    """

    def __init__(self, cfg: GameConfig, prior: str = "informed"):
        super().__init__(cfg)
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {PRIORS}")
        self.prior = prior
        if prior == "informed":
            base = cfg.traps / cfg.n
            self.prior_params = (PRIOR_STRENGTH * base, PRIOR_STRENGTH * (1.0 - base))
        else:
            self.prior_params = (1.0, 1.0)

    def _degenerate_estimates(self) -> np.ndarray | None:
        """pi is certain when there are no traps (0) or every server is trapped (1)."""
        cfg = self.cfg
        if cfg.traps in (0, cfg.n):
            return np.full((cfg.n, cfg.defense_units + 1), float(cfg.traps == cfg.n))
        return None

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        attacks, traps = self.cell_counts()
        alpha0, beta0 = self.prior_params
        return alpha0 + traps, beta0 + attacks - traps

    def posterior_mean(self) -> np.ndarray:
        alpha, beta = self.posterior()
        return alpha / (alpha + beta)

    def trap_estimates(self, rng: random.Random) -> np.ndarray:
        fixed = self._degenerate_estimates()
        if fixed is not None:
            return fixed
        alpha, beta = self.posterior()
        return np.vectorize(rng.betavariate)(alpha, beta)


class StrategicAttacker(BayesianAttacker):
    """Level 2: models the adaptive defender's lure rule, averaged with the posterior mean."""

    def __init__(self, cfg: GameConfig, prior: str = "informed"):
        super().__init__(cfg, prior)
        self.freq = np.full(cfg.n, 1.0 / cfg.n)

    def predicted_cells(self) -> np.ndarray:
        """Lure-rule prediction: traps sit on undefended servers."""
        cfg = self.cfg
        w = self.freq + TRAP_FLOOR
        q = np.minimum(1.0, cfg.traps * w / w.sum())
        pred = np.zeros((cfg.n, cfg.defense_units + 1))
        pred[:, 0] = np.minimum(1.0, q / self.defense_level_probs()[:, 0])
        return pred

    def trap_estimates(self, rng: random.Random | None = None) -> np.ndarray:
        fixed = self._degenerate_estimates()
        if fixed is not None:
            return fixed
        return 0.5 * (self.predicted_cells() + self.posterior_mean())

    def observe(self, obs: Observation) -> None:
        super().observe(obs)
        total = sum(obs.attack)
        if total > 0:
            share = np.asarray(obs.attack, dtype=float) / total
            self.freq = (1.0 - SMOOTHING) * self.freq + SMOOTHING * share


class OmniscientAttacker(Attacker):
    """Best-responds to the defender's true current mixed strategy.

    Ties are broken in the defender's favour, which reproduces the SSE.
    """

    def __init__(self, cfg: GameConfig, defender: Defender):
        super().__init__(cfg)
        self.defender = defender
        self._UA, self._UD = payoff_matrices(cfg)

    def act(self, rng: random.Random) -> AttackAlloc:
        x = self.defender.mixed_strategy()
        att = x @ self._UA
        tied = np.flatnonzero(att >= att.max() - TIE_TOL)
        dfn = (x @ self._UD)[tied]
        best_for_defender = tied[dfn >= dfn.max() - TIE_TOL]
        return self.allocations[rng.choice(best_for_defender.tolist())]

    def observe(self, obs: Observation) -> None:
        pass
