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


def prior_params(cfg: GameConfig, prior: str) -> tuple[float, float]:
    """Beta prior per cell. informed: mean T/N, strength 4; uniform: Beta(1, 1)."""
    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {PRIORS}")
    if prior == "informed":
        base = cfg.traps / cfg.n
        return (PRIOR_STRENGTH * base, PRIOR_STRENGTH * (1.0 - base))
    return (1.0, 1.0)


def degenerate_estimates(cfg: GameConfig) -> np.ndarray | None:
    """pi is certain when there are no traps (0) or every server is trapped (1)."""
    if cfg.traps in (0, cfg.n):
        return np.full((cfg.n, cfg.defense_units + 1), float(cfg.traps == cfg.n))
    return None


class BayesianAttacker(LearningAttacker):
    """Beta prior per cell; Thompson-samples pi_iL each round.

    prior="informed": mean T/N, strength 4, i.e. Beta(4T/N, 4(1 - T/N)).
    prior="uniform":  Beta(1, 1).
    """

    def __init__(self, cfg: GameConfig, prior: str = "informed"):
        super().__init__(cfg)
        self.prior_params = prior_params(cfg, prior)
        self.prior = prior

    def _degenerate_estimates(self) -> np.ndarray | None:
        return degenerate_estimates(self.cfg)

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
    """Level 2: models the adaptive defender's lure rule, averaged with the posterior mean.

    model_rate and model_floor are the smoothing rate and trap weight floor the
    attacker assumes the defender uses. They only affect the attacker's prediction;
    the defaults equal the adaptive defender's true constants (a correct model).
    """

    def __init__(self, cfg: GameConfig, prior: str = "informed",
                 model_rate: float = SMOOTHING, model_floor: float = TRAP_FLOOR):
        super().__init__(cfg, prior)
        if not 0.0 < model_rate <= 1.0:
            raise ValueError("model_rate must be in (0, 1]")
        if model_floor < 0.0:
            raise ValueError("model_floor must be non-negative")
        self.model_rate = model_rate
        self.model_floor = model_floor
        self.freq = np.full(cfg.n, 1.0 / cfg.n)

    def predicted_cells(self) -> np.ndarray:
        """Lure-rule prediction: traps sit on undefended servers."""
        cfg = self.cfg
        w = self.freq + self.model_floor
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
            self.freq = (1.0 - self.model_rate) * self.freq + self.model_rate * share


# --- learned level-2 attacker -------------------------------------------------
# These are the attacker's own modelling choices. They are NOT taken from the
# defender: this attacker is given no defender rule and no defender constants.
LEARNED_PRESSURE_RATE = 0.3   # smoothing rate of the attacker's own attack pressure g_i
PRESSURE_LOW = 0.15           # g_i < PRESSURE_LOW               -> bin 0 (low)
PRESSURE_HIGH = 0.35          # PRESSURE_LOW <= g_i <= PRESSURE_HIGH -> bin 1 (medium)
PRESSURE_BINS = ("low", "medium", "high")   # g_i > PRESSURE_HIGH -> bin 2 (high)


@dataclass(frozen=True)
class PressureObservation(Observation):
    """An Observation plus the attacker's pressure bin per server when it chose the attack."""

    bins: tuple[int, ...] = ()


def pressure_bin(g: float) -> int:
    if g < PRESSURE_LOW:
        return 0
    if g <= PRESSURE_HIGH:
        return 1
    return 2


class LearnedStrategicAttacker(LearningAttacker):
    """Level 2 without a given defender model: learns whether traps follow its own attacks.

    It tracks its own attack pressure g_i (smoothed share of its attack units,
    rate LEARNED_PRESSURE_RATE) and keeps a Beta posterior on the trap
    probability per cell (server i, defended?, pressure bin of i when the attack
    was chosen), with defended meaning defense level >= 1. Each round it
    Thompson-samples the (i, undefended) and (i, defended) cells at server i's
    current bin and uses the shared decision step (level 0 -> undefended cell,
    levels >= 1 -> defended cell). If traps follow its attacks, high-pressure
    undefended cells learn a high trap probability; if not, the bins look alike.
    """

    def __init__(self, cfg: GameConfig, prior: str = "informed"):
        super().__init__(cfg)
        self.prior_params = prior_params(cfg, prior)
        self.prior = prior
        self.pressure = np.full(cfg.n, 1.0 / cfg.n)

    def current_bins(self) -> tuple[int, ...]:
        return tuple(pressure_bin(g) for g in self.pressure)

    def cell_counts(self) -> tuple[np.ndarray, np.ndarray]:
        """(attacks, traps) per (server, defended?, bin), shape (n, 2, 3)."""
        attacks = np.zeros((self.cfg.n, 2, len(PRESSURE_BINS)))
        traps = np.zeros_like(attacks)
        for obs in self.memory:
            for i, trapped in obs.trap_seen.items():
                cell = (i, int(obs.defense[i] >= 1), obs.bins[i])
                attacks[cell] += 1
                traps[cell] += trapped
        return attacks, traps

    def posterior(self) -> tuple[np.ndarray, np.ndarray]:
        attacks, traps = self.cell_counts()
        alpha0, beta0 = self.prior_params
        return alpha0 + traps, beta0 + attacks - traps

    def posterior_mean(self) -> np.ndarray:
        alpha, beta = self.posterior()
        return alpha / (alpha + beta)

    def trap_estimates(self, rng: random.Random) -> np.ndarray:
        fixed = degenerate_estimates(self.cfg)
        if fixed is not None:
            return fixed
        alpha, beta = self.posterior()
        pi = np.empty((self.cfg.n, self.cfg.defense_units + 1))
        for i, b in enumerate(self.current_bins()):
            pi[i, 0] = rng.betavariate(alpha[i, 0, b], beta[i, 0, b])
            pi[i, 1:] = rng.betavariate(alpha[i, 1, b], beta[i, 1, b])
        return pi

    def observe(self, obs: Observation) -> None:
        # The pressure has not changed since act(), so these are the bins at choice time.
        super().observe(PressureObservation(
            defense=obs.defense, trap_seen=obs.trap_seen, outcomes=obs.outcomes,
            attack=obs.attack, bins=self.current_bins()))
        total = sum(obs.attack)
        if total > 0:
            share = np.asarray(obs.attack, dtype=float) / total
            self.pressure = ((1.0 - LEARNED_PRESSURE_RATE) * self.pressure
                             + LEARNED_PRESSURE_RATE * share)


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
