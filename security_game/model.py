"""Stage-game model: strategy spaces, outcome probabilities, and payoffs.

Pure functions only. Randomness enters only through an explicitly passed
``random.Random``. See DESIGN.md, section "Stage game".
"""

from __future__ import annotations

import itertools
import random
from enum import Enum
from functools import lru_cache
from typing import Iterator, Sequence

import numpy as np

from .config import GameConfig

DefenseAlloc = tuple[int, ...]
TrapPlacement = tuple[int, ...]
AttackAlloc = tuple[int, ...]
DefenderPure = tuple[DefenseAlloc, TrapPlacement]


class Outcome(Enum):
    NOT_ATTACKED = 0
    CAUGHT = 1
    SUCCESS = 2
    BLOCKED = 3


# --------------------------------------------------------------------------
# Strategy spaces
# --------------------------------------------------------------------------

def _compositions(total: int, bins: int) -> Iterator[tuple[int, ...]]:
    """All tuples of ``bins`` non-negative ints summing to ``total``.

    Lexicographic order on the first bin, then the second, and so on.
    """
    if bins == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, bins - 1):
            yield (first,) + rest


@lru_cache(maxsize=None)
def _defense_allocations(cfg: GameConfig) -> tuple[DefenseAlloc, ...]:
    return tuple(_compositions(cfg.defense_units, cfg.n))


def defense_allocations(cfg: GameConfig) -> list[DefenseAlloc]:
    """Every split of exactly ``defense_units`` over the n servers."""
    return list(_defense_allocations(cfg))


def trap_placements(cfg: GameConfig) -> list[TrapPlacement]:
    """Every set of ``traps`` servers, as sorted index tuples. traps=0 gives [()]."""
    return list(itertools.combinations(range(cfg.n), cfg.traps))


@lru_cache(maxsize=None)
def _defender_pure_strategies(cfg: GameConfig) -> tuple[DefenderPure, ...]:
    return tuple(itertools.product(_defense_allocations(cfg), trap_placements(cfg)))


def defender_pure_strategies(cfg: GameConfig) -> list[DefenderPure]:
    """All (defense allocation, trap placement) pairs."""
    return list(_defender_pure_strategies(cfg))


@lru_cache(maxsize=None)
def _attack_allocations(cfg: GameConfig) -> tuple[AttackAlloc, ...]:
    # Split over n servers plus one hold-back bin, then drop the hold-back bin.
    return tuple(c[:-1] for c in _compositions(cfg.attack_units, cfg.n + 1))


def attack_allocations(cfg: GameConfig) -> list[AttackAlloc]:
    """Every allocation of at most ``attack_units`` over the n servers.

    Includes the all-zero allocation (hold everything back).
    """
    return list(_attack_allocations(cfg))


# --------------------------------------------------------------------------
# Outcome probabilities
# --------------------------------------------------------------------------

def success_prob(attack: float, defense: float, k: float) -> float:
    """Contest success probability a / (a + d + k); 0 if nothing is sent."""
    if attack <= 0:
        return 0.0
    return attack / (attack + defense + k)


def server_outcome_probs(a_i: int, d_i: int, trapped: bool,
                         cfg: GameConfig) -> tuple[float, float, float]:
    """(p_caught, p_success, p_blocked) for one server.

    Unattacked servers give (0, 0, 0). On a trapped server the attacker is
    caught with probability c; otherwise the contest decides success vs blocked.
    """
    if a_i <= 0:
        return (0.0, 0.0, 0.0)
    p_caught = cfg.catch_prob if trapped else 0.0
    p = success_prob(a_i, d_i, cfg.resilience)
    p_success = (1.0 - p_caught) * p
    p_blocked = (1.0 - p_caught) * (1.0 - p)
    return (p_caught, p_success, p_blocked)


# --------------------------------------------------------------------------
# Payoffs
# --------------------------------------------------------------------------

def expected_payoffs(cfg: GameConfig, defense: Sequence[int], traps: Sequence[int],
                     attack: Sequence[int]) -> tuple[float, float]:
    """Exact expected (attacker, defender) payoff for one pure defender
    strategy against one attack allocation."""
    trap_set = set(traps)
    u_att = 0.0
    u_def = 0.0
    for i in range(cfg.n):
        a_i = attack[i]
        if a_i <= 0:
            continue
        p_c, p_s, p_b = server_outcome_probs(a_i, defense[i], i in trap_set, cfg)
        v = cfg.values[i]
        u_att += (-cfg.caught_penalty * p_c + v * p_s - cfg.block_cost * p_b
                  - cfg.effort_cost * a_i)
        u_def += cfg.catch_reward * p_c - v * p_s + cfg.block_reward * p_b
    return u_att, u_def


@lru_cache(maxsize=None)
def payoff_matrices(cfg: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """Attacker and defender payoff matrices UA, UD.

    Rows follow ``defender_pure_strategies(cfg)``, columns follow
    ``attack_allocations(cfg)``. The arrays are cached and read-only.
    """
    rows = _defender_pure_strategies(cfg)
    cols = _attack_allocations(cfg)
    UA = np.empty((len(rows), len(cols)))
    UD = np.empty((len(rows), len(cols)))
    for r, (defense, traps) in enumerate(rows):
        for j, attack in enumerate(cols):
            UA[r, j], UD[r, j] = expected_payoffs(cfg, defense, traps, attack)
    UA.setflags(write=False)
    UD.setflags(write=False)
    return UA, UD


# --------------------------------------------------------------------------
# Realized outcomes
# --------------------------------------------------------------------------

def sample_outcomes(cfg: GameConfig, defense: Sequence[int], traps: Sequence[int],
                    attack: Sequence[int], rng: random.Random) -> tuple[Outcome, ...]:
    """Draw one realized Outcome per server.

    Exactly one uniform is drawn per server, attacked or not, so the stream
    stays aligned across different attack allocations (common random numbers).
    """
    trap_set = set(traps)
    outcomes = []
    for i in range(cfg.n):
        u = rng.random()
        if attack[i] <= 0:
            outcomes.append(Outcome.NOT_ATTACKED)
            continue
        p_c, p_s, _ = server_outcome_probs(attack[i], defense[i], i in trap_set, cfg)
        if u < p_c:
            outcomes.append(Outcome.CAUGHT)
        elif u < p_c + p_s:
            outcomes.append(Outcome.SUCCESS)
        else:
            outcomes.append(Outcome.BLOCKED)
    return tuple(outcomes)


def realized_payoffs(cfg: GameConfig, attack: Sequence[int],
                     outcomes: Sequence[Outcome]) -> tuple[float, float]:
    """(attacker, defender) payoff for realized per-server outcomes."""
    u_att = 0.0
    u_def = 0.0
    for i, outcome in enumerate(outcomes):
        if outcome is Outcome.NOT_ATTACKED:
            continue
        u_att -= cfg.effort_cost * attack[i]
        if outcome is Outcome.CAUGHT:
            u_att -= cfg.caught_penalty
            u_def += cfg.catch_reward
        elif outcome is Outcome.SUCCESS:
            u_att += cfg.values[i]
            u_def -= cfg.values[i]
        else:
            u_att -= cfg.block_cost
            u_def += cfg.block_reward
    return u_att, u_def
