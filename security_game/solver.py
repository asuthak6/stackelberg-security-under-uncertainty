"""Strong Stackelberg Equilibrium solver and mixed-strategy helpers.

See DESIGN.md, section "Equilibrium (formal solution)". The helpers take
(cfg, distribution) over ``defender_pure_strategies(cfg)`` so they apply to
any defender mixed strategy, not only the equilibrium one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import linprog

from .config import GameConfig
from .model import AttackAlloc, attack_allocations, defender_pure_strategies, payoff_matrices

BEST_RESPONSE_TOL = 1e-7
_VALUE_TIE_TOL = 1e-9


@dataclass(frozen=True, eq=False)
class SSEResult:
    """Equilibrium defender distribution and the attacker's best response."""

    distribution: np.ndarray       # over defender_pure_strategies(cfg), read-only
    response_index: int            # column of the attacker's response
    response: AttackAlloc
    defender_value: float
    attacker_value: float


@lru_cache(maxsize=None)
def solve_sse(cfg: GameConfig) -> SSEResult:
    """Strong Stackelberg Equilibrium by the multiple-LP method."""
    UA, UD = payoff_matrices(cfg)
    m, n_cols = UA.shape
    A_eq = np.ones((1, m))
    b_eq = np.array([1.0])
    b_ub = np.zeros(n_cols)

    best_value = -np.inf
    best_j = -1
    best_x = None
    for j in range(n_cols):
        # x . (UA[:, j'] - UA[:, j]) <= 0 for all j': j is a best response.
        A_ub = (UA - UA[:, [j]]).T
        res = linprog(-UD[:, j], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=(0, None), method="highs")
        if res.status != 0:
            continue
        value = -res.fun
        if value > best_value + _VALUE_TIE_TOL:
            best_value, best_j, best_x = value, j, res.x

    if best_x is None:
        raise RuntimeError("no feasible attacker response found")

    x = np.clip(best_x, 0.0, None)
    x = x / x.sum()
    x.setflags(write=False)

    values = attacker_values(cfg, x)
    if values[best_j] < values.max() - BEST_RESPONSE_TOL:
        raise RuntimeError(
            f"response {best_j} is not a best response: "
            f"{values[best_j]:.10f} < {values.max():.10f}")

    return SSEResult(
        distribution=x,
        response_index=best_j,
        response=attack_allocations(cfg)[best_j],
        defender_value=float(x @ UD[:, best_j]),
        attacker_value=float(values[best_j]),
    )


# --------------------------------------------------------------------------
# Helpers for any defender mixed strategy
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _strategy_arrays(cfg: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """Defense levels (m, n) and trap indicators (m, n) per pure strategy."""
    pures = defender_pure_strategies(cfg)
    defense = np.array([d for d, _ in pures], dtype=int).reshape(len(pures), cfg.n)
    trapped = np.zeros((len(pures), cfg.n))
    for r, (_, traps) in enumerate(pures):
        trapped[r, list(traps)] = 1.0
    defense.setflags(write=False)
    trapped.setflags(write=False)
    return defense, trapped


def trap_marginals(cfg: GameConfig, distribution: np.ndarray) -> np.ndarray:
    """P(trap on server i), shape (n,)."""
    _, trapped = _strategy_arrays(cfg)
    return distribution @ trapped


def expected_defense(cfg: GameConfig, distribution: np.ndarray) -> np.ndarray:
    """E[defense units on server i], shape (n,)."""
    defense, _ = _strategy_arrays(cfg)
    return distribution @ defense


def defense_level_marginals(cfg: GameConfig, distribution: np.ndarray) -> np.ndarray:
    """P(defense level L on server i), shape (n, D + 1)."""
    defense, _ = _strategy_arrays(cfg)
    out = np.zeros((cfg.n, cfg.defense_units + 1))
    for i in range(cfg.n):
        out[i] = np.bincount(defense[:, i], weights=distribution,
                             minlength=cfg.defense_units + 1)
    return out


def trap_given_defense(cfg: GameConfig, distribution: np.ndarray) -> np.ndarray:
    """P(trap on i | defense level on i = L), shape (n, D + 1).

    NaN where level L never occurs on server i.
    """
    defense, trapped = _strategy_arrays(cfg)
    levels = defense_level_marginals(cfg, distribution)
    joint = np.zeros_like(levels)
    for i in range(cfg.n):
        joint[i] = np.bincount(defense[:, i], weights=distribution * trapped[:, i],
                               minlength=cfg.defense_units + 1)
    out = np.full_like(levels, np.nan)
    occurs = levels > 1e-12
    out[occurs] = joint[occurs] / levels[occurs]
    return out


def attacker_values(cfg: GameConfig, distribution: np.ndarray) -> np.ndarray:
    """Expected attacker payoff of every attack allocation, shape (70,) by default."""
    UA, _ = payoff_matrices(cfg)
    return distribution @ UA
