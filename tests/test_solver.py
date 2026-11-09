"""Tests for the Strong Stackelberg Equilibrium solver and its helpers."""

import itertools

import numpy as np
import pytest

from security_game.config import GameConfig
from security_game.model import defender_pure_strategies, payoff_matrices
from security_game.solver import (
    attacker_values,
    defense_level_marginals,
    expected_defense,
    solve_sse,
    trap_given_defense,
    trap_marginals,
)

CFG = GameConfig()


def defender_value_vs_best_response(UA_row, UD_row, tol=1e-9):
    """Defender value when the attacker best-responds, ties broken for the defender."""
    best = UA_row.max()
    return UD_row[UA_row >= best - tol].max()


def test_distribution_is_valid():
    x = solve_sse(CFG).distribution
    assert x.shape == (140,)
    assert np.all(x >= 0)
    assert x.sum() == pytest.approx(1.0)


def test_response_is_best_response():
    sse = solve_sse(CFG)
    values = attacker_values(CFG, sse.distribution)
    assert values[sse.response_index] >= values.max() - 1e-7


def test_values_match_distribution():
    sse = solve_sse(CFG)
    UA, UD = payoff_matrices(CFG)
    assert sse.defender_value == pytest.approx(sse.distribution @ UD[:, sse.response_index])
    assert sse.attacker_value == pytest.approx(sse.distribution @ UA[:, sse.response_index])


def test_regression_default_values():
    sse = solve_sse(CFG)
    assert sse.defender_value == pytest.approx(-1.134, abs=0.01)
    assert sse.attacker_value == pytest.approx(0.384, abs=0.01)
    assert solve_sse(GameConfig(traps=0)).defender_value == pytest.approx(-6.793, abs=0.01)


def test_commitment_beats_pure_and_uniform():
    sse = solve_sse(CFG)
    UA, UD = payoff_matrices(CFG)
    for r in range(UA.shape[0]):
        assert defender_value_vs_best_response(UA[r], UD[r]) <= sse.defender_value + 1e-7
    uniform = np.full(UA.shape[0], 1.0 / UA.shape[0])
    assert (defender_value_vs_best_response(uniform @ UA, uniform @ UD)
            <= sse.defender_value + 1e-7)


def test_brute_force_tiny_game():
    cfg = GameConfig(values=(5.0, 3.0), defense_units=1, attack_units=1, traps=1)
    assert len(defender_pure_strategies(cfg)) == 4
    UA, UD = payoff_matrices(cfg)
    steps = 50  # simplex step 0.02
    grid = np.array([c for c in itertools.product(range(steps + 1), repeat=3)
                     if sum(c) <= steps])
    X = np.column_stack([grid, steps - grid.sum(axis=1)]) / steps
    att, dfn = X @ UA, X @ UD
    best_grid = max(defender_value_vs_best_response(a, d) for a, d in zip(att, dfn))
    sse = solve_sse(cfg)
    assert best_grid <= sse.defender_value + 1e-7
    assert best_grid >= sse.defender_value - 0.05


@pytest.mark.parametrize("traps", [0, 1, 2])
def test_solves_for_trap_counts(traps):
    cfg = GameConfig(traps=traps)
    sse = solve_sse(cfg)
    assert sse.distribution.sum() == pytest.approx(1.0)
    assert trap_marginals(cfg, sse.distribution).sum() == pytest.approx(traps)


def test_helpers_consistent():
    x = solve_sse(CFG).distribution
    levels = defense_level_marginals(CFG, x)
    assert levels.shape == (CFG.n, CFG.defense_units + 1)
    assert np.allclose(levels.sum(axis=1), 1.0)
    assert np.allclose(levels @ np.arange(CFG.defense_units + 1), expected_defense(CFG, x))
    assert expected_defense(CFG, x).sum() == pytest.approx(CFG.defense_units)
    # Law of total probability: sum_L P(L) P(trap | L) = P(trap).
    cond = np.nan_to_num(trap_given_defense(CFG, x))
    assert np.allclose((levels * cond).sum(axis=1), trap_marginals(CFG, x))


def test_trap_given_defense_nan_for_missing_levels():
    # A point mass on one pure strategy: only its own level occurs per server.
    pures = defender_pure_strategies(CFG)
    x = np.zeros(len(pures))
    x[pures.index(((2, 1, 1, 0), (3,)))] = 1.0
    cond = trap_given_defense(CFG, x)
    assert cond[3, 0] == 1.0
    assert cond[0, 2] == 0.0
    assert np.isnan(cond[0, 0])
