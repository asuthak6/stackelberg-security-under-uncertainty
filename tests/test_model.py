"""Tests for strategy spaces, outcome probabilities, and payoffs."""

import random

import pytest

from security_game.config import GameConfig
from security_game.model import (
    Outcome,
    attack_allocations,
    defender_pure_strategies,
    defense_allocations,
    expected_payoffs,
    payoff_matrices,
    realized_payoffs,
    sample_outcomes,
    server_outcome_probs,
    trap_placements,
)

CFG = GameConfig()

# Hand-calculated case (see DESIGN.md payoff rules).
DEFENSE = (2, 1, 1, 0)
TRAPS = (3,)
ATTACK = (2, 0, 0, 2)
EXPECTED_ATT = 2.9 + (-6.5)   # server 1 contest + server 4 caught = -3.6
EXPECTED_DEF = -3.4 + 6.0     # = 2.6


def test_strategy_counts_default():
    assert len(defense_allocations(CFG)) == 35
    assert len(trap_placements(CFG)) == 4
    assert len(defender_pure_strategies(CFG)) == 140
    assert len(attack_allocations(CFG)) == 70


def test_trap_placement_counts():
    assert trap_placements(GameConfig(traps=0)) == [()]
    assert len(trap_placements(GameConfig(traps=2))) == 6


def test_strategy_spaces_unique():
    assert len(set(defense_allocations(CFG))) == 35
    assert len(set(attack_allocations(CFG))) == 70


def test_allocation_sums():
    for a in attack_allocations(CFG):
        assert len(a) == CFG.n
        assert all(x >= 0 for x in a)
        assert sum(a) <= CFG.attack_units
    assert (0, 0, 0, 0) in attack_allocations(CFG)
    for d in defense_allocations(CFG):
        assert len(d) == CFG.n
        assert all(x >= 0 for x in d)
        assert sum(d) == CFG.defense_units


@pytest.mark.parametrize("catch_prob", [1.0, 0.5])
@pytest.mark.parametrize("trapped", [False, True])
def test_outcome_probs_sum_to_one(catch_prob, trapped):
    cfg = GameConfig(catch_prob=catch_prob)
    for a in range(1, cfg.attack_units + 1):
        for d in range(cfg.defense_units + 1):
            probs = server_outcome_probs(a, d, trapped, cfg)
            assert all(p >= 0 for p in probs)
            assert sum(probs) == pytest.approx(1.0)


def test_unattacked_server_probs_are_zero():
    assert server_outcome_probs(0, 2, True, CFG) == (0.0, 0.0, 0.0)


def test_hand_calculated_case():
    u_att, u_def = expected_payoffs(CFG, DEFENSE, TRAPS, ATTACK)
    assert u_att == pytest.approx(-3.6)
    assert u_def == pytest.approx(2.6)
    assert u_att == pytest.approx(EXPECTED_ATT)
    assert u_def == pytest.approx(EXPECTED_DEF)


def test_payoff_matrix_shape_and_entry():
    UA, UD = payoff_matrices(CFG)
    assert UA.shape == (140, 70)
    assert UD.shape == (140, 70)
    row = defender_pure_strategies(CFG).index((DEFENSE, TRAPS))
    col = attack_allocations(CFG).index(ATTACK)
    u_att, u_def = expected_payoffs(CFG, DEFENSE, TRAPS, ATTACK)
    assert UA[row, col] == pytest.approx(u_att)
    assert UD[row, col] == pytest.approx(u_def)


def test_monte_carlo_matches_expected():
    rng = random.Random(12345)
    n = 20000
    tot_att = tot_def = 0.0
    for _ in range(n):
        outcomes = sample_outcomes(CFG, DEFENSE, TRAPS, ATTACK, rng)
        u_att, u_def = realized_payoffs(CFG, ATTACK, outcomes)
        tot_att += u_att
        tot_def += u_def
    exp_att, exp_def = expected_payoffs(CFG, DEFENSE, TRAPS, ATTACK)
    assert abs(tot_att / n - exp_att) < 0.1
    assert abs(tot_def / n - exp_def) < 0.1


def test_sample_outcomes_unattacked_and_trapped():
    rng = random.Random(0)
    outcomes = sample_outcomes(CFG, DEFENSE, TRAPS, ATTACK, rng)
    assert outcomes[1] is Outcome.NOT_ATTACKED
    assert outcomes[2] is Outcome.NOT_ATTACKED
    assert outcomes[3] is Outcome.CAUGHT  # trapped with c = 1


def test_all_zero_attack():
    zero = (0, 0, 0, 0)
    assert expected_payoffs(CFG, DEFENSE, TRAPS, zero) == (0.0, 0.0)
    outcomes = sample_outcomes(CFG, DEFENSE, TRAPS, zero, random.Random(1))
    assert realized_payoffs(CFG, zero, outcomes) == (0.0, 0.0)


@pytest.mark.parametrize("kwargs", [
    {"defense_units": 0},
    {"attack_units": 0},
    {"traps": 5},
    {"traps": -1},
    {"catch_prob": 1.5},
    {"catch_prob": -0.1},
    {"memory": 0},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        GameConfig(**kwargs)
