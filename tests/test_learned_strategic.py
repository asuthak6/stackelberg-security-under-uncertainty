"""Tests for the learned level-2 attacker (strategic_learned)."""

import ast
import inspect
import random
import textwrap
import types

import numpy as np
import pytest

from security_game import attackers as attackers_module
from security_game.attackers import (
    LEARNED_PRESSURE_RATE,
    PRESSURE_HIGH,
    PRESSURE_LOW,
    BayesianAttacker,
    LearnedStrategicAttacker,
    Observation,
    StrategicAttacker,
    pressure_bin,
)
from security_game.config import GameConfig
from security_game.game import make_attacker, make_defender, play
from security_game.model import Outcome

CFG = GameConfig()
NA = Outcome.NOT_ATTACKED


def obs(defense, attack, traps):
    trap_seen = {i: i in traps for i in range(len(attack)) if attack[i] > 0}
    outcomes = tuple(
        NA if a == 0 else (Outcome.CAUGHT if i in traps else Outcome.BLOCKED)
        for i, a in enumerate(attack))
    return Observation(defense=tuple(defense), trap_seen=trap_seen,
                       outcomes=outcomes, attack=tuple(attack))


# --- not given the defender's rule ---------------------------------------------

def _names_from_defenders() -> set[str]:
    """Names attackers.py imports from defenders.py, plus the module name itself."""
    src = inspect.getsource(attackers_module)
    line = next(l for l in src.splitlines() if l.startswith("from .defenders import"))
    names = {n.strip() for n in line.split("import", 1)[1].split(",")}
    return names | {"defenders"}


def _source_names(func: types.FunctionType) -> set[str]:
    """Every identifier in a function's source, including its signature defaults."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            names |= {node.module or ""} | {a.name for a in node.names}
    return names


def _reachable_functions(cls: type) -> list[types.FunctionType]:
    """Methods of cls (own and inherited) plus module helpers they call, transitively."""
    todo = [f for k in cls.__mro__ if k is not object
            for f in vars(k).values() if isinstance(f, types.FunctionType)]
    out, seen = [], set()
    while todo:
        func = todo.pop()
        if func.__qualname__ in seen:
            continue
        seen.add(func.__qualname__)
        out.append(func)
        for name in _source_names(func):
            target = vars(attackers_module).get(name)
            if isinstance(target, types.FunctionType):
                todo.append(target)
    return out


def test_learned_attacker_uses_nothing_from_defenders():
    forbidden = _names_from_defenders()
    assert {"SMOOTHING", "TRAP_FLOOR"} <= forbidden
    assert not issubclass(LearnedStrategicAttacker, (StrategicAttacker, BayesianAttacker))
    for func in _reachable_functions(LearnedStrategicAttacker):
        used = _source_names(func)
        assert not used & forbidden, (func.__qualname__, used & forbidden)
    assert "defenders" not in inspect.getsource(LearnedStrategicAttacker)


def test_import_check_detects_the_rule_given_attacker():
    # The check above must be able to fail: the rule-given attacker uses SMOOTHING
    # and TRAP_FLOOR (as signature defaults).
    forbidden = _names_from_defenders()
    hits = set().union(*(_source_names(f) & forbidden
                         for f in _reachable_functions(StrategicAttacker)))
    assert {"SMOOTHING", "TRAP_FLOOR"} <= hits


# --- pressure ------------------------------------------------------------------

def test_pressure_bin_edges():
    assert (PRESSURE_LOW, PRESSURE_HIGH, LEARNED_PRESSURE_RATE) == (0.15, 0.35, 0.3)
    assert pressure_bin(0.0) == 0
    assert pressure_bin(0.1499) == 0
    assert pressure_bin(0.15) == 1
    assert pressure_bin(0.35) == 1
    assert pressure_bin(0.3501) == 2
    assert pressure_bin(1.0) == 2


def test_pressure_updates_and_stored_bins():
    a = LearnedStrategicAttacker(CFG)
    assert np.allclose(a.pressure, 0.25)
    assert a.current_bins() == (1, 1, 1, 1)

    a.observe(obs((2, 1, 1, 0), (4, 0, 0, 0), traps=(3,)))
    assert np.allclose(a.pressure, [0.475, 0.175, 0.175, 0.175])
    assert a.current_bins() == (2, 1, 1, 1)

    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 0), traps=(3,)))   # no attack: unchanged
    assert np.allclose(a.pressure, [0.475, 0.175, 0.175, 0.175])

    a.observe(obs((2, 1, 1, 0), (3, 1, 0, 0), traps=(3,)))
    assert np.allclose(a.pressure, [0.7 * 0.475 + 0.3 * 0.75, 0.7 * 0.175 + 0.3 * 0.25,
                                    0.7 * 0.175, 0.7 * 0.175])
    assert a.current_bins() == (2, 1, 0, 0)

    # Each remembered observation carries the bins at the time the attack was chosen.
    assert [o.bins for o in a.memory] == [(1, 1, 1, 1), (2, 1, 1, 1), (2, 1, 1, 1)]


# --- beliefs -------------------------------------------------------------------

def test_posterior_counts_match_hand_counts():
    a = LearnedStrategicAttacker(CFG)
    # bins (1,1,1,1): server 0 defended, not trapped; server 3 undefended, trapped
    a.observe(obs((2, 1, 1, 0), (2, 0, 0, 2), traps=(3,)))
    # pressure after round 1: [0.325, 0.175, 0.175, 0.325] -> bins (1, 1, 1, 1)
    assert a.current_bins() == (1, 1, 1, 1)
    # server 0 undefended and trapped
    a.observe(obs((0, 2, 2, 0), (4, 0, 0, 0), traps=(0,)))
    # pressure [0.5275, 0.1225, 0.1225, 0.2275] -> (2, 0, 0, 1)
    assert a.current_bins() == (2, 0, 0, 1)
    # server 0 undefended, not trapped, high bin; server 2 defended, low bin, trapped
    a.observe(obs((0, 2, 2, 0), (3, 0, 1, 0), traps=(2,)))

    attacks, traps = a.cell_counts()
    assert attacks.shape == (4, 2, 3)
    exp_attacks = np.zeros((4, 2, 3))
    exp_traps = np.zeros((4, 2, 3))
    exp_attacks[0, 1, 1] = 1                                 # round 1, server 0
    exp_attacks[3, 0, 1] = 1; exp_traps[3, 0, 1] = 1         # round 1, server 3
    exp_attacks[0, 0, 1] = 1; exp_traps[0, 0, 1] = 1         # round 2, server 0
    exp_attacks[0, 0, 2] = 1                                 # round 3, server 0
    exp_attacks[2, 1, 0] = 1; exp_traps[2, 1, 0] = 1         # round 3, server 2
    assert np.array_equal(attacks, exp_attacks)
    assert np.array_equal(traps, exp_traps)

    alpha, beta = a.posterior()
    assert np.array_equal(alpha, 1.0 + exp_traps)            # informed prior Beta(1, 3)
    assert np.array_equal(beta, 3.0 + exp_attacks - exp_traps)
    assert a.prior_params == BayesianAttacker(CFG).prior_params


def test_estimates_use_current_bin_and_map_levels():
    a = LearnedStrategicAttacker(CFG)
    for _ in range(30):   # server 0 always trapped when undefended
        a.observe(obs((0, 2, 1, 1), (4, 0, 0, 0), traps=(0,)))
    assert a.current_bins()[0] == 2
    pi = a.trap_estimates(random.Random(0))
    assert pi.shape == (CFG.n, CFG.defense_units + 1)
    assert np.all((pi >= 0) & (pi <= 1))
    for i in range(CFG.n):   # levels >= 1 share the defended cell
        assert np.all(pi[i, 1:] == pi[i, 1])
    alpha, beta = a.posterior()
    assert alpha[0, 0, 2] > 20 and beta[0, 0, 2] == pytest.approx(3.0)


def test_memory_window_forgets():
    cfg = GameConfig(memory=2)
    a = LearnedStrategicAttacker(cfg)
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 2), traps=(3,)))
    a.observe(obs((2, 1, 1, 0), (2, 0, 0, 0), traps=(3,)))
    a.observe(obs((2, 1, 1, 0), (2, 0, 0, 0), traps=(3,)))
    attacks, traps = a.cell_counts()
    assert attacks[3].sum() == 0 and traps.sum() == 0
    assert attacks[0, 1].sum() == 2
    assert len(a.memory) == 2


@pytest.mark.parametrize("traps, value", [(0, 0.0), (4, 1.0)])
def test_degenerate_trap_counts(traps, value):
    a = LearnedStrategicAttacker(GameConfig(traps=traps))
    a.observe(obs((2, 1, 1, 0), (1, 1, 1, 1), traps=tuple(range(traps))))
    assert np.all(a.trap_estimates(random.Random(0)) == value)
    a.act(random.Random(0))


def test_requires_catch_prob_one():
    with pytest.raises(ValueError):
        LearnedStrategicAttacker(GameConfig(catch_prob=0.5))


# --- in the game -----------------------------------------------------------------

def _run(defender, rounds, seed):
    d = make_defender(defender, CFG)
    a = make_attacker("strategic_learned", CFG, d)
    assert isinstance(a, LearnedStrategicAttacker)
    return play(CFG, d, a, rounds, seed)


@pytest.mark.parametrize("defender", ["sse", "adaptive"])
def test_determinism_and_runs_50_rounds(defender):
    a, b, c = _run(defender, 50, 3), _run(defender, 50, 3), _run(defender, 50, 4)
    assert len(a) == 50
    assert a == b
    assert a != c


# --- within-server diagnostic (experiments) ---------------------------------------

def test_within_server_differences_hand_counted():
    import math
    from security_game.experiments import within_server_differences
    a = LearnedStrategicAttacker(CFG)
    # Round 1, bins (1,1,1,1): server 0 undefended, trapped -> cell (0, und, medium).
    a.observe(obs((0, 2, 1, 1), (4, 0, 0, 0), traps=(0,)))
    assert a.current_bins() == (2, 1, 1, 1)
    # Round 2, bins (2,1,1,1): server 0 undefended, not trapped -> (0, und, high);
    # server 1 undefended, not trapped -> (1, und, medium).
    a.observe(obs((0, 0, 2, 2), (2, 2, 0, 0), traps=(3,)))
    d = within_server_differences(a)
    # Prior Beta(1, 3). Server 0: medium Beta(2, 3) -> 0.4; high Beta(1, 4) -> 0.2.
    # Only server 0 has both medium and high; no server has a low cell.
    assert d["diff_high_medium"] == pytest.approx(0.2 - 0.4)
    assert d["servers_high_medium"] == 1
    assert math.isnan(d["diff_medium_low"]) and d["servers_medium_low"] == 0
    assert math.isnan(d["diff_high_low"]) and d["servers_high_low"] == 0
