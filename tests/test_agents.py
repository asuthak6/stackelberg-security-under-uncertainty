"""Tests for defender and attacker agents."""

import random
from collections import Counter

import numpy as np
import pytest

from security_game.attackers import (
    BayesianAttacker,
    NaiveAttacker,
    Observation,
    StrategicAttacker,
)
from security_game.config import GameConfig
from security_game.defenders import AdaptiveDefender, SSEDefender
from security_game.model import Outcome
from security_game.solver import trap_marginals

CFG = GameConfig()
NA = Outcome.NOT_ATTACKED


def obs(defense, attack, traps):
    """Observation for a round with the given realized defense, attack, and traps."""
    trap_seen = {i: i in traps for i in range(len(attack)) if attack[i] > 0}
    outcomes = tuple(
        NA if a == 0 else (Outcome.CAUGHT if i in traps else Outcome.BLOCKED)
        for i, a in enumerate(attack))
    return Observation(defense=tuple(defense), trap_seen=trap_seen,
                       outcomes=outcomes, attack=tuple(attack))


@pytest.mark.parametrize("make", [SSEDefender, AdaptiveDefender])
def test_mixed_strategy_is_distribution(make):
    x = make(CFG).mixed_strategy()
    assert x.shape == (140,)
    assert np.all(x >= 0)
    assert x.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("traps", [1, 2])
def test_adaptive_never_defends_trapped_server(traps):
    cfg = GameConfig(traps=traps)
    d = AdaptiveDefender(cfg)
    for attack in [(4, 0, 0, 0), (0, 2, 2, 0), (1, 0, 0, 3)]:
        d.observe(attack, ())
    x = d.mixed_strategy()
    assert x.sum() == pytest.approx(1.0)
    for p, (defense, trap_set) in zip(x, d.pure_strategies):
        if any(defense[i] > 0 for i in trap_set):
            assert p == 0.0


def test_adaptive_trap_distribution_t0():
    d = AdaptiveDefender(GameConfig(traps=0))
    assert d.mixed_strategy().sum() == pytest.approx(1.0)
    assert np.allclose(d.trap_distribution(), [1.0])


def test_adaptive_sampling_matches_mixed_strategy():
    d = AdaptiveDefender(CFG)
    for _ in range(5):
        d.observe((4, 0, 0, 0), ())
    x = d.mixed_strategy()
    rng = random.Random(7)
    n = 20000
    counts = Counter(d.act(rng) for _ in range(n))
    checked = 0
    for idx, p in enumerate(x):
        if p > 0.05:
            assert abs(counts[idx] / n - p) < 0.02
            checked += 1
    assert checked > 0


def test_adaptive_shifts_trap_toward_attacked_server():
    d = AdaptiveDefender(CFG)
    before = trap_marginals(CFG, d.mixed_strategy())[1]
    for _ in range(10):
        d.observe((0, 3, 0, 0), ())
    after = trap_marginals(CFG, d.mixed_strategy())
    assert after[1] > before + 0.3
    assert after.argmax() == 1


def test_adaptive_ignores_empty_attack():
    d = AdaptiveDefender(CFG)
    x = d.mixed_strategy().copy()
    d.observe((0, 0, 0, 0), ())
    assert np.allclose(d.mixed_strategy(), x)


def test_learning_attackers_require_c_equal_one():
    with pytest.raises(ValueError):
        NaiveAttacker(GameConfig(catch_prob=0.5))


def test_naive_cells():
    a = NaiveAttacker(CFG)
    assert np.all(a.trap_estimates() == 0.0)
    a.observe(obs(defense=(2, 1, 1, 0), attack=(0, 0, 0, 2), traps=(3,)))
    pi = a.trap_estimates()
    assert pi[3, 0] == 1.0
    assert pi[3, 1] == 0.0          # other level on the same server still untested
    assert pi[0, 2] == 0.0          # server not attacked
    a.observe(obs(defense=(2, 1, 1, 0), attack=(0, 0, 0, 2), traps=(0,)))
    assert a.trap_estimates()[3, 0] == pytest.approx(0.5)


def feed_hand_counted(a):
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 2), traps=(3,)))   # caught at (3, 0)
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 2), traps=(0,)))   # safe at (3, 0)
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 2), traps=(3,)))   # caught at (3, 0)
    a.observe(obs((1, 1, 1, 1), (2, 0, 0, 1), traps=(2,)))   # safe at (0, 1) and (3, 1)


@pytest.mark.parametrize("prior, a0, b0", [("informed", 1.0, 3.0), ("uniform", 1.0, 1.0)])
def test_bayesian_posterior_counts(prior, a0, b0):
    a = BayesianAttacker(CFG, prior=prior)
    assert a.prior_params == pytest.approx((a0, b0))
    feed_hand_counted(a)
    alpha, beta = a.posterior()
    # (3, 0): 2 catches, 1 safe; (0, 1) and (3, 1): 1 safe each; (1, 1): untested.
    assert (alpha[3, 0], beta[3, 0]) == pytest.approx((a0 + 2, b0 + 1))
    assert (alpha[0, 1], beta[0, 1]) == pytest.approx((a0, b0 + 1))
    assert (alpha[3, 1], beta[3, 1]) == pytest.approx((a0, b0 + 1))
    assert (alpha[1, 1], beta[1, 1]) == pytest.approx((a0, b0))
    pi = a.trap_estimates(random.Random(0))
    assert pi.shape == (CFG.n, CFG.defense_units + 1)
    assert np.all((pi >= 0) & (pi <= 1))


def test_uniform_prior_reproduces_old_behaviour():
    a = BayesianAttacker(CFG, prior="uniform")
    feed_hand_counted(a)
    attacks, traps = a.cell_counts()
    alpha, beta = a.posterior()
    assert np.array_equal(alpha, 1.0 + traps)
    assert np.array_equal(beta, 1.0 + attacks - traps)
    s = StrategicAttacker(CFG, prior="uniform")
    feed_hand_counted(s)
    expected = 0.5 * (s.predicted_cells() + (1.0 + traps) / (2.0 + attacks))
    assert np.allclose(s.trap_estimates(), expected)


@pytest.mark.parametrize("traps", [1, 2, 3])
@pytest.mark.parametrize("cls", [BayesianAttacker, StrategicAttacker])
def test_informed_prior_mean_is_base_rate(cls, traps):
    cfg = GameConfig(traps=traps)
    a = cls(cfg)
    assert np.allclose(a.posterior_mean(), traps / cfg.n)
    assert sum(a.prior_params) == pytest.approx(4.0)


def test_invalid_prior_rejected():
    with pytest.raises(ValueError):
        BayesianAttacker(CFG, prior="jeffreys")


@pytest.mark.parametrize("make", [
    NaiveAttacker,
    BayesianAttacker,
    StrategicAttacker,
    lambda cfg: BayesianAttacker(cfg, prior="uniform"),
    lambda cfg: StrategicAttacker(cfg, prior="uniform"),
])
def test_no_traps_means_pi_zero(make):
    cfg = GameConfig(traps=0)
    a = make(cfg)
    a.observe(obs((2, 1, 1, 0), (1, 1, 1, 1), traps=()))
    assert np.all(a.trap_estimates(random.Random(0)) == 0.0)
    a.act(random.Random(0))  # must not fail (Beta(0, 4) would be invalid)


def test_memory_window_forgets():
    cfg = GameConfig(memory=2)
    a = NaiveAttacker(cfg)
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 2), traps=(3,)))
    a.observe(obs((2, 1, 1, 0), (2, 0, 0, 0), traps=(3,)))
    a.observe(obs((2, 1, 1, 0), (2, 0, 0, 0), traps=(3,)))
    assert a.trap_estimates()[3, 0] == 0.0


def test_strategic_prediction_targets_level_zero():
    a = StrategicAttacker(CFG)
    for _ in range(10):
        a.observe(obs((2, 2, 0, 0), (0, 0, 3, 0), traps=(0,)))
    pred = a.predicted_cells()
    assert np.all(pred[:, 1:] == 0.0)
    assert pred[2, 0] == pred[:, 0].max()
    pi = a.trap_estimates()
    assert np.all((pi >= 0) & (pi <= 1))


def test_allocation_values_match_payoff_formula():
    # With no traps believed and certain defense, values equal exact expected payoffs.
    from security_game.model import expected_payoffs
    a = NaiveAttacker(CFG)
    defense = (2, 1, 1, 0)
    levels = np.zeros((CFG.n, CFG.defense_units + 1))
    levels[np.arange(CFG.n), defense] = 1.0
    values = a.allocation_values(levels, np.zeros_like(levels))
    for j, attack in enumerate(a.allocations):
        assert values[j] == pytest.approx(expected_payoffs(CFG, defense, (), attack)[0])


# --- strategic attacker with a (mis)specified model of the defender ----------

class _OldStrategic(StrategicAttacker):
    """The strategic attacker as written before model_rate/model_floor existed."""

    def predicted_cells(self):
        from security_game.defenders import TRAP_FLOOR
        w = self.freq + TRAP_FLOOR
        q = np.minimum(1.0, self.cfg.traps * w / w.sum())
        pred = np.zeros((self.cfg.n, self.cfg.defense_units + 1))
        pred[:, 0] = np.minimum(1.0, q / self.defense_level_probs()[:, 0])
        return pred

    def observe(self, o):
        from security_game.defenders import SMOOTHING
        BayesianAttacker.observe(self, o)
        total = sum(o.attack)
        if total > 0:
            share = np.asarray(o.attack, dtype=float) / total
            self.freq = (1.0 - SMOOTHING) * self.freq + SMOOTHING * share


# Total attacker payoff and attack units per server over 200 rounds, computed with
# the code before model_rate/model_floor were added (git HEAD 6ed3c65).
OLD_STRATEGIC_RUNS = {
    ("sse", "strategic", 0): (76.0, (388, 304, 108, 0)),
    ("sse", "strategic_uniform", 0): (-21.0, (12, 8, 0, 0)),
    ("sse", "strategic", 1): (75.0, (406, 252, 142, 0)),
    ("sse", "strategic_uniform", 1): (97.0, (404, 348, 12, 0)),
    ("adaptive", "strategic", 0): (56.0, (332, 264, 200, 4)),
    ("adaptive", "strategic_uniform", 0): (-16.0, (8, 4, 0, 0)),
    ("adaptive", "strategic", 1): (228.0, (324, 278, 190, 8)),
    ("adaptive", "strategic_uniform", 1): (-7.0, (12, 4, 0, 0)),
}


@pytest.mark.parametrize("key", sorted(OLD_STRATEGIC_RUNS))
def test_strategic_defaults_reproduce_old_runs(key):
    from security_game.game import make_attacker, make_defender, play
    defender, attacker, seed = key
    d = make_defender(defender, CFG)
    logs = play(CFG, d, make_attacker(attacker, CFG, d), 200, seed)
    payoff, units = OLD_STRATEGIC_RUNS[key]
    assert sum(log.attacker_payoff for log in logs) == pytest.approx(payoff, abs=1e-9)
    assert tuple(sum(log.attack[i] for log in logs) for i in range(CFG.n)) == units


@pytest.mark.parametrize("defender", ["sse", "adaptive"])
@pytest.mark.parametrize("prior", ["informed", "uniform"])
def test_strategic_defaults_match_old_rule_round_by_round(defender, prior):
    from security_game.game import make_defender, play
    runs = []
    for make in (lambda: StrategicAttacker(CFG, prior=prior),
                 lambda: StrategicAttacker(CFG, prior=prior, model_rate=0.2,
                                           model_floor=0.05),
                 lambda: _OldStrategic(CFG, prior=prior)):
        d = make_defender(defender, CFG)
        runs.append(play(CFG, d, make(), 150, 7))
    assert runs[0] == runs[1] == runs[2]


def test_strategic_model_rate_sets_its_frequency_update():
    a = StrategicAttacker(CFG, model_rate=0.8)
    a.observe(obs((2, 1, 1, 0), (4, 0, 0, 0), traps=(3,)))
    assert np.allclose(a.freq, [0.2 * 0.25 + 0.8, 0.05, 0.05, 0.05])
    a.observe(obs((2, 1, 1, 0), (0, 0, 0, 0), traps=(3,)))  # no attack: unchanged
    assert np.allclose(a.freq, [0.2 * 0.25 + 0.8, 0.05, 0.05, 0.05])


def test_strategic_model_floor_flattens_prediction():
    sharp = StrategicAttacker(CFG, model_floor=0.0)
    flat = StrategicAttacker(CFG, model_floor=10.0)
    for a in (sharp, flat):
        for _ in range(5):
            a.observe(obs((2, 2, 0, 0), (0, 0, 4, 0), traps=(2,)))
    spread = lambda a: np.ptp(a.predicted_cells()[:, 0] * a.defense_level_probs()[:, 0])
    assert spread(sharp) > spread(flat)


@pytest.mark.parametrize("kwargs", [{"model_rate": 0.0}, {"model_rate": 1.5},
                                    {"model_floor": -0.1}])
def test_strategic_invalid_model_rejected(kwargs):
    with pytest.raises(ValueError):
        StrategicAttacker(CFG, **kwargs)
