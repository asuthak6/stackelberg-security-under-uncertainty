"""Tests for the repeated game loop and summary metrics."""

import math
import statistics

import pytest

from security_game.attackers import NaiveAttacker
from security_game.config import GameConfig
from security_game.defenders import AdaptiveDefender
from security_game.model import Outcome
from security_game.game import (
    ATTACKERS,
    DEFENDERS,
    RoundLog,
    make_attacker,
    make_defender,
    play,
    summarize,
)

CFG = GameConfig()


class RecordingNaive(NaiveAttacker):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.seen = []

    def observe(self, obs):
        super().observe(obs)
        self.seen.append(obs)


def run(defender_name, attacker_name, rounds, seed):
    defender = make_defender(defender_name, CFG)
    attacker = make_attacker(attacker_name, CFG, defender)
    return play(CFG, defender, attacker, rounds, seed)


def test_trap_seen_only_for_attacked_servers():
    attacker = RecordingNaive(CFG)
    logs = play(CFG, AdaptiveDefender(CFG), attacker, 200, seed=3)
    assert len(attacker.seen) == len(logs)
    for obs, log in zip(attacker.seen, logs):
        attacked = {i for i, a in enumerate(log.attack) if a > 0}
        assert set(obs.trap_seen) == attacked
        for i, trapped in obs.trap_seen.items():
            assert trapped == (i in log.traps)
        assert obs.defense == log.defense


def test_omniscient_vs_sse_matches_equilibrium():
    att, dfn = [], []
    for seed in range(20):
        s = summarize(run("sse", "omniscient", 500, seed))
        att.append(s["attacker_payoff"])
        dfn.append(s["defender_payoff"])
    assert statistics.mean(dfn) == pytest.approx(-1.134, abs=0.15)
    assert statistics.mean(att) == pytest.approx(0.384, abs=0.15)


def test_omniscient_vs_sse_plays_defender_favourable_response():
    logs = run("sse", "omniscient", 20, seed=0)
    assert all(log.attack == (0, 0, 0, 3) for log in logs)


@pytest.mark.parametrize("attacker", ["random", "bayesian", "strategic"])
def test_determinism(attacker):
    a = run("adaptive", attacker, 100, seed=5)
    b = run("adaptive", attacker, 100, seed=5)
    c = run("adaptive", attacker, 100, seed=6)
    assert a == b
    assert a != c


@pytest.mark.parametrize("defender", sorted(DEFENDERS))
@pytest.mark.parametrize("attacker", ATTACKERS)
def test_every_pair_runs(defender, attacker):
    logs = run(defender, attacker, 50, seed=1)
    assert len(logs) == 50
    s = summarize(logs)
    assert set(s) == {"attacker_payoff", "defender_payoff", "catch_rate",
                      "takedown_rate", "attack_share", "undefended_attack_share",
                      "undefended_catch_rate", "defended_takedown_rate"}


def test_make_attacker_priors():
    d = make_defender("sse", CFG)
    assert make_attacker("bayesian", CFG, d).prior == "informed"
    assert make_attacker("strategic", CFG, d).prior == "informed"
    assert make_attacker("bayesian_uniform", CFG, d).prior == "uniform"
    assert make_attacker("strategic_uniform", CFG, d).prior == "uniform"
    for bad in ("naive_uniform", "bayesian_flat", "unknown"):
        with pytest.raises(ValueError):
            make_attacker(bad, CFG, d)


def test_summarize_hand_counted():
    C, S, B, N = Outcome.CAUGHT, Outcome.SUCCESS, Outcome.BLOCKED, Outcome.NOT_ATTACKED
    logs = [
        # undefended+caught on server 4, defended+success on server 1
        RoundLog(0, (2, 1, 1, 0), (3,), (2, 0, 0, 2), (S, N, N, C), 0.0, 0.0),
        # undefended+blocked on server 3, defended+blocked on server 1
        RoundLog(1, (3, 1, 0, 0), (1,), (1, 0, 1, 0), (B, N, B, N), 0.0, 0.0),
        # no attack
        RoundLog(2, (2, 1, 1, 0), (3,), (0, 0, 0, 0), (N, N, N, N), 0.0, 0.0),
    ]
    s = summarize(logs)
    assert s["catch_rate"] == pytest.approx(1 / 4)
    assert s["takedown_rate"] == pytest.approx(1 / 4)
    assert s["attack_share"] == pytest.approx(2 / 3)
    assert s["undefended_attack_share"] == pytest.approx(2 / 4)
    assert s["undefended_catch_rate"] == pytest.approx(1 / 2)
    assert s["defended_takedown_rate"] == pytest.approx(1 / 2)


def test_summarize_nan_when_no_undefended_attacks():
    N, B = Outcome.NOT_ATTACKED, Outcome.BLOCKED
    s = summarize([RoundLog(0, (4, 0, 0, 0), (1,), (1, 0, 0, 0), (B, N, N, N), 0.0, 0.0)])
    assert math.isnan(s["undefended_catch_rate"])
    assert s["defended_takedown_rate"] == 0.0
