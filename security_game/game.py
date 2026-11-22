"""Repeated game loop. See DESIGN.md, section "Repeated game and perception".

Usage: python -m security_game.game --defender adaptive --attacker naive --rounds 500 --seed 0
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

from .attackers import (
    Attacker,
    BayesianAttacker,
    LearnedStrategicAttacker,
    NaiveAttacker,
    Observation,
    OmniscientAttacker,
    RandomAttacker,
    StrategicAttacker,
)
from .config import GameConfig
from .defenders import AdaptiveDefender, Defender, SSEDefender
from .model import AttackAlloc, Outcome, realized_payoffs, sample_outcomes

DEFENDERS = {"sse": SSEDefender, "adaptive": AdaptiveDefender}
ATTACKERS = ("random", "naive", "bayesian", "strategic", "omniscient",
             "bayesian_uniform", "strategic_uniform", "strategic_learned")


@dataclass(frozen=True)
class RoundLog:
    round: int
    defense: tuple[int, ...]
    traps: tuple[int, ...]
    attack: AttackAlloc
    outcomes: tuple[Outcome, ...]
    attacker_payoff: float
    defender_payoff: float


def make_defender(name: str, cfg: GameConfig) -> Defender:
    try:
        return DEFENDERS[name](cfg)
    except KeyError:
        raise ValueError(f"unknown defender {name!r}") from None


def make_attacker(name: str, cfg: GameConfig, defender: Defender, **kwargs) -> Attacker:
    """Build an attacker by name. Extra keyword arguments go to the attacker's
    constructor (e.g. model_rate / model_floor for the strategic attacker)."""
    if name == "omniscient":
        return OmniscientAttacker(cfg, defender, **kwargs)
    if name in ("random", "naive"):
        return {"random": RandomAttacker, "naive": NaiveAttacker}[name](cfg, **kwargs)
    if name == "strategic_learned":
        return LearnedStrategicAttacker(cfg, **kwargs)
    base, _, prior = name.partition("_")
    classes = {"bayesian": BayesianAttacker, "strategic": StrategicAttacker}
    if base not in classes or prior not in ("", "uniform"):
        raise ValueError(f"unknown attacker {name!r}")
    return classes[base](cfg, prior=prior or "informed", **kwargs)


def play(cfg: GameConfig, defender: Defender, attacker: Attacker,
         rounds: int, seed: int,
         before_round: Callable[[int], None] | None = None) -> list[RoundLog]:
    """Play ``rounds`` rounds and return one log entry per round.

    ``before_round(t)``, if given, is called at the start of round t, before
    either agent acts. It may inspect the agents' state but must not change it;
    it draws no random numbers, so the game is unchanged.
    """
    def_rng = random.Random(f"defender:{seed}")
    att_rng = random.Random(f"attacker:{seed}")
    env_rng = random.Random(f"environment:{seed}")

    logs = []
    for t in range(rounds):
        if before_round is not None:
            before_round(t)
        defense, traps = defender.pure_strategies[defender.act(def_rng)]
        attack = attacker.act(att_rng)
        outcomes = sample_outcomes(cfg, defense, traps, attack, env_rng)
        u_att, u_def = realized_payoffs(cfg, attack, outcomes)
        trap_set = set(traps)
        trap_seen = {i: i in trap_set for i in range(cfg.n) if attack[i] > 0}
        attacker.observe(Observation(defense=defense, trap_seen=trap_seen,
                                     outcomes=outcomes, attack=attack))
        defender.observe(attack, outcomes)
        logs.append(RoundLog(t, defense, traps, attack, outcomes, u_att, u_def))
    return logs


def summarize(logs: list[RoundLog], per_server: bool = False) -> dict[str, float]:
    """Per-run metrics. Rates are NaN when their denominator is zero.

    With per_server=True, also per server i (1-based in the key):
    attack_unit_share_s{i}, the share of all attack units over the run sent to
    server i (NaN if no units were sent), and attack_rate_s{i}, the share of
    rounds in which server i was attacked.
    """
    attacked = catches = successes = attack_rounds = 0
    undefended = undefended_catches = defended = defended_successes = 0
    n_servers = len(logs[0].attack) if logs else 0
    units = [0] * n_servers
    rounds_hit = [0] * n_servers
    for log in logs:
        for i, a in enumerate(log.attack):
            units[i] += a
            rounds_hit[i] += a > 0
        hit = False
        for i, outcome in enumerate(log.outcomes):
            if outcome is Outcome.NOT_ATTACKED:
                continue
            hit = True
            attacked += 1
            caught = outcome is Outcome.CAUGHT
            success = outcome is Outcome.SUCCESS
            catches += caught
            successes += success
            if log.defense[i] == 0:
                undefended += 1
                undefended_catches += caught
            else:
                defended += 1
                defended_successes += success
        attack_rounds += hit

    def ratio(a: int, b: int) -> float:
        return a / b if b else float("nan")

    n = len(logs)
    summary = {
        "attacker_payoff": sum(l.attacker_payoff for l in logs) / n,
        "defender_payoff": sum(l.defender_payoff for l in logs) / n,
        "catch_rate": ratio(catches, attacked),
        "takedown_rate": ratio(successes, attacked),
        "attack_share": attack_rounds / n,
        "undefended_attack_share": ratio(undefended, attacked),
        "undefended_catch_rate": ratio(undefended_catches, undefended),
        "defended_takedown_rate": ratio(defended_successes, defended),
    }
    if per_server:
        for i in range(n_servers):
            summary[f"attack_unit_share_s{i + 1}"] = ratio(units[i], sum(units))
        for i in range(n_servers):
            summary[f"attack_rate_s{i + 1}"] = rounds_hit[i] / n
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Play the repeated security game")
    p.add_argument("--defender", choices=sorted(DEFENDERS), default="adaptive")
    p.add_argument("--attacker", choices=ATTACKERS, default="naive")
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = GameConfig()
    defender = make_defender(args.defender, cfg)
    attacker = make_attacker(args.attacker, cfg, defender)
    summary = summarize(play(cfg, defender, attacker, args.rounds, args.seed))
    print(f"{args.attacker} attacker vs {args.defender} defender, "
          f"{args.rounds} rounds, seed {args.seed}")
    for key, value in summary.items():
        print(f"  {key:<26}{value:>8.3f}")


if __name__ == "__main__":
    main()
