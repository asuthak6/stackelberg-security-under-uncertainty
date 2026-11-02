"""The game loop. It runs the agents against ANY environment.

This is the single model. The environment answers "did the attack succeed?"
(by formula or by real measurement); everything else here -- the agents, the
payoffs, the learning, the round structure -- is identical either way. That is
what makes the fast simulation and the live demo the same game, not two.

Payoffs are computed HERE, at the model level, from the environment's per-server
outcomes -- so they mean the same thing whichever backend produced the outcome:
    success : attacker +v_i,       defender -v_i
    blocked : attacker -block_cost, defender +block_reward
    effort  : attacker -effort_cost * a_i   (always)
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

from config import GameConfig
from agents import Attacker, Defender
from environment import Environment, FormulaEnvironment


@dataclass
class RoundLog:
    rnd: int
    defense: list[float]
    attack: list[float]
    successes: list[int]
    attacker_payoff: float
    defender_payoff: float
    value_captured: float


def _payoffs(cfg: GameConfig, attack: list[float], successes: list[int]):
    a_pay = d_pay = captured = 0.0
    for i in range(cfg.n):
        a_pay -= cfg.effort_cost * attack[i]
        if attack[i] <= 0:
            continue
        if successes[i]:
            a_pay += cfg.values[i]
            d_pay -= cfg.values[i]
            captured += cfg.values[i]
        else:
            a_pay -= cfg.block_cost
            d_pay += cfg.block_reward
    return a_pay, d_pay, captured


def play(cfg: GameConfig, env: Environment, rounds: int, defender: Defender,
         attacker: Attacker, seed: int = 0) -> list[RoundLog]:
    """Run `rounds` of the game. `env` resolves each round (formula or live)."""
    rng = random.Random(seed)
    logs: list[RoundLog] = []
    for r in range(1, rounds + 1):
        defense = defender.allocate(rng)              # leader commits
        observed = attacker.perceive(defense, rng)    # noisy observation
        attack = attacker.choose(observed, rng)       # follower best-responds
        successes = env.resolve(defense, attack, rng)  # formula OR real server
        a_pay, d_pay, captured = _payoffs(cfg, attack, successes)
        logs.append(RoundLog(r, defense, attack, successes, a_pay, d_pay, captured))
        defender.learn(attack)                        # both adapt
    return logs


def summarize(logs: list[RoundLog]) -> dict:
    n = len(logs)
    attacked = sum(1 for lg in logs for a in lg.attack if a > 1e-6)
    succ = sum(sum(lg.successes) for lg in logs)
    return {
        "rounds": n,
        "attacker_payoff_avg": sum(lg.attacker_payoff for lg in logs) / n,
        "defender_payoff_avg": sum(lg.defender_payoff for lg in logs) / n,
        "value_captured_avg": sum(lg.value_captured for lg in logs) / n,
        "attacks_launched": attacked,
        "attacks_succeeded": succ,
        "success_rate": (succ / attacked) if attacked else 0.0,
    }


def make_defender(cfg: GameConfig, strategy: str) -> Defender:
    d = Defender(cfg, strategy=strategy)
    if strategy == "sse":
        from baseline import solve_sse
        d._sse_shape = list(solve_sse(cfg.values, min(2.0, cfg.n - 0.5), 1.0).coverage)
    return d


def main() -> None:
    p = argparse.ArgumentParser(description="Run the security game on the FORMULA environment")
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--defender", choices=["adaptive", "sse", "uniform"], default="adaptive")
    p.add_argument("--attacker", choices=["distributed", "single", "random"], default="distributed")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = GameConfig(noise=args.noise)
    env = FormulaEnvironment(cfg)
    defender = make_defender(cfg, args.defender)
    attacker = Attacker(cfg, mode=args.attacker)
    s = summarize(play(cfg, env, args.rounds, defender, attacker, args.seed))

    print(f"Security game (FORMULA environment) - defender={args.defender}, "
          f"attacker={args.attacker}, noise={args.noise}")
    print(f"  rounds: {s['rounds']}")
    print(f"  attacker payoff / round : {s['attacker_payoff_avg']:+.2f}")
    print(f"  defender payoff / round : {s['defender_payoff_avg']:+.2f}")
    print(f"  attack success rate     : {s['success_rate']:.1%} "
          f"({s['attacks_succeeded']}/{s['attacks_launched']})")
    print("\n  (This is the fast backend. For the real-server backend, see run_live.py;\n"
          "   for formula-vs-real agreement, see validate.py.)")


if __name__ == "__main__":
    main()
