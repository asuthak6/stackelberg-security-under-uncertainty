"""Run the game against the REAL server (the live backend).

This is the "real DDoS" demonstration -- but it is the SAME game as game.py and
experiments.py, just with the live environment plugged in. The defender's agent
commits an allocation, the attacker's agent floods, and the outcome of each round
is MEASURED on a real loopback HTTP server instead of drawn from the formula.

It is slower than the formula backend (each attacked server is really attacked),
so run a handful of rounds, not thousands.

    python run_live.py --rounds 8

Everything is loopback-only; the traffic clients refuse any non-loopback target.
"""
from __future__ import annotations

import argparse

from config import GameConfig
from agents import Attacker
from environment import LiveEnvironment
from game import play, summarize, make_defender


def main() -> None:
    p = argparse.ArgumentParser(description="Run the security game on the REAL-server backend")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--noise", type=float, default=0.3)
    p.add_argument("--defender", choices=["adaptive", "sse", "uniform"], default="adaptive")
    p.add_argument("--attacker", choices=["distributed", "single", "random"], default="distributed")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = GameConfig(noise=args.noise)
    defender = make_defender(cfg, args.defender)
    attacker = Attacker(cfg, mode=args.attacker)

    print(f"Security game (LIVE server backend) - defender={args.defender}, "
          f"attacker={args.attacker}")
    print(f"  Each round, every attacked server is really flooded on 127.0.0.1 and\n"
          f"  its outage is measured. {args.rounds} rounds...\n")

    with LiveEnvironment(cfg) as env:
        logs = play(cfg, env, args.rounds, defender, attacker, args.seed)

    hdr = f"  {'round':>5}  {'defense':>18}  {'attack':>18}  outcome"
    print(hdr)
    for lg in logs:
        d = "[" + ",".join(f"{x:.0f}" for x in lg.defense) + "]"
        a = "[" + ",".join(f"{x:.0f}" for x in lg.attack) + "]"
        marks = " ".join(f"S{i+1}:{'OUT' if s else 'def'}"
                         for i, s in enumerate(lg.successes) if lg.attack[i] > 0)
        print(f"  {lg.rnd:>5}  {d:>18}  {a:>18}  {marks}")

    s = summarize(logs)
    print(f"\n  measured attack success rate: {s['success_rate']:.0%} "
          f"({s['attacks_succeeded']}/{s['attacks_launched']})")
    print(f"  attacker payoff/round {s['attacker_payoff_avg']:+.2f}  "
          f"defender payoff/round {s['defender_payoff_avg']:+.2f}")
    print("\n  Same agents and payoffs as the formula game (game.py) -- only the\n"
          "  environment changed. validate.py shows the two backends agree.")


if __name__ == "__main__":
    main()
