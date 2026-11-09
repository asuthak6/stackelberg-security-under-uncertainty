"""Print a readable summary of the Strong Stackelberg Equilibrium.

Usage: python -m security_game.inspect_equilibrium [--traps 2] ...
"""

from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from .config import GameConfig
from .model import attack_allocations
from .solver import (
    attacker_values,
    defense_level_marginals,
    expected_defense,
    solve_sse,
    trap_given_defense,
    trap_marginals,
)

TOP_K = 8
SUPPORT_TOL = 1e-6


def build_config(args: argparse.Namespace) -> GameConfig:
    overrides = {
        "traps": args.traps,
        "caught_penalty": args.caught_penalty,
        "catch_reward": args.catch_reward,
        "defense_units": args.defense_units,
        "attack_units": args.attack_units,
        "catch_prob": args.catch_prob,
    }
    return GameConfig(**{k: v for k, v in overrides.items() if v is not None})


def report(cfg: GameConfig) -> None:
    sse = solve_sse(cfg)
    no_traps = solve_sse(dataclasses.replace(cfg, traps=0))
    x = sse.distribution

    print("Config")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name:<15} {getattr(cfg, field.name)}")

    print("\nValues")
    print(f"  {'':<22}{'defender':>10}{'attacker':>10}")
    print(f"  {f'SSE (traps={cfg.traps})':<22}{sse.defender_value:>10.3f}{sse.attacker_value:>10.3f}")
    print(f"  {'SSE (traps=0)':<22}{no_traps.defender_value:>10.3f}{no_traps.attacker_value:>10.3f}")

    print(f"\nAttacker best response: {sse.response}")

    e_def = expected_defense(cfg, x)
    p_trap = trap_marginals(cfg, x)
    levels = defense_level_marginals(cfg, x)
    p_trap_given = trap_given_defense(cfg, x)
    print("\nPer server")
    print(f"  {'server':<8}{'value':>7}{'E[def]':>9}{'P(trap)':>9}   P(defense = L) / P(trap | defense = L)")
    for i in range(cfg.n):
        cells = [f"L={L}: {levels[i, L]:.3f}/{p_trap_given[i, L]:.3f}"
                 for L in range(cfg.defense_units + 1) if levels[i, L] > SUPPORT_TOL]
        print(f"  {i + 1:<8}{cfg.values[i]:>7.1f}{e_def[i]:>9.3f}{p_trap[i]:>9.3f}   "
              + "  ".join(cells))

    values = attacker_values(cfg, x)
    allocs = attack_allocations(cfg)
    order = np.argsort(-values, kind="stable")
    best = values.max()
    print(f"\nTop {TOP_K} attacker allocations against the equilibrium")
    print(f"  {'allocation':<16}{'payoff':>9}{'gap':>10}")
    for j in order[:TOP_K]:
        marker = "  <- response" if j == sse.response_index else ""
        print(f"  {str(allocs[j]):<16}{values[j]:>9.4f}{best - values[j]:>10.4f}{marker}")
    n_close = int(np.sum(values >= best - 0.05))
    print(f"  allocations within 0.05 of the best: {n_close}")

    support = int(np.sum(x > SUPPORT_TOL))
    print(f"\nDefender pure strategies with probability > {SUPPORT_TOL:g}: {support} of {len(x)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize the Strong Stackelberg Equilibrium")
    p.add_argument("--traps", type=int)
    p.add_argument("--caught-penalty", type=float)
    p.add_argument("--catch-reward", type=float)
    p.add_argument("--defense-units", type=int)
    p.add_argument("--attack-units", type=int)
    p.add_argument("--catch-prob", type=float)
    report(build_config(p.parse_args()))


if __name__ == "__main__":
    main()
