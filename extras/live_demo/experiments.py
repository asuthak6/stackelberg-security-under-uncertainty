"""Controlled experiments, run on the FORMULA environment.

Why the formula and not the real server here: these experiments need thousands
of rounds across many seeds and parameter settings. That is only practical on
the fast backend -- and validate.py is what justifies trusting it. Same game,
same agents (game.py); only the environment differs.

Produces CSVs and, if matplotlib is present, figures:
  1. noise sweep        -> fig_noise.png        (the core research result)
  2. strategy grid       -> fig_strategies.png
  3. learning curve      -> fig_learning.png

    python experiments.py --seeds 20
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from config import GameConfig
from agents import Attacker
from environment import FormulaEnvironment
from game import play, summarize, make_defender


def run_cell(noise, defender, attacker, rounds, seeds):
    atk, dfn, suc = [], [], []
    for seed in range(seeds):
        cfg = GameConfig(noise=noise)
        env = FormulaEnvironment(cfg)
        d = make_defender(cfg, defender)
        a = Attacker(cfg, mode=attacker)
        s = summarize(play(cfg, env, rounds, d, a, seed))
        atk.append(s["attacker_payoff_avg"])
        dfn.append(s["defender_payoff_avg"])
        suc.append(s["success_rate"])
    return {"atk_payoff": st.mean(atk), "def_payoff": st.mean(dfn), "success": st.mean(suc)}


def exp_noise(rounds, seeds, outdir, plt):
    levels = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    rows = [{"noise": nz, **run_cell(nz, "adaptive", "distributed", rounds, seeds)}
            for nz in levels]
    with (outdir / "results_noise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nExperiment 1 - noise sweep (adaptive defender, distributed attacker)")
    print(f"  {'noise':>6} {'atk payoff':>11} {'def payoff':>11} {'success':>8}")
    for r in rows:
        print(f"  {r['noise']:>6.2f} {r['atk_payoff']:>+11.2f} {r['def_payoff']:>+11.2f} {r['success']:>7.1%}")
    if plt:
        fig, ax = plt.subplots(figsize=(7, 4.3))
        ax.plot([r["noise"] for r in rows], [r["atk_payoff"] for r in rows], "o-",
                color="#c62828", label="attacker payoff")
        ax.plot([r["noise"] for r in rows], [r["def_payoff"] for r in rows], "s-",
                color="#1565c0", label="defender payoff")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("attacker observation noise (std dev)")
        ax.set_ylabel("average payoff per round")
        ax.set_title("More perception noise: attacker loses, defender gains")
        ax.legend(); fig.tight_layout(); fig.savefig(outdir / "fig_noise.png", dpi=130); plt.close(fig)


def exp_strategies(rounds, seeds, outdir, plt):
    defs, atks = ["adaptive", "sse", "uniform"], ["distributed", "single"]
    rows = []
    for d in defs:
        for a in atks:
            rows.append({"def_strategy": d, "atk_strategy": a, **run_cell(1.0, d, a, rounds, seeds)})
    with (outdir / "results_strategies.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nExperiment 2 - strategy grid (noise = 1.0)")
    print(f"  {'defender':>10} {'attacker':>12} {'atk payoff':>11} {'def payoff':>11}")
    for r in rows:
        print(f"  {r['def_strategy']:>10} {r['atk_strategy']:>12} "
              f"{r['atk_payoff']:>+11.2f} {r['def_payoff']:>+11.2f}")
    if plt:
        import numpy as np
        fig, ax = plt.subplots(figsize=(7.5, 4.3))
        x = np.arange(len(defs)); w = 0.38
        for k, a in enumerate(atks):
            vals = [next(r["def_payoff"] for r in rows
                         if r["def_strategy"] == d and r["atk_strategy"] == a) for d in defs]
            ax.bar(x + (k - 0.5) * w, vals, w, label=f"vs {a} attacker")
        ax.set_xticks(x); ax.set_xticklabels(defs)
        ax.set_ylabel("defender payoff per round (higher = better)")
        ax.set_title("Defender strategy comparison (noise = 1.0)")
        ax.legend(); fig.tight_layout(); fig.savefig(outdir / "fig_strategies.png", dpi=130); plt.close(fig)


def exp_learning(rounds, seeds, outdir, plt):
    if not plt:
        return
    import numpy as np
    series = {}
    for d in ("adaptive", "uniform"):
        acc = np.zeros(rounds)
        for seed in range(seeds):
            cfg = GameConfig(noise=1.0)
            env = FormulaEnvironment(cfg)
            dd = make_defender(cfg, d); aa = Attacker(cfg, "distributed")
            logs = play(cfg, env, rounds, dd, aa, seed)
            acc += np.array([lg.defender_payoff for lg in logs])
        series[d] = acc / seeds
    roll = lambda x, w=20: np.convolve(x, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for d, c in (("adaptive", "#1565c0"), ("uniform", "#9e9e9e")):
        ax.plot(roll(series[d]), color=c, label=f"{d} defender")
    ax.set_xlabel("round (20-round rolling average)")
    ax.set_ylabel("defender payoff")
    ax.set_title("The adaptive defender improves as it learns the attacker")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "fig_learning.png", dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Security-game experiments (formula backend)")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("(matplotlib not installed - CSVs only)")
    exp_noise(args.rounds, args.seeds, outdir, plt)
    exp_strategies(args.rounds, args.seeds, outdir, plt)
    exp_learning(args.rounds, args.seeds, outdir, plt)
    print(f"\nDone. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
