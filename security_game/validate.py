"""Validation: does the fast formula predict what the real server does?

This is the script that proves the two backends are ONE model. For a grid of
(defense, attack) levels on a single server, it:

  1. MEASURES the real outage probability on the live loopback server
     (many short real attacks per cell, fraction that cause an outage), and
  2. fits the formula's single resilience parameter k to those measurements, then
  3. reports how closely the fitted formula a/(a+d+k) matches the measured curve.

A close match means the formula is a trustworthy fast surrogate for the real
system: we can solve for equilibrium and run thousands of rounds on the formula
(game.py, experiments.py) knowing it behaves like the real server (run_live.py).

    python validate.py            # ~1 min
    python validate.py --trials 12 --duration 0.5

Needs scipy (fit) and, for the figure, matplotlib.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import minimize_scalar

from config import GameConfig, success_prob
from environment import LiveEnvironment


def measure_grid(defenses, attacks, trials, duration):
    """Return measured outage probability for each (defense, attack) cell."""
    cfg = GameConfig()
    live = np.zeros((len(defenses), len(attacks)))
    # The default LiveEnvironment is already tuned so the outage gradient is
    # visible across the measured range (a too-robust server would never fall).
    with LiveEnvironment(cfg, duration=duration) as env:
        for di, d in enumerate(defenses):
            for ai, a in enumerate(attacks):
                outages = sum(env._measure_one(d, a) for _ in range(trials))
                live[di, ai] = outages / trials
    return live


def fit_resilience(defenses, attacks, live):
    """Find the k that makes a/(a+d+k) best match the measured outage grid."""
    def sse(k):
        total = 0.0
        for di, d in enumerate(defenses):
            for ai, a in enumerate(attacks):
                total += (success_prob(a, d, k) - live[di, ai]) ** 2
        return total
    res = minimize_scalar(sse, bounds=(0.01, 20.0), method="bounded")
    return res.x, res.fun


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the formula against the real server")
    ap.add_argument("--trials", type=int, default=8, help="real attacks measured per cell")
    ap.add_argument("--duration", type=float, default=0.5, help="seconds per measured attack")
    ap.add_argument("--out", default="fig_validation.png")
    args = ap.parse_args()

    defenses = [0.0, 1.5, 3.0]
    attacks = [1.0, 3.0, 5.0]
    print(f"Measuring {len(defenses)*len(attacks)} cells x {args.trials} real attacks "
          f"({args.duration:g}s each)... this takes a minute.")
    live = measure_grid(defenses, attacks, args.trials, args.duration)

    k, sse = fit_resilience(defenses, attacks, live)
    # Goodness of fit: 1 - SSE/TSS over all cells.
    flat = live.flatten()
    tss = float(((flat - flat.mean()) ** 2).sum()) or 1e-9
    r2 = 1 - sse / tss

    print(f"\nFitted contest resilience k = {k:.2f}   (fit R^2 = {r2:.3f})\n")
    print("  outage probability: measured (real server)  vs  formula a/(a+d+k)")
    header = "  defense \\ attack  " + "  ".join(f"a={a:g}" for a in attacks)
    print(header)
    for di, d in enumerate(defenses):
        meas = "  ".join(f"{live[di, ai]:.2f}" for ai in range(len(attacks)))
        pred = "  ".join(f"{success_prob(a, d, k):.2f}" for a in attacks)
        print(f"  d={d:<3g} measured  {meas}")
        print(f"  d={d:<3g} formula   {pred}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = np.linspace(0.2, attacks[-1], 100)
        colors = ["#c62828", "#1565c0", "#2e7d32"]
        fig, ax = plt.subplots(figsize=(7, 4.4))
        for di, d in enumerate(defenses):
            c = colors[di % len(colors)]
            ax.plot(attacks, live[di], "o", color=c, label=f"measured, defense={d:g}")
            ax.plot(xs, [success_prob(a, d, k) for a in xs], "-", color=c, alpha=0.7,
                    label=f"formula, defense={d:g}")
        ax.set_xlabel("attack intensity")
        ax.set_ylabel("outage probability")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Formula vs. real server (fitted k={k:.2f}, R²={r2:.2f})")
        ax.legend(fontsize=8, ncol=1)
        fig.tight_layout()
        fig.savefig(args.out, dpi=130)
        plt.close(fig)
        print(f"\nSaved {args.out}")
    except ImportError:
        print("\n(matplotlib not installed - skipped the figure)")


if __name__ == "__main__":
    main()
