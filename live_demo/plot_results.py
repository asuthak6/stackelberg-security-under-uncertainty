"""Turn demo_raw.csv into figures for the report.

Produces two PNGs:
  fig_goodput.png  -- legit availability (goodput %) per phase, a bar chart.
  fig_latency.png  -- legit request latency over time, one panel per phase,
                      so you can literally see the server degrade and recover.

Usage:
    python plot_results.py            # reads demo_raw.csv / demo_results.csv
    python plot_results.py --raw demo_raw.csv

Requires matplotlib (pip install matplotlib). This is the only part of the
project that needs a package outside the standard library; the simulation and
the live demo do not.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless; just write files
import matplotlib.pyplot as plt

PHASE_LABEL = {
    "A_baseline": "A: baseline\n(no attack)",
    "B_attack": "B: attack\n(no defense)",
    "C_mitigate_capacity": "C: +capacity",
    "D_mitigate_admission": "D: +admission\ncontrol",
}
PHASE_ORDER = ["A_baseline", "B_attack", "C_mitigate_capacity", "D_mitigate_admission"]


def load_raw(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "phase": r["phase"],
                "t": float(r["t_start_s"]),
                "latency_ms": float(r["latency_ms"]),
                "status": int(r["status"]),
                "kind": r["kind"],
            })
    return rows


def plot_goodput(rows, out: Path) -> None:
    legit = [r for r in rows if r["kind"] == "legit"]
    total = defaultdict(int)
    ok = defaultdict(int)
    for r in legit:
        total[r["phase"]] += 1
        if r["status"] == 200:
            ok[r["phase"]] += 1
    phases = [p for p in PHASE_ORDER if p in total]
    vals = [100 * ok[p] / total[p] for p in phases]
    colors = ["#2e7d32", "#c62828", "#1565c0", "#f9a825"][:len(phases)]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar([PHASE_LABEL[p] for p in phases], vals, color=colors)
    ax.set_ylabel("Legitimate-user goodput (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Service availability for a normal user, by phase")
    ax.axhline(50, ls="--", lw=1, color="gray")
    ax.text(len(phases) - 0.4, 52, "outage threshold", color="gray", fontsize=8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_latency(rows, out: Path) -> None:
    legit = [r for r in rows if r["kind"] == "legit"]
    phases = [p for p in PHASE_ORDER if any(r["phase"] == p for r in legit)]
    fig, axes = plt.subplots(1, len(phases), figsize=(3.3 * len(phases), 4),
                             sharey=True)
    if len(phases) == 1:
        axes = [axes]
    for ax, p in zip(axes, phases):
        pts = [r for r in legit if r["phase"] == p]
        ok = [(r["t"], r["latency_ms"]) for r in pts if r["status"] == 200]
        bad = [(r["t"], r["latency_ms"]) for r in pts if r["status"] != 200]
        if ok:
            ax.scatter([t for t, _ in ok], [l for _, l in ok], s=14,
                       color="#1565c0", label="served (200)")
        if bad:
            ax.scatter([t for t, _ in bad], [l for _, l in bad], s=20,
                       color="#c62828", marker="x", label="failed")
        ax.set_title(PHASE_LABEL[p], fontsize=10)
        ax.set_xlabel("time (s)")
        ax.set_yscale("log")
    axes[0].set_ylabel("legit request latency (ms, log scale)")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Legitimate-user latency over time (failed requests in red)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot DDoS demo results")
    ap.add_argument("--raw", default="demo_raw.csv")
    ap.add_argument("--goodput-out", default="fig_goodput.png")
    ap.add_argument("--latency-out", default="fig_latency.png")
    args = ap.parse_args()

    rows = load_raw(Path(args.raw))
    plot_goodput(rows, Path(args.goodput_out))
    plot_latency(rows, Path(args.latency_out))
    print(f"wrote {args.goodput_out} and {args.latency_out}")


if __name__ == "__main__":
    main()
