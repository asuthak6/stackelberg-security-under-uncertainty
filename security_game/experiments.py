"""Experiments E1-E6, paired comparisons, figures, and summary tables.

See DESIGN.md, section "Experiments". Every cell uses seeds 0..S-1, so seed s
is shared by all attacker/defender pairs (paired design, common random numbers).

Usage: python -m security_game.experiments [--seeds 50] [--rounds 500] [--quick] [--out results/]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import t as t_dist  # noqa: E402

from .config import GameConfig  # noqa: E402
from .game import make_attacker, make_defender, play, summarize  # noqa: E402
from .model import Outcome  # noqa: E402
from .solver import solve_sse  # noqa: E402

ATTACKERS = ("random", "naive", "bayesian", "bayesian_uniform",
             "strategic", "strategic_uniform", "omniscient")
DEFENDERS = ("sse", "adaptive")
METRICS = ("attacker_payoff", "defender_payoff", "catch_rate", "takedown_rate",
           "attack_share", "undefended_attack_share", "undefended_catch_rate")
LEARNING_ATTACKERS = ("naive", "bayesian", "strategic", "omniscient")
TRAP_COUNTS = (0, 1, 2)
TRAP_ATTACKERS = ("naive", "bayesian", "strategic", "omniscient")
PRIOR_ATTACKERS = ("bayesian", "bayesian_uniform", "strategic", "strategic_uniform")
LURE_ATTACKERS = ATTACKERS
MEMORY_WINDOWS = (10, 50, 200)
MEMORY_ATTACKERS = ("naive", "bayesian", "strategic")
ROLLING_WINDOW = 25
PAIRED_ATTACKER_PAIRS = (("strategic", "naive"), ("bayesian", "naive"),
                         ("strategic", "bayesian"))

DEFAULT_SEEDS, DEFAULT_ROUNDS = 50, 500
QUICK_SEEDS, QUICK_ROUNDS = 3, 60

# One fixed color per attacker in every figure (validated categorical palette).
# Uniform-prior ablations are also hatched; line charts add distinct markers.
COLORS = {
    "random": "#eda100",
    "naive": "#e34948",
    "bayesian": "#2a78d6",
    "bayesian_uniform": "#1baf7a",
    "strategic": "#4a3aa7",
    "strategic_uniform": "#e87ba4",
    "omniscient": "#008300",
}
HATCHES = {"bayesian_uniform": "///", "strategic_uniform": "///"}
MARKERS = {"random": "v", "naive": "o", "bayesian": "s", "bayesian_uniform": "P",
           "strategic": "^", "strategic_uniform": "X", "omniscient": "D"}
LINESTYLES = {"omniscient": "--"}
LABELS = {
    "random": "Random",
    "naive": "Naive",
    "bayesian": "Bayesian",
    "bayesian_uniform": "Bayesian\n(uniform prior)",
    "strategic": "Strategic",
    "strategic_uniform": "Strategic\n(uniform prior)",
    "omniscient": "Omniscient",
}
DEFENDER_LABELS = {"sse": "SSE defender", "adaptive": "Adaptive defender"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DPI = 150

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------
# Running games
# --------------------------------------------------------------------------

@dataclass
class Run:
    summary: dict[str, float]
    payoff: np.ndarray   # attacker payoff per round
    caught: np.ndarray   # 1.0 if the attacker was caught on any server that round


class Runner:
    """Runs and caches one game per (config, defender, attacker, seed)."""

    def __init__(self, seeds: int, rounds: int):
        self.seeds = list(range(seeds))
        self.rounds = rounds
        self._cache: dict[tuple, Run] = {}

    def run(self, cfg: GameConfig, defender: str, attacker: str, seed: int) -> Run:
        key = (cfg, defender, attacker, seed)
        if key not in self._cache:
            d = make_defender(defender, cfg)
            a = make_attacker(attacker, cfg, d)
            logs = play(cfg, d, a, self.rounds, seed)
            self._cache[key] = Run(
                summary=summarize(logs),
                payoff=np.array([log.attacker_payoff for log in logs]),
                caught=np.array([float(Outcome.CAUGHT in log.outcomes) for log in logs]),
            )
        return self._cache[key]

    def cell(self, cfg: GameConfig, defender: str, attacker: str) -> list[Run]:
        return [self.run(cfg, defender, attacker, s) for s in self.seeds]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def mean_ci(values: Iterable[float]) -> tuple[float, float, float, int]:
    """Mean and 95% t-interval, ignoring NaN. Returns (mean, low, high, n)."""
    v = np.array([x for x in values if not math.isnan(x)], dtype=float)
    n = len(v)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    m = float(v.mean())
    if n < 2:
        return m, math.nan, math.nan, n
    half = float(t_dist.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n))
    return m, m - half, m + half, n


def paired_stats(diffs: Sequence[float]) -> dict[str, float]:
    m, lo, hi, n = mean_ci(diffs)
    d = np.asarray(diffs, dtype=float)
    return {"n": n, "mean_diff": m, "ci_low": lo, "ci_high": hi,
            "seeds_first_higher": int(np.sum(d > 0)),
            "seeds_second_higher": int(np.sum(d < 0)),
            "ties": int(np.sum(d == 0))}


def rolling_band(per_seed: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    """Trailing rolling mean per seed, then mean and 95% band across seeds."""
    c = np.cumsum(np.pad(per_seed, ((0, 0), (1, 0))), axis=1)
    roll = (c[:, window:] - c[:, :-window]) / window
    n = roll.shape[0]
    mean = roll.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    half = t_dist.ppf(0.975, n - 1) * roll.std(axis=0, ddof=1) / math.sqrt(n)
    return mean, mean - half, mean + half


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------

def _fmt(x) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else f"{x:.6f}"
    return str(x)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def summary_rows(seed_rows: list[dict], keys: Sequence[str],
                 metrics: Sequence[str]) -> list[dict]:
    """Long-format summary: one row per group and metric with mean and 95% t-interval."""
    groups: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for group, rows in groups.items():
        for metric in metrics:
            m, lo, hi, n = mean_ci(r[metric] for r in rows)
            out.append({**dict(zip(keys, group)), "metric": metric, "n": n,
                        "mean": m, "ci_low": lo, "ci_high": hi})
    return out


def lookup(summary: list[dict], metric: str, **keys) -> dict:
    for row in summary:
        if row["metric"] == metric and all(row[k] == v for k, v in keys.items()):
            return row
    raise KeyError((metric, keys))


def _pm(row: dict) -> str:
    """'mean ± half-width' for markdown tables."""
    if math.isnan(row["mean"]):
        return "n/a"
    if math.isnan(row["ci_low"]):
        return f"{row['mean']:+.3f}"
    return f"{row['mean']:+.3f} ± {(row['ci_high'] - row['ci_low']) / 2:.3f}"


def md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------

RATE_METRICS = {"catch_rate", "takedown_rate", "attack_share",
                "undefended_attack_share", "undefended_catch_rate"}


def _yerr(rows: list[dict]) -> np.ndarray:
    """Asymmetric error bars; intervals of rate metrics are clipped to [0, 1] for display."""
    lo, hi = [], []
    for r in rows:
        low, high = r["ci_low"], r["ci_high"]
        if r.get("metric") in RATE_METRICS:
            low, high = max(low, 0.0), min(high, 1.0)
        lo.append(0.0 if math.isnan(low) else r["mean"] - low)
        hi.append(0.0 if math.isnan(high) else high - r["mean"])
    return np.array([lo, hi])


def bar_panel(ax, attackers: Sequence[str], rows: list[dict], title: str,
              ylabel: str | None, reference: tuple[float, str] | None = None) -> None:
    x = np.arange(len(attackers))
    means = [r["mean"] for r in rows]
    ax.bar(x, means, width=0.7, color=[COLORS[a] for a in attackers],
           hatch=[HATCHES.get(a, "") for a in attackers], edgecolor="white",
           linewidth=0.8, zorder=2)
    ax.errorbar(x, means, yerr=_yerr(rows), fmt="none", ecolor=INK,
                elinewidth=1.2, capsize=3, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    if reference is not None:
        ax.axhline(reference[0], color=INK, linestyle=":", linewidth=1.2,
                   label=reference[1], zorder=1)
        ax.legend(loc="best", fontsize=8)
    ax.set_xticks(x, [LABELS[a] for a in attackers], rotation=35, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def save(fig, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def e1_main_grid(runner: Runner, cfg: GameConfig) -> tuple[list[dict], list[dict]]:
    rows = []
    for d in DEFENDERS:
        omni = runner.cell(cfg, d, "omniscient")
        for a in ATTACKERS:
            for seed, run, ref in zip(runner.seeds, runner.cell(cfg, d, a), omni):
                rows.append({"defender": d, "attacker": a, "seed": seed,
                             **{m: run.summary[m] for m in METRICS},
                             "perception_gap": ref.summary["attacker_payoff"]
                             - run.summary["attacker_payoff"]})
    return rows, summary_rows(rows, ("defender", "attacker"), METRICS + ("perception_gap",))


def e2_learning(runner: Runner, cfg: GameConfig) -> list[dict]:
    window = min(ROLLING_WINDOW, runner.rounds)
    rows = []
    for d in DEFENDERS:
        for a in LEARNING_ATTACKERS:
            runs = runner.cell(cfg, d, a)
            pay = rolling_band(np.array([r.payoff for r in runs]), window)
            catch = rolling_band(np.array([r.caught for r in runs]), window)
            for k in range(len(pay[0])):
                rows.append({"defender": d, "attacker": a, "round": k + window,
                             "payoff_mean": pay[0][k], "payoff_ci_low": pay[1][k],
                             "payoff_ci_high": pay[2][k], "catch_mean": catch[0][k],
                             "catch_ci_low": catch[1][k], "catch_ci_high": catch[2][k]})
    return rows


def e3_traps(runner: Runner, cfg: GameConfig) -> tuple[list[dict], list[dict]]:
    rows, eq = [], {}
    for T in TRAP_COUNTS:
        cfg_t = dataclasses.replace(cfg, traps=T)
        sse = solve_sse(cfg_t)
        eq[T] = (sse.defender_value, sse.attacker_value)
        for d in DEFENDERS:
            for a in TRAP_ATTACKERS:
                for seed, run in zip(runner.seeds, runner.cell(cfg_t, d, a)):
                    rows.append({"traps": T, "defender": d, "attacker": a, "seed": seed,
                                 "defender_payoff": run.summary["defender_payoff"],
                                 "attacker_payoff": run.summary["attacker_payoff"]})
    summary = summary_rows(rows, ("traps", "defender", "attacker"),
                           ("defender_payoff", "attacker_payoff"))
    for row in summary:
        row["eq_defender_value"], row["eq_attacker_value"] = eq[row["traps"]]
    return rows, summary


def e6_memory(runner: Runner, cfg: GameConfig) -> tuple[list[dict], list[dict]]:
    rows = []
    for W in MEMORY_WINDOWS:
        cfg_w = dataclasses.replace(cfg, memory=W)
        for a in MEMORY_ATTACKERS:
            for seed, run in zip(runner.seeds, runner.cell(cfg_w, "adaptive", a)):
                rows.append({"memory": W, "defender": "adaptive", "attacker": a,
                             "seed": seed,
                             "attacker_payoff": run.summary["attacker_payoff"]})
    return rows, summary_rows(rows, ("memory", "defender", "attacker"), ("attacker_payoff",))


def paired_comparisons(e1_rows: list[dict], eq_attacker_value: float) -> list[dict]:
    def series(d: str, a: str, metric: str) -> np.ndarray:
        return np.array([r[metric] for r in e1_rows
                         if r["defender"] == d and r["attacker"] == a])

    out = []
    for first, second in PAIRED_ATTACKER_PAIRS:
        for d in DEFENDERS:
            diffs = series(d, first, "attacker_payoff") - series(d, second, "attacker_payoff")
            out.append({"comparison": f"{first} - {second}", "defender": d,
                        "attacker": f"{first} vs {second}", "metric": "attacker_payoff",
                        **paired_stats(diffs)})
    for a in ATTACKERS:
        diffs = series("sse", a, "defender_payoff") - series("adaptive", a, "defender_payoff")
        out.append({"comparison": "sse - adaptive", "defender": "sse vs adaptive",
                    "attacker": a, "metric": "defender_payoff", **paired_stats(diffs)})
    for a in ATTACKERS:
        diffs = series("sse", a, "attacker_payoff") - eq_attacker_value
        out.append({"comparison": "attacker - equilibrium value", "defender": "sse",
                    "attacker": a, "metric": "attacker_payoff", **paired_stats(diffs)})
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_e1(summary: list[dict], metric: str, ylabel: str, path: Path,
           attackers: Sequence[str] = ATTACKERS,
           references: dict[str, tuple[float, str]] | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, d in zip(axes, DEFENDERS):
        rows = [lookup(summary, metric, defender=d, attacker=a) for a in attackers]
        bar_panel(ax, attackers, rows, DEFENDER_LABELS[d],
                  ylabel if ax is axes[0] else None, (references or {}).get(d))
    save(fig, path)


def fig_e2(rows: list[dict], key: str, ylabel: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, d in zip(axes, DEFENDERS):
        for a in LEARNING_ATTACKERS:
            sub = [r for r in rows if r["defender"] == d and r["attacker"] == a]
            x = [r["round"] for r in sub]
            ax.fill_between(x, [r[f"{key}_ci_low"] for r in sub],
                            [r[f"{key}_ci_high"] for r in sub],
                            color=COLORS[a], alpha=0.18, linewidth=0)
            ax.plot(x, [r[f"{key}_mean"] for r in sub], color=COLORS[a],
                    linestyle=LINESTYLES.get(a, "-"), linewidth=2, marker=MARKERS[a],
                    markersize=6, markevery=max(1, len(x) // 8), label=LABELS[a])
        ax.set_title(DEFENDER_LABELS[d])
        ax.set_xlabel("Round")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="best", fontsize=8)
    save(fig, path)


def fig_e3(summary: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    offsets = np.linspace(-0.12, 0.12, len(TRAP_ATTACKERS))
    for ax, d in zip(axes, DEFENDERS):
        for off, a in zip(offsets, TRAP_ATTACKERS):
            rows = [lookup(summary, "defender_payoff", traps=T, defender=d, attacker=a)
                    for T in TRAP_COUNTS]
            ax.errorbar(np.array(TRAP_COUNTS) + off, [r["mean"] for r in rows],
                        yerr=_yerr(rows), color=COLORS[a], marker=MARKERS[a],
                        markersize=7, linewidth=2, linestyle=LINESTYLES.get(a, "-"),
                        capsize=3, label=LABELS[a])
        if d == "sse":
            eq = [lookup(summary, "defender_payoff", traps=T, defender=d,
                         attacker=TRAP_ATTACKERS[0])["eq_defender_value"]
                  for T in TRAP_COUNTS]
            ax.plot(TRAP_COUNTS, eq, color=INK, linestyle=":", marker="_",
                    markersize=14, linewidth=1.2, label="SSE equilibrium value")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(TRAP_COUNTS)
        ax.set_xlabel("Traps per round (T)")
        ax.set_title(DEFENDER_LABELS[d])
    axes[0].set_ylabel("Defender payoff (per round)")
    axes[0].legend(loc="best", fontsize=8)
    save(fig, path)


def fig_grid(summary: list[dict], attackers: Sequence[str],
             metrics: Sequence[tuple[str, str]], path: Path) -> None:
    """Rows = metrics, columns = defenders, bars = attackers."""
    fig, axes = plt.subplots(len(metrics), 2, figsize=(10, 3.9 * len(metrics)),
                             sharey="row", squeeze=False)
    for i, (metric, ylabel) in enumerate(metrics):
        for j, d in enumerate(DEFENDERS):
            rows = [lookup(summary, metric, defender=d, attacker=a) for a in attackers]
            bar_panel(axes[i, j], attackers, rows, DEFENDER_LABELS[d],
                      ylabel if j == 0 else None)
    fig.tight_layout()
    save(fig, path)


def fig_e6(summary: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    x = np.arange(len(MEMORY_WINDOWS))
    offsets = np.linspace(-0.08, 0.08, len(MEMORY_ATTACKERS))
    for off, a in zip(offsets, MEMORY_ATTACKERS):
        rows = [lookup(summary, "attacker_payoff", memory=W, defender="adaptive", attacker=a)
                for W in MEMORY_WINDOWS]
        ax.errorbar(x + off, [r["mean"] for r in rows], yerr=_yerr(rows),
                    color=COLORS[a], marker=MARKERS[a], markersize=7, linewidth=2,
                    capsize=3, label=LABELS[a])
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, [str(W) for W in MEMORY_WINDOWS])
    ax.set_xlabel("Attacker memory W (rounds)")
    ax.set_ylabel("Attacker payoff (per round)")
    ax.set_title("Adaptive defender")
    ax.legend(loc="best", fontsize=8)
    save(fig, path)


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------

def write_summary_md(path: Path, seeds: int, rounds: int, e1: list[dict],
                     paired: list[dict], e3: list[dict], e6: list[dict]) -> None:
    scale = f"{seeds} seeds x {rounds} rounds per cell"
    cols = METRICS + ("perception_gap",)
    e1_rows = [[d, a] + [_pm(lookup(e1, m, defender=d, attacker=a)) for m in cols]
               for d in DEFENDERS for a in ATTACKERS]

    def ci_text(r: dict) -> str:
        if math.isnan(r["ci_low"]):
            return "n/a"
        return f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"

    paired_rows = [[r["comparison"], r["defender"], r["attacker"], r["metric"],
                    f"{r['mean_diff']:+.3f}", ci_text(r), str(r["seeds_first_higher"]),
                    str(r["seeds_second_higher"]), str(r["ties"])] for r in paired]
    e3_rows = []
    for T in TRAP_COUNTS:
        for d in DEFENDERS:
            for a in TRAP_ATTACKERS:
                r = lookup(e3, "defender_payoff", traps=T, defender=d, attacker=a)
                e3_rows.append([str(T), d, a, _pm(r), f"{r['eq_defender_value']:+.3f}",
                                f"{r['eq_attacker_value']:+.3f}"])
    e6_rows = [[str(W), a, _pm(lookup(e6, "attacker_payoff", memory=W,
                                      defender="adaptive", attacker=a))]
               for W in MEMORY_WINDOWS for a in MEMORY_ATTACKERS]

    text = "\n\n".join([
        "# Results summary",
        f"**E1 main grid.** Mean ± 95% t-interval half-width across seeds, {scale}.",
        md_table(["defender", "attacker", *cols], e1_rows),
        f"**Paired comparisons.** Seed-paired differences (first minus second), "
        f"mean and 95% t-interval, {scale}.",
        md_table(["comparison", "defender", "attacker", "metric", "mean diff", "95% CI",
                  "seeds first higher", "seeds second higher", "ties"], paired_rows),
        f"**E3 value of traps.** Defender payoff (mean ± 95% half-width) by number of "
        f"traps T, with the SSE equilibrium values re-solved per T, {scale}.",
        md_table(["T", "defender", "attacker", "defender payoff",
                  "SSE defender value", "SSE attacker value"], e3_rows),
        f"**E6 memory robustness.** Attacker payoff (mean ± 95% half-width) against the "
        f"adaptive defender by memory window W, {scale}.",
        md_table(["W", "attacker", "attacker payoff"], e6_rows),
    ]) + "\n"
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def run_all(out: Path, seeds: int, rounds: int, quick: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig()
    runner = Runner(seeds, rounds)
    eq = solve_sse(cfg)

    (out / "config.json").write_text(json.dumps({
        "game": dataclasses.asdict(cfg),
        "seeds": seeds,
        "rounds": rounds,
        "quick": quick,
        "attackers": ATTACKERS,
        "defenders": DEFENDERS,
        "rolling_window": ROLLING_WINDOW,
        "trap_counts": TRAP_COUNTS,
        "memory_windows": MEMORY_WINDOWS,
        "sse_defender_value": eq.defender_value,
        "sse_attacker_value": eq.attacker_value,
        "sse_response": eq.response,
    }, indent=2), encoding="utf-8")

    print("E1 main grid")
    e1_rows, e1 = e1_main_grid(runner, cfg)
    write_csv(out / "e1_main_grid_seeds.csv", e1_rows)
    write_csv(out / "e1_main_grid_summary.csv", e1)
    ylabel = "Attacker payoff (per round)"
    fig_e1(e1, "attacker_payoff", ylabel, out / "fig_e1a_attacker_payoff.png",
           references={"sse": (eq.attacker_value, "SSE equilibrium value")})
    fig_e1(e1, "defender_payoff", "Defender payoff (per round)",
           out / "fig_e1b_defender_payoff.png",
           references={"sse": (eq.defender_value, "SSE equilibrium value")})
    fig_e1(e1, "perception_gap", "Perception gap (payoff per round)",
           out / "fig_e1c_perception_gap.png", attackers=ATTACKERS[:-1])

    print("E2 learning over time")
    e2_rows = e2_learning(runner, cfg)
    write_csv(out / "e2_learning_summary.csv", e2_rows)
    fig_e2(e2_rows, "payoff", f"Attacker payoff per round\n({ROLLING_WINDOW}-round rolling mean)",
           out / "fig_e2_learning_payoff.png")
    fig_e2(e2_rows, "catch", f"Share of rounds with a catch\n({ROLLING_WINDOW}-round rolling mean)",
           out / "fig_e2_learning_catch.png")

    print("E3 value of traps")
    e3_rows, e3 = e3_traps(runner, cfg)
    write_csv(out / "e3_traps_seeds.csv", e3_rows)
    write_csv(out / "e3_traps_summary.csv", e3)
    fig_e3(e3, out / "fig_e3_traps.png")

    print("E4 prior sensitivity, E5 lure")
    fig_grid(e1, PRIOR_ATTACKERS, [("attacker_payoff", "Attacker payoff (per round)"),
                                   ("attack_share", "Share of rounds with an attack")],
             out / "fig_e4_prior.png")
    fig_grid(e1, LURE_ATTACKERS,
             [("undefended_attack_share", "Share of attacks on\nundefended servers"),
              ("undefended_catch_rate", "Catch rate on\nundefended servers")],
             out / "fig_e5_lure.png")

    print("E6 memory robustness")
    e6_rows, e6 = e6_memory(runner, cfg)
    write_csv(out / "e6_memory_seeds.csv", e6_rows)
    write_csv(out / "e6_memory_summary.csv", e6)
    fig_e6(e6, out / "fig_e6_memory.png")

    paired = paired_comparisons(e1_rows, eq.attacker_value)
    write_csv(out / "paired_comparisons.csv", paired)
    write_summary_md(out / "summary.md", seeds, rounds, e1, paired, e3, e6)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run experiments E1-E6")
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--quick", action="store_true",
                   help=f"{QUICK_SEEDS} seeds x {QUICK_ROUNDS} rounds, for testing")
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    seeds, rounds = (QUICK_SEEDS, QUICK_ROUNDS) if args.quick else (args.seeds, args.rounds)

    start = time.perf_counter()
    run_all(args.out, seeds, rounds, quick=args.quick)
    print(f"Done: {seeds} seeds x {rounds} rounds in {time.perf_counter() - start:.0f} s "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
