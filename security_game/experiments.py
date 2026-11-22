"""Experiments E1-E10, paired comparisons, figures, and summary tables.

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

from .attackers import PRESSURE_BINS, LearnedStrategicAttacker, NaiveAttacker  # noqa: E402
from .config import GameConfig  # noqa: E402
from .defenders import SMOOTHING, AdaptiveDefender  # noqa: E402
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
# E7: rate the strategic attacker assumes for the adaptive defender (true rate 0.2).
MODEL_RATES = (0.05, 0.1, 0.2, 0.4, 0.8)
E7_METRICS = ("attacker_payoff", "defender_payoff", "catch_rate", "attack_share",
              "undefended_attack_share", "undefended_catch_rate")
E7_BASELINES = ("naive", "bayesian")
E7_REFERENCES = ("naive", "bayesian", "omniscient")
# E9: naive-attacker belief diagnostic against the adaptive defender.
E9_METRICS = ("belief_error", "belief_error_tested", "untested_cell_share",
              "unsafe_untested_round_share", "undefended_attack_share",
              "undefended_catch_rate", "attacker_payoff")
UNSAFE_THRESHOLD = 0.5
# E10: learned level-2 attacker (no defender rule given).
LEARNED = "strategic_learned"
E10_SETTINGS = ((50, "sse"), (50, "adaptive"), (200, "adaptive"))   # (memory W, defender)
E10_BASELINES = ("naive", "bayesian", "strategic", "omniscient")
E10_FIG_ATTACKERS = ("naive", "bayesian", LEARNED, "strategic", "omniscient")
E10_METRICS = ("attacker_payoff", "defender_payoff", "catch_rate", "attack_share",
               "undefended_attack_share", "undefended_catch_rate")
E10_LEARNED_KEYS = tuple(f"learned_trap_{b}" for b in PRESSURE_BINS)
# E10 within-server comparison: both defenders at both memory windows.
E10W_SETTINGS = ((50, "sse"), (50, "adaptive"), (200, "sse"), (200, "adaptive"))
E10W_PAIRS = (("medium", "low"), ("high", "medium"), ("high", "low"))
E10W_KEYS = tuple(f"diff_{hi}_{lo}" for hi, lo in E10W_PAIRS)

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
    # Unused slot of the reference palette; validated (adjacent pairs, light mode)
    # in the E10 bar order naive, bayesian, strategic_learned, strategic, omniscient.
    "strategic_learned": "#eb6834",
}
HATCHES = {"bayesian_uniform": "///", "strategic_uniform": "///"}
MARKERS = {"random": "v", "naive": "o", "bayesian": "s", "bayesian_uniform": "P",
           "strategic": "^", "strategic_uniform": "X", "omniscient": "D",
           "strategic_learned": "<"}
LINESTYLES = {"omniscient": "--"}
LABELS = {
    "random": "Random",
    "naive": "Naive",
    "bayesian": "Bayesian",
    "bayesian_uniform": "Bayesian\n(uniform prior)",
    "strategic": "Strategic",
    "strategic_uniform": "Strategic\n(uniform prior)",
    "omniscient": "Omniscient",
    "strategic_learned": "Strategic\n(learned)",
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

    def run(self, cfg: GameConfig, defender: str, attacker: str, seed: int,
            **attacker_kwargs) -> Run:
        key = (cfg, defender, attacker, seed, tuple(sorted(attacker_kwargs.items())))
        if key not in self._cache:
            d = make_defender(defender, cfg)
            a = make_attacker(attacker, cfg, d, **attacker_kwargs)
            logs = play(cfg, d, a, self.rounds, seed)
            self._cache[key] = Run(
                summary=summarize(logs, per_server=True),
                payoff=np.array([log.attacker_payoff for log in logs]),
                caught=np.array([float(Outcome.CAUGHT in log.outcomes) for log in logs]),
            )
        return self._cache[key]

    def cell(self, cfg: GameConfig, defender: str, attacker: str,
             **attacker_kwargs) -> list[Run]:
        return [self.run(cfg, defender, attacker, s, **attacker_kwargs) for s in self.seeds]


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
RATE_PREFIXES = ("attack_unit_share_s", "attack_rate_s")


def _yerr(rows: list[dict]) -> np.ndarray:
    """Asymmetric error bars; intervals of rate metrics are clipped to [0, 1] for display."""
    lo, hi = [], []
    for r in rows:
        low, high = r["ci_low"], r["ci_high"]
        metric = r.get("metric", "")
        if metric in RATE_METRICS or metric.startswith(RATE_PREFIXES):
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


def _model_kwargs(rate: float) -> dict:
    """The true rate needs no override, so that E7 cell is exactly the E1 strategic run."""
    return {} if rate == SMOOTHING else {"model_rate": rate}


def e7_misspecified(runner: Runner, cfg: GameConfig
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Strategic attacker with a wrong model of the defender's smoothing rate.

    Returns (per-seed rows, summary, seed-paired differences vs E7_BASELINES).
    """
    rows, paired = [], []
    for d in DEFENDERS:
        base = {b: np.array([r.summary["attacker_payoff"] for r in runner.cell(cfg, d, b)])
                for b in E7_BASELINES}
        for rate in MODEL_RATES:
            runs = runner.cell(cfg, d, "strategic", **_model_kwargs(rate))
            for seed, run in zip(runner.seeds, runs):
                rows.append({"model_rate": rate, "defender": d, "attacker": "strategic",
                             "seed": seed, **{m: run.summary[m] for m in E7_METRICS}})
            pay = np.array([r.summary["attacker_payoff"] for r in runs])
            for b in E7_BASELINES:
                paired.append({"model_rate": rate, "defender": d,
                               "comparison": f"strategic - {b}", "metric": "attacker_payoff",
                               **paired_stats(pay - base[b])})
    return rows, summary_rows(rows, ("model_rate", "defender"), E7_METRICS), paired


def server_keys(cfg: GameConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(attack-unit share keys, attack-rate keys) from summarize(), servers 1..n."""
    return (tuple(f"attack_unit_share_s{i + 1}" for i in range(cfg.n)),
            tuple(f"attack_rate_s{i + 1}" for i in range(cfg.n)))


def e8_targets(runner: Runner, cfg: GameConfig) -> tuple[list[dict], list[dict]]:
    """Per-server target selection for every E1 attacker x defender (E1 runs)."""
    unit_keys, rate_keys = server_keys(cfg)
    keys = unit_keys + rate_keys
    rows = []
    for d in DEFENDERS:
        for a in ATTACKERS:
            for seed, run in zip(runner.seeds, runner.cell(cfg, d, a)):
                rows.append({"defender": d, "attacker": a, "seed": seed,
                             **{k: run.summary[k] for k in keys}})
    return rows, summary_rows(rows, ("defender", "attacker"), keys)


def naive_belief_diagnostic(cfg: GameConfig, seed: int, rounds: int) -> dict[str, float]:
    """One naive-vs-adaptive run, recording the naive attacker's level-0 cell beliefs.

    Before each round's actions it compares the naive estimate of P(trap | defense 0)
    on every server with the true conditional probability under the defender's
    current mixed strategy. The recording draws no random numbers, so the run is
    identical to the plain E6 run with the same seed.
    """
    defender, attacker = AdaptiveDefender(cfg), NaiveAttacker(cfg)
    n = cfg.n
    trapped = np.array([[i in traps for i in range(n)]
                        for _, traps in defender.pure_strategies], dtype=float)
    undefended = np.array([[d[i] == 0 for i in range(n)]
                           for d, _ in defender.pure_strategies], dtype=float)
    errors, tested_flags, unsafe = [], [], []

    def record(_t: int) -> None:
        x = defender.mixed_strategy()
        true = (x @ (trapped * undefended)) / (x @ undefended)
        attacks, _ = attacker.cell_counts()
        tested = attacks[:, 0] > 0
        errors.append(np.abs(attacker.trap_estimates()[:, 0] - true))
        tested_flags.append(tested)
        unsafe.append(bool(np.any(~tested & (true > UNSAFE_THRESHOLD))))

    logs = play(cfg, defender, attacker, rounds, seed, before_round=record)
    errors, tested_flags = np.array(errors), np.array(tested_flags)
    summary = summarize(logs, per_server=True)
    return {
        "belief_error": float(errors.mean()),
        "belief_error_tested": (float(errors[tested_flags].mean())
                                if tested_flags.any() else math.nan),
        "untested_cell_share": float(1.0 - tested_flags.mean()),
        "unsafe_untested_round_share": float(np.mean(unsafe)),
        **{m: summary[m] for m in ("undefended_attack_share", "undefended_catch_rate",
                                   "attacker_payoff")},
    }


def e9_naive_memory(runner: Runner, cfg: GameConfig) -> tuple[list[dict], list[dict]]:
    rows = []
    for W in MEMORY_WINDOWS:
        cfg_w = dataclasses.replace(cfg, memory=W)
        for seed in runner.seeds:
            diag = naive_belief_diagnostic(cfg_w, seed, runner.rounds)
            # Same seed and config as E6: the recorded run must be the E6 run.
            e6_pay = runner.run(cfg_w, "adaptive", "naive", seed).summary["attacker_payoff"]
            if diag["attacker_payoff"] != e6_pay:
                raise RuntimeError(f"E9 run diverged from E6 (W={W}, seed={seed})")
            rows.append({"memory": W, "defender": "adaptive", "attacker": "naive",
                         "seed": seed, **diag})
    return rows, summary_rows(rows, ("memory", "defender", "attacker"), E9_METRICS)


def learned_run(cfg: GameConfig, defender: str, seed: int, rounds: int) -> dict[str, float]:
    """One strategic_learned run (same seed streams as every other run).

    Besides the usual metrics, reports the learned-dependence diagnostic at the
    end of the run: the posterior mean trap probability of the undefended cells,
    averaged over servers, per pressure bin, using only cells with at least one
    remembered observation (NaN if a bin has none).
    """
    d = make_defender(defender, cfg)
    attacker = make_attacker(LEARNED, cfg, d)
    assert isinstance(attacker, LearnedStrategicAttacker)
    summary = summarize(play(cfg, d, attacker, rounds, seed))
    attacks, _ = attacker.cell_counts()
    mean = attacker.posterior_mean()
    learned = {}
    for b, name in enumerate(PRESSURE_BINS):
        seen = attacks[:, 0, b] > 0
        learned[f"learned_trap_{name}"] = (float(mean[seen, 0, b].mean())
                                           if seen.any() else math.nan)
    return {**{m: summary[m] for m in E10_METRICS}, **learned}


def within_server_differences(attacker: LearnedStrategicAttacker) -> dict[str, float]:
    """Within-server comparison of learned P(trap | undefended) across pressure bins.

    For each server, the posterior mean of its undefended cell in bin `hi` minus
    that in bin `lo`, using only servers where both cells have at least one
    remembered observation; then averaged over those servers. NaN if no server
    qualifies. Also reports how many servers contributed to each difference.
    """
    attacks, _ = attacker.cell_counts()
    mean = attacker.posterior_mean()
    index = {name: b for b, name in enumerate(PRESSURE_BINS)}
    out: dict[str, float] = {}
    for (hi, lo), key in zip(E10W_PAIRS, E10W_KEYS):
        both = (attacks[:, 0, index[hi]] > 0) & (attacks[:, 0, index[lo]] > 0)
        diffs = mean[both, 0, index[hi]] - mean[both, 0, index[lo]]
        out[key] = float(diffs.mean()) if both.any() else math.nan
        out[f"servers_{hi}_{lo}"] = int(both.sum())
    return out


def e10_within_server(runner: Runner, cfg: GameConfig, e10_rows: list[dict]
                      ) -> tuple[list[dict], list[dict]]:
    """Within-server learned dependence at the end of each strategic_learned run.

    Runs strategic_learned with the same seeds and streams as E10; where E10 has
    the same setting, the run must reproduce E10's attacker payoff.
    """
    e10_pay = {(r["memory"], r["defender"], r["seed"]): r["attacker_payoff"]
               for r in e10_rows}
    rows = []
    for W, d in E10W_SETTINGS:
        cfg_w = dataclasses.replace(cfg, memory=W)
        for seed in runner.seeds:
            defender = make_defender(d, cfg_w)
            attacker = make_attacker(LEARNED, cfg_w, defender)
            assert isinstance(attacker, LearnedStrategicAttacker)
            pay = summarize(play(cfg_w, defender, attacker, runner.rounds, seed)
                            )["attacker_payoff"]
            if (W, d, seed) in e10_pay and pay != e10_pay[(W, d, seed)]:
                raise RuntimeError(f"within-server run diverged from E10 ({W}, {d}, {seed})")
            rows.append({"memory": W, "defender": d, "attacker": LEARNED, "seed": seed,
                         "attacker_payoff": pay, **within_server_differences(attacker)})
    keys = E10W_KEYS + tuple(k.replace("diff_", "servers_") for k in E10W_KEYS)
    return rows, summary_rows(rows, ("memory", "defender"), keys)


def e10_learned(runner: Runner, cfg: GameConfig
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """strategic_learned per E10 setting, and seed-paired differences vs the baselines.

    At W = 50 the baselines are the E1 runs; at W = 200 only the E6 runs exist
    (naive, bayesian, strategic against the adaptive defender).
    """
    rows, paired = [], []
    for W, d in E10_SETTINGS:
        cfg_w = dataclasses.replace(cfg, memory=W)
        runs = [learned_run(cfg_w, d, s, runner.rounds) for s in runner.seeds]
        for seed, run in zip(runner.seeds, runs):
            rows.append({"memory": W, "defender": d, "attacker": LEARNED, "seed": seed, **run})
        pay = np.array([r["attacker_payoff"] for r in runs])
        baselines = E10_BASELINES if W == cfg.memory else MEMORY_ATTACKERS
        for b in baselines:
            base = runner.cell(cfg_w, d, b)
            base_pay = np.array([r.summary["attacker_payoff"] for r in base])
            m, lo, hi, _ = mean_ci(base_pay)
            paired.append({"memory": W, "defender": d,
                           "comparison": f"{LEARNED} - {b}", "metric": "attacker_payoff",
                           "learned_mean": float(pay.mean()), "baseline_mean": m,
                           **paired_stats(pay - base_pay)})
    summary = summary_rows(rows, ("memory", "defender"), E10_METRICS + E10_LEARNED_KEYS)
    return rows, summary, paired


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


def fig_e7(e7: list[dict], e1: list[dict], path: Path) -> None:
    """Strategic attacker payoff vs assumed model rate, with E1 reference lines."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    rates = np.array(MODEL_RATES)
    for ax, d in zip(axes, DEFENDERS):
        for a in E7_REFERENCES:
            ref = lookup(e1, "attacker_payoff", defender=d, attacker=a)
            ax.axhspan(ref["ci_low"], ref["ci_high"], color=COLORS[a], alpha=0.12,
                       linewidth=0, zorder=1)
            ax.axhline(ref["mean"], color=COLORS[a], linestyle=(0, (5, 3)),
                       linewidth=1.5, label=LABELS[a] + " (E1)", zorder=2)
        rows = [lookup(e7, "attacker_payoff", model_rate=r, defender=d) for r in MODEL_RATES]
        ax.errorbar(rates, [r["mean"] for r in rows], yerr=_yerr(rows),
                    color=COLORS["strategic"], marker=MARKERS["strategic"], markersize=8,
                    linewidth=2, capsize=3, label="Strategic (assumed rate)", zorder=4)
        ax.axvline(SMOOTHING, color=MUTED, linestyle=":", linewidth=1.2,
                   label=f"True defender rate ({SMOOTHING})", zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [f"{r:g}" for r in rates])
        ax.minorticks_off()
        ax.set_xlabel("Smoothing rate assumed by the strategic attacker")
        ax.set_title(DEFENDER_LABELS[d])
    axes[0].set_ylabel("Attacker payoff (per round)")
    handles, labels = axes[1].get_legend_handles_labels()
    first = labels.index("Strategic (assumed rate)")
    order = [first] + [i for i in range(len(labels)) if i != first]
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                   title="Bands: 95% CI of E1 baselines", title_fontsize=8)
    save(fig, path)


def fig_e8(e8: list[dict], cfg: GameConfig, eq_response: Sequence[int], path: Path,
           attackers: Sequence[str] = ATTACKERS) -> None:
    """Attack-unit share per server, one bar per attacker, one panel per defender."""
    unit_keys, _ = server_keys(cfg)
    eq_share = np.asarray(eq_response, dtype=float) / max(sum(eq_response), 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    x = np.arange(cfg.n)
    group = 0.84
    width = group / len(attackers)
    for ax, d in zip(axes, DEFENDERS):
        for k, a in enumerate(attackers):
            rows = [lookup(e8, key, defender=d, attacker=a) for key in unit_keys]
            pos = x - group / 2 + (k + 0.5) * width
            ax.bar(pos, [r["mean"] for r in rows], width=width, color=COLORS[a],
                   hatch=HATCHES.get(a, ""), edgecolor="white", linewidth=0.6,
                   label=LABELS[a].replace("\n", " "), zorder=2)
            ax.errorbar(pos, [r["mean"] for r in rows],
                        yerr=_yerr(rows),
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5, zorder=3)
        ax.hlines(eq_share, x - group / 2, x + group / 2, color=INK, linewidth=1.6,
                  linestyle=(0, (4, 2)), zorder=4)
        ax.plot(x, eq_share, linestyle="none", marker="D", markersize=6, color=INK,
                markeredgecolor="white", zorder=5,
                label=f"SSE equilibrium response {tuple(eq_response)}")
        ax.set_xticks(x, [f"Server {i + 1}\n(v = {cfg.values[i]:g})" for i in range(cfg.n)])
        ax.set_ylim(0, 1.05)
        ax.set_title(DEFENDER_LABELS[d])
    axes[0].set_ylabel("Share of attack units sent to server")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    save(fig, path)


def fig_e10_payoff(e10: list[dict], e1: list[dict], cfg: GameConfig,
                   eq_attacker_value: float, path: Path) -> None:
    """Attacker payoff at W = 50 for the E10 attackers, one panel per defender."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, d in zip(axes, DEFENDERS):
        rows = [lookup(e10, "attacker_payoff", memory=cfg.memory, defender=d) if a == LEARNED
                else lookup(e1, "attacker_payoff", defender=d, attacker=a)
                for a in E10_FIG_ATTACKERS]
        ref = (eq_attacker_value, "SSE equilibrium value") if d == "sse" else None
        bar_panel(ax, E10_FIG_ATTACKERS, rows, DEFENDER_LABELS[d],
                  "Attacker payoff (per round)" if ax is axes[0] else None, ref)
    save(fig, path)


def fig_e10_dependence(e10: list[dict], cfg: GameConfig, path: Path) -> None:
    """Learned trap probability of undefended cells by pressure bin, one panel per defender."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    x = np.arange(len(PRESSURE_BINS))
    prior_mean = cfg.traps / cfg.n
    for ax, d in zip(axes, DEFENDERS):
        rows = [lookup(e10, key, memory=cfg.memory, defender=d) for key in E10_LEARNED_KEYS]
        means = [r["mean"] for r in rows]
        ax.bar(x, means, width=0.6, color=COLORS[LEARNED], edgecolor="white", zorder=2)
        ax.errorbar(x, means, yerr=_yerr(rows), fmt="none", ecolor=INK,
                    elinewidth=1.2, capsize=3, zorder=3)
        for xi, r in zip(x, rows):
            top = r["ci_high"] if not math.isnan(r["ci_high"]) else r["mean"]
            if not math.isnan(top):
                ax.annotate(f"{r['n']} runs", (xi, top), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=MUTED)
        ax.axhline(prior_mean, color=INK, linestyle=":", linewidth=1.2,
                   label=f"Prior mean T/N = {prior_mean:g}", zorder=1)
        ax.set_xticks(x, [f"{b}\npressure" for b in PRESSURE_BINS])
        ax.set_title(DEFENDER_LABELS[d])
    axes[0].set_ylabel("Learned P(trap | undefended)\n(posterior mean, end of run)")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, path)


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------

def write_summary_md(path: Path, seeds: int, rounds: int, e1: list[dict],
                     paired: list[dict], e3: list[dict], e6: list[dict],
                     e7: list[dict], e7_paired: list[dict], e8: list[dict],
                     e9: list[dict], cfg: GameConfig, eq_response: Sequence[int]) -> None:
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

    def paired_text(r: dict) -> list[str]:
        return [f"{r['mean_diff']:+.3f}", ci_text(r),
                f"{r['seeds_first_higher']}/{r['seeds_second_higher']}/{r['ties']}"]

    e7_rows = []
    for d in DEFENDERS:
        for rate in MODEL_RATES:
            row = [d, f"{rate:g}" + (" (true)" if rate == SMOOTHING else ""),
                   _pm(lookup(e7, "attacker_payoff", model_rate=rate, defender=d)),
                   _pm(lookup(e7, "catch_rate", model_rate=rate, defender=d)),
                   _pm(lookup(e7, "attack_share", model_rate=rate, defender=d))]
            for b in E7_BASELINES:
                row += paired_text(next(r for r in e7_paired if r["model_rate"] == rate
                                        and r["defender"] == d
                                        and r["comparison"] == f"strategic - {b}"))
            e7_rows.append(row)
    e7_refs = [[d] + [_pm(lookup(e1, "attacker_payoff", defender=d, attacker=a))
                      for a in E7_REFERENCES] for d in DEFENDERS]

    unit_keys, rate_keys = server_keys(cfg)
    e8_rows = [[d, a] + [_pm(lookup(e8, k, defender=d, attacker=a))
                         for k in unit_keys + rate_keys]
               for d in DEFENDERS for a in ATTACKERS]
    total = max(sum(eq_response), 1)
    e8_rows.append(["(equilibrium)", f"SSE response {tuple(eq_response)}"]
                   + [f"{u / total:.3f}" for u in eq_response]
                   + [f"{float(u > 0):.3f}" for u in eq_response])
    server_cols = [f"s{i + 1} (v={cfg.values[i]:g})" for i in range(cfg.n)]

    e9_rows = [[str(W)] + [_pm(lookup(e9, m, memory=W, defender="adaptive", attacker="naive"))
                           for m in E9_METRICS] for W in MEMORY_WINDOWS]

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
        f"**E7 misspecified strategic attacker.** Strategic attacker whose model of the "
        f"adaptive defender assumes smoothing rate model_rate (the defender's true rate "
        f"stays {SMOOTHING}; floor unchanged). Mean ± 95% half-width; paired columns are "
        f"seed-paired differences in attacker payoff (strategic minus baseline, E1 runs "
        f"with the same seeds) with 95% t-interval and seeds strategic higher / baseline "
        f"higher / tied, {scale}.",
        md_table(["defender", "model_rate", "attacker payoff", "catch rate", "attack share",
                  "vs naive: diff", "95% CI", "won/lost/tied",
                  "vs bayesian: diff", "95% CI", "won/lost/tied"], e7_rows),
        "E7 reference values: attacker payoff of the baselines against the same defender (E1).",
        md_table(["defender", *(f"{a} payoff" for a in E7_REFERENCES)], e7_refs),
        f"**E8 target selection.** Per server: share of all attack units sent to it "
        f"(unit share) and share of rounds in which it was attacked (attack rate); "
        f"mean ± 95% half-width across seeds (E1 runs), {scale}. Seeds with no attack "
        f"at all have no unit share and are left out of that mean. The last row is "
        f"the attacker's response in the Strong Stackelberg Equilibrium.",
        md_table(["defender", "attacker", *(f"unit share {c}" for c in server_cols),
                  *(f"attack rate {c}" for c in server_cols)], e8_rows),
        f"**E9 naive memory diagnostic.** Naive attacker against the adaptive defender by "
        f"memory W. Belief error: mean over rounds and servers of |naive estimate of "
        f"P(trap | defense 0) - true P(trap on server | server undefended)| under the "
        f"defender's current mixed strategy, measured before each round; 'tested' "
        f"restricts it to cells with at least one remembered attack. Unsafe untested "
        f"rounds: share of rounds with an untested (so assumed safe) level-0 cell whose "
        f"true trap probability is above {UNSAFE_THRESHOLD}. Mean ± 95% half-width, "
        f"{scale}; runs are identical to E6.",
        md_table(["W", *E9_METRICS], e9_rows),
    ]) + "\n"
    path.write_text(text, encoding="utf-8")


def append_e10_md(path: Path, seeds: int, rounds: int, e10: list[dict],
                  e10_paired: list[dict]) -> None:
    """Append the E10 tables to summary.md, after the existing sections."""
    scale = f"{seeds} seeds x {rounds} rounds per cell"

    def ci_text(r: dict) -> str:
        if math.isnan(r["ci_low"]):
            return "n/a"
        return f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"

    result_rows = [[str(W), d] + [_pm(lookup(e10, m, memory=W, defender=d))
                                  for m in E10_METRICS] for W, d in E10_SETTINGS]
    paired_rows = [[str(r["memory"]), r["defender"], r["comparison"],
                    f"{r['learned_mean']:+.3f}", f"{r['baseline_mean']:+.3f}",
                    f"{r['mean_diff']:+.3f}", ci_text(r),
                    f"{r['seeds_first_higher']}/{r['seeds_second_higher']}/{r['ties']}"]
                   for r in e10_paired]

    def learned_cell(r: dict) -> str:
        return f"{_pm(r)} (n = {r['n']})"

    learned_rows = [[str(W), d] + [learned_cell(lookup(e10, k, memory=W, defender=d))
                                   for k in E10_LEARNED_KEYS] for W, d in E10_SETTINGS]
    text = "\n\n".join([
        "",
        f"**E10 learned strategic attacker.** strategic_learned (level 2, not given the "
        f"defender's rule; learns trap probability per server, defended?, and its own "
        f"attack-pressure bin). Mean ± 95% half-width, {scale}.",
        md_table(["W", "defender", *E10_METRICS], result_rows),
        f"**E10 paired comparisons.** Seed-paired differences in attacker payoff "
        f"(strategic_learned minus baseline, same seeds and random streams), mean and "
        f"95% t-interval, and seeds strategic_learned higher / baseline higher / tied. "
        f"W = 50 baselines are the E1 runs; W = 200 baselines are the E6 runs (naive, "
        f"bayesian, strategic only), {scale}.",
        md_table(["W", "defender", "comparison", "strategic_learned payoff",
                  "baseline payoff", "mean diff", "95% CI", "won/lost/tied"], paired_rows),
        f"**E10 learned dependence.** Posterior mean trap probability of undefended "
        f"cells at the end of each run, averaged over servers with at least one "
        f"remembered observation in that pressure bin; mean ± 95% half-width across "
        f"the n runs that had such a cell. The prior mean is T/N. {scale}.",
        md_table(["W", "defender", *(f"{b} pressure" for b in PRESSURE_BINS)], learned_rows),
    ]) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def append_e10_within_md(path: Path, seeds: int, rounds: int, e10w: list[dict]) -> None:
    """Append the E10 within-server table to summary.md, after the existing sections."""
    scale = f"{seeds} seeds x {rounds} rounds per cell"

    def cell(W: int, d: str, key: str) -> str:
        r = lookup(e10w, key, memory=W, defender=d)
        servers = lookup(e10w, key.replace("diff_", "servers_"), memory=W, defender=d)
        return f"{_pm(r)} (n = {r['n']}; {servers['mean']:.1f} servers)"

    rows = [[str(W), d] + [cell(W, d, k) for k in E10W_KEYS] for W, d in E10W_SETTINGS]
    text = "\n\n".join([
        "",
        f"**E10 within-server learned dependence.** For each server, the learned "
        f"P(trap | undefended) (posterior mean at the end of the run) in one pressure "
        f"bin minus that in another, using only servers with at least one remembered "
        f"observation in both bins; averaged over those servers per run, then mean ± "
        f"95% half-width across the n runs with at least one such server (mean number "
        f"of servers per run in brackets). Same seeds and streams as E10; the W = 200 "
        f"SSE runs exist only for this table. {scale}.",
        md_table(["W", "defender", *(f"{hi} - {lo}" for hi, lo in E10W_PAIRS)], rows),
    ]) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


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
        "e7_model_rates": MODEL_RATES,
        "e7_true_defender_rate": SMOOTHING,
        "e9_unsafe_threshold": UNSAFE_THRESHOLD,
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

    print("E7 misspecified strategic attacker")
    e7_rows, e7, e7_paired = e7_misspecified(runner, cfg)
    write_csv(out / "e7_misspecified_seeds.csv", e7_rows)
    write_csv(out / "e7_misspecified_summary.csv", e7)
    write_csv(out / "e7_paired_comparisons.csv", e7_paired)
    fig_e7(e7, e1, out / "fig_e7_misspecified.png")

    print("E8 target selection")
    e8_rows, e8 = e8_targets(runner, cfg)
    write_csv(out / "e8_targets_seeds.csv", e8_rows)
    write_csv(out / "e8_targets_summary.csv", e8)
    fig_e8(e8, cfg, eq.response, out / "fig_e8_targets.png")

    print("E9 naive memory diagnostic")
    e9_rows, e9 = e9_naive_memory(runner, cfg)
    write_csv(out / "e9_naive_memory_seeds.csv", e9_rows)
    write_csv(out / "e9_naive_memory_summary.csv", e9)

    paired = paired_comparisons(e1_rows, eq.attacker_value)
    write_csv(out / "paired_comparisons.csv", paired)
    write_summary_md(out / "summary.md", seeds, rounds, e1, paired, e3, e6,
                     e7, e7_paired, e8, e9, cfg, eq.response)

    print("E10 learned strategic attacker")
    e10_rows, e10, e10_paired = e10_learned(runner, cfg)
    write_csv(out / "e10_learned_seeds.csv", e10_rows)
    write_csv(out / "e10_learned_summary.csv", e10)
    write_csv(out / "e10_paired_comparisons.csv", e10_paired)
    fig_e10_payoff(e10, e1, cfg, eq.attacker_value, out / "fig_e10_payoff.png")
    fig_e10_dependence(e10, cfg, out / "fig_e10_learned_dependence.png")
    append_e10_md(out / "summary.md", seeds, rounds, e10, e10_paired)

    print("E10 within-server learned dependence")
    e10w_rows, e10w = e10_within_server(runner, cfg, e10_rows)
    write_csv(out / "e10_within_server_seeds.csv", e10w_rows)
    write_csv(out / "e10_within_server_summary.csv", e10w)
    append_e10_within_md(out / "summary.md", seeds, rounds, e10w)


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run experiments E1-E10")
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
