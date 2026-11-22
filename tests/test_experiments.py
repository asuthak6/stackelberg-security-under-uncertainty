"""Smoke test of the experiments pipeline in quick mode."""

import csv

import pytest

from security_game.experiments import (
    ATTACKERS,
    DEFENDERS,
    MEMORY_WINDOWS,
    MODEL_RATES,
    main,
)

CSVS = [
    "e1_main_grid_seeds.csv",
    "e1_main_grid_summary.csv",
    "e2_learning_summary.csv",
    "e3_traps_seeds.csv",
    "e3_traps_summary.csv",
    "e6_memory_seeds.csv",
    "e6_memory_summary.csv",
    "paired_comparisons.csv",
    "e7_misspecified_seeds.csv",
    "e7_misspecified_summary.csv",
    "e7_paired_comparisons.csv",
    "e8_targets_seeds.csv",
    "e8_targets_summary.csv",
    "e9_naive_memory_seeds.csv",
    "e9_naive_memory_summary.csv",
    "e10_learned_seeds.csv",
    "e10_learned_summary.csv",
    "e10_paired_comparisons.csv",
    "e10_within_server_seeds.csv",
    "e10_within_server_summary.csv",
]
PNGS = [
    "fig_e1a_attacker_payoff.png",
    "fig_e1b_defender_payoff.png",
    "fig_e1c_perception_gap.png",
    "fig_e2_learning_payoff.png",
    "fig_e2_learning_catch.png",
    "fig_e3_traps.png",
    "fig_e4_prior.png",
    "fig_e5_lure.png",
    "fig_e6_memory.png",
    "fig_e7_misspecified.png",
    "fig_e8_targets.png",
    "fig_e10_payoff.png",
    "fig_e10_learned_dependence.png",
]


@pytest.fixture(scope="module")
def quick_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("results")
    main(["--quick", "--out", str(out)])
    return out


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_quick_run_creates_outputs(quick_out):
    for name in CSVS + PNGS + ["config.json", "summary.md"]:
        path = quick_out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_e1_has_every_pair_and_seed(quick_out):
    rows = read_csv(quick_out / "e1_main_grid_seeds.csv")
    assert len(rows) == len(ATTACKERS) * len(DEFENDERS) * 3
    for row in rows:
        if row["attacker"] == "omniscient":
            assert float(row["perception_gap"]) == 0.0


def test_paired_comparisons_rows(quick_out):
    rows = read_csv(quick_out / "paired_comparisons.csv")
    keys = {(r["comparison"], r["defender"], r["attacker"]) for r in rows}
    for first, second in [("strategic", "naive"), ("bayesian", "naive"),
                          ("strategic", "bayesian")]:
        for d in DEFENDERS:
            assert (f"{first} - {second}", d, f"{first} vs {second}") in keys
    for a in ATTACKERS:
        assert ("sse - adaptive", "sse vs adaptive", a) in keys
        assert ("attacker - equilibrium value", "sse", a) in keys
    assert len(rows) == 3 * len(DEFENDERS) + 2 * len(ATTACKERS)
    for r in rows:
        assert int(r["n"]) == 3
        assert (int(r["seeds_first_higher"]) + int(r["seeds_second_higher"])
                + int(r["ties"])) == 3


def test_summary_md_has_tables(quick_out):
    text = (quick_out / "summary.md").read_text(encoding="utf-8")
    for caption in ("E1 main grid", "Paired comparisons", "E3 value of traps",
                    "E6 memory robustness", "E7 misspecified strategic attacker",
                    "E8 target selection", "E9 naive memory diagnostic",
                    "E10 learned strategic attacker", "E10 paired comparisons",
                    "E10 learned dependence", "E10 within-server learned dependence"):
        assert caption in text
    assert "3 seeds x 60 rounds" in text


def test_e7_rows_and_true_rate_matches_e1(quick_out):
    rows = read_csv(quick_out / "e7_misspecified_seeds.csv")
    assert len(rows) == len(MODEL_RATES) * len(DEFENDERS) * 3
    e1 = {(r["defender"], r["seed"]): r["attacker_payoff"]
          for r in read_csv(quick_out / "e1_main_grid_seeds.csv")
          if r["attacker"] == "strategic"}
    true_rate = [r for r in rows if float(r["model_rate"]) == 0.2]
    assert len(true_rate) == len(DEFENDERS) * 3
    for r in true_rate:
        assert r["attacker_payoff"] == e1[(r["defender"], r["seed"])]
    paired = read_csv(quick_out / "e7_paired_comparisons.csv")
    assert len(paired) == len(MODEL_RATES) * len(DEFENDERS) * 2
    assert {r["comparison"] for r in paired} == {"strategic - naive", "strategic - bayesian"}


def test_e8_unit_shares_sum_to_one(quick_out):
    rows = read_csv(quick_out / "e8_targets_seeds.csv")
    assert len(rows) == len(ATTACKERS) * len(DEFENDERS) * 3
    for r in rows:
        shares = [float(r[f"attack_unit_share_s{i}"]) for i in range(1, 5)]
        if all(x == x for x in shares):  # not NaN: at least one attack happened
            assert sum(shares) == pytest.approx(1.0, abs=1e-5)


def test_e9_matches_e6(quick_out):
    rows = read_csv(quick_out / "e9_naive_memory_seeds.csv")
    assert len(rows) == len(MEMORY_WINDOWS) * 3
    e6 = {(r["memory"], r["seed"]): r["attacker_payoff"]
          for r in read_csv(quick_out / "e6_memory_seeds.csv") if r["attacker"] == "naive"}
    for r in rows:
        assert r["attacker_payoff"] == e6[(r["memory"], r["seed"])]
        assert 0.0 <= float(r["belief_error"]) <= 1.0


def test_e10_rows_and_pairs(quick_out):
    rows = read_csv(quick_out / "e10_learned_seeds.csv")
    assert {(r["memory"], r["defender"]) for r in rows} == {
        ("50", "sse"), ("50", "adaptive"), ("200", "adaptive")}
    assert len(rows) == 3 * 3
    assert all(r["attacker"] == "strategic_learned" for r in rows)
    paired = read_csv(quick_out / "e10_paired_comparisons.csv")
    keys = {(r["memory"], r["defender"], r["comparison"]) for r in paired}
    for d in DEFENDERS:
        for b in ("naive", "bayesian", "strategic", "omniscient"):
            assert ("50", d, f"strategic_learned - {b}") in keys
    for b in ("naive", "bayesian", "strategic"):
        assert ("200", "adaptive", f"strategic_learned - {b}") in keys
    assert len(paired) == 2 * 4 + 3
    # Paired against the existing runs: the baseline means are the E1 / E6 means.
    e1 = read_csv(quick_out / "e1_main_grid_seeds.csv")
    naive_sse = [float(r["attacker_payoff"]) for r in e1
                 if r["defender"] == "sse" and r["attacker"] == "naive"]
    row = next(r for r in paired if r["memory"] == "50" and r["defender"] == "sse"
               and r["comparison"] == "strategic_learned - naive")
    assert float(row["baseline_mean"]) == pytest.approx(sum(naive_sse) / 3, abs=1e-5)


def test_e10_within_server_rows(quick_out):
    rows = read_csv(quick_out / "e10_within_server_seeds.csv")
    assert {(r["memory"], r["defender"]) for r in rows} == {
        ("50", "sse"), ("50", "adaptive"), ("200", "sse"), ("200", "adaptive")}
    assert len(rows) == 4 * 3
    e10 = {(r["memory"], r["defender"], r["seed"]): r["attacker_payoff"]
           for r in read_csv(quick_out / "e10_learned_seeds.csv")}
    for r in rows:
        key = (r["memory"], r["defender"], r["seed"])
        if key in e10:   # same run as E10
            assert r["attacker_payoff"] == e10[key]
        for pair in ("medium_low", "high_medium", "high_low"):
            assert 0 <= int(r[f"servers_{pair}"]) <= 4
