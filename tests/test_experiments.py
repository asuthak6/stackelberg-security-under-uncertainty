"""Smoke test of the experiments pipeline in quick mode."""

import csv

import pytest

from security_game.experiments import ATTACKERS, DEFENDERS, main

CSVS = [
    "e1_main_grid_seeds.csv",
    "e1_main_grid_summary.csv",
    "e2_learning_summary.csv",
    "e3_traps_seeds.csv",
    "e3_traps_summary.csv",
    "e6_memory_seeds.csv",
    "e6_memory_summary.csv",
    "paired_comparisons.csv",
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
                    "E6 memory robustness"):
        assert caption in text
    assert "3 seeds x 60 rounds" in text
