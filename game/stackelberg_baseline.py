"""Strong Stackelberg Equilibrium for the single-target coverage baseline.

This is the *solvable anchor* of our project: the simplest honest version of the
attacker/defender game, for which an exact optimal defender strategy can be
computed. The full game (security_game.py) is richer -- distributed attacks,
noisy observation, traps, learning -- and is studied by simulation. This file
gives us the theoretical optimum to compare those simulations against.

THE GAME (one shot, perfect information)
----------------------------------------
- N servers, server i worth value v_i.
- Defender (the LEADER) commits to a coverage vector c = (c_1..c_N),
  c_i in [0,1], sum(c_i) <= budget. c_i is the probability server i is protected.
- Attacker (the FOLLOWER) sees c and attacks the single server that maximizes
  its own expected payoff.
- Payoffs when server i is attacked (classic zero-sum security game):
      uncovered (prob 1 - c_i):  attacker +v_i,     defender -v_i
      covered   (prob   c_i  ):  attacker -penalty, defender +penalty
  Attacker's expected payoff on i:  EUa(i) = (1-c_i)*v_i + c_i*(-penalty)
  Defender's expected payoff if i is attacked:
                                    EUd(i) = (1-c_i)*(-v_i) + c_i*(+penalty)

STRONG STACKELBERG EQUILIBRIUM (SSE)
------------------------------------
The defender wants the coverage c that maximizes its own payoff, anticipating
that the attacker will then attack its best-response target. We find it with the
classic MULTIPLE-LP method (Conitzer & Sandholm): assume in turn that each
target t is the one the attacker ends up attacking, and solve a linear program
for the best coverage that makes t a best response. Pick the best t.

For a fixed "attacked target" t, the LP is:

    maximize   EUd(t) = (1-c_t)*(-v_t) + c_t*penalty        (defender payoff)
    over       c in R^N
    subject to EUa(t) >= EUa(i)   for all i   (t is the attacker's best response)
               sum_i c_i <= budget
               0 <= c_i <= 1

Both the objective and all constraints are linear in c, so each is a small LP.
We solve N of them (one per possible attacked target) and keep the best.

Needs scipy (pip install scipy).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


# Default instance: four servers, same values as the original project.
DEFAULT_VALUES = (10.0, 8.0, 6.0, 4.0)


@dataclass
class SSEResult:
    coverage: np.ndarray      # optimal c_i per server
    attacked: int             # target the attacker best-responds to (0-indexed)
    defender_value: float     # defender's expected payoff at the SSE
    attacker_value: float     # attacker's expected payoff at the SSE


def _attacker_eu(coverage: np.ndarray, values: np.ndarray, penalty: float) -> np.ndarray:
    """Expected attacker payoff for attacking each server, given coverage."""
    return (1 - coverage) * values + coverage * (-penalty)


def solve_sse(values=DEFAULT_VALUES, budget: float = 2.0, penalty: float = 1.0) -> SSEResult:
    values = np.asarray(values, dtype=float)
    n = len(values)
    best: SSEResult | None = None

    for t in range(n):
        # linprog MINIMIZES c . x, so we minimize the NEGATIVE of defender payoff.
        # Defender payoff if t is attacked: (1-c_t)*(-v_t) + c_t*penalty
        #   = -v_t + c_t*(v_t + penalty).  Only c_t varies; -v_t is a constant we
        #   must add back when reporting the payoff and when comparing targets.
        obj = np.zeros(n)
        obj[t] = -(values[t] + penalty)  # minimizing this maximizes c_t
        const = -values[t]               # the constant part of the defender payoff

        # Best-response constraints: EUa(t) >= EUa(i)  for every i.
        #   EUa(j) = v_j - c_j*(v_j + penalty)
        #   EUa(t) - EUa(i) >= 0
        #   [v_t - c_t*(v_t+penalty)] - [v_i - c_i*(v_i+penalty)] >= 0
        # linprog wants  A_ub @ c <= b_ub, so multiply by -1:
        #   c_t*(v_t+penalty) - c_i*(v_i+penalty) <= v_t - v_i
        A_ub, b_ub = [], []
        for i in range(n):
            if i == t:
                continue
            row = np.zeros(n)
            row[t] += (values[t] + penalty)
            row[i] -= (values[i] + penalty)
            A_ub.append(row)
            b_ub.append(values[t] - values[i])

        # Budget constraint: sum(c_i) <= budget.
        A_ub.append(np.ones(n))
        b_ub.append(budget)

        res = linprog(
            c=obj,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=[(0.0, 1.0)] * n,
            method="highs",
        )
        if not res.success:
            continue  # no feasible coverage makes t the best response

        coverage = res.x
        eua = _attacker_eu(coverage, values, penalty)
        # Numerical guard: confirm t really is (one of) the attacker's best targets.
        if eua[t] < eua.max() - 1e-6:
            continue
        defender_value = -res.fun + const  # undo the minimization, add the -v_t constant
        candidate = SSEResult(coverage, t, defender_value, float(eua[t]))
        if best is None or candidate.defender_value > best.defender_value:
            best = candidate

    assert best is not None, "no SSE found -- check the instance parameters"
    return best


def main() -> None:
    p = argparse.ArgumentParser(description="Compute the Stackelberg baseline (single-target SSG)")
    p.add_argument("--values", type=float, nargs="+", default=list(DEFAULT_VALUES),
                   help="server values, highest first")
    p.add_argument("--budget", type=float, default=2.0,
                   help="total coverage budget (sum of coverage probabilities)")
    p.add_argument("--penalty", type=float, default=1.0,
                   help="attacker penalty for hitting a covered server")
    args = p.parse_args()

    sse = solve_sse(args.values, args.budget, args.penalty)
    values = np.asarray(args.values)

    print("Single-target Stackelberg security game -- Strong Stackelberg Equilibrium")
    print(f"  servers (value):   {', '.join(f'S{i+1}={v:g}' for i, v in enumerate(values))}")
    print(f"  coverage budget:   {args.budget:g}   attacker penalty: {args.penalty:g}\n")
    print("  optimal defender coverage (probability each server is protected):")
    for i, c in enumerate(sse.coverage):
        bar = "#" * int(round(c * 20))
        print(f"    S{i+1} (v={values[i]:g}): c={c:0.3f}  {bar}")
    print(f"\n  attacker best-responds by hitting S{sse.attacked + 1} "
          f"(value {values[sse.attacked]:g})")
    print(f"  expected payoff  -- defender: {sse.defender_value:+.3f}  "
          f"attacker: {sse.attacker_value:+.3f}")
    print("\n  Read: the defender spreads coverage so the attacker is INDIFFERENT across\n"
          "  several servers (their expected attacker-payoffs are equalized), which is the\n"
          "  signature of a Stackelberg security-game solution.")


if __name__ == "__main__":
    main()
