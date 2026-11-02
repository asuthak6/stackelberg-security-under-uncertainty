# Attacker vs. Defender — one Stackelberg security game, two environments

This is the whole project in one package. It is **one model**: the same attacker
and defender agents, the same payoffs, the same rounds — played against a
swappable **environment** that decides each round whether an attack caused an
outage.

- **Formula environment** — answers with the contest probability `a/(a+d+k)`.
  Fast and analyzable: used to solve for the equilibrium and run thousands of
  rounds.
- **Live environment** — answers by *measuring a real loopback HTTP server*. The
  defense becomes real serving capacity, the attack becomes a real request flood,
  and an outage is observed when a legitimate user can't get through. This is the
  "real DDoS" — wired into the game as its environment, not a separate thing.

`validate.py` shows the formula predicts what the real server does, so the fast
backend is a trustworthy stand-in for the real one.

Read **`MODEL.md`** for the formal model (and your Sprint-3 study guide).

## Why one package and not two folders

Earlier this was split into a `game/` (theory) and a `live_demo/` (real server),
which looked like two models. It isn't — the real server is just the *environment*
the one game runs in. This package makes that explicit: `environment.py` holds
both backends behind the same interface, and every other file is backend-agnostic.

## Files

| File | Role |
|---|---|
| `config.py` | shared parameters + the contest helper |
| `agents.py` | the Attacker and Defender (environment-agnostic) |
| `environment.py` | **the key file** — `FormulaEnvironment` and `LiveEnvironment` |
| `server.py` | the real loopback HTTP server (used by the live environment) |
| `clients.py` | real traffic sources; loopback-only safety guard |
| `game.py` | the game loop + payoffs; runs agents against any environment |
| `baseline.py` | exact Stackelberg equilibrium for the single-target case |
| `experiments.py` | the controlled experiments (formula backend) |
| `validate.py` | formula-vs-real-server agreement (the unification check) |
| `run_live.py` | the game played against the real server (the live demo) |

## Install

```bash
pip install scipy matplotlib     # numpy comes with scipy
```

Only `baseline.py`, `validate.py` and `experiments.py` need scipy/matplotlib; the
game and the live server are pure standard library.

## Run

```bash
# the solvable equilibrium baseline
python baseline.py

# the game on the FAST backend (formula)
python game.py --defender adaptive --attacker distributed --noise 1.0

# the game on the REAL server (the live DDoS demo) — a few rounds, loopback only
python run_live.py --rounds 8

# prove the formula matches the real server (~1 min of real attacks)
python validate.py

# the full experiments + figures (formula backend)
python experiments.py --seeds 20
```

## What you get

- **`run_live.py`** — rounds where undefended servers really fall and well-defended
  ones hold, measured on a real server. Its aggregate payoffs land almost on top of
  the formula game's — same game, two backends.
- **`validate.py` / `fig_validation.png`** — measured outage probability vs. the
  fitted formula. The direction matches cleanly; the fit is moderate because the
  real server is more threshold-like than the smooth formula (an honest caveat).
- **`experiments.py`** — the core findings on the fast backend:
  - noise sweep: more perception noise → attacker loses, defender gains
  - adaptive defender beats uniform, edges out the Stackelberg-shaped baseline
  - the adaptive defender improving as it learns

## Honesty notes (for the report)
- Agents are adaptive **heuristics**; only the single-target baseline is exact.
- The formula is a **surrogate** validated against the real server, not identical
  to it.
- All headline numbers are means over many seeds; the game is stochastic.
- Everything in the live backend is loopback-only; `clients.assert_loopback`
  refuses any non-loopback target.
- Traps/deception (`MODEL.md` §11) are specified but not yet built.
