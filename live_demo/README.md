# Live DDoS Demonstration (loopback sandbox)

This folder is the "make it real" extension of our SE 4457 attacker/defender
project. Instead of deciding outages with a formula, we stand up a **real HTTP
server on this machine** and knock it over with **real concurrent request
traffic**, then measure what actually happens to a normal user. Then we turn on
mitigation and measure the recovery.

Nothing here touches the outside world. The server only ever binds to
`127.0.0.1`, and the traffic clients **refuse to send to any address that isn't
loopback** (see `assert_loopback` in `clients.py`). Point them at a public host
and they raise an error before sending a single packet. This is a lab rig for
our own machine, not an attack tool.

## How it maps onto our game

| In `security_agents.py` (the model) | Here (the live system) |
|---|---|
| Defender's mitigation budget | Real server **capacity** (worker slots) and/or an **admission rate limit** |
| Attacker intensity | Number of real flooding connections |
| `outage = intensity > capacity` (a formula) | `outage` = measured: the legit user's requests actually fail / time out |
| Attacker payoff | Not needed for the demo, but the same agents can drive it (see "Agent-driven mode") |

So this doesn't replace the model — it's the Sprint 4 "write code that implements
the model and run simulations / compare designs" step, with the outage grounded
in a real measurement.

## The files

- `target_server.py` — the defended asset. A real `ThreadingHTTPServer` on
  loopback. Capacity is a semaphore of `N` concurrent slots; each request holds a
  slot for `service_time` (simulated work). When slots stay full longer than
  `queue_timeout`, requests are shed with HTTP 503 — that is the outage. An
  optional token bucket adds admission control (HTTP 429 when over the rate).
- `clients.py` — two traffic sources that speak real HTTP: a steady
  **legitimate user** and an **attacker flood**. Both record real latency and
  real status codes. Contains the loopback-only safety guard.
- `run_demo.py` — the orchestrator. Runs four phases and prints a table + two
  CSVs. **Start here.**
- `plot_results.py` — turns the raw CSV into `fig_goodput.png` and
  `fig_latency.png` for the report (needs `matplotlib`).

## Running it

```bash
# 1. The headline demo (four phases, ~20s total):
python run_demo.py

# 2. Make the figures for the report:
python plot_results.py
```

You can also run the pieces by hand in two terminals to see it live:

```bash
# terminal 1 — start the undefended server
python target_server.py --port 8080 --capacity 8 --service-time 0.025

# terminal 2 — flood it and watch the server's served/429/503 counters
python -c "from clients import TrafficSource; \
print(len(TrafficSource('127.0.0.1',8080,'attack',0,400).run(5)))"
```

## What the four phases show

`run_demo.py` runs these back to back against the same server, measuring the
**legitimate user** each time (goodput = % of their requests that got 200 OK):

- **A — Baseline.** Legit user only. ~100% goodput, ~25 ms latency. This is
  "healthy".
- **B — Attack, no defense.** The flood turns on. Legit goodput collapses and
  latency explodes — a real, observed outage.
- **C — Mitigate with capacity.** The defender spends its budget on more worker
  slots (exactly the budget→capacity idea from our model). Goodput recovers and
  median latency drops back to normal.
- **D — Mitigate with admission control.** Instead, the defender caps intake with
  a token bucket. This keeps the server *healthy* (low latency for the requests
  it admits) but it drops traffic **indiscriminately**, so the legit user's
  goodput stays low.

### Representative result (defaults, one run)

| Phase | Legit goodput | Legit p50 | Legit p95 | Verdict |
|---|---|---|---|---|
| A baseline | 100% | 27 ms | 28 ms | available |
| B attack | 40% | 224 ms | 1448 ms | **OUTAGE** |
| C +capacity | 95% | 31 ms | 1265 ms | available |
| D +admission | 20% | 31 ms | 683 ms | **OUTAGE** |

Numbers vary run to run (it's real timing on a real machine, and it depends on
your CPU), but the shape is stable. Re-run a few times / several seeds before
quoting figures in the report.

## The insight (this is the part worth discussing in the report)

1. **The outage is real and easy to reproduce.** Offered load above capacity, a
   bounded queue, and a shed-on-timeout policy is all it takes. We can point to
   measured latency and failures, not a formula.

2. **Capacity scaling (C) restores goodput but not the tail.** p50 is back to
   normal, yet p95 stays high. That tail comes from a *different* bottleneck —
   the cost of accepting hundreds of attacker connections — which adding
   service capacity doesn't fix. Real systems hit exactly this: you can scale the
   app tier and still suffer at the connection layer.

3. **Admission control (D) protects the server but not the user.** A plain rate
   limiter keeps latency low for what it admits, but with no way to tell the
   attacker apart from a normal user it drops both in proportion. The lesson:
   effective DDoS defense needs a **discriminator** (per-source limits, behavioral
   fingerprinting, challenges) — not just a smaller intake. Our loopback rig
   can't key on source IP (all traffic is 127.0.0.1), which is itself a good
   thing to state as a limitation.

Put together: **the best single lever here is capacity, but the real answer is to
combine admission control with per-source discrimination** so you shed the
attacker and keep the user. That's a clean motivation for the next sprint.

## Agent-driven mode (optional extension)

The heuristic agents in `../security_agents.py` can drive this rig instead of
fixed phases: map the defender's allocation to `set_capacity` / `set_rate_limit`
and the attacker's chosen intensity to `attack_workers`, run a round, and read
the *measured* outage back as the payoff. That turns the adaptive game into a
closed loop over a real service — a strong Sprint 4 result if we have time.

## Honesty / limitations (say these in the report)

- All traffic is loopback, so there is no network latency, no packet loss, and no
  source-IP diversity. We measure *server-side* saturation, not network
  saturation.
- `ThreadingHTTPServer` spawns a thread per connection; under a few hundred
  connections, thread/accept overhead is part of what we measure (realistic, but
  it adds noise to tail latency).
- The attack is volumetric (lots of cheap GETs). We don't model protocol-specific
  floods (SYN, slowloris) — those are listed as future work in the main project.
- Results are timing-dependent; use multiple runs before reporting means.
