"""Orchestrator: run a real DDoS demonstration against our own loopback server.

Four phases, each measured on live traffic. The metric that matters is the
LEGITIMATE user's availability -- can a normal user still use the service while
the attacker floods it?

  A  BASELINE            legit traffic only, no attack. (What "healthy" means.)
  B  ATTACK              legit + attacker flood, defender does nothing.
  C  MITIGATE: CAPACITY  defender spends its budget on more service capacity
                         (this is exactly how mitigation works in our existing
                         security_agents.py game: budget -> effective capacity).
  D  MITIGATE: ADMISSION defender instead caps intake with a token-bucket rate
                         limiter, protecting the server's health.

For each phase we report, for the legit user:
  goodput   : fraction of legit requests that got 200 OK
  p50 / p95 : median and tail latency (ms)
  verdict   : OUTAGE if legit goodput < --outage-threshold

Everything is 127.0.0.1 only; the clients refuse any other target.

    python run_demo.py
    python run_demo.py --attack-workers 150 --duration 6
"""
from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

from clients import RequestResult, TrafficSource
from target_server import ServerState, make_server

HOST = "127.0.0.1"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


class PhaseStats:
    def __init__(self, name: str, legit: list[RequestResult],
                 attack: list[RequestResult]) -> None:
        self.name = name
        self.legit = legit
        self.attack = attack

    def goodput(self) -> float:
        if not self.legit:
            return float("nan")
        return sum(1 for r in self.legit if r.status == 200) / len(self.legit)

    def legit_latency_ms(self, pct: float) -> float:
        lats = [r.latency * 1000 for r in self.legit if r.status == 200]
        return percentile(lats, pct)

    def row(self, outage_threshold: float) -> dict:
        gp = self.goodput()
        return {
            "phase": self.name,
            "legit_requests": len(self.legit),
            "legit_goodput_pct": round(gp * 100, 1),
            "legit_p50_ms": round(self.legit_latency_ms(0.50), 1),
            "legit_p95_ms": round(self.legit_latency_ms(0.95), 1),
            "attack_requests": len(self.attack),
            "verdict": "OUTAGE" if gp < outage_threshold else "available",
        }


def run_phase(name: str, state: ServerState, port: int, duration: float,
              legit_rate: float, legit_workers: int,
              attack_workers: int) -> PhaseStats:
    """Start fresh traffic sources, run them concurrently for `duration`.

    The attacker runs its workers flat out (rate=0 -> no self-pacing), which is
    what a flood is: as many requests as the connections can push.
    """
    legit = TrafficSource(HOST, port, "legit", legit_rate, legit_workers)
    sources = [legit]
    if attack_workers > 0:
        sources.append(TrafficSource(HOST, port, "attack", rate=0.0,
                                     workers=attack_workers))

    results: dict[str, list[RequestResult]] = {"legit": [], "attack": []}

    def drive(src: TrafficSource) -> None:
        results[src.kind] = src.run(duration)

    threads = [threading.Thread(target=drive, args=(s,)) for s in sources]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return PhaseStats(name, results["legit"], results["attack"])


def set_capacity(state: ServerState, capacity: int) -> None:
    """Resize the server's concurrent-slot semaphore between phases."""
    state.slots = threading.BoundedSemaphore(capacity)


def set_rate_limit(state: ServerState, rate: float) -> None:
    state.bucket.rate = rate
    state.bucket.capacity = max(rate, 1.0)
    state.bucket._tokens = state.bucket.capacity
    state.bucket._last = time.monotonic()


def main() -> None:
    p = argparse.ArgumentParser(description="Real DDoS demo on a loopback server (4 phases)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--duration", type=float, default=5.0, help="seconds per phase")
    p.add_argument("--base-capacity", type=int, default=8,
                   help="server slots with no mitigation (the undefended asset)")
    p.add_argument("--mitigation-budget", type=int, default=80,
                   help="extra slots the defender adds in phase C")
    p.add_argument("--service-time", type=float, default=0.025)
    p.add_argument("--queue-timeout", type=float, default=0.2,
                   help="max seconds a request waits for a slot before 503")
    p.add_argument("--legit-rate", type=float, default=20.0, help="legit requests/sec")
    p.add_argument("--legit-workers", type=int, default=4)
    p.add_argument("--attack-workers", type=int, default=400,
                   help="attacker concurrent connections (flood strength)")
    p.add_argument("--admission-rate", type=float, default=150.0,
                   help="token-bucket intake cap used in phase D (below base throughput)")
    p.add_argument("--outage-threshold", type=float, default=0.5)
    p.add_argument("--output", default="demo_results.csv")
    p.add_argument("--raw-output", default="demo_raw.csv")
    args = p.parse_args()

    state = ServerState(args.base_capacity, args.service_time, args.queue_timeout,
                        rate_limit=0.0)
    httpd = make_server(HOST, args.port, state)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)

    base_tput = args.base_capacity / args.service_time
    print(f"Target: http://{HOST}:{args.port}")
    print(f"  base capacity {args.base_capacity} slots x {args.service_time*1000:.0f}ms "
          f"=> ~{base_tput:.0f} req/s sustainable")
    print(f"  legit user {args.legit_rate:.0f} req/s | attacker {args.attack_workers} "
          f"flooding connections\n")

    phases: list[PhaseStats] = []

    # A: baseline
    set_capacity(state, args.base_capacity)
    set_rate_limit(state, 0.0)
    phases.append(run_phase("A_baseline", state, args.port, args.duration,
                            args.legit_rate, args.legit_workers, 0))

    # B: attack, no mitigation
    set_capacity(state, args.base_capacity)
    set_rate_limit(state, 0.0)
    phases.append(run_phase("B_attack", state, args.port, args.duration,
                            args.legit_rate, args.legit_workers, args.attack_workers))

    # C: mitigation by capacity (matches the game's budget->capacity model)
    set_capacity(state, args.base_capacity + args.mitigation_budget)
    set_rate_limit(state, 0.0)
    phases.append(run_phase("C_mitigate_capacity", state, args.port, args.duration,
                            args.legit_rate, args.legit_workers, args.attack_workers))

    # D: mitigation by admission control (token bucket), base capacity
    set_capacity(state, args.base_capacity)
    set_rate_limit(state, args.admission_rate)
    phases.append(run_phase("D_mitigate_admission", state, args.port, args.duration,
                            args.legit_rate, args.legit_workers, args.attack_workers))

    httpd.shutdown()

    rows = [ph.row(args.outage_threshold) for ph in phases]
    cols = ["phase", "legit_requests", "legit_goodput_pct", "legit_p50_ms",
            "legit_p95_ms", "attack_requests", "verdict"]
    width = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(width[c]) for c in cols))

    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with Path(args.raw_output).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "t_start_s", "latency_ms", "status", "kind"])
        for ph in phases:
            for r in ph.legit + ph.attack:
                w.writerow([ph.name, round(r.t_start, 4), round(r.latency * 1000, 2),
                            r.status, r.kind])

    print(f"\nSaved summary -> {Path(args.output).resolve()}")
    print(f"Saved raw log -> {Path(args.raw_output).resolve()}")
    a, b, c, d = (ph.goodput() for ph in phases)
    print(f"\nLegit availability: baseline {a:.0%} -> attack {b:.0%} -> "
          f"+capacity {c:.0%} -> +admission {d:.0%}")


if __name__ == "__main__":
    main()
