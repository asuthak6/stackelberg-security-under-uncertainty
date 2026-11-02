"""The environment: the one piece that makes this a single model, not two.

The game asks the environment ONE question each round, for each server:
    "given this much defense and this much attack, did the attack cause an outage?"

Two backends answer that question the same way, so the agents and the game loop
are identical whichever is plugged in:

  FormulaEnvironment  -- answers with the Tullock probability a/(a+d+k). Fast and
                         analyzable: use it to solve for equilibrium and to run
                         thousands of rounds (baseline.py, experiments.py).

  LiveEnvironment     -- answers by MEASURING a real loopback HTTP server. The
                         defense allocated to a server becomes real serving
                         capacity; the attack becomes a real flood of requests;
                         an "outage" is observed when a legitimate user can no
                         longer get through. This is the TA's "real DDoS", now
                         wired into the model as its environment.

The formula is a surrogate for the live measurement; validate.py checks that the
two agree. Everything is loopback-only (see clients.assert_loopback).
"""
from __future__ import annotations

import random
import socket
import threading
import time
from abc import ABC, abstractmethod

from config import GameConfig, success_prob
from clients import TrafficSource
from server import ServerState, make_server


class Environment(ABC):
    """Resolves one round of the game into per-server outcomes."""

    @abstractmethod
    def resolve(self, defense: list[float], attack: list[float],
                rng: random.Random) -> list[int]:
        """Return a list of per-server outcomes: 1 = attack succeeded (outage),
        0 = defended (blocked or not attacked)."""
        raise NotImplementedError


class FormulaEnvironment(Environment):
    """The fast, analyzable backend: outcome drawn from the contest probability."""

    def __init__(self, cfg: GameConfig) -> None:
        self.cfg = cfg

    def resolve(self, defense, attack, rng):
        out = []
        for i in range(self.cfg.n):
            if attack[i] <= 0:
                out.append(0)
                continue
            p = success_prob(attack[i], defense[i], self.cfg.resilience)
            out.append(1 if rng.random() < p else 0)
        return out


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class LiveEnvironment(Environment):
    """The real backend: outcome MEASURED on a loopback server.

    Each attacked server is exercised one at a time on a single reused server
    process whose capacity is set from the defense allocated to it. The attack
    allocation becomes a real flood; a steady legitimate client runs alongside.
    If the legitimate client's success rate falls below `outage_threshold`, the
    server is deemed to have suffered an outage (attack succeeded).

    Use it as a context manager so the server starts once and is reused:
        with LiveEnvironment(cfg) as env:
            env.resolve(defense, attack, rng)
    """

    def __init__(self, cfg: GameConfig, *, base_capacity: int = 1,
                 capacity_per_defense: float = 2.0, workers_per_attack: float = 75.0,
                 service_time: float = 0.02, queue_timeout: float = 0.12,
                 legit_rate: float = 20.0, legit_workers: int = 3,
                 duration: float = 0.6, outage_threshold: float = 0.5) -> None:
        self.cfg = cfg
        self.base_capacity = base_capacity
        self.capacity_per_defense = capacity_per_defense
        self.workers_per_attack = workers_per_attack
        self.service_time = service_time
        self.queue_timeout = queue_timeout
        self.legit_rate = legit_rate
        self.legit_workers = legit_workers
        self.duration = duration
        self.outage_threshold = outage_threshold
        self._httpd = None
        self._state = None
        self._port = None

    def __enter__(self) -> "LiveEnvironment":
        self._port = _free_port()
        self._state = ServerState(self.base_capacity, self.service_time,
                                  self.queue_timeout, rate_limit=0.0)
        self._httpd = make_server("127.0.0.1", self._port, self._state)
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        time.sleep(0.2)
        return self

    def __exit__(self, *exc) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None

    def _capacity_for(self, defense_i: float) -> int:
        return max(1, round(self.base_capacity + self.capacity_per_defense * defense_i))

    def _measure_one(self, defense_i: float, attack_i: float) -> int:
        """Run a real attack on one server; return 1 if it caused an outage."""
        # Resize the server's capacity to reflect the defense on this server.
        # (Handlers hold their own semaphore reference, so this swap is safe.)
        self._state.slots = threading.BoundedSemaphore(self._capacity_for(defense_i))
        time.sleep(0.03)  # let any in-flight requests from the previous cell drain
        workers = max(1, round(self.workers_per_attack * attack_i))

        legit = TrafficSource("127.0.0.1", self._port, "legit",
                              self.legit_rate, self.legit_workers, timeout=1.0)
        attack = TrafficSource("127.0.0.1", self._port, "attack",
                               rate=0.0, workers=workers, timeout=1.0)
        results: dict[str, list] = {"legit": [], "attack": []}

        def drive(src):
            results[src.kind] = src.run(self.duration)

        threads = [threading.Thread(target=drive, args=(s,)) for s in (legit, attack)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        legit_results = results["legit"]
        if not legit_results:
            return 0
        goodput = sum(1 for r in legit_results if r.status == 200) / len(legit_results)
        return 1 if goodput < self.outage_threshold else 0

    def resolve(self, defense, attack, rng):
        if self._httpd is None:
            raise RuntimeError("use LiveEnvironment as a context manager (with ...:)")
        out = []
        for i in range(self.cfg.n):
            out.append(0 if attack[i] <= 0 else self._measure_one(defense[i], attack[i]))
        return out
