"""Traffic sources for the demo: a legitimate user and an attacker (flood).

Both talk real HTTP over real sockets to the target server. The key safety
property lives in `assert_loopback`: the attacker (and the legit client) refuse
to send traffic anywhere except this machine's loopback interface. Point it at a
public host and it raises before a single packet is sent. That is deliberate --
it makes this a sandbox demonstration rig, not a tool that can attack anyone.

Each request returns a RequestResult so the orchestrator can measure *real*
latency and *real* success/failure, which is what turns "simulated DDoS" into
"observed DDoS on local infrastructure".
"""
from __future__ import annotations

import http.client
import ipaddress
import socket
import threading
import time
from dataclasses import dataclass


def assert_loopback(host: str) -> None:
    """Raise unless every address `host` resolves to is a loopback address.

    This guard keeps the demo self-contained: load is never generated against
    anything but the local machine.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ValueError(f"cannot resolve {host!r}: {e}") from e
    for info in infos:
        addr = info[4][0]
        if not ipaddress.ip_address(addr).is_loopback:
            raise ValueError(
                f"refusing to send traffic to {host!r} ({addr}): "
                "this demo only ever targets loopback (127.0.0.1 / ::1)."
            )


@dataclass
class RequestResult:
    t_start: float      # monotonic time the request was sent (relative set later)
    latency: float      # seconds until response (or until failure)
    status: int         # HTTP status, or 0 for a connection-level failure
    kind: str           # "legit" or "attack"


def _one_request(host: str, port: int, kind: str, timeout: float) -> RequestResult:
    start = time.monotonic()
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", "/")
        resp = conn.getresponse()
        resp.read()
        status = resp.status
        conn.close()
    except Exception:
        # Connection refused, reset, or timed out -> the asset was unreachable
        # for this request. Status 0 marks a hard failure (worse than a 503).
        status = 0
    return RequestResult(start, time.monotonic() - start, status, kind)


class TrafficSource:
    """A pool of threads firing requests at a target rate for a fixed duration."""

    def __init__(self, host: str, port: int, kind: str,
                 rate: float, workers: int, timeout: float = 2.0) -> None:
        assert_loopback(host)  # <-- fails fast if target is not loopback
        self.host = host
        self.port = port
        self.kind = kind
        self.rate = rate          # requests/sec this source aims to generate
        self.workers = workers    # concurrent connections it will use
        self.timeout = timeout
        self.results: list[RequestResult] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def _worker(self, per_worker_interval: float) -> None:
        # Each worker paces itself so the pool's combined rate ~= self.rate.
        while not self._stop.is_set():
            r = _one_request(self.host, self.port, self.kind, self.timeout)
            with self._lock:
                self.results.append(r)
            if per_worker_interval > 0:
                # Sleep the remainder of this worker's slice (never negative).
                self._stop.wait(max(0.0, per_worker_interval - r.latency))

    def run(self, duration: float) -> list[RequestResult]:
        self.results = []
        self._stop.clear()
        per_worker_interval = (self.workers / self.rate) if self.rate > 0 else 0.0
        threads = [threading.Thread(target=self._worker, args=(per_worker_interval,),
                                    daemon=True) for _ in range(self.workers)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        # Let it run for `duration`, then signal stop and drain.
        time.sleep(duration)
        self._stop.set()
        for t in threads:
            t.join(timeout=self.timeout + 1)
        # Rebase t_start to seconds-since-start for easy plotting.
        for r in self.results:
            r.t_start -= t0
        return self.results
