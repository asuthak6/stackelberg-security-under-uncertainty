"""SE 4457 project: the DEFENDER's asset -- a real HTTP server, bound to loopback only.

This is a genuine socket server (it really accepts TCP connections and really
serves HTTP), but it only ever listens on 127.0.0.1, so the only thing that can
reach it is another process on this same machine. That is what makes the
"attack" in this project legal and self-contained: we flood a server we own,
over the loopback interface, and nothing leaves the laptop.

Two knobs model the defender's mitigation budget, and both affect real behaviour
that we measure rather than assume:

1. capacity  -- the number of requests the backend can process *at the same
                time*. Implemented as a semaphore with `capacity` slots. Each
                request holds a slot for `service_time` seconds (simulated work:
                a sleep). So sustainable throughput is roughly
                capacity / service_time requests per second. Push past that and
                the slots stay full, new requests wait, latency climbs, and once
                the wait exceeds `queue_timeout` the server sheds the request
                with HTTP 503. That 503 storm under load IS the outage.

2. rate_limit -- an optional token-bucket admission control. Before a request is
                allowed to compete for a slot, it must take a token. Tokens
                refill at `rate_limit` per second. Excess requests are rejected
                immediately with HTTP 429 -- cheaply, without tying up a slot.
                This is the realistic DDoS defense: cap intake so the backend
                never collapses, trading "some requests refused" for "the server
                stays alive".

Run it directly to serve until Ctrl+C:
    python target_server.py --port 8080 --capacity 8 --service-time 0.02
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class TokenBucket:
    """Classic token bucket. Thread-safe. Returns True if a token was available.

    `rate` tokens are added per second up to a burst ceiling of `capacity`
    tokens. A rate of 0 disables the limiter (every request is admitted).
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(rate, 1.0)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> bool:
        if self.rate <= 0:
            return True  # limiter disabled
        with self._lock:
            now = time.monotonic()
            # Refill based on elapsed time since the last check.
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class ServerState:
    """Shared mitigation configuration and live counters for one server."""

    def __init__(self, capacity: int, service_time: float, queue_timeout: float,
                 rate_limit: float) -> None:
        self.service_time = service_time
        self.queue_timeout = queue_timeout
        # `capacity` concurrent processing slots. Acquire blocks until a slot is
        # free or `queue_timeout` passes.
        self.slots = threading.BoundedSemaphore(capacity)
        self.bucket = TokenBucket(rate_limit)
        # Live counters (atomic enough under CPython's GIL for integer += here,
        # but we guard with a lock to be correct about it).
        self._lock = threading.Lock()
        self.served = 0       # 200 OK
        self.rejected = 0     # 429 (rate limited)
        self.overloaded = 0   # 503 (no slot within timeout)

    def bump(self, field: str) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + 1)


class Handler(BaseHTTPRequestHandler):
    # Injected by the factory below.
    state: ServerState

    # Silence the default per-request stderr logging; we do our own accounting.
    def log_message(self, *args) -> None:  # noqa: D401, ANN002
        return

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        state = self.state

        # --- Admission control (the defender's active mitigation) -------------
        if not state.bucket.take():
            state.bump("rejected")
            self._respond(429, b"rate limited\n")
            return

        # --- Capacity (the defender's effective service capacity) -------------
        acquired = state.slots.acquire(timeout=state.queue_timeout)
        if not acquired:
            # Waited in the queue too long -> the asset is effectively down for
            # this request. This is a real, measured outage, not a formula.
            state.bump("overloaded")
            self._respond(503, b"overloaded\n")
            return
        try:
            # Simulated backend work. Real wall-clock time, so real latency.
            time.sleep(state.service_time)
            state.bump("served")
            self._respond(200, b"ok\n")
        finally:
            state.slots.release()

    def _respond(self, code: int, body: bytes) -> None:
        try:
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client gave up (timed out and hung up) while we were replying.
            # Under a flood this is normal; just drop it.
            pass


def make_server(host: str, port: int, state: ServerState) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(
            f"refusing to bind to {host!r}: this demo server only serves loopback"
        )
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd


def run(host: str, port: int, capacity: int, service_time: float,
        queue_timeout: float, rate_limit: float) -> None:
    state = ServerState(capacity, service_time, queue_timeout, rate_limit)
    httpd = make_server(host, port, state)
    print(f"target server listening on http://{host}:{port}")
    print(f"  capacity={capacity} slots, service_time={service_time*1000:.0f}ms, "
          f"queue_timeout={queue_timeout*1000:.0f}ms, "
          f"rate_limit={'off' if rate_limit <= 0 else f'{rate_limit:.0f}/s'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.shutdown()
        print(f"served={state.served} rejected(429)={state.rejected} "
              f"overloaded(503)={state.overloaded}")


def main() -> None:
    p = argparse.ArgumentParser(description="Loopback-only target HTTP server (the defended asset)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--capacity", type=int, default=8,
                   help="concurrent processing slots (the defender's service capacity)")
    p.add_argument("--service-time", type=float, default=0.02,
                   help="seconds of work per request")
    p.add_argument("--queue-timeout", type=float, default=0.5,
                   help="max seconds a request waits for a slot before 503")
    p.add_argument("--rate-limit", type=float, default=0.0,
                   help="token-bucket admission rate per second (0 = disabled)")
    args = p.parse_args()
    run(args.host, args.port, args.capacity, args.service_time,
        args.queue_timeout, args.rate_limit)


if __name__ == "__main__":
    main()
