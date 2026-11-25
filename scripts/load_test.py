"""Load test: N concurrent users for a fixed duration, with percentiles.

    python scripts/load_test.py --users 50 --duration 120 --base-url https://localhost -k

Each virtual user registers an account, then loops through a mixed workload —
list threads, one SSE chat turn, poll the render state — measuring latency and
counting 429s. Exits non-zero when the error rate exceeds --max-error-rate.

Point it at a stack started with SYSML_SMOKE=1 so a run tests *this* deployment
rather than the model provider's throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import secrets
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import httpx

PASSWORD = "load-test-password"


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile: good enough, and impossible to misread."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def sse_types(text: str) -> List[str]:
    kinds: List[str] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                try:
                    kinds.append(str(json.loads(line[5:].strip()).get("type")))
                except ValueError:
                    pass
    return kinds


class Sample:
    __slots__ = ("latencies", "errors", "auth_limited", "server_errors")

    def __init__(self) -> None:
        self.latencies: Dict[str, List[float]] = {"threads": [], "chat": [], "render": []}
        self.errors = 0
        self.auth_limited = 0
        self.server_errors = 0

    @property
    def requests(self) -> int:
        return sum(len(values) for values in self.latencies.values())

    def record(self, kind: str, status: int, elapsed: float) -> None:
        self.latencies[kind].append(elapsed)
        if status == 429:
            self.auth_limited += 1
        elif status >= 500:
            self.server_errors += 1
        elif status >= 400:
            self.errors += 1


async def one_user(
    base_url: str,
    verify: bool,
    timeout: float,
    deadline: float,
    sample: Sample,
    lock: asyncio.Lock,
    bootstrap: asyncio.Lock,
) -> None:
    """One virtual user, with its own session.

    The client has to be per user: an httpx client owns exactly one cookie jar,
    and sharing one makes every task act as whichever account registered last.
    That collapse hides real behaviour — a shared jar produced 404s on threads
    the *same* task had just created, because another task had meanwhile
    re-registered and replaced the cookie.
    """
    email = f"load-{secrets.token_hex(4)}@example.test"
    async with httpx.AsyncClient(
        base_url=base_url,
        verify=verify,
        timeout=httpx.Timeout(timeout, connect=10.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
    ) as client:
        # Sign-up is serialised and 429-aware: /api/auth/ is limited per
        # address and every virtual user shares one here, so a 429 means the
        # limiter is working. Bounded, so a broken deploy still fails.
        async with bootstrap:
            for _ in range(20):
                registered = await client.post(
                    "/api/auth/register", json={"email": email, "password": PASSWORD}
                )
                if registered.status_code != 429:
                    break
                await asyncio.sleep(6.5)
        if registered.status_code != 200:
            async with lock:
                sample.errors += 1
            return
        created = await client.post("/api/threads", json={})
        if created.status_code != 200:
            async with lock:
                sample.errors += 1
            return
        thread_id = created.json()["thread"]["id"]

        while time.monotonic() < deadline:
            for kind, method, path, body in (
                ("threads", "GET", "/api/threads", None),
                ("chat", "POST", f"/api/threads/{thread_id}/chat", {"message": "load probe"}),
                ("render", "GET", "/api/render/state", None),
            ):
                started = time.perf_counter()
                try:
                    response = await client.request(method, path, json=body)
                    status = response.status_code
                except httpx.HTTPError:
                    status = 0
                elapsed = time.perf_counter() - started
                async with lock:
                    sample.record(kind, status, elapsed)
                if time.monotonic() >= deadline:
                    break


async def run(args: argparse.Namespace) -> int:
    async with httpx.AsyncClient(
        base_url=args.base_url,
        verify=not args.insecure,
        timeout=httpx.Timeout(args.timeout, connect=10.0),
        follow_redirects=True,
    ) as probe:
        ready = await probe.get("/api/ready")
        if ready.status_code != 200:
            print(f"load: /api/ready is {ready.status_code}: {ready.text[:300]}", file=sys.stderr)
            return 2

    sample = Sample()
    lock = asyncio.Lock()
    # One gate for sign-up: the auth limiter counts per address, and all
    # virtual users share one.
    bootstrap = asyncio.Lock()
    deadline = time.monotonic() + args.duration
    started = time.monotonic()
    await asyncio.gather(
        *[
            one_user(
                args.base_url,
                not args.insecure,
                args.timeout,
                deadline,
                sample,
                lock,
                bootstrap,
            )
            for _ in range(args.users)
        ]
    )
    wall = time.monotonic() - started

    total = sample.requests
    failed = sample.errors + sample.server_errors
    rate = (failed / total) if total else 1.0
    everything = [value for values in sample.latencies.values() for value in values]

    print(f"load: {args.users} users for {args.duration}s ({wall:.1f}s wall)")
    print(f"  requests      {total} ({total / wall:.1f}/s)")
    print(f"  errors        {failed}  (rate {rate:.2%})")
    print(f"  429s          {sample.auth_limited}  (rate limits only count past the budget)")
    print(f"  p50/p95/p99   {percentile(everything, 0.50) * 1000:.0f} / "
          f"{percentile(everything, 0.95) * 1000:.0f} / "
          f"{percentile(everything, 0.99) * 1000:.0f} ms")
    for kind, values in sample.latencies.items():
        if values:
            print(f"  {kind:<12}  p95 {percentile(values, 0.95) * 1000:.0f} ms "
                  f"(mean {statistics.fmean(values) * 1000:.0f} ms, n={len(values)})")

    if rate > args.max_error_rate:
        print(f"load: FAILED — error rate {rate:.2%} > {args.max_error_rate:.2%}", file=sys.stderr)
        return 1
    print("load: ok")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--duration", type=float, default=120.0, help="seconds")
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("-k", "--insecure", action="store_true")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
