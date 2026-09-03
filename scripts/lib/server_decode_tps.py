#!/usr/bin/env python3
"""Read one request's decode rate off a running rmlx server.

The server times every generation's inter-token gaps and appends the aggregate
to a ring exposed as JSON at `GET /metrics/cache` under `itl`, newest last. The
`step_mean_ms` there is the mean of the gaps between consecutive tokens, so
`1000 / step_mean_ms` is `(n - 1) / (t_last - t_first)` — the same window the
speculative round loops report as `decode_tps` and the same one `rmlx baseline`
reports. A bench arm with no round-loop record still has this one, and taking it
is why the no-drafter and speculative arms of a comparison mean the same thing.

The ring is server-wide and one entry deep per request, so "this request's
sample" is only the last entry if exactly one entry appeared. `--after N`, with
N the ring length observed before the request, enforces that: a request that
produced fewer than two tokens appends nothing and would otherwise be read as
the previous request's rate.

Output (stdout), one `key=value` per line:

    ring_len=<n>        entries in the ITL ring right now
    decode_tps=<f>      only with --after, and only once the ring grew by one
    step_count=<n>      tokens behind that sample

Exit codes: 0 — read; 2 — the server did not answer; 5 — the ring did not grow
by exactly one; 6 — the new sample carries no usable interval.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


def itl_ring(base_url, timeout):
    """The server's ITL ring, oldest first."""
    with urllib.request.urlopen(f"{base_url}/metrics/cache", timeout=timeout) as r:
        body = json.load(r)
    samples = body.get("itl")
    if not isinstance(samples, list):
        raise ValueError("/metrics/cache carries no 'itl' array")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="e.g. http://127.0.0.1:8090")
    parser.add_argument(
        "--after",
        type=int,
        default=None,
        help="ring length observed before the request this call is reading",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        samples = itl_ring(args.base_url.rstrip("/"), args.timeout)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"server_decode_tps: {args.base_url}: {exc}", file=sys.stderr)
        return 2

    lines = [f"ring_len={len(samples)}"]
    if args.after is None:
        print("\n".join(lines))
        return 0

    if len(samples) != args.after + 1:
        print(
            f"server_decode_tps: the ITL ring went from {args.after} to "
            f"{len(samples)} entries across one request; exactly one sample has "
            "to be attributable to it",
            file=sys.stderr,
        )
        return 5

    sample = samples[-1]
    mean_ms = sample.get("step_mean_ms")
    step_count = sample.get("step_count", 0)
    if not isinstance(mean_ms, (int, float)) or mean_ms <= 0 or step_count < 2:
        print(
            f"server_decode_tps: newest ITL sample has step_mean_ms={mean_ms!r} "
            f"step_count={step_count!r}, no interval to read a rate from",
            file=sys.stderr,
        )
        return 6

    lines.append(f"decode_tps={1000.0 / mean_ms:.6f}")
    lines.append(f"step_count={step_count}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
