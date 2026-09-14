#!/usr/bin/env python3
"""Read whether a run's requests were sampled, out of its rmlx run log.

The engine writes one `host categorical sampler active` event per request, and
only when the sampler is live — a greedy request produces none. That is the only
place the question is answered: a request asking for `temperature: 0` does not
prove the engine ran greedy, because the checkpoint's own generation config and
the engine's fallbacks both reach the resolved sampling and neither is visible
from the request.

The distinction decides what a caller comparing two arms' answers is entitled to
say. Two greedy arms of one verifier compute one answer, so a difference between
them is a defect. Two sampled arms are two draws and may differ with nothing
wrong, so there is no comparison to make and the caller has to record that rather
than pass silently.

A log whose sampler events cover some of its requests and not the others is
refused: those requests did not share one sampling setup, so the run has no
single disposition to record.

Output (stdout):

    sampler_events=<n>
    sampled=<true|false>

Exit codes: 0 — read; 2 — log unreadable; 5 — the sampler covered some requests
and not the others.
"""

import argparse
import json
import sys

MARKER = "generate: host categorical sampler active"


def sampler_events(path):
    """How many requests the log says the engine resolved a sampler for."""
    count = 0
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if MARKER not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            fields = record.get("fields")
            if isinstance(fields, dict) and MARKER in fields.get("message", ""):
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="an rmlx run log (<RMLX_HOME>/logs/*.jsonl)")
    parser.add_argument(
        "--expect-requests",
        type=int,
        required=True,
        help="requests served against this log, warmups included",
    )
    args = parser.parse_args()

    try:
        count = sampler_events(args.log)
    except OSError as exc:
        print(f"server_sampling: cannot read {args.log}: {exc}", file=sys.stderr)
        return 2

    if count not in (0, args.expect_requests):
        print(
            f"server_sampling: {args.log} holds {count} sampler events for "
            f"{args.expect_requests} requests; the sampler covered some of them and "
            "not the others, so this run has no one sampling setup to record",
            file=sys.stderr,
        )
        return 5

    print(f"sampler_events={count}")
    print(f"sampled={'true' if count else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
