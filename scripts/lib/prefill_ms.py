#!/usr/bin/env python3
"""Read the prefill time of an rmlx run out of its log: `generate: prefill{prefill_ms}`.

Nothing in a Metal System Trace marks where prefill ends: weight load submits no
GPU work, so the traced process's very first row is already prefill. The run
itself knows the boundary and writes it as the `generate: prefill` event of each
request, which lands in `<RMLX_HOME>/logs/<run-id>.jsonl` even when
`xctrace --launch` has swallowed the child's stdout. `scripts/mst_capture.sh`
uses this to default `--skip-ms` to a measured value rather than a guessed one.

The event is matched by its exact message, so no other event that happens to
carry a `prefill_ms` field can be read as this one.

Prints the rounded integer milliseconds, or nothing at all when the log carries
no such event — the caller treats empty as "could not measure" and says so,
rather than silently summarising a window that still contains prefill.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import published_run_log  # noqa: E402


def prefill_ms(path):
    """Last `prefill_ms` in the log, or None.

    The last one, not the first: a run may serve more than one request (a warmup
    ahead of the measured one), and the trace window covers all of them. Taking
    the first would place the boundary before work the summary still counts.
    """
    found = None
    for fields in published_run_log.events_matching(
        path, published_run_log.PREFILL_MESSAGE, exact=True
    ):
        try:
            found = float(fields["prefill_ms"])
        except (KeyError, TypeError, ValueError):
            continue
    return found


def main():
    if len(sys.argv) != 2:
        print("usage: prefill_ms.py <run-log.jsonl>", file=sys.stderr)
        return 2
    value = prefill_ms(sys.argv[1])
    if value is None:
        return 0
    print(int(round(value)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
