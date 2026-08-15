#!/usr/bin/env python3
"""Read `decode_profile{prefill_ms}` back out of an rmlx run log.

Nothing in a Metal System Trace marks where prefill ends: weight load submits no
GPU work, so the traced process's very first row is already prefill. The run
itself knows the boundary and emits it as a plain `info!` event, which lands in
`<RMLX_HOME>/logs/<run-id>.jsonl` even when `xctrace --launch` has swallowed the
child's stdout. `scripts/mst_capture.sh` uses this to default `--skip-ms` to a
measured value rather than a guessed one.

Prints the rounded integer milliseconds, or nothing at all when the log carries
no such event — the caller treats empty as "could not measure" and says so,
rather than silently summarising a window that still contains prefill.
"""

import json
import sys


def prefill_ms(path):
    """Last `prefill_ms` in the log, or None.

    The last one, not the first: a run may decode more than once (a warmup pass
    ahead of the measured one), and the trace window covers all of them. Taking
    the first would place the boundary before work the summary still counts.
    """
    found = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            # Cheap reject before the JSON parse: these logs run to megabytes
            # and only a couple of lines carry the field.
            if '"prefill_ms"' not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            fields = record.get("fields")
            if isinstance(fields, dict) and "prefill_ms" in fields:
                try:
                    found = float(fields["prefill_ms"])
                except (TypeError, ValueError):
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
