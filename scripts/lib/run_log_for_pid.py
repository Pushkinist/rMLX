#!/usr/bin/env python3
"""Pick the run log a given process wrote, out of the candidates on stdin.

A bench phase starts a server and then has to read that server's log. "The
newest file", or "the lexicographically last new file", answers a different
question: two runs starting in the same second share a run-id and therefore a
log path, and any other rmlx process can create a file in the same directory
between the two listings. Every number the phase then reports would belong to
someone else's run, with nothing in the output to say so.

The `rmlx start` event carries the writing process's pid. This reads it and
returns the one candidate that matches, refusing when zero or more than one do
— either way the phase does not have a log it can honestly attribute.

Candidates come on stdin, one path per line. Prints the matching path.

Exit codes: 0 — one match; 4 — no candidate carries that pid; 5 — more than one
does.
"""

import argparse
import json
import sys

EVENT = "rmlx start"


def writer_pids(path):
    """Pids the `rmlx start` events in `path` report."""
    pids = set()
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return pids
    with handle:
        for line in handle:
            if EVENT not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            fields = record.get("fields")
            if not isinstance(fields, dict) or fields.get("message") != EVENT:
                continue
            pid = fields.get("pid")
            if isinstance(pid, int):
                pids.add(pid)
    return pids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()

    candidates = [line.strip() for line in sys.stdin if line.strip()]
    matches = [p for p in candidates if args.pid in writer_pids(p)]

    if not matches:
        print(
            f"run_log_for_pid: none of {len(candidates)} candidate log(s) carries "
            f"an 'rmlx start' event from pid {args.pid}",
            file=sys.stderr,
        )
        return 4

    if len(matches) > 1:
        print(
            f"run_log_for_pid: {matches} all carry pid {args.pid}; one process "
            "cannot have written two logs this phase can read as its own",
            file=sys.stderr,
        )
        return 5

    print(matches[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
