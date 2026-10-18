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

Candidates come on stdin, one `<offset><TAB><path>` per line: the byte offset
is where the log stood when the phase began, 0 for a log that did not exist
then. Only the bytes from that offset are the phase's. A log the phase shares
with an earlier process (same second, same run id) is handed on as a copy of
those bytes in `--slice-dir`, so a reader of the printed path sees nothing the
earlier process wrote; a log that began with the phase is printed as it is.

A log whose phase bytes carry an `rmlx start` from any other pid is refused:
two processes wrote to the phase, and the reader cannot tell their lines apart.

Prints the path to read.

Exit codes: 0 — one match; 4 — no candidate carries that pid; 5 — more than one
does; 6 — the matching log's phase carries another pid's `rmlx start`.
"""

import argparse
import json
import os
import sys

EVENT = "rmlx start"


def read_from(path, offset):
    """The text of `path` from byte `offset`, or None when it cannot be read."""
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None


def writer_pids(text):
    """Pids the `rmlx start` events in `text` report."""
    pids = set()
    for line in (text or "").splitlines():
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
    parser.add_argument("--slice-dir", required=True)
    args = parser.parse_args()

    candidates = []
    for line in sys.stdin:
        offset, _, path = line.rstrip("\n").partition("\t")
        if path:
            candidates.append((path, int(offset), read_from(path, int(offset))))
    matches = [c for c in candidates if args.pid in writer_pids(c[2])]

    if not matches:
        print(
            f"run_log_for_pid: none of {len(candidates)} candidate log(s) carries "
            f"an 'rmlx start' event from pid {args.pid}",
            file=sys.stderr,
        )
        return 4

    if len(matches) > 1:
        print(
            f"run_log_for_pid: {[m[0] for m in matches]} all carry pid {args.pid}; one process "
            "cannot have written two logs this phase can read as its own",
            file=sys.stderr,
        )
        return 5

    path, offset, text = matches[0]
    others = writer_pids(text) - {args.pid}
    if others:
        print(
            f"run_log_for_pid: {path} carries 'rmlx start' events from "
            f"{sorted(others)} as well as pid {args.pid} in this phase; the "
            "lines of the two processes cannot be told apart",
            file=sys.stderr,
        )
        return 6

    if offset == 0:
        print(path)
        return 0
    sliced = os.path.join(args.slice_dir, f"{os.path.basename(path)}.from{offset}")
    with open(sliced, "w", encoding="utf-8") as out:
        out.write(text)
    print(sliced)
    return 0


if __name__ == "__main__":
    sys.exit(main())
