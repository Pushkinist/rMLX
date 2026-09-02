#!/usr/bin/env python3
"""Read the KV codec a run resolved out of its rmlx run log.

A bench row's `kv_quant` is part of the `bests` cell key, so a label that does
not describe the codec the server actually ran files the measurement under
another codec's name. Passing no `--kv-quant` does not make the field unknown —
the CLI resolves one and says so once, at startup, in the `cache-type resolved`
event. That field is written through `KvQuant`'s `Display`, which is the same
spelling the flag accepts and the metrics DB records, so it is used verbatim
rather than mapped here: a second mapping is a second thing to drift.

A log carrying more than one distinct value is refused. That means either two
servers wrote to one file or the codec changed mid-run, and neither leaves one
label the whole run can honestly carry.

Every canonical codec name is lowercase (`none`, `k8v8`, `mixed_k8g64_v4g64`),
so a value carrying an upper-case letter or a struct body was rendered through
`Debug` instead — `None`, `K8V8`, `Mixed { k_bits: 8, .. }`. None of those is a
name the flag accepts or the DB records, and filing a row under one puts the
measurement in a cell nothing else will ever land in. Refused.

Output (stdout): `kv_quant=<name>`.

Exit codes: 0 — read; 2 — log unreadable; 4 — no `cache-type resolved` event;
5 — the log resolved more than one codec; 6 — the codec is not written under
its canonical name.
"""

import argparse
import json
import sys

EVENT = "cache-type resolved"


def resolved(path):
    """Every distinct `kv_quant` the log resolved, in first-seen order."""
    seen = []
    with open(path, encoding="utf-8", errors="replace") as handle:
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
            name = fields.get("kv_quant")
            if isinstance(name, str) and name not in seen:
                seen.append(name)
    return seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="an rmlx run log (<RMLX_HOME>/logs/*.jsonl)")
    args = parser.parse_args()

    try:
        names = resolved(args.log)
    except OSError as exc:
        print(f"server_kv_quant: cannot read {args.log}: {exc}", file=sys.stderr)
        return 2

    if not names:
        print(
            f"server_kv_quant: no '{EVENT}' event in {args.log}, so which KV codec "
            "the run used is not recorded anywhere",
            file=sys.stderr,
        )
        return 4

    if len(names) > 1:
        print(
            f"server_kv_quant: {args.log} resolved {names}; one run log cannot "
            "carry one kv_quant label",
            file=sys.stderr,
        )
        return 5

    name = names[0]
    if any(c.isupper() for c in name) or "{" in name:
        print(
            f"server_kv_quant: {args.log} names the codec {name!r}, which is a "
            "Debug rendering rather than the canonical lower-case name the flag "
            "accepts and the metrics DB records",
            file=sys.stderr,
        )
        return 6

    print(f"kv_quant={name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
