#!/usr/bin/env python3
"""Say which binary a bench run measured, and prove it can produce the readings.

A digest alone answers "is this the same file as last time". It does not answer
"is this the build I think it is": a stash-build-unstash cycle leaves the
previous binary in `target/`, both arms hash the same, and a comparison of a
build against itself looks clean. What separates the two is the presence of the
thing the run depends on.

So this reports both:

  sha256    of the file as it is on disk, taken again at ingest so a rebuild
            between the measurement and the record is caught rather than pasted
            over.
  markers   the exact log-message literals the run's readers grep for. A binary
            that does not contain one cannot write the event, so the reading
            that event carries cannot come out of this run — and that is
            knowable before the server is started rather than after three
            passes produced nothing.

The literals are imported from the readers that consume them, never restated
here: a second copy would let the binary check pass while the reader that
motivated it looks for something else.

`release-perf` sets `strip = "symbols"`, so a symbol table is not available on
the profile this harness measures and `nm` reports nothing to key on. String
literals survive stripping, which is why the marker is the message text. The
search is a byte scan of the file, so it needs no external tool and no minimum
run length.

Exit codes: 0 — every marker is present; 1 — a marker is absent, so this binary
cannot produce a reading the run needs; 2 — the binary could not be read.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import published_run_log  # noqa: E402
import server_kv_quant  # noqa: E402
import spec_round_log  # noqa: E402

# Each entry is one reading and the message literals that can carry it. A group
# with more than one alternative is satisfied by any of them: the round loop
# writes the greedy or the stochastic name depending on how the request was
# sampled, and `spec_round_log` matches either.
PLAIN_MARKERS = {
    "ttft": (published_run_log.TTFT_MARKER,),
    "decode_rate": (published_run_log.ITL_MESSAGE,),
    "sampling": (published_run_log.SAMPLER_MARKER,),
    "kv_quant": (server_kv_quant.EVENT,),
}
SPECULATIVE_MARKERS = {
    "ttft": (published_run_log.TTFT_MARKER,),
    "sampling": (published_run_log.SAMPLER_MARKER,),
    "kv_quant": (server_kv_quant.EVENT,),
    "round_loop": tuple(f"{m}: done" for m in spec_round_log.DONE_MARKERS),
}


def markers_for(arm):
    return PLAIN_MARKERS if arm == "plain" else SPECULATIVE_MARKERS


def identify(path, arm):
    """`(record, missing)` — the identity block and the readings it cannot give."""
    blob = pathlib.Path(path).read_bytes()
    counts = {}
    missing = []
    for reading, alternatives in sorted(markers_for(arm).items()):
        found = {m: blob.count(m.encode("utf-8")) for m in alternatives}
        counts.update(found)
        if not any(found.values()):
            missing.append((reading, alternatives))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "arm": arm,
        "markers": counts,
    }, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("binary")
    ap.add_argument("--arm", required=True, choices=("plain", "speculative"))
    args = ap.parse_args()

    try:
        record, missing = identify(args.binary, args.arm)
    except OSError as exc:
        print(f"binary_identity: cannot read {args.binary}: {exc}", file=sys.stderr)
        return 2

    for reading, alternatives in missing:
        print(
            f"binary_identity: {args.binary} contains none of "
            f"{', '.join(repr(m) for m in alternatives)}, so it cannot write the "
            f"event this run reads its {reading} from. It is not a build of this "
            "tree, or the message was renamed and the reader was not.",
            file=sys.stderr,
        )
    if missing:
        return 1

    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
