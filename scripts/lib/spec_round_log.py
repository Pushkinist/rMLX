#!/usr/bin/env python3
"""Read a speculative round loop's own numbers back out of an rmlx run log.

Every speculative round loop closes a request with one
`<kind>_generate_{greedy,stochastic}...: done` event carrying the round counts,
the draft/accept totals and `decode_tps` — the rate over the window from the
first emitted token to the last, prefill excluded, which is the same window
`rmlx baseline` reports. The same line also carries `emitted` and `elapsed_ms`,
and `elapsed_ms` covers the prompt prefill too: dividing one by the other
reproduces the prefill-contaminated rate the engine stopped reporting, which on
a 4k prompt reads roughly half the real one. There is one decode rate on that
line, the engine computed it, and this is the only place it is read.

`decode_tps` is an `Option<f64>` rendered through `Debug`, so the JSON field is
the string `Some(20.98)` or `None` (docs/SPECULATIVE.md). `None` means the run
emitted fewer than two tokens and has no interval to measure — not a rate of
zero, and not something to substitute another number for.

A `done` line whose `decode_tps` is a bare number was written by a binary older
than that field's correction and carries the prefill-inclusive value under the
corrected name. Reading it is refused rather than guessed at: silently treating
it as the corrected number is the whole defect this module exists to close.

Output (stdout), one `key=value` per line:

    events=<n>                  round-loop done events considered
    rounds_total=<n>
    draft_tokens_total=<n>
    accept_tokens_total=<n>
    accept_rate=<f>             accept_tokens_total / draft_tokens_total
    accepted_per_step=<f>       accept_tokens_total / rounds_total
    decode_tps=<f>              one line per event with a measurable rate

Aggregating the `decode_tps` lines is the caller's job: a bench script that
already has a median/stddev helper must not grow a second one here.

Exit codes: 0 — read; 2 — log unreadable; 3 — a `done` event's `decode_tps` is
not the documented shape; 4 — no `done` event in the log.
"""

import argparse
import json
import sys

DONE_MARKERS = ("generate_greedy", "generate_stochastic")


class SpecLogError(Exception):
    """A `done` event carries a `decode_tps` this reader must not interpret."""


def done_events(path):
    """The round-loop `done` events in `path`, in file order.

    Lines that are not JSON are skipped: a log can be truncated mid-write by a
    killed server, and a partial last line is not a reason to lose the rest.
    """
    events = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "done" not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            fields = record.get("fields")
            if not isinstance(fields, dict):
                continue
            message = fields.get("message", "")
            if any(m in message for m in DONE_MARKERS) and "done" in message:
                events.append(fields)
    return events


def decode_tps(fields):
    """The rate the round loop measured, or None when it measured none.

    Raises SpecLogError for anything else, including a missing field — both
    mean the log predates the corrected field and the only rate in it is the
    contaminated one.
    """
    raw = fields.get("decode_tps")
    message = fields.get("message", "<no message>")
    if raw is None:
        raise SpecLogError(
            f"'{message}' carries no decode_tps field: this log was written by a "
            "binary that reported only the prefill-inclusive rate"
        )
    if not isinstance(raw, str):
        raise SpecLogError(
            f"'{message}' has decode_tps={raw!r}, a bare number where the "
            "corrected field renders Some(x) or None: this log was written by a "
            "binary whose decode_tps still counted prefill"
        )
    if raw == "None":
        return None
    if raw.startswith("Some(") and raw.endswith(")"):
        try:
            return float(raw[len("Some(") : -1])
        except ValueError as exc:
            raise SpecLogError(
                f"'{message}' has decode_tps={raw!r}, which is not a number"
            ) from exc
    raise SpecLogError(
        f"'{message}' has decode_tps={raw!r}, neither Some(x) nor None"
    )


def summarize(events):
    """The `key=value` lines for `events`."""
    rounds = sum(e.get("rounds", 0) for e in events)
    draft = sum(e.get("total_draft", 0) for e in events)
    # The Gemma4 assistant and MTP sidecar loops name it `total_accept`; the
    # cached spec loops name it `total_accept_count`.
    accept = sum(e.get("total_accept", e.get("total_accept_count", 0)) for e in events)

    lines = [
        f"events={len(events)}",
        f"rounds_total={rounds}",
        f"draft_tokens_total={draft}",
        f"accept_tokens_total={accept}",
        f"accept_rate={accept / draft if draft > 0 else 0.0:.6f}",
        f"accepted_per_step={accept / rounds if rounds > 0 else 0.0:.6f}",
    ]
    for event in events:
        rate = decode_tps(event)
        if rate is not None:
            lines.append(f"decode_tps={rate:.6f}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="an rmlx run log (<RMLX_HOME>/logs/*.jsonl)")
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="consider only the last N done events (0 = all)",
    )
    args = parser.parse_args()

    try:
        events = done_events(args.log)
    except OSError as exc:
        print(f"spec_round_log: cannot read {args.log}: {exc}", file=sys.stderr)
        return 2

    if not events:
        print(
            f"spec_round_log: no speculative round-loop 'done' event in {args.log}",
            file=sys.stderr,
        )
        return 4

    if args.last > 0:
        events = events[-args.last :]

    try:
        lines = summarize(events)
    except SpecLogError as exc:
        print(f"spec_round_log: {exc}", file=sys.stderr)
        return 3

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
