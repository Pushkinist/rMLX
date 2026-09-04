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
    emitted_total=<n>
    draft_tokens_total=<n>
    accept_tokens_total=<n>
    accept_rate=<f>             accept_tokens_total / draft_tokens_total
    accepted_per_step=<f>       accept_tokens_total / rounds_total
    tokens_per_round=<f>        emitted_total / rounds_total
    draft_ms_per_round=<f>      draft_ms summed / rounds_total
    verify_ms_per_round=<f>     verifier_ms summed / rounds_total
    loop_ms_per_round=<f>       (round_ms - draft_ms - verifier_ms) / rounds_total
    block_size=<n>              the block the engine actually ran
    decode_config=<s>           the cell every event agreed it belongs to
    decode_tps=<f>              one line per event with a measurable rate

The ratios are the engine's own formulas applied to the summed counters. The
engine also derives them per request and puts them on the same line, and every
event is checked against this module's arithmetic before it is aggregated: two
expressions of one formula drift silently otherwise, and the row that reaches
the append-only store cannot be taken back out.

`block_size` is the block the engine ran, which is not always the one asked
for — a sidecar caps it at its own — and `decode_config` is the cell key the
round loop composed
(`rmlx_metrics::cell::decode_config`) — the drafter, its block and, when the
loop resizes the block, the policy it resizes by. Reading it back rather than
spelling it here is why a new drafter reaches the metrics store without the
bench script learning about it. Events that disagree are refused: a log holding
two configurations has no one cell for its aggregate to belong to.

Aggregating the `decode_tps` lines is the caller's job: a bench script that
already has a median/stddev helper must not grow a second one here.

Exit codes: 0 — read; 2 — log unreadable; 3 — a `done` event's `decode_tps` is
not the documented shape, or the events name more than one cell; 4 — no `done`
event in the log; 5 — the log holds a different number of `done` events than
the caller served requests.
"""

import argparse
import json
import sys

DONE_MARKERS = ("generate_greedy", "generate_stochastic")


class SpecLogError(Exception):
    """A `done` event carries something this reader must not interpret."""


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


def one_value(events, field):
    """The value of `field` that every event agrees on.

    Raises SpecLogError when they disagree, or when an event carries none. An
    aggregate over two configurations belongs to neither cell, and a missing
    field means the log predates it and the value would have to be guessed at
    from the caller's own flags — which is how a row gets filed under a
    configuration the run did not use.
    """
    values = {e.get(field) for e in events}
    if None in values:
        raise SpecLogError(
            f"a round-loop 'done' event carries no {field} field: this log was "
            "written by a binary that did not report it"
        )
    if len(values) > 1:
        named = ", ".join(sorted(str(v) for v in values))
        raise SpecLogError(
            f"the round-loop events report {len(values)} values for {field} "
            f"({named}); an aggregate over them describes no one run"
        )
    return values.pop()


# The engine's own name for each derived field, and the raw counters it comes
# from. `rounds` is the denominator for all of them; `loop_ms_per_round` is a
# residual, so its numerator is a difference.
DERIVED_FIELDS = (
    ("accept_rate", ("total_accept",), ("total_draft",)),
    ("accepted_per_step", ("total_accept",), ("rounds",)),
    ("tokens_per_round", ("emitted",), ("rounds",)),
    ("draft_ms_per_round", ("draft_ms",), ("rounds",)),
    ("verify_ms_per_round", ("verifier_ms",), ("rounds",)),
    ("loop_ms_per_round", ("round_ms", "-draft_ms", "-verifier_ms"), ("rounds",)),
)


def _sum_terms(fields, terms):
    """Sum `terms`, where a leading `-` negates the named field."""
    total = 0.0
    for term in terms:
        if term.startswith("-"):
            total -= float(fields.get(term[1:], 0.0))
        else:
            total += float(fields.get(term, 0.0))
    return total


def check_derived(fields):
    """Refuse an event whose derived fields disagree with its own counters.

    The engine derives these per request and this module derives them per run;
    the two are one formula and this is where that is enforced. An event that
    carries none of them is left alone — the field set is checked by
    `one_value`, which names the missing one.
    """
    message = fields.get("message", "<no message>")
    for name, numerator, denominator in DERIVED_FIELDS:
        if name not in fields:
            continue
        bottom = _sum_terms(fields, denominator)
        want = _sum_terms(fields, numerator) / bottom if bottom > 0 else 0.0
        got = float(fields[name])
        if abs(got - want) > max(1e-6, abs(want) * 1e-6):
            raise SpecLogError(
                f"'{message}' reports {name}={got!r} but its own counters give "
                f"{want!r}: the engine and this reader do not agree on the formula"
            )


def summarize(events):
    """The `key=value` lines for `events`.

    Every ratio is the engine's formula over the summed counters. Zero rounds
    gives zero rather than a division error: a request whose stop token arrived
    before the first round has no per-round figure.
    """
    rounds = sum(e.get("rounds", 0) for e in events)
    emitted = sum(e.get("emitted", 0) for e in events)
    draft = sum(e.get("total_draft", 0) for e in events)
    # The Gemma4 assistant and MTP sidecar loops name it `total_accept`; the
    # cached spec loops name it `total_accept_count`.
    accept = sum(e.get("total_accept", e.get("total_accept_count", 0)) for e in events)
    draft_ms = sum(e.get("draft_ms", 0.0) for e in events)
    verify_ms = sum(e.get("verifier_ms", 0.0) for e in events)
    round_ms = sum(e.get("round_ms", 0.0) for e in events)

    def per_round(total):
        return total / rounds if rounds > 0 else 0.0

    for event in events:
        check_derived(event)

    lines = [
        f"events={len(events)}",
        f"rounds_total={rounds}",
        f"emitted_total={emitted}",
        f"draft_tokens_total={draft}",
        f"accept_tokens_total={accept}",
        f"accept_rate={accept / draft if draft > 0 else 0.0:.6f}",
        f"accepted_per_step={per_round(accept):.6f}",
        f"tokens_per_round={per_round(emitted):.6f}",
        f"draft_ms_per_round={per_round(draft_ms):.6f}",
        f"verify_ms_per_round={per_round(verify_ms):.6f}",
        f"loop_ms_per_round={per_round(round_ms - draft_ms - verify_ms):.6f}",
        f"block_size={one_value(events, 'block_size')}",
        f"decode_config={one_value(events, 'decode_config')}",
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
    parser.add_argument(
        "--expect-total",
        type=int,
        default=0,
        help=(
            "the exact number of done events the log must hold (0 = any). "
            "One per request served against this log, warmups included: a "
            "smaller number means a request produced no round-loop record and "
            "the events that remain do not line up with the runs the caller "
            "thinks it measured."
        ),
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

    if args.expect_total > 0 and len(events) != args.expect_total:
        print(
            f"spec_round_log: {args.log} holds {len(events)} round-loop 'done' "
            f"events, expected {args.expect_total}",
            file=sys.stderr,
        )
        return 5

    if args.last > 0:
        if len(events) < args.last:
            print(
                f"spec_round_log: {args.log} holds {len(events)} round-loop "
                f"'done' events, fewer than the {args.last} asked for",
                file=sys.stderr,
            )
            return 5
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
