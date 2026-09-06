#!/usr/bin/env python3
"""Read one rmlx run log back as a per-request table.

The published-protocol harness sends one chat request per sample and needs each
request's own numbers, not an aggregate over the pass. The two rings the server
publishes at `GET /metrics/cache` hold twenty entries, so neither survives a
128-sample dataset; the run log does. Every streaming request leaves exactly one
`generate_streaming: TTFT` event, every request the engine could time leaves
exactly one `generate: ITL stats` event, and every speculative request leaves
exactly one round-loop `done` event. All three are written in request order,
which is the order this reader zips them in.

The decode rate is the engine's own reading of the window from the first emitted
token to the last, prefill excluded, on both arms:

  plain         `1000 / mean_ms` off the ITL event — the mean gap between
                consecutive tokens, so this is `(n - 1) / (t_last - t_first)`.
  speculative   `decode_tps` off the round-loop `done` event.

The `done` line's contract — the `Some(x)` / `None` rendering, the required
counters, and the derived figures agreeing with them — has one implementation,
in `spec_round_log.py`, and this module calls it rather than restating it.

A request the engine could not time is refused, not dropped: a published mean
over "the samples that worked" is not a mean over the sample set.

Output (stdout): one JSON object.

    {"arm": "...", "requests": [{...}, ...], "sampling": {...} | null,
     "decode_config": "...", "block_size": n, "charged": false}

`sampling` is the temperature / top_p / top_k / min_p / seed the engine says it
resolved, read back rather than re-derived from the checkpoint's file — the
request sends no sampling field, so what actually ran is only knowable from the
engine. It is null for a greedy checkpoint, which resolves no sampler and
writes no such event.

`decode_config`, `block_size` and `charged` are present on the speculative arm
only. Each request carries `ttft_ms` and `decode_tps`, plus `step_count` on the
plain arm and the round loop's own counters and per-round figures on the
speculative one.

Exit codes: 0 — read; 2 — the log could not be read; 3 — an event carries
something this reader must not interpret; 4 — the log holds no event of a kind
the arm needs; 5 — the log holds a different number of events than the caller
served requests.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import spec_round_log  # noqa: E402

TTFT_MARKER = "generate_streaming: TTFT"
# Anchored, not a substring test: the speculative path writes
# "spec generate: ITL stats (M30)", which contains this one.
ITL_MESSAGE = "generate: ITL stats (M30)"
SAMPLER_MARKER = "generate: host categorical sampler active"
# The fields the engine says it resolved for a request. `seed` is on the list
# because the request sends none and the engine substitutes a fixed default, so
# three passes replay one RNG stream rather than sampling independently — a
# fact a published number has to carry.
SAMPLER_FIELDS = ("temperature", "top_p", "top_k", "min_p", "seed")

# The round loop's own figures, copied onto each request as it reported them.
# `spec_round_log.check_derived` has already refused any that contradict the
# counters beside them.
ROUND_FIELDS = (
    "rounds",
    "total_draft",
    "total_accept",
    "accept_rate",
    "accepted_per_step",
    "tokens_per_round",
    "draft_ms_per_round",
    "verify_ms_per_round",
    "loop_ms_per_round",
)


class LogError(Exception):
    """The log carries something this reader must not interpret."""


class MissingEventError(Exception):
    """The log holds no event of a kind the arm needs."""


class CountError(Exception):
    """The log holds a different number of events than requests were served."""


def events_matching(path, marker, exact=False):
    """The `fields` of every event whose message holds `marker`, in file order.

    `exact` compares the whole message, for a marker that is a substring of
    another event's.

    A log can be truncated mid-write by a killed server, so a line that is not
    JSON is skipped rather than losing the rest of the file.
    """
    found = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if marker not in line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            fields = record.get("fields")
            if not isinstance(fields, dict):
                continue
            message = fields.get("message", "")
            if message == marker if exact else marker in message:
                found.append(fields)
    return found


def sampler_config(path, expect_total, last):
    """The sampling the engine says it resolved, if it resolved any.

    The event is written per request and only when the sampler is active
    (`temperature > 0`), so a greedy checkpoint produces none — which is why the
    caller states how many it expects rather than this deciding. Every event
    must agree: three passes of one protocol are one sampling setup, and a
    request that got another is not a repetition of the others.
    """
    events = events_matching(path, SAMPLER_MARKER)
    if expect_total == 0:
        if events:
            raise LogError(
                f"the log holds {len(events)} sampler events on a run whose "
                "checkpoint states no sampling temperature: the engine resolved a "
                "sampler this run was not supposed to have"
            )
        return None
    events = take(events, "sampler", expect_total, last)
    resolved = {}
    for name in SAMPLER_FIELDS:
        values = {e.get(name) for e in events}
        if None in values:
            raise LogError(
                f"a sampler event carries no {name}: the sampling this run used "
                "cannot be read back, only guessed at from the checkpoint's file"
            )
        if len(values) > 1:
            raise LogError(
                f"the sampler events report {len(values)} values for {name} "
                f"({', '.join(sorted(str(v) for v in values))}); the requests of "
                "one pass did not share one sampling setup"
            )
        resolved[name] = values.pop()
    return resolved


def take(events, kind, expect_total, last):
    """The last `last` of `events`, once there are exactly `expect_total`."""
    if not events:
        raise MissingEventError(f"the log holds no {kind} event")
    if len(events) != expect_total:
        raise CountError(
            f"the log holds {len(events)} {kind} events, expected {expect_total}: "
            "one per request served against it, warmups included"
        )
    return events[-last:] if last > 0 else events


def ttft_of(fields):
    """The TTFT this event reports, in milliseconds."""
    value = fields.get("ttft_ms")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LogError(
            f"a TTFT event reports ttft_ms={value!r}, which is not a number"
        )
    if value < 0:
        raise LogError(f"a TTFT event reports ttft_ms={value!r}, which is not a time")
    return value


def itl_rate(fields):
    """The decode rate this ITL event reports, and the tokens behind it."""
    mean_ms = fields.get("mean_ms")
    step_count = fields.get("step_count")
    if not isinstance(mean_ms, (int, float)) or isinstance(mean_ms, bool) or mean_ms <= 0:
        raise LogError(
            f"an ITL event reports mean_ms={mean_ms!r}, which is no interval to "
            "read a rate from"
        )
    if not isinstance(step_count, int) or step_count < 2:
        raise LogError(
            f"an ITL event reports step_count={step_count!r}; a rate over the "
            "decode window needs at least two tokens"
        )
    return 1000.0 / mean_ms, step_count


def plain_requests(path, expect_total, last, sampler_events):
    """One row per request, for a server that was given no drafter."""
    intruders = spec_round_log.done_events(path)
    if intruders:
        raise LogError(
            f"the log holds {len(intruders)} speculative round-loop 'done' events "
            "on a run that was given no drafter: this log belongs to a different "
            "server, or the drafter reached it another way"
        )
    ttfts = take(events_matching(path, TTFT_MARKER), "TTFT", expect_total, last)
    itls = take(events_matching(path, ITL_MESSAGE, exact=True), "ITL stats", expect_total, last)
    rows = []
    for ttft, itl in zip(ttfts, itls):
        rate, step_count = itl_rate(itl)
        rows.append(
            {
                "ttft_ms": ttft_of(ttft),
                "decode_tps": rate,
                "step_count": step_count,
            }
        )
    return {
        "arm": "plain",
        "requests": rows,
        "sampling": sampler_config(path, sampler_events, last),
    }


def speculative_requests(path, expect_total, last, sampler_events):
    """One row per request, for a server that was given a drafter."""
    ttfts = take(events_matching(path, TTFT_MARKER), "TTFT", expect_total, last)
    events = spec_round_log.done_events(path)
    if not events:
        raise MissingEventError(
            "the log holds no speculative round-loop 'done' event"
        )
    events = take(events, "round-loop 'done'", expect_total, last)

    spec_round_log.require_counters(events)
    for event in events:
        spec_round_log.check_seed(event)
        spec_round_log.check_derived(event)

    rows = []
    for ttft, event in zip(ttfts, events):
        rate = spec_round_log.decode_tps(event)
        if rate is None:
            raise LogError(
                f"a round-loop 'done' event reports no measurable decode rate "
                f"(it emitted {event.get('emitted')!r} tokens); a published mean "
                "over the samples that happened to work is not a mean over the "
                "sample set"
            )
        row = {"ttft_ms": ttft_of(ttft), "decode_tps": rate}
        row.update({name: event[name] for name in ROUND_FIELDS})
        rows.append(row)

    return {
        "arm": "speculative",
        "requests": rows,
        "sampling": sampler_config(path, sampler_events, last),
        "decode_config": spec_round_log.one_value(events, "decode_config"),
        "block_size": spec_round_log.one_value(events, "block_size"),
        "charged": spec_round_log.charged_flag(events),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="an rmlx run log (<RMLX_HOME>/logs/*.jsonl)")
    parser.add_argument(
        "--arm",
        required=True,
        choices=("plain", "speculative"),
        help="whether the server this log belongs to was given a drafter",
    )
    parser.add_argument(
        "--expect-total",
        type=int,
        required=True,
        help="requests served against this log, warmups included",
    )
    parser.add_argument(
        "--last",
        type=int,
        default=0,
        help="keep only the last N rows (0 = all), dropping the warmups",
    )
    parser.add_argument(
        "--expect-sampler-events",
        type=int,
        required=True,
        help=(
            "sampler-resolution events the log must hold — the request count "
            "when the checkpoint states a temperature above zero, 0 when it is "
            "greedy and the engine writes none"
        ),
    )
    args = parser.parse_args()

    read = plain_requests if args.arm == "plain" else speculative_requests
    try:
        result = read(
            args.log, args.expect_total, args.last, args.expect_sampler_events
        )
    except OSError as exc:
        print(f"published_run_log: cannot read {args.log}: {exc}", file=sys.stderr)
        return 2
    except MissingEventError as exc:
        print(f"published_run_log: {args.log}: {exc}", file=sys.stderr)
        return 4
    except CountError as exc:
        print(f"published_run_log: {args.log}: {exc}", file=sys.stderr)
        return 5
    except (LogError, spec_round_log.SpecLogError) as exc:
        print(f"published_run_log: {args.log}: {exc}", file=sys.stderr)
        return 3

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
