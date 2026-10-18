#!/usr/bin/env python3
"""Assemble one pass's fixed-length-prompt run.

The published protocol's second figure is autoregressive output speed, input
speed and resident memory on one prompt of a stated token length. This is one
run of it: the engine's own reading of the decode window, the client's reading
of the same window, the prompt count the server gave the body, and the peak the
process-memory gauges reached while the request was in flight.

INPUT SPEED HAS TWO DEFINITIONS, on a cold prompt cache. The harness sends this
request once per pass, before any cell request and after a warmup whose prompt
shares no prefix with it, so the count in the denominator is a prefill that
actually happened. `prefill_tps` is prompt tokens over TTFT: what a client sees.
`prefill_engine_tps` is the tokens the prefill ran the model over, over the
prefill time the engine timed itself, off its `generate: prefill` event: what
the engine reports as its own prefill rate. The engine's event also says how
many prompt tokens a cache supplied, so a prompt that was not cold is refused
instead of published under a name that says it was. A run whose TTFT or prefill
time is not a positive duration has no input speed and is refused rather than
divided by.

RESIDENT MEMORY IS A SAMPLED PEAK. `phys_footprint` is a gauge, so the largest
value seen at the poll interval is a lower bound on the true peak. The interval
travels with the figure. A run with no sample at all is refused: a peak over
nothing is not a small peak.

THE METAL PEAK IS THE ENGINE'S OWN, and it is a different kind of figure: the
allocator's high-water mark since the server started, read when the request
ended (`row["metal_peak_alloc_mb"]`, MiB). It is exact, not sampled, and it
covers the whole process life — the warmup before this request, not only this
request.

Exit codes: 0 — assembled; 1 — the readings do not describe one run of the
fitted prompt; 2 — an input could not be read.
"""

import argparse
import json
import pathlib
import sys


class RunError(Exception):
    """The readings do not describe one run of the fitted prompt."""


def read_kv(path):
    values = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key] = value
    return values


def number(values, key, cast=float):
    if key not in values or values[key] == "":
        raise RunError(f"the client reading carries no {key}")
    try:
        return cast(values[key])
    except ValueError as exc:
        raise RunError(f"the client read {key}={values[key]!r}, not a number") from exc


def peak_memory(path, what="the fixed-prompt request"):
    """`(phys_bytes, rss_bytes, samples)` over the poll file."""
    peaks = {"phys": 0, "rss": 0}
    samples = 0
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2 or parts[0] not in peaks:
            continue
        try:
            value = int(float(parts[1]))
        except ValueError:
            continue
        peaks[parts[0]] = max(peaks[parts[0]], value)
        samples += 1
    if not samples:
        raise RunError(
            f"the server published no process-memory gauge while {what} was in "
            "flight, so there is no resident figure for it. A peak over no sample "
            "is not a small peak"
        )
    for name, value in peaks.items():
        if value <= 0:
            raise RunError(
                f"every {name} sample read {value}, which is not a resident "
                "figure; the gauge was published but carries nothing"
            )
    return peaks["phys"], peaks["rss"], samples


def cold_request(row, client, what, cross_check_pct):
    """The readings of one request on a cold prompt cache, checked against each other.

    `row` is the engine's account of the request (`published_run_log.py`) and
    `client` the client's. The prompt count is the server's, the engine's prefill
    event must be of the same request, and the engine's decode rate must agree with
    the client's reading of the same window. `what` names the request in a refusal.
    """
    prompt_tokens = number(client, "prompt_tokens", int)

    ttft_ms = row["ttft_ms"]
    if not isinstance(ttft_ms, (int, float)) or ttft_ms <= 0:
        raise RunError(
            f"the engine reported ttft_ms={ttft_ms!r}, which is not a prefill "
            "duration; input speed is prompt tokens over it"
        )

    for name in ("prompt_tokens", "reused_tokens", "forwarded_tokens", "prefill_ms"):
        if name not in row:
            raise RunError(f"the engine's prefill event for this request carries no {name}")
    if row["prompt_tokens"] != prompt_tokens:
        raise RunError(
            f"the engine's prefill event counts {row['prompt_tokens']} prompt tokens "
            f"where the server counted {prompt_tokens} for this request: the event "
            "belongs to another request"
        )
    if row["reused_tokens"] != 0:
        raise RunError(
            f"the engine took {row['reused_tokens']} of {what}'s tokens "
            "from a prompt cache, so its input speed is not over a cold prompt"
        )
    if row["forwarded_tokens"] != prompt_tokens:
        raise RunError(
            f"the engine ran the model over {row['forwarded_tokens']} of the "
            f"{prompt_tokens} prompt tokens, so its prefill is not of the whole prompt"
        )
    prefill_ms = row["prefill_ms"]
    if not isinstance(prefill_ms, (int, float)) or prefill_ms <= 0:
        raise RunError(
            f"the engine reported prefill_ms={prefill_ms!r}, which is not a prefill "
            "duration; its input speed is the forwarded tokens over it"
        )

    engine_tps = row["decode_tps"]
    client_tps = number(client, "decode_tps")
    if engine_tps <= 0:
        raise RunError(
            f"the engine reported decode_tps={engine_tps!r} for {what}, "
            "which is not a rate"
        )
    off = abs(client_tps - engine_tps) / engine_tps * 100.0
    if off > cross_check_pct:
        raise RunError(
            f"the engine read {engine_tps:.3f} tok/s over {what}'s decode "
            f"window and the client read {client_tps:.3f} over the same window, "
            f"{off:.1f}% apart and past the {cross_check_pct:.0f}% band two "
            "readings of one window are allowed"
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": number(client, "completion_tokens", int),
        "ttft_ms": ttft_ms,
        "decode_tps": engine_tps,
        "client_decode_tps": client_tps,
        "prefill_tps": prompt_tokens / (ttft_ms / 1000.0),
        "prefill_ms": prefill_ms,
        "prefill_engine_tps": row["forwarded_tokens"] / (prefill_ms / 1000.0),
        "metal_peak_alloc_mb": row["metal_peak_alloc_mb"],
    }


def assemble(args):
    fit = json.loads(pathlib.Path(args.fit).read_text(encoding="utf-8"))
    engine = json.loads(pathlib.Path(args.engine).read_text(encoding="utf-8"))
    rows = engine["requests"]
    if not rows:
        raise RunError("the engine reported no request for this pass")
    # The fixed-prompt request is sent between the warmups and the cells, and
    # the caller keeps one row more than there are cells, so it is the first.
    row = rows[0]
    client = read_kv(args.client)

    prompt_tokens = number(client, "prompt_tokens", int)
    if prompt_tokens != fit["prompt_tokens"]:
        raise RunError(
            f"the server counted this request's prompt at {prompt_tokens} tokens "
            f"where the fit measured {fit['prompt_tokens']} for the same body; "
            "the two counts are of different prompts"
        )
    figures = cold_request(row, client, "the fixed prompt", args.cross_check_pct)

    phys, rss, samples = peak_memory(args.memory)
    return {
        "pass": args.pass_number,
        "target_tokens": fit["target_tokens"],
        "max_tokens": fit["max_tokens"],
        "body_sha256": fit["body_sha256"],
        "corpus": fit["corpus"],
        "corpus_sha256": fit["corpus_sha256"],
        # The body travels with the run. A resident figure recorded against a
        # prompt nobody can reproduce is a figure attributed to nothing, and the
        # fitted body is not checked in — it belongs to this tokenizer.
        "messages": fit["messages"],
        "words": fit["words"],
        "filler_word": fit["filler_word"],
        "filler_reps": fit["filler_reps"],
        **figures,
        "phys_footprint_bytes": phys,
        "rss_bytes": rss,
        "memory_samples": samples,
        "memory_poll_ms": args.memory_poll_ms,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, help="published_run_log.py output")
    ap.add_argument("--client", required=True, help="the client-side key=value block")
    ap.add_argument("--memory", required=True, help="the memory poll file")
    ap.add_argument("--fit", required=True, help="published_fixed_prompt.py record")
    ap.add_argument("--pass-number", type=int, required=True, dest="pass_number")
    ap.add_argument("--memory-poll-ms", type=int, required=True)
    ap.add_argument("--cross-check-pct", type=float, default=10.0)
    args = ap.parse_args()

    try:
        block = assemble(args)
    except RunError as exc:
        print(f"published_fixed_run: pass {args.pass_number}: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"published_fixed_run: cannot read an input: {exc}", file=sys.stderr)
        return 2

    json.dump(block, sys.stdout)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
