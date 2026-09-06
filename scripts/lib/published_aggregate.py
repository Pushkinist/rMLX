#!/usr/bin/env python3
"""Assemble and summarise a published-protocol speculative-decoding run.

Two subcommands, one row schema:

  pass    zip one pass's client-side blocks with the engine's per-request rows
          and emit that pass as JSON. Both readings of each request's decode
          window are present here, so this is where they are required to agree.
  report  read three pass files and emit the per-dataset and macro-average
          means with their run-to-run range, plus the table a human reads.

The macro average is the mean of the per-dataset means, not a mean over the
pooled samples: the datasets have different sizes and pooling would weight
MT-Bench's 80 samples below HumanEval's 128 without saying so.

The range of a mean is `(max - min) / mean` over the three passes, as a
percentage. A mean whose range exceeds `--range-pct` is not a clean mean: the
three passes measured three different things and averaging them publishes a
number none of them produced. Such a cell is reported as unstable, its mean is
withheld from the mean column, and `report` exits 3.

Exit codes: 0 — every cell is stable; 1 — an input could not be read or the
passes do not describe the same sample set; 3 — the run is readable and at
least one cell's mean is refused for its range.
"""

import argparse
import json
import statistics
import sys

SCHEMA_VERSION = 1

# Per-round figures the speculative arm carries through to the report. The
# plain arm has no round loop and carries none — a zero there would rank as a
# measured one.
ROUND_FIGURES = ("tokens_per_round", "accepted_per_step", "accept_rate")


class InputError(Exception):
    """An input could not be read, or the inputs do not describe one run."""


# ── pass ──────────────────────────────────────────────────────────────────────


def read_kv(path):
    """The `key=value` block a client-side reader wrote for one request."""
    values = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            key, sep, value = line.rstrip("\n").partition("=")
            if sep:
                values[key] = value
    return values


def require_number(values, key, where, cast=float):
    if key not in values or values[key] == "":
        raise InputError(f"{where} carries no {key}")
    try:
        return cast(values[key])
    except ValueError as exc:
        raise InputError(f"{where} has {key}={values[key]!r}, not a number") from exc


def cmd_pass(args):
    with open(args.engine, encoding="utf-8") as handle:
        engine = json.load(handle)
    rows_in = engine["requests"]

    # A charged round loop forces an evaluation at every phase boundary,
    # draining a pipeline it otherwise keeps full. Its rate describes a slower,
    # differently scheduled engine, and it is reachable from an ambient RUST_LOG
    # that the harness's own filter does not clear.
    if engine.get("charged"):
        raise InputError(
            f"pass {args.pass_number}: the round loop reported charged=true — it "
            "forced an evaluation at every phase boundary and its decode rate "
            "describes a differently scheduled engine. Unset RUST_LOG and re-run"
        )

    index = []
    with open(args.index, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            index.append(line.split("\t"))

    if len(index) != len(rows_in):
        raise InputError(
            f"pass {args.pass_number}: {len(index)} requests were sent and the "
            f"engine reported {len(rows_in)}; the two are zipped by order, so a "
            "mismatch means no request's numbers can be attributed"
        )

    samples = []
    for entry, engine_row in zip(index, rows_in):
        cell, dataset, max_tokens, sample_id, body_sha256, kv_path = entry
        where = f"pass {args.pass_number} {cell}/{sample_id}"
        client = read_kv(kv_path)
        client_tps = require_number(client, "decode_tps", where)
        engine_tps = engine_row["decode_tps"]
        off = abs(client_tps - engine_tps) / engine_tps * 100.0
        if off > args.cross_check_pct:
            raise InputError(
                f"{where}: the engine read {engine_tps:.3f} tok/s over the decode "
                f"window and the client read {client_tps:.3f} over the same "
                f"window, {off:.1f}% apart and past the {args.cross_check_pct:.0f}% "
                "band two readings of one window are allowed"
            )
        row = {
            "pass": args.pass_number,
            "cell": cell,
            "dataset": dataset,
            "sample_id": sample_id,
            "body_sha256": body_sha256,
            "max_tokens": int(max_tokens),
            "prompt_tokens": require_number(client, "prompt_tokens", where, int),
            "completion_tokens": require_number(client, "tokens", where, int),
            "ttft_ms": engine_row["ttft_ms"],
            "decode_tps": engine_tps,
            "client_decode_tps": client_tps,
        }
        for name in ROUND_FIGURES:
            if name in engine_row:
                row[name] = engine_row[name]
        samples.append(row)

    out = {"pass": args.pass_number, "arm": engine["arm"], "samples": samples}
    for name in ("decode_config", "block_size", "charged"):
        if name in engine:
            out[name] = engine[name]
    json.dump(out, sys.stdout)
    print()
    return 0


# ── report ────────────────────────────────────────────────────────────────────


def cell_index(pass_obj):
    """`{cell: [rows]}` for one pass, in first-seen cell order."""
    cells = {}
    for row in pass_obj["samples"]:
        cells.setdefault(row["cell"], []).append(row)
    return cells


def check_same_sample_set(passes):
    """Every pass must have measured the same samples in the same cells."""
    indices = [cell_index(p) for p in passes]
    first = indices[0]
    for name in first:
        if not first[name]:
            raise InputError(f"cell {name} holds no sample; there is no mean of none")
    for other, pass_obj in zip(indices[1:], passes[1:]):
        if set(other) != set(first):
            raise InputError(
                f"pass {pass_obj['pass']} measured cells "
                f"{sorted(other)} where pass {passes[0]['pass']} measured "
                f"{sorted(first)}; a mean over them is a mean over two sample sets"
            )
        for name in first:
            want = sorted(r["sample_id"] for r in first[name])
            got = sorted(r["sample_id"] for r in other[name])
            if got != want:
                raise InputError(
                    f"cell {name}: pass {pass_obj['pass']} measured "
                    f"{len(got)} samples against pass {passes[0]['pass']}'s "
                    f"{len(want)}, and their ids differ; the passes are not "
                    "repetitions of one measurement"
                )
    return indices


def spread(values):
    """Mean of `values` and the spread across them, as a percent of the mean."""
    mean = statistics.fmean(values)
    if mean == 0:
        raise InputError("a pass mean of zero has no range to express as a percent")
    return mean, (max(values) - min(values)) / mean * 100.0


def summarise(values, range_pct):
    mean, rng = spread(values)
    return {
        "pass_means": values,
        "mean": mean,
        "range_pct": rng,
        "stable": rng <= range_pct,
    }


def mean_of(rows, key):
    present = [r[key] for r in rows if key in r]
    return statistics.fmean(present) if present else None


def cmd_report(args):
    passes = []
    for path in args.passes:
        with open(path, encoding="utf-8") as handle:
            passes.append(json.load(handle))
    if len(passes) != 3:
        raise InputError(
            f"{len(passes)} pass files were given; the protocol reports the mean "
            "of three consecutive runs and a mean of any other count is a "
            "different figure"
        )
    numbers = [p["pass"] for p in passes]
    if sorted(numbers) != [1, 2, 3]:
        raise InputError(f"the pass files are numbered {numbers}, not 1, 2 and 3")
    arms = {p["arm"] for p in passes}
    if len(arms) != 1:
        raise InputError(f"the passes ran different arms ({sorted(arms)})")

    indices = check_same_sample_set(passes)
    cell_names = list(indices[0])

    cells = {}
    for name in cell_names:
        per_pass = [idx[name] for idx in indices]
        entry = summarise(
            [statistics.fmean([r["decode_tps"] for r in rows]) for rows in per_pass],
            args.range_pct,
        )
        entry["dataset"] = per_pass[0][0]["dataset"]
        entry["max_tokens"] = per_pass[0][0]["max_tokens"]
        entry["samples"] = len(per_pass[0])
        flat = [r for rows in per_pass for r in rows]
        entry["ttft_ms_mean"] = mean_of(flat, "ttft_ms")
        entry["prompt_tokens_mean"] = mean_of(flat, "prompt_tokens")
        entry["completion_tokens_mean"] = mean_of(flat, "completion_tokens")
        for figure in ROUND_FIGURES:
            value = mean_of(flat, figure)
            if value is not None:
                entry[figure] = value
        cells[name] = entry

    macro = summarise(
        [
            statistics.fmean(
                [
                    statistics.fmean([r["decode_tps"] for r in idx[name]])
                    for name in cell_names
                ]
            )
            for idx in indices
        ],
        args.range_pct,
    )

    meta = {}
    if args.meta:
        with open(args.meta, encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("arm") not in (None, passes[0]["arm"]):
            raise InputError(
                f"the run metadata says arm={meta['arm']!r} where the passes ran "
                f"{passes[0]['arm']!r}"
            )

    result = {
        "schema_version": SCHEMA_VERSION,
        **meta,
        "arm": passes[0]["arm"],
        "range_refusal_pct": args.range_pct,
        "cells": cells,
        "macro": macro,
        "samples": [r for p in passes for r in p["samples"]],
    }
    for name in ("decode_config", "block_size", "charged"):
        if name in passes[0]:
            result[name] = passes[0][name]

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")

    print_table(result, args.range_pct)
    return 0 if all(c["stable"] for c in cells.values()) and macro["stable"] else 3


def mean_column(entry):
    """The mean, or the refusal that stands in for it."""
    return f"{entry['mean']:.2f}" if entry["stable"] else "UNSTABLE"


def print_table(result, range_pct):
    fmt = "%-16s  %7s  %9s  %8s  %8s  %10s  %s\n"
    print("=" * 96)
    print(f"  PUBLISHED-PROTOCOL DECODE RATE — arm={result['arm']}")
    print("=" * 96)
    sys.stdout.write(
        fmt % ("cell", "samples", "max_tok", "mean t/s", "range %", "TTFT ms", "pass means")
    )
    sys.stdout.write(fmt % ("-" * 16, "-" * 7, "-" * 9, "-" * 8, "-" * 8, "-" * 10, "-" * 30))
    for name, entry in result["cells"].items():
        sys.stdout.write(
            fmt
            % (
                name,
                entry["samples"],
                entry["max_tokens"],
                mean_column(entry),
                f"{entry['range_pct']:.2f}",
                f"{entry['ttft_ms_mean']:.1f}",
                " ".join(f"{v:.2f}" for v in entry["pass_means"]),
            )
        )
    macro = result["macro"]
    sys.stdout.write(
        fmt
        % (
            "MACRO",
            "-",
            "-",
            mean_column(macro),
            f"{macro['range_pct']:.2f}",
            "-",
            " ".join(f"{v:.2f}" for v in macro["pass_means"]),
        )
    )
    print("=" * 96)

    refused = [n for n, c in result["cells"].items() if not c["stable"]]
    if not macro["stable"]:
        refused.append("MACRO")
    if refused:
        print()
        print(
            f"RANGE REFUSAL: {', '.join(refused)} — the three passes disagree by "
            f"more than {range_pct:.0f}% of the mean."
        )
        print(
            "  The mean is withheld: averaging three passes that measured "
            "different things publishes a number none of them produced."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pass", help="assemble one pass from client and engine reads")
    p.add_argument("--index", required=True, help="one tab-separated row per request")
    p.add_argument("--engine", required=True, help="published_run_log.py output")
    p.add_argument("--pass-number", type=int, required=True, dest="pass_number")
    p.add_argument("--cross-check-pct", type=float, default=10.0)
    p.set_defaults(func=cmd_pass)

    r = sub.add_parser("report", help="summarise three passes")
    r.add_argument("passes", nargs="+", help="the pass JSON files")
    r.add_argument("--range-pct", type=float, default=5.0)
    r.add_argument("--json", default=None, help="write the full result here")
    r.add_argument(
        "--meta",
        default=None,
        help="a JSON object describing the run, merged into the result",
    )
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    try:
        return args.func(args)
    except InputError as exc:
        print(f"published_aggregate: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"published_aggregate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
