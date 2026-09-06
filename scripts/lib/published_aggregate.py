#!/usr/bin/env python3
"""Assemble and summarise a published-protocol speculative-decoding run.

Two subcommands, one row schema:

  pass    zip one pass's client-side blocks with the engine's per-request rows
          and emit that pass as JSON. Both readings of each request's decode
          window are present here, so this is where they are required to agree.
  report  read three pass files and emit the per-dataset and macro-average
          means with their run-to-run range, plus the table a human reads.
          `--fixed` adds the fixed-length-prompt block: three runs of one
          prompt, whose rates are refused on the same band and whose resident
          figures are peaks — reported as the maximum over the three runs,
          because a mean of three peaks is not a peak of anything.

THE MACRO AVERAGE IS OVER DATASETS, AT ONE OUTPUT-TOKEN BUDGET.

The protocol macro-averages three datasets. A dataset measured at a second
budget is a second column beside that headline, not a fourth dataset: folding
it in would give that dataset twice the weight of the others and the headline
would no longer be the figure a third party's number can sit next to. So the
macro is the mean of one cell per dataset — the cell at `--macro-max-tokens` —
and a dataset with no cell there, or two, is refused rather than averaged over.
Pooling the samples instead would weight the datasets by size; they do not have
the same size, and nothing in the number would say so.

WHAT THE RANGE BOUNDS, AND WHAT IT DOES NOT.

The range of a mean is `(max - min) / mean` over the three passes, as a
percentage. A dataset mean whose range exceeds `--range-pct` is not a clean
mean: the three passes measured three different things and averaging them
publishes a number none of them produced. Such a cell is reported as unstable,
its mean is withheld from the mean column, and `report` exits 3.

WHAT THE RANGE IS A RANGE OF.

The request sends no seed, so the engine substitutes its fixed default and
seeds one RNG per request from it. Three passes of one sample are therefore the
same generation replayed, and the run-to-run range is a reading of machine
variance with the sampling held still — which is the tighter estimator, and the
one that makes a 5% band a statement about measurement stability. That claim is
checked rather than asserted: `divergent_samples` counts the samples that did
not generate the same length in all three passes, and is reported, never
refused.

That bound is over PASS MEANS, each already a mean over 80-128 samples, so
per-sample noise has divided by sqrt(n) before it is seen. A run where one
sample is fast in pass 1 and slow in pass 2 while another does the reverse has
a pass-mean range of zero, and so does a thermal ramp that slows the second half
of every pass identically. `sample_range_pct_max` — the widest across-pass range
of any single sample in the cell — is reported for exactly that reason and is
NOT a refusal: at a sampled temperature two passes of one prompt generate
different text of different length, so a wide per-sample range is ordinary.

The macro has no refusal of its own. Its range is bounded by the widest dataset
range (`macro_p` is their mean, so `max - min` cannot exceed the mean of their
spans), so a macro test with the same band could never fire while the datasets
passed. Its mean is instead withheld whenever any dataset's is, because it is
derived from them and it is the number a reader copies.

Exit codes: 0 — every dataset mean is clean; 1 — an input could not be read, or
the inputs do not describe one measurement; 3 — the run is readable and at least
one dataset mean is refused for its range.
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
        if engine_tps <= 0:
            raise InputError(
                f"{where}: the engine reported decode_tps={engine_tps!r}, which is "
                "not a rate; there is nothing for the client reading to be checked "
                "against"
            )
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
            # `completion_tokens`, not `tokens`: the reader falls back to a count
            # of content chunks for the second when the usage chunk carried no
            # count, and that is a client derivation under an engine field's
            # name. Content chunks miss every token whose visible piece is empty,
            # so it reads low with nothing saying so.
            "completion_tokens": require_number(client, "completion_tokens", where, int),
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


def divergent_samples(per_pass):
    """How many samples in this cell did not generate the same length twice.

    The request sends no seed, so the engine substitutes its fixed default and
    seeds one RNG per request from it. Three passes of one sample are therefore
    meant to be the same generation replayed, which is what makes the
    run-to-run range a reading of machine variance rather than of sampling.
    This counts the samples where that did not hold — a float tie flipped, and
    the passes generated different text of different length. It is a measured
    check on the claim, reported and never refused: the claim is about what the
    range means, and a reader has to be told when it means something else.
    """
    by_sample = {}
    for rows in per_pass:
        for row in rows:
            by_sample.setdefault(row["sample_id"], set()).add(row["completion_tokens"])
    return sum(1 for lengths in by_sample.values() if len(lengths) > 1)


def widest_sample_range(per_pass):
    """The widest across-pass range of any one sample in this cell, as a percent.

    What the pass-mean range cannot see: two samples moving in opposite
    directions between passes, or a within-pass ramp that repeats identically.
    Reported, never refused — at a sampled temperature one prompt generates
    different text of different length on each pass.
    """
    by_sample = {}
    for rows in per_pass:
        for row in rows:
            by_sample.setdefault(row["sample_id"], []).append(row["decode_tps"])
    widest = 0.0
    for rates in by_sample.values():
        mean = statistics.fmean(rates)
        if mean > 0:
            widest = max(widest, (max(rates) - min(rates)) / mean * 100.0)
    return widest


def macro_cell_per_dataset(cells, macro_max_tokens):
    """One cell name per dataset — the one at the headline output budget.

    A dataset with none there would silently drop out of the macro; a dataset
    with two would enter it twice. Both are a mis-specified cell table, and the
    macro that came out of either is not the figure it is labelled as.
    """
    by_dataset = {}
    for name, entry in cells.items():
        if entry["max_tokens"] == macro_max_tokens:
            by_dataset.setdefault(entry["dataset"], []).append(name)
    for dataset in sorted({e["dataset"] for e in cells.values()}):
        found = by_dataset.get(dataset, [])
        if len(found) != 1:
            raise InputError(
                f"dataset {dataset} has {len(found)} cells at the macro budget of "
                f"{macro_max_tokens} output tokens ({', '.join(sorted(found)) or 'none'}); "
                "the macro is one cell per dataset, so it can neither drop this "
                "dataset nor count it twice"
            )
    return [found[0] for found in (by_dataset[d] for d in sorted(by_dataset))]


# The fixed-prompt block's rate figures — the two the protocol publishes as
# scores, and so the two the range band refuses. `ttft_ms` is what `prefill_tps`
# is derived from, so refusing the rate already covers it.
FIXED_RATES = ("decode_tps", "prefill_tps")
# Peaks, not scores. A peak over three runs is their maximum; a mean of three
# peaks is not a peak of anything, and a range band over a resident figure would
# refuse an allocator's ordinary behaviour.
FIXED_PEAKS = ("phys_footprint_bytes", "rss_bytes")
FIXED_REPORTED = ("ttft_ms", "completion_tokens")


def fixed_block(paths, range_pct):
    """The fixed-length-prompt block: three runs of one prompt, summarised.

    The three runs must have measured one prompt, so the body's content address
    and the token count the server gave it have to agree across them, and that
    count has to be the target the protocol names. A block assembled out of
    three different prompts would still produce a mean.
    """
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    if len(runs) != 3:
        raise InputError(
            f"the fixed-prompt block was given {len(runs)} runs; the protocol "
            "reports the mean of three consecutive runs"
        )
    for field in ("body_sha256", "prompt_tokens", "target_tokens", "max_tokens"):
        seen = {r[field] for r in runs}
        if len(seen) != 1:
            raise InputError(
                f"the fixed-prompt runs disagree on {field} ({sorted(seen)}); "
                "they did not measure one prompt"
            )
    first = runs[0]
    if first["prompt_tokens"] != first["target_tokens"]:
        raise InputError(
            f"the fixed prompt tokenized to {first['prompt_tokens']} tokens where "
            f"the protocol names {first['target_tokens']}; a figure published "
            "under the target's name would be a figure measured on another prompt"
        )

    block = {
        "prompt_tokens": first["prompt_tokens"],
        "target_tokens": first["target_tokens"],
        "max_tokens": first["max_tokens"],
        "body_sha256": first["body_sha256"],
        "corpus": first["corpus"],
        "corpus_sha256": first["corpus_sha256"],
        "memory_poll_ms": first["memory_poll_ms"],
    }
    for name in FIXED_RATES:
        block[name] = summarise([r[name] for r in runs], range_pct)
    for name in FIXED_REPORTED:
        entry = summarise([r[name] for r in runs], range_pct)
        # Reported with its spread, never refused: it is not a published score.
        entry.pop("stable")
        block[name] = entry
    for name in FIXED_PEAKS:
        values = [r[name] for r in runs]
        block[name] = {"run_values": values, "max": max(values), "min": min(values)}
    return block


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
        entry["sample_range_pct_max"] = widest_sample_range(per_pass)
        entry["divergent_samples"] = divergent_samples(per_pass)
        for figure in ROUND_FIGURES:
            value = mean_of(flat, figure)
            if value is not None:
                entry[figure] = value
        cells[name] = entry

    macro_cells = macro_cell_per_dataset(cells, args.macro_max_tokens)
    macro = summarise(
        [
            statistics.fmean(
                [
                    statistics.fmean([r["decode_tps"] for r in idx[name]])
                    for name in macro_cells
                ]
            )
            for idx in indices
        ],
        args.range_pct,
    )
    macro["cells"] = macro_cells
    macro["max_tokens"] = args.macro_max_tokens
    # Derived from the dataset means, so it is withheld whenever one of them is.
    # Its own range cannot exceed the widest of theirs, so testing it against the
    # same band would be a refusal that could not fire.
    macro["stable"] = all(cells[name]["stable"] for name in macro_cells)

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

    stable = all(c["stable"] for c in cells.values())
    if args.fixed:
        block = fixed_block(args.fixed, args.range_pct)
        result["fixed_prompt"] = block
        stable = stable and all(block[name]["stable"] for name in FIXED_RATES)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
            handle.write("\n")

    print_table(result, args.range_pct)
    return 0 if stable else 3


def mean_column(entry):
    """The mean, or the refusal that stands in for it."""
    return f"{entry['mean']:.2f}" if entry["stable"] else "UNSTABLE"


def print_table(result, range_pct):
    macro = result["macro"]
    fmt = "%-16s  %7s  %9s  %8s  %8s  %9s  %8s  %s\n"
    print("=" * 112)
    print(f"  PUBLISHED-PROTOCOL DECODE RATE — arm={result['arm']}")
    print("=" * 112)
    sys.stdout.write(
        fmt
        % ("cell", "samples", "max_tok", "mean t/s", "range %", "worst s %",
           "TTFT ms", "pass means")
    )
    sys.stdout.write(
        fmt % ("-" * 16, "-" * 7, "-" * 9, "-" * 8, "-" * 8, "-" * 9, "-" * 8, "-" * 30)
    )
    for name, entry in result["cells"].items():
        sys.stdout.write(
            fmt
            % (
                name,
                entry["samples"],
                entry["max_tokens"],
                mean_column(entry),
                f"{entry['range_pct']:.2f}",
                f"{entry['sample_range_pct_max']:.1f}",
                f"{entry['ttft_ms_mean']:.1f}",
                " ".join(f"{v:.2f}" for v in entry["pass_means"]),
            )
        )
    sys.stdout.write(
        fmt
        % (
            "MACRO",
            len(macro["cells"]),
            macro["max_tokens"],
            mean_column(macro),
            f"{macro['range_pct']:.2f}",
            "-",
            "-",
            " ".join(f"{v:.2f}" for v in macro["pass_means"]),
        )
    )
    print("=" * 112)
    print(
        f"  MACRO covers {', '.join(macro['cells'])} — one cell per dataset at "
        f"{macro['max_tokens']} output tokens. Any other cell is a column beside it."
    )
    print(
        "  range % is over the three pass means; worst s % is the widest "
        "across-pass range of any one sample, reported and never refused."
    )

    diverged = sum(c["divergent_samples"] for c in result["cells"].values())
    total = sum(c["samples"] for c in result["cells"].values())
    if diverged:
        print(
            f"  {diverged} of {total} samples generated a different length across "
            "the passes, so for those the range carries sampling variance too."
        )
    else:
        print(
            f"  all {total} samples generated the same length in all three passes, "
            "so the range is machine variance and not sampling variance."
        )

    block = result.get("fixed_prompt")
    if block:
        print()
        print(
            f"  FIXED PROMPT — {block['prompt_tokens']} tokens, plain decode, "
            f"{block['max_tokens']} output budget, three runs"
        )
        for name, unit in (("decode_tps", "tok/s out"), ("prefill_tps", "tok/s in")):
            entry = block[name]
            print(
                f"    {unit:<10} {mean_column(entry):>10}   range "
                f"{entry['range_pct']:.2f}%   runs "
                + " ".join(f"{v:.2f}" for v in entry["pass_means"])
            )
        for name in FIXED_PEAKS:
            peak = block[name]
            print(
                f"    {name:<10} {peak['max'] / 1e6:>10.1f} MB peak over the three "
                "runs, sampled every " + f"{block['memory_poll_ms']} ms"
            )

    refused = [n for n, c in result["cells"].items() if not c["stable"]]
    refused += [
        f"fixed_prompt/{n}" for n in FIXED_RATES if block and not block[n]["stable"]
    ]
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
        if not macro["stable"]:
            print(
                "  MACRO is withheld too — it is the mean of the dataset means and "
                "one of them was refused."
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
    r.add_argument(
        "--macro-max-tokens",
        type=int,
        required=True,
        help="the output-token budget the macro average is taken at",
    )
    r.add_argument("--json", default=None, help="write the full result here")
    r.add_argument(
        "--fixed",
        nargs="+",
        default=None,
        metavar="RUN.json",
        help="the three fixed-length-prompt runs, one per pass",
    )
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
