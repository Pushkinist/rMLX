#!/usr/bin/env python3
"""Re-query a recorded published-protocol run from `runs.db` and compare.

`ingest/published_ingest.py` records the measured things and deliberately not
their means: the per-dataset and macro averages are derived, and a stored
derivation can disagree with what it was derived from. So the headline of a run
lives in its result file, and the rows live in the database. This reads the rows
back by the run's label and start time with `rmlx metrics query`, rebuilds the
means from them, and requires the two to agree. A mismatch means the rows are
not the measurement the result describes: a record that was only partly
ingested, one ingested twice, or a number that moved on its way into the store.

What is compared:

  per cell      the mean `decode_tps_warm` and `ttft_warm_ms` over the cell's
                samples, and the number of samples that have a row
  macro         the mean of the cell means the result names for the macro
  fixed prompt  its `decode_tps_warm`, `prefill_tps`, `ttft_warm_ms` and
                `metal_peak_alloc_mb` rows, one of each

The rows are found by `label=<label>;` at the start of `notes` and by the run's
`ts_utc`. The re-queried figures are written to `--out` (default
`db_summary.json` beside the result), so a later write-up cites what the
database returned.

Exit codes: 0 — the rows reproduce the result; 1 — they do not; 2 — the result
or the query could not be read.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

# The label names a directory and enters a LIKE pattern, so `_` and `%` are not
# in it.
LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
METRICS = ("decode_tps_warm", "ttft_warm_ms", "prefill_tps", "metal_peak_alloc_mb")
# Values travel through JSON, a REAL column and a printed float. Each leg is
# exact for a double; the allowance is for the order a mean is summed in.
REL_TOL = 1e-9
FIXED_PREFIX = "published/fixed_"


class Unreadable(Exception):
    """The result or the query could not be read."""


def query_rows(result: dict, rmlx_bin: str, db: str | None) -> list[dict]:
    label, ts = result.get("label"), result.get("ts_utc")
    if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label):
        raise Unreadable(f"the result carries no valid label (got {label!r})")
    if not isinstance(ts, str) or not TS_PATTERN.fullmatch(ts):
        raise Unreadable(f"the result carries no valid ts_utc (got {ts!r})")
    metrics = ", ".join(f"'{m}'" for m in METRICS)
    sql = (
        "SELECT p.name, o.metric, o.max_tokens, o.value "
        "FROM observations o JOIN prompts p ON p.id = o.prompt_id "
        f"WHERE o.notes LIKE 'label={label};%' AND o.ts_utc = '{ts}' "
        f"AND o.metric IN ({metrics})"
    )
    cmd = [rmlx_bin, "metrics"] + (["--db", db] if db else []) + ["query", sql]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise Unreadable(
            f"`rmlx metrics query` exited {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if not lines or lines[0].split("\t") != ["name", "metric", "max_tokens", "value"]:
        raise Unreadable(f"`rmlx metrics query` printed an unexpected header: {lines[:1]}")
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != 4:
            raise Unreadable(f"`rmlx metrics query` printed a malformed row: {line!r}")
        name, metric, max_tokens, value = fields
        try:
            rows.append(
                {
                    "name": name,
                    "metric": metric,
                    "max_tokens": int(max_tokens) if max_tokens else None,
                    "value": float(value),
                }
            )
        except ValueError as exc:
            raise Unreadable(f"`rmlx metrics query` printed a malformed row: {line!r}") from exc
    return rows


def close(got: float, want: float) -> bool:
    return abs(got - want) <= REL_TOL * max(abs(want), 1.0)


def requery(result: dict, rows: list[dict]) -> tuple[dict, list[str]]:
    """`(figures from the database, disagreements with the result)`."""
    problems: list[str] = []
    fixed_rows = [r for r in rows if r["name"].startswith(FIXED_PREFIX)]
    sample_rows = [r for r in rows if not r["name"].startswith(FIXED_PREFIX)]
    if not rows:
        return {}, [
            f"the database holds no row for label {result['label']!r} at "
            f"{result['ts_utc']}; the run was not recorded"
        ]

    by_cell: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in sample_rows:
        parts = row["name"].split("/", 2)
        if len(parts) != 3 or parts[0] != "published":
            problems.append(f"a row is filed under {row['name']!r}, which is no sample")
            continue
        cell = by_cell.setdefault((parts[1], row["max_tokens"]), {})
        cell.setdefault(row["metric"], []).append(row["value"])

    figures: dict = {"label": result["label"], "ts_utc": result["ts_utc"], "cells": {}}
    cell_means: dict[str, float] = {}
    for name, entry in result["cells"].items():
        values = by_cell.get((entry["dataset"], entry["max_tokens"]), {})
        decode = values.get("decode_tps_warm", [])
        ttft = values.get("ttft_warm_ms", [])
        figures["cells"][name] = {
            "samples": len(decode),
            "decode_tps_mean": statistics.fmean(decode) if decode else None,
            "ttft_ms_mean": statistics.fmean(ttft) if ttft else None,
        }
        if len(decode) > entry["samples"] or len(ttft) > entry["samples"]:
            problems.append(
                f"cell {name}: the database holds {len(decode)} decode rows and "
                f"{len(ttft)} TTFT rows for {entry['samples']} samples; this result "
                "is already written. The store is append-only, so there is nothing "
                "to undo and it must not be recorded again"
            )
            continue
        if len(decode) != entry["samples"] or len(ttft) != entry["samples"]:
            problems.append(
                f"cell {name}: the database holds {len(decode)} decode rows and "
                f"{len(ttft)} TTFT rows where the result measured {entry['samples']} "
                "samples; the run was recorded partly"
            )
            continue
        cell_means[name] = figures["cells"][name]["decode_tps_mean"]
        for figure, got, want in (
            ("decode_tps", cell_means[name], entry["mean"]),
            ("ttft_ms", figures["cells"][name]["ttft_ms_mean"], entry["ttft_ms_mean"]),
        ):
            if not close(got, want):
                problems.append(
                    f"cell {name}: the rows give a mean {figure} of {got!r} where the "
                    f"result says {want!r}"
                )

    macro = result["macro"]
    if all(name in cell_means for name in macro["cells"]):
        rebuilt = statistics.fmean([cell_means[name] for name in macro["cells"]])
        figures["macro_decode_tps"] = rebuilt
        if not close(rebuilt, macro["mean"]):
            problems.append(
                f"macro: the rows give a mean of {rebuilt!r} where the result says "
                f"{macro['mean']!r}"
            )

    block = result.get("fixed_prompt")
    if block:
        fixed = {}
        for metric in ("decode_tps_warm", "prefill_tps", "ttft_warm_ms", "metal_peak_alloc_mb"):
            values = [r["value"] for r in fixed_rows if r["metric"] == metric]
            if len(values) != 1:
                problems.append(
                    f"fixed prompt: the database holds {len(values)} {metric} rows "
                    "where the result recorded one"
                )
                continue
            fixed[metric] = values[0]
        figures["fixed_prompt"] = fixed
        for metric, want in (
            ("decode_tps_warm", block["decode_tps"]["mean"]),
            ("prefill_tps", block["prefill_tps"]["mean"]),
            ("ttft_warm_ms", block["ttft_ms"]["mean"]),
            ("metal_peak_alloc_mb", block["metal_peak_alloc_mb"]["max"]),
        ):
            if metric in fixed and not close(fixed[metric], want):
                problems.append(
                    f"fixed prompt: the row gives {metric} {fixed[metric]!r} where "
                    f"the result says {want!r}"
                )
    elif fixed_rows:
        problems.append("the database holds a fixed-prompt row the result has no block for")
    return figures, problems


def check(result_path: Path, rmlx_bin: str, db: str | None, out: Path) -> list[str]:
    """Re-query one result's rows, write what came back, return the disagreements."""
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = query_rows(result, rmlx_bin, db)
        figures, problems = requery(result, rows)
    except (OSError, KeyError, ValueError) as exc:
        raise Unreadable(f"cannot read {result_path}: {exc!r}") from exc
    out.write_text(json.dumps(figures, indent=1) + "\n", encoding="utf-8")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("result", help="JSON emitted by spec_bench_published.sh")
    ap.add_argument("--rmlx-bin", default="rmlx")
    ap.add_argument("--db", default=None, help="a runs.db other than the default")
    ap.add_argument("--out", default=None, help="default: db_summary.json beside the result")
    args = ap.parse_args()
    result_path = Path(args.result)
    out = Path(args.out) if args.out else result_path.parent / "db_summary.json"
    try:
        problems = check(result_path, args.rmlx_bin, args.db, out)
    except Unreadable as exc:
        print(f"published_requery: {exc}", file=sys.stderr)
        return 2
    for problem in problems:
        print(f"published_requery: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"published_requery: the rows reproduce the result; figures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
