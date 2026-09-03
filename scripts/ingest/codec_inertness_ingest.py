#!/usr/bin/env python3
"""Ingest `codec_inertness_probe.sh` residency cells into the rMLX metrics DB.

The probe writes a CSV and never touches `runs.db` — a sweep is an experiment
and the store is append-only. This is the separate, explicit step that promotes
its cells into §8.5 RunRecords, one per (model, context, codec).

**Only `kv_cache_bytes` is recorded.** The probe's `decode_tps` / `ttft_ms` /
`prefill_tps` columns are single unpaired runs on a shared host, where the same
binary and flags have read 11% apart thirty minutes apart; promoting one into an
append-only table would make an unusable number permanent. `kv_cache_bytes` is a
byte count and does not care about host load, which is why it is the column the
codec disposition rests on and the only one that comes through here. Throughput
belongs to `scripts/perf_ab.sh` and `perf_ab_ingest.py`.

The greedy token-id digest travels in `notes`, not as a metric: it is an
identity, not a measurement, and `observations` stores reals.

A sweep run at non-default `--kv-boundary-layers` lands in a cell of its own:
the probe records the counts in a `kv_boundary` column and the record carries
them as `decode_config` (`docs/METRICS_DB.md` §3.2). Those cells do not rank
against a default-boundary sweep, which is the point — a different head/tail
count is a different engine configuration, not another sample of one.

`notes` also carries `any_layer_skipped_store`, which is a per-RUN flag and not
a codec classification — see the field's comment below. Rows written before that
rename spell it `packed_store_skipped` and carry no caveat; `observations` is
append-only, so both spellings exist and a query over the column has to accept
either. Nothing derived from it should classify a codec.

Usage:
    # Inspect the records without writing anything:
    python3 scripts/ingest/codec_inertness_ingest.py --dry-run CSV \\
        --backend-version 0.3.0 --build-profile release-perf

    # Write buffer files and ingest them:
    python3 scripts/ingest/codec_inertness_ingest.py --record CSV \\
        --backend-version 0.3.0 --build-profile release-perf
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
RMLX_REPO_ROOT = Path(os.environ.get("RMLX_REPO_ROOT", str(_SCRIPT_DIR.parents[1])))
# Runtime state lives under a single root (CLAUDE.md). `RMLX_HOME` wins; the
# in-repo `.rmlx/` is the dev default.
RMLX_HOME = Path(os.environ.get("RMLX_HOME", str(RMLX_REPO_ROOT / ".rmlx")))
BUFFER_PENDING = RMLX_HOME / "metrics" / "buffer" / "pending"
# Assembled here first, moved into `pending/` only for the moment the recorder
# is invoked. Anything left sitting in `pending/` is, by contract, work the next
# `--replay-pending` sweep will claim and quarantine if it cannot ingest.
STAGING = RMLX_HOME / "bench" / "codec_inertness" / "records"

# The probe's model column is a snapshot directory name. Split it the way the
# identity columns want it: `<namespace>__<model>`.
NAMESPACES = {"mlx-community", "z-lab", "prism-ml", "paramind", "paro-team"}

# §5.2 whitelist tokens, most specific first — same order as
# `rmlx_metrics::identity::infer_weight_quant`, which this mirrors on purpose so
# a row ingested here lands in the same cell as one the binary recorded.
WEIGHT_QUANT_TOKENS = [
    "mxfp8", "mxfp4", "nvfp4", "q4_k_m", "q8_0",
    "8bit", "4bit", "2bit", "3bit", "5bit", "6bit", "fp16", "bf16", "paro",
]


def split_model(dirname: str) -> tuple[str, str]:
    """`mlx-community__gemma-4-e2b-it-mxfp8` -> (`mlx-community`, `gemma-...`)."""
    if "__" in dirname:
        ns, model = dirname.split("__", 1)
        if ns in NAMESPACES:
            return ns, model
    return "local", dirname


def weight_quant_of(model: str) -> str:
    lower = model.lower()
    for token in WEIGHT_QUANT_TOKENS:
        if token in lower:
            return token
    return "bf16"


# The engine's own default, `rmlx_models::kv_cache::KvBoundary::default`.
KV_BOUNDARY_DEFAULT = (2, 8)


def decode_config_of(row: dict[str, str]) -> str | None:
    """The `decode_config` cell term this row's boundary-layer counts imply.

    Read off the `kv_boundary` column the probe writes from its own argument,
    so the term describes the sweep that ran rather than one the operator
    typed. `None` is the engine at its defaults — which is also the right
    answer for a CSV written before the probe had the column, because there was
    no way to run at anything else then. That is a fact about those sweeps, not
    a substituted value.
    """
    raw = (row.get("kv_boundary") or "").strip()
    if not raw:
        return None
    try:
        head, tail = (int(x) for x in raw.split(","))
    except ValueError:
        raise SystemExit(
            f"unparseable kv_boundary column {raw!r}: expected '<head>,<tail>'. "
            "Refusing rather than recording the row under the default cell — a "
            "sweep filed in the wrong cell is permanent."
        ) from None
    if (head, tail) == KV_BOUNDARY_DEFAULT:
        return None
    return f"kv_boundary/head={head},kv_boundary/tail={tail}"


def prompt_for(fixture_tokens: int, prompts_dir: Path) -> tuple[str, str]:
    """Resolve the canonical bench prompt the probe's `--prompt-tokens` names.

    The body is the fixture file's own text, which is what `perf_ab_ingest.py`
    records, so the `prompts` row this record keys on is the one every other
    cell at that length already keys on. Deriving a body some other way would
    fork the content-addressed prompt table and put this sweep in a cell of its
    own.
    """
    name = f"longctx_{fixture_tokens // 1024}k"
    path = prompts_dir / f"{name}.json"
    if not path.is_file():
        raise SystemExit(f"prompt fixture not found: {path}")
    return name, path.read_text(encoding="utf-8")


def build_record(row: dict[str, str], args: argparse.Namespace, prompts_dir: Path) -> dict[str, Any]:
    namespace, model = split_model(row["model"])
    fixture_tokens = int(row["prompt_tokens"])
    # The fixture NAME is not a token count: the chat template pushes a
    # `longctx_32k` prompt to 34 355 tokens on gemma-4. Recording the name in
    # the `prompt_tokens` column would make every cell of this sweep disagree
    # with the same cell recorded by any other harness. Refuse rather than
    # substitute — a plausible wrong number is the failure mode here.
    measured = row.get("prompt_tokens_measured") or ""
    if not measured:
        raise SystemExit(
            "this CSV has no `prompt_tokens_measured` column (it predates the "
            "probe recording one). Re-run the probe; the fixture name in "
            "`prompt_tokens` is not a token count and is not substituted."
        )
    prompt_tokens = int(measured)
    # `ctx_max` is a PK column. The probe accepts `--max-ctx`, so re-deriving it
    # from its default formula here would silently record a plausible wrong
    # ceiling for any sweep that overrode one — permanently, in an append-only
    # table, with no error. Read what the run used; refuse if the CSV predates
    # the column, for the same reason the token count above is not substituted.
    raw_ctx = row.get("max_ctx") or ""
    if not raw_ctx:
        raise SystemExit(
            "this CSV has no `max_ctx` column (it predates the probe recording "
            "one). Re-run the probe; the default ceiling formula is not "
            "re-derived here, because a sweep that passed --max-ctx would then "
            "be recorded under a ceiling it never ran at."
        )
    max_ctx = int(raw_ctx)
    prompt_name, prompt_body = prompt_for(fixture_tokens, prompts_dir)
    return {
        "schema_version": 1,
        "backend": "rmlx",
        "backend_version": args.backend_version,
        "model_namespace": namespace,
        "model": model,
        "weight_quant": weight_quant_of(model),
        "kv_quant": row["codec"],
        "decode_config": decode_config_of(row),
        "ctx_max": max_ctx,
        "prompt": {"name": prompt_name, "body": prompt_body, "notes": None},
        "ts_utc": row["timestamp"],
        "git_sha": args.git_sha,
        "build_profile": args.build_profile,
        "hardware_tag": args.hardware_tag,
        "prompt_tokens": prompt_tokens,
        "max_tokens": int(row["max_tokens"]),
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 0,
        "n_measure": 1,
        "output_first_64": None,
        "notes": (
            # `any_layer_skipped_store` is per-RUN, not per-codec: the
            # layer-adaptive head/tail promotion types some layers `K8V8`, so a
            # store-reading codec sets it too. Named so no later reader mistakes
            # it for a classification. `observations` is append-only, so the
            # caveat has to travel with the value.
            f"codec_inertness_probe; binary_sha={row['binary_sha']}; "
            f"token_ids_sha256_16={row['ids_sha']}; "
            f"any_layer_skipped_store={row['store_skipped']}; "
            "residency only — the probe's throughput columns are single unpaired "
            "runs on a shared host and are deliberately not recorded"
        ),
        "description": args.description,
        "metrics": [{"name": "kv_cache_bytes", "value": float(row["kv_cache_bytes"])}],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", help="codec_inertness.csv written by the probe")
    p.add_argument(
        "--backend-version",
        required=True,
        help="semver of the rmlx binary that produced the cells (§8.5.1). Required, "
        "not inferred: a wrong identity value is worse than no value.",
    )
    p.add_argument(
        "--build-profile",
        required=True,
        help="how the measured binary was built, e.g. 'release-perf'.",
    )
    p.add_argument("--git-sha", default=None, help="caller-supplied provenance (§8.5.1)")
    p.add_argument("--hardware-tag", default="m5_max_128gb")
    p.add_argument("--description", default=None)
    p.add_argument("--prompts-dir", default=str(RMLX_REPO_ROOT / "prompts"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record", action="store_true", help="ingest via `rmlx metrics record --file`")
    p.add_argument("--rmlx-bin", default=str(RMLX_REPO_ROOT / "target" / "release-perf" / "rmlx"))
    args = p.parse_args()

    prompts_dir = Path(args.prompts_dir)
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    # A failed cell emits an empty `kv_cache_bytes` and an empty digest — the
    # sha256 of nothing, which reads exactly like a real result. The exit code
    # is the column that tells the two apart, so it is the filter.
    usable = [r for r in rows if r["exit_code"] == "0" and r["kv_cache_bytes"]]
    dropped = len(rows) - len(usable)
    if dropped:
        print(f"skipping {dropped} row(s) with a non-zero exit code or no byte count")
    if not usable:
        print("no usable rows", file=sys.stderr)
        return 2

    records = [build_record(r, args, prompts_dir) for r in usable]

    if args.dry_run:
        for rec in records:
            trimmed = dict(rec)
            trimmed["prompt"] = dict(rec["prompt"], body=f"<{len(rec['prompt']['body'])} chars>")
            print(json.dumps(trimmed, indent=2))
        return 0

    STAGING.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for rec in records:
        stamp = rec["ts_utc"].replace("-", "").replace(":", "")
        path = STAGING / f"{stamp}-{rec['kv_quant']}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        staged.append(path)
    print(f"staged {len(staged)} record(s) in {STAGING}")

    if not args.record:
        print(
            "not ingested (pass --record). The files above are OUTSIDE\n"
            f"  {BUFFER_PENDING},\n"
            "  so no --replay-pending sweep can claim them."
        )
        return 0

    BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in staged:
        queued = BUFFER_PENDING / path.name
        path.rename(queued)
        try:
            proc = subprocess.run(
                [args.rmlx_bin, "metrics", "record", "--file", str(queued)], check=False
            )
        finally:
            # The recorder deletes on success; on failure the file is ours to
            # retrieve, not the next sweep's to quarantine.
            if queued.exists():
                queued.rename(path)
        if proc.returncode != 0:
            print(
                f"ingest FAILED for {path.name} (exit {proc.returncode}); "
                "record kept in staging, not in the pending queue",
                file=sys.stderr,
            )
            failed += 1
    print(f"ingested {len(staged) - failed} of {len(staged)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
