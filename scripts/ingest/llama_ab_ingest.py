#!/usr/bin/env python3
"""Ingest a `scripts/bench_llama_ab.sh` result into the rMLX metrics DB.

`bench_llama_ab.sh` deliberately never touches `runs.db` — an A/B run is an
experiment and the store is append-only. This is the separate, explicit step
that promotes one accepted comparison into two §8.5 RunRecords, one per arm.

Sibling of `llama_bench_ingest.py`, which reads `llama-bench -o json`. The two
inputs are unrelated shapes: `llama-bench` measures synthetic token generation
and registers a placeholder prompt, while this reads real `/completion` timings
over a real prompt body and carries the resident-memory columns
(`kv_cache_bytes`, `peak_rss_mb`) that `llama-bench` cannot report at all.

Usage:
    # Inspect the records without writing anything:
    python3 scripts/ingest/llama_ab_ingest.py --dry-run RESULT.json \
        --arm-a-backend llama_cpp --arm-b-backend llama_cpp_tq \
        --model Qwen3-8B-Q8_0 --weight-quant q8_0 \
        --arm-a-kv-quant none --arm-b-kv-quant turbo3 \
        --prompt-name longctx_8k_text --prompt-file PROMPT.txt

    # Write buffer files and ingest them:
    python3 scripts/ingest/llama_ab_ingest.py --record RESULT.json ...

Every identity field a cross-backend row needs is caller-supplied and none is
guessed. `backend_version` is free-form for non-rMLX backends (§8.5.1) — pass
the build commit. A run the harness marked TAINTED is refused unless
`--accept-tainted` is given, and the taint text is then carried into `notes`:
promoting a tainted number silently is how an experiment becomes a permanent
row nobody can take back out.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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

MIB = 1024 * 1024


def _arm_record(
    result: dict[str, Any],
    arm: str,
    args: argparse.Namespace,
    prompt_body: str,
) -> dict[str, Any]:
    """Build one §8.5 RunRecord from one arm of an A/B result."""
    slots = [s for s in result["slots"] if s["arm"] == arm]
    if not slots:
        raise SystemExit(f"no slots for arm {arm} in {args.result}")

    tps = [s["decode_tps"] for s in slots]
    backend = args.arm_a_backend if arm == "A" else args.arm_b_backend
    version = args.arm_a_version if arm == "A" else args.arm_b_version
    kv_quant = args.arm_a_kv_quant if arm == "A" else args.arm_b_kv_quant

    # One measurement per slot, so a single-slot arm has no sample stddev.
    stddev = statistics.stdev(tps) if len(tps) > 1 else None

    # Every slot of one arm allocates the same KV buffer (same n_ctx, same
    # codec); differing values mean the arm was not one configuration.
    kv_mib = {s["kv_mib"] for s in slots}
    if len(kv_mib) != 1:
        raise SystemExit(f"arm {arm} slots disagree on kv_mib: {sorted(kv_mib)}")

    notes = (
        f"bench_llama_ab.sh ABBA n={len(slots)}/arm; "
        f"decode_tps median over slots, min={min(tps):.3f} max={max(tps):.3f}; "
        f"verdict={result['verdict']}"
    )
    if result.get("tainted"):
        notes += f"; TAINTED: {result['tainted']}"

    return {
        "schema_version": 1,
        "backend": backend,
        "backend_version": version,
        "model_namespace": args.model_namespace,
        "model": args.model,
        "weight_quant": args.weight_quant,
        "kv_quant": kv_quant,
        "ctx_max": result["n_ctx"],
        "prompt": {"name": args.prompt_name, "body": prompt_body, "notes": args.prompt_notes},
        "ts_utc": result["ts_utc"].replace("T", "T").replace("Z", "Z"),
        "git_sha": args.git_sha,
        "build_profile": "release",
        "hardware_tag": args.hardware_tag,
        "prompt_tokens": slots[0]["prompt_n"],
        "max_tokens": result["n_predict"],
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 1,
        "n_measure": len(slots),
        "output_first_64": slots[0].get("output_first_64") or None,
        "notes": notes,
        "description": args.description,
        "metrics": [
            {"name": "decode_tps_warm", "value": statistics.median(tps), "stddev": stddev},
            {
                "name": "prefill_tps",
                "value": statistics.median([s["prompt_tps"] for s in slots]),
            },
            {"name": "kv_cache_bytes", "value": kv_mib.pop() * MIB},
            {"name": "peak_rss_mb", "value": statistics.median([s["peak_rss_mb"] for s in slots])},
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result", help="JSON emitted by bench_llama_ab.sh")
    p.add_argument("--prompt-file", required=True, help="the prompt body that was sent")
    p.add_argument("--prompt-name", required=True)
    p.add_argument("--prompt-notes", default=None)
    p.add_argument("--model", required=True, help="§5.1 model label")
    p.add_argument("--model-namespace", default="hf")
    p.add_argument("--weight-quant", required=True, help="§5.2 whitelist value, e.g. q8_0")
    p.add_argument("--arm-a-backend", required=True, help="§5.4 whitelist value")
    p.add_argument("--arm-b-backend", required=True)
    p.add_argument("--arm-a-version", default=None, help="free-form for non-rMLX backends")
    p.add_argument("--arm-b-version", default=None)
    p.add_argument("--arm-a-kv-quant", required=True)
    p.add_argument("--arm-b-kv-quant", required=True)
    p.add_argument("--hardware-tag", default="m5_max_128gb")
    p.add_argument("--git-sha", default=None, help="caller-supplied provenance (§8.5.1)")
    p.add_argument("--description", default=None)
    p.add_argument("--accept-tainted", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--record", action="store_true", help="ingest via `rmlx metrics record --file`")
    p.add_argument("--rmlx-bin", default=str(RMLX_REPO_ROOT / "target" / "release" / "rmlx"))
    args = p.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    if result.get("tainted") and not args.accept_tainted:
        print(
            f"refusing: run is TAINTED ({result['tainted']}).\n"
            "  Re-run on a quiet host, or pass --accept-tainted to record it with the\n"
            "  taint carried into `notes`.",
            file=sys.stderr,
        )
        return 2

    prompt_body = Path(args.prompt_file).read_text(encoding="utf-8")
    records = [_arm_record(result, arm, args, prompt_body) for arm in ("A", "B")]

    if args.dry_run:
        for rec in records:
            trimmed = dict(rec)
            trimmed["prompt"] = dict(rec["prompt"], body=f"<{len(prompt_body)} chars>")
            print(json.dumps(trimmed, indent=2))
        return 0

    BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rec in records:
        path = BUFFER_PENDING / f"{result['ts_utc']}-{rec['backend']}-{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        written.append(path)
        print(f"wrote {path}")

    if not args.record:
        print("not ingested (pass --record). Inspect the files above first.")
        return 0

    failed = 0
    for path in written:
        proc = subprocess.run(
            [args.rmlx_bin, "metrics", "record", "--file", str(path)], check=False
        )
        if proc.returncode != 0:
            print(f"ingest FAILED for {path} (exit {proc.returncode})", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
