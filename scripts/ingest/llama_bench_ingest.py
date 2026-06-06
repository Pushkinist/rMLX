#!/usr/bin/env python3
"""Ingest llama-bench JSON output into the rMLX metrics DB.

Converts `llama-bench -o json` result rows into the §8.5 universal ingest
record shape and writes each one to `metrics/buffer/pending/` for pickup by
`rmlx metrics record --file <path>`.

Usage:
    # Write pending buffer file(s) only — human can inspect before ingest:
    python3 scripts/ingest/llama_bench_ingest.py llama_bench_output.json

    # Write buffer file then immediately call `rmlx metrics record`:
    python3 scripts/ingest/llama_bench_ingest.py --record llama_bench_output.json

    # Read from stdin:
    llama-bench -o json ... | python3 scripts/ingest/llama_bench_ingest.py -

    # Override hardware tag (default: m5_max_128gb):
    python3 scripts/ingest/llama_bench_ingest.py --hardware-tag m4_max_64gb run.json

    # Dry-run: print the §8.5 JSON records without writing files:
    python3 scripts/ingest/llama_bench_ingest.py --dry-run run.json

Input:
    llama-bench -o json produces a JSON array of result objects.  Each object
    contains one test configuration (n_prompt / n_gen pair).  This script
    groups rows by model + quant configuration and emits one §8.5 record per
    group, collecting prefill_tps (n_prompt>0, n_gen==0) and decode_tps_warm
    (n_prompt==0, n_gen>0) into the same record where they share the same
    model/quant/config.  Ungrouped rows (e.g. mixed pp+tg) each emit their own
    record with only the metric(s) that apply.

§8.5 mapping:
    llama-bench field      → observations column / §8.5 key
    model_filename         → model_namespace + model  (via §5.1 path rules)
    test_time              → ts_utc
    type_k / type_v        → kv_quant  (mapped via GGML_KV_MAP below)
    n_gpu_layers           → notes (informational, not a §8.5 field)
    avg_ts  (n_gen>0)      → decode_tps_warm  (value)
    stddev_ts (n_gen>0)    → decode_tps_warm  (stddev)
    avg_ts  (n_prompt>0, n_gen==0) → prefill_tps  (value)
    1000.0 / avg_ts (n_gen>0)      → step_ms_mean  (value, ms per token)
    n_prompt               → prompt_tokens
    n_gen                  → max_tokens
    build_commit           → git_sha

Fields with no llama-bench source (set to null / omitted):
    backend_version  — llama-bench emits build_commit, not semver
    build_profile    — no concept in llama-bench
    output_first_64  — decode text not captured by llama-bench
    temperature, seed, n_warmups, n_measure — not emitted
    metal_peak_alloc_mb, kv_cache_bytes, peak_rss_mb, ttft_*,
    itl_*, overall_tps — not emitted by llama-bench
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any


# ── Paths ──────────────────────────────────────────────────────────────────────

# rMLX repo root: this file lives at scripts/ingest/llama_bench_ingest.py.
_SCRIPT_DIR = Path(__file__).resolve().parent
RMLX_REPO_ROOT = Path(os.environ.get("RMLX_REPO_ROOT", str(_SCRIPT_DIR.parents[1])))
_BUFFER_PENDING = RMLX_REPO_ROOT / "metrics" / "buffer" / "pending"
_BUFFER_FAILED = RMLX_REPO_ROOT / "metrics" / "buffer" / "failed"
RMLX_DB_PATH = Path(
    os.environ.get("RMLX_METRICS_DB", str(RMLX_REPO_ROOT / "metrics" / "runs.db"))
)

DEFAULT_HARDWARE_TAG = "m5_max_128gb"

# Prompt body used as a placeholder when llama-bench provides no prompt text.
# llama-bench measures synthetic token generation, not a real prompt body — we
# register a minimal synthetic prompt in the prompts table so the FK is valid.
_SYNTHETIC_PROMPT: dict[str, Any] = {
    "name": "llama_bench_synthetic",
    "body": "<llama-bench synthetic benchmark — no real prompt text>",
    "notes": "Placeholder for llama-bench runs; body is not a real prompt.",
}


# ── GGML type → kv_quant canonical ─────────────────────────────────────────────

# llama-bench emits GGML type names in type_k / type_v fields.
# Map to the §5.3 canonical kv_quant strings.  Both fields must agree for a
# single kv_quant label; if they differ, we encode as "<type_k>/<type_v>" with
# the raw GGML names (still lowercase).
GGML_TO_CANONICAL: dict[str, str] = {
    "f16": "none",   # f16 KV = no quantization in rMLX terms
    "f32": "none",
    "bf16": "none",
    "q8_0": "k8v8",  # heuristic: if both K and V are q8_0 → k8v8
    "q4_0": "k8v4",  # heuristic: if K=q8_0 + V=q4_0 → k8v4 (see _map_kv_quant)
    "q4_k": "k8v4",
    "q5_k": "k8v4",
    "q6_k": "k8v8",
}


def _map_kv_quant(type_k: str, type_v: str) -> str:
    """Map GGML type_k + type_v strings to a §5.3 canonical kv_quant string.

    The exact GGML type names vary across llama.cpp versions.  Where the pair
    maps cleanly to a rMLX canonical (none / k8v4 / k8v8), use that.  Otherwise
    encode verbatim as "<type_k>_<type_v>" (lowercase, underscored).

    Note: the rMLX whitelist only includes none/k8v4/k8v8/planar/turbo4/turbo8.
    A non-canonical string will be rejected by `rmlx metrics record`.  In that
    case the pending file is moved to buffer/failed/ for human triage.
    """
    k = type_k.lower().strip() if type_k else "f16"
    v = type_v.lower().strip() if type_v else "f16"

    # Both unquantized → none.
    if k in ("f16", "f32", "bf16") and v in ("f16", "f32", "bf16"):
        return "none"

    # Symmetric q8_0 → k8v8.
    if k == "q8_0" and v == "q8_0":
        return "k8v8"

    # Asymmetric: K=q8_0, V=q4_0 (or similar 4-bit) → k8v4.
    if k == "q8_0" and v in ("q4_0", "q4_k", "q5_k"):
        return "k8v4"

    # Use per-key canonical if both agree.
    ck = GGML_TO_CANONICAL.get(k)
    cv = GGML_TO_CANONICAL.get(v)
    if ck and cv and ck == cv:
        return ck

    # Fall back: encode verbatim so the pending file records what happened.
    # rmlx metrics record will reject it; operator can inspect + fix.
    return f"{k}_{v}"


# ── Model path parsing (§5.1) ──────────────────────────────────────────────────

_NAMESPACE_WHITELIST = frozenset([
    "mlx-community", "z-lab", "prism-ml", "paramind", "paro-team",
    "ollama", "hf", "local",
])


def _split_model_path(model_filename: str) -> tuple[str, str]:
    """Split a model_filename into (model_namespace, model) per §5.1 rules.

    llama-bench emits the path to the GGUF file, e.g.:
        <open-models>/mlx-community__gemma-4-e2b-it-mxfp8/model.gguf
        ~/.cache/llama.cpp/Qwen2.5-7B-Instruct-Q4_K_M.gguf
        meta-llama/Llama-3.2-3B-Instruct  (HF id)
    """
    p = Path(model_filename.strip())

    if p.is_absolute() or model_filename.startswith("~"):
        # Filesystem path: inspect the parent directory name (the model dir),
        # or the file stem if directly a .gguf in root.
        # Walk up to find a directory name containing '__'.
        for part in reversed(p.parts):
            if "__" in part:
                ns, mdl = part.split("__", 1)
                # Strip GGUF extension from mdl if present.
                mdl = mdl.removesuffix(".gguf")
                if ns in _NAMESPACE_WHITELIST:
                    return ns, mdl
                # Unknown namespace → fall through to local.
                break
        # No namespaced directory found: use basename without extension.
        stem = p.stem if p.suffix else p.name
        return "local", stem

    # HF id: one slash, no leading slash.
    slash_count = model_filename.count("/")
    if slash_count == 1:
        return "hf", model_filename

    # Ollama tag: colon, no slash.
    if ":" in model_filename and "/" not in model_filename:
        return "ollama", model_filename

    # Anything else: treat as local bare name.
    return "local", model_filename


# ── Row classification ─────────────────────────────────────────────────────────

def _classify_row(row: dict[str, Any]) -> str:
    """Return 'prefill', 'decode', or 'mixed' based on n_prompt / n_gen."""
    n_prompt = int(row.get("n_prompt", 0))
    n_gen = int(row.get("n_gen", 0))
    if n_prompt > 0 and n_gen == 0:
        return "prefill"
    if n_prompt == 0 and n_gen > 0:
        return "decode"
    return "mixed"


# ── §8.5 record builder ────────────────────────────────────────────────────────

def _build_record(rows: list[dict[str, Any]], hardware_tag: str) -> dict[str, Any]:
    """Build a §8.5 record dict from one or more llama-bench result rows.

    All rows in the list must share the same model / quant configuration.
    Multiple rows (e.g. one prefill + one decode) are merged into one record.
    """
    if not rows:
        raise ValueError("rows list must not be empty")

    first = rows[0]

    model_filename = first.get("model_filename", "")
    model_namespace, model = _split_model_path(model_filename)

    type_k = first.get("type_k", "f16")
    type_v = first.get("type_v", "f16")
    kv_quant = _map_kv_quant(type_k, type_v)

    # weight_quant: llama-bench operates on GGUF files; the quant scheme is
    # encoded in the model name (Q4_K_M, Q8_0, etc.).  We normalise the model
    # stem for common GGUF suffixes.  Unknown → "unknown" (recorder will warn).
    weight_quant = _infer_weight_quant(model_filename)

    # ts_utc: use test_time from the first row (ISO-8601).
    ts_utc = first.get("test_time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if ts_utc and not ts_utc.endswith("Z") and "+" not in ts_utc:
        ts_utc = ts_utc.rstrip() + "Z"

    git_sha: str | None = first.get("build_commit") or None

    # n_gpu_layers: informational only; goes into notes.
    n_gpu_layers = first.get("n_gpu_layers")
    notes_parts = ["llama-bench"]
    if n_gpu_layers is not None:
        notes_parts.append(f"n_gpu_layers={n_gpu_layers}")

    # Prompt context: llama-bench uses synthetic tokens, no real text.
    prompt_tokens: int | None = None
    max_tokens: int | None = None

    metrics: list[dict[str, Any]] = []

    for row in rows:
        kind = _classify_row(row)
        avg_ts = row.get("avg_ts")
        stddev_ts = row.get("stddev_ts")

        if avg_ts is None or float(avg_ts) <= 0:
            continue

        avg_ts_f = float(avg_ts)
        stddev_f = float(stddev_ts) if stddev_ts is not None else None

        if kind == "prefill":
            metrics.append({"name": "prefill_tps", "value": avg_ts_f})
            if prompt_tokens is None:
                n_p = row.get("n_prompt")
                if n_p is not None:
                    prompt_tokens = int(n_p)

        elif kind == "decode":
            metrics.append({
                "name": "decode_tps_warm",
                "value": avg_ts_f,
                "stddev": stddev_f,
            })
            # step_ms_mean = ms per generated token = 1000 / tg_tps.
            metrics.append({
                "name": "step_ms_mean",
                "value": round(1000.0 / avg_ts_f, 4),
            })
            if max_tokens is None:
                n_g = row.get("n_gen")
                if n_g is not None:
                    max_tokens = int(n_g)

        # mixed: emit prefill_tps for the pp component, decode_tps_warm for tg.
        elif kind == "mixed":
            # llama-bench reports combined avg_ts for mixed pp+tg; we cannot
            # cleanly separate them.  Emit as overall_tps (best approximation).
            metrics.append({"name": "overall_tps", "value": avg_ts_f})
            n_p = row.get("n_prompt")
            n_g = row.get("n_gen")
            if n_p is not None and prompt_tokens is None:
                prompt_tokens = int(n_p)
            if n_g is not None and max_tokens is None:
                max_tokens = int(n_g)

    # Require at least one non-null metric entry.
    if not metrics:
        raise ValueError(
            f"no usable metrics found in rows for model={model_filename!r}"
        )

    return {
        "backend":          "llama_cpp",
        "backend_version":  None,
        "model_namespace":  model_namespace,
        "model":            model,
        "weight_quant":     weight_quant,
        "kv_quant":         kv_quant,
        "ctx_max":          _infer_ctx_max(rows),
        "prompt":           _SYNTHETIC_PROMPT,
        "ts_utc":           ts_utc,
        "git_sha":          git_sha,
        "build_profile":    None,
        "hardware_tag":     hardware_tag,
        "prompt_tokens":    prompt_tokens,
        "max_tokens":       max_tokens,
        "temperature":      None,
        "seed":             None,
        "n_warmups":        None,
        "n_measure":        len(rows),
        "output_first_64":  None,
        "notes":            " ".join(notes_parts),
        "description":      None,
        "metrics":          metrics,
    }


def _infer_weight_quant(model_filename: str) -> str:
    """Infer weight quant from the GGUF filename suffix."""
    stem = Path(model_filename).stem.lower()
    # Common GGUF quant suffixes — check longest matches first.
    patterns: list[tuple[str, str]] = [
        ("q4_k_m", "q4_k_m"),
        ("q4_k_s", "q4_k_m"),  # treat _s as _m for now
        ("q5_k_m", "q4_k_m"),  # no exact match; nearest is q4_k_m — flag unknown
        ("q8_0",   "q8_0"),
        ("q4_0",   "q4_0"),
        ("f16",    "fp16"),
        ("fp16",   "fp16"),
        ("bf16",   "bf16"),
        ("f32",    "bf16"),    # approximate — no rMLX canonical for f32 weights
        ("2bit",   "2bit"),
        ("3bit",   "3bit"),
        ("4bit",   "4bit"),
        ("5bit",   "5bit"),
        ("6bit",   "6bit"),
        ("8bit",   "8bit"),
    ]
    for suffix, canonical in patterns:
        if stem.endswith(suffix) or f"-{suffix}" in stem or f"_{suffix}" in stem:
            return canonical
    return "unknown"


def _infer_ctx_max(rows: list[dict[str, Any]]) -> int:
    """Return context length from rows, defaulting to 4096."""
    for row in rows:
        # llama-bench has n_ctx in older versions; fallback to prompt+gen.
        n_ctx = row.get("n_ctx")
        if n_ctx:
            return int(n_ctx)
        n_p = int(row.get("n_prompt", 0))
        n_g = int(row.get("n_gen", 0))
        if n_p + n_g > 0:
            return max(n_p + n_g, 512)
    return 4096


# ── Grouping ───────────────────────────────────────────────────────────────────

def _group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Group rows that should be merged into one §8.5 record."""
    return (
        row.get("model_filename", ""),
        row.get("type_k", ""),
        row.get("type_v", ""),
        str(row.get("n_gpu_layers", "")),
        str(row.get("n_ctx", "")),
    )


# ── Buffer write + recorder invocation ────────────────────────────────────────

def _find_rmlx_bin() -> str | None:
    result = subprocess.run(["which", "rmlx"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    release = RMLX_REPO_ROOT / "target" / "release" / "rmlx"
    if release.exists():
        return str(release)
    debug = RMLX_REPO_ROOT / "target" / "debug" / "rmlx"
    if debug.exists():
        return str(debug)
    return None


def _write_pending(payload: dict[str, Any]) -> Path:
    """Write one §8.5 payload to metrics/buffer/pending/ and return the path."""
    _BUFFER_PENDING.mkdir(parents=True, exist_ok=True)
    ts_tag = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    uid8 = uuid.uuid4().hex[:8]
    record_path = _BUFFER_PENDING / f"{ts_tag}-{uid8}.json"
    record_path.write_text(json.dumps(payload, indent=2))
    return record_path


def _invoke_recorder(record_path: Path) -> bool:
    """Call `rmlx metrics record --file <path>`.  Return True on success."""
    rmlx_bin = _find_rmlx_bin()
    if rmlx_bin is None:
        print(
            f"[llama_bench_ingest] ERROR: rmlx binary not found; "
            f"pending file kept at {record_path}",
            file=sys.stderr,
        )
        return False

    result = subprocess.run(
        [rmlx_bin, "metrics", "--db", str(RMLX_DB_PATH), "record", "--file", str(record_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        record_path.unlink(missing_ok=True)
        return True

    # Recorder rejected: move to failed/.
    _BUFFER_FAILED.mkdir(parents=True, exist_ok=True)
    failed_path = _BUFFER_FAILED / record_path.name
    try:
        record_path.rename(failed_path)
    except OSError:
        pass
    print(
        f"[llama_bench_ingest] WARN: recorder rejected record; "
        f"see {failed_path}\n  stderr: {result.stderr.strip()[:400]}",
        file=sys.stderr,
    )
    return False


# ── Entry point ────────────────────────────────────────────────────────────────

def ingest(
    source_path: str,
    hardware_tag: str,
    invoke_recorder: bool,
    dry_run: bool,
) -> int:
    """Parse llama-bench JSON, build §8.5 records, write to buffer.

    Returns the number of records successfully written (or printed for dry-run).
    Raises SystemExit with code 1 on fatal input errors.
    """
    if source_path == "-":
        raw = sys.stdin.read()
    else:
        p = Path(source_path)
        if not p.exists():
            print(
                f"[llama_bench_ingest] ERROR: input file not found: {source_path}",
                file=sys.stderr,
            )
            sys.exit(1)
        raw = p.read_text()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(
            f"[llama_bench_ingest] ERROR: invalid JSON: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not isinstance(data, list):
        # llama-bench may emit a single object in some versions; wrap it.
        data = [data]

    if not data:
        print("[llama_bench_ingest] WARN: empty input — no rows to ingest", file=sys.stderr)
        return 0

    # Group rows by (model, quant, config) to merge prefill + decode pairs.
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in data:
        if not isinstance(row, dict):
            print(
                f"[llama_bench_ingest] WARN: skipping non-object row: {row!r}",
                file=sys.stderr,
            )
            continue
        key = _group_key(row)
        groups.setdefault(key, []).append(row)

    n_written = 0
    for key, rows in groups.items():
        try:
            payload = _build_record(rows, hardware_tag)
        except ValueError as exc:
            print(
                f"[llama_bench_ingest] WARN: skipping group {key}: {exc}",
                file=sys.stderr,
            )
            continue

        if dry_run:
            print(json.dumps(payload, indent=2))
            n_written += 1
            continue

        record_path = _write_pending(payload)
        print(f"[llama_bench_ingest] wrote {record_path}", flush=True)

        if invoke_recorder:
            ok = _invoke_recorder(record_path)
            if ok:
                print(f"[llama_bench_ingest] recorded OK", flush=True)
        n_written += 1

    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest llama-bench -o json output into the rMLX metrics DB.\n\n"
            "Writes §8.5 universal JSON records to metrics/buffer/pending/.\n"
            "Use --record to also invoke `rmlx metrics record --file <path>`."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="FILE",
        help='Path to llama-bench JSON output file, or "-" for stdin.',
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=False,
        help="After writing the buffer file, invoke `rmlx metrics record --file <path>`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print §8.5 JSON records to stdout without writing files or calling recorder.",
    )
    parser.add_argument(
        "--hardware-tag",
        default=DEFAULT_HARDWARE_TAG,
        help=f"hardware_tag for the observations (default: {DEFAULT_HARDWARE_TAG}).",
    )
    args = parser.parse_args()

    if args.dry_run and args.record:
        parser.error("--dry-run and --record are mutually exclusive")

    n = ingest(
        source_path=args.input,
        hardware_tag=args.hardware_tag,
        invoke_recorder=args.record,
        dry_run=args.dry_run,
    )
    print(f"[llama_bench_ingest] done: {n} record(s) processed", flush=True)


if __name__ == "__main__":
    main()
