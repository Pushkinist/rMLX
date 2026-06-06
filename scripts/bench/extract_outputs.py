"""Extract per-(model, backend, quant) greedy-output tuples from CBB summary.csv.

Computes string-similarity quality drift across cells.
The bench was run with temperature=0 + 32 max_tokens on `prompts/longctx_4k.json`,
so `output_first_64` is a deterministic decode fingerprint per cell.

Usage:
    python -m scripts.bench.extract_outputs
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(
    os.environ.get("RMLX_ROOT")
    or Path(__file__).resolve().parents[2]
)
CBB_SUMMARY = Path(
    os.environ.get("CROSS_BENCH_ROOT")
    or ROOT.parents[1] / "Cross-Backend-Bench"
) / "metrics" / "summary.csv"
OUT_DIR = ROOT / "metrics" / "ppl_drift"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 9-model scope per BENCHMARK_RECORDS (drop DR-Venus + Laguna).
MODELS_IN_SCOPE = {
    "mlx-community__Qwen3.6-35B-A3B-8bit": "Qwen3.5MoE / affine8",
    "z-lab__Qwen3.6-27B-PARO": "Qwen3.5MoE / paro4",
    "prism-ml__Ternary-Bonsai-8B-mlx-2bit": "Qwen3 dense / 2bit",
    "mlx-community__medgemma-1.5-4b-it-8bit": "Gemma3 / affine8",
    "mlx-community__gemma-4-e2b-it-mxfp8": "Gemma4 small / mxfp8",
    "mlx-community__gemma-4-e4b-it-mxfp8": "Gemma4 small / mxfp8",
    "mlx-community__gemma-4-26b-a4b-it-mxfp8": "Gemma4 MoE / mxfp8",
    "mlx-community__gemma-4-31b-it-mxfp8": "Gemma4 dense / mxfp8",
    "z-lab__gemma-4-31B-it-PARO": "Gemma4 dense / paro4",
}


def short_model(model_id: str) -> str:
    """Normalize model_id to BENCHMARK_RECORDS naming.

    Bench runners store either:
        - filesystem path:  <RMLX_O_MODELS_ROOT>/mlx-community__foo
        - HF-style id:      mlx-community/foo  or  z-lab/Qwen3.6-27B-PARO
    Both must collapse to the canonical underscore form.
    """
    base = model_id.rsplit("/", 1)[-1]
    # If we got `mlx-community/foo`, the prior split returned `foo` only — re-stitch.
    if "/" in model_id and "__" not in base:
        prefix, suffix = model_id.rsplit("/", 1)
        # Take only the last 2 path components (org / repo).
        org = prefix.rsplit("/", 1)[-1]
        return f"{org}__{suffix}"
    return base


def kv_label(quant_signature: str) -> str:
    """Extract kv-quant token from quant_signature, or 'bf16' if absent."""
    s = quant_signature.lower()
    if "k8v4" in s:
        return "k8v4"
    if "k8v8" in s:
        return "k8v8"
    if "planar" in s:
        return "planar"
    return "bf16"


def main() -> None:
    if not CBB_SUMMARY.exists():
        sys.exit(f"summary.csv not found: {CBB_SUMMARY}")

    # Map (model, backend, kv) -> list of (run_id, decode_tps, output_first_64)
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    with CBB_SUMMARY.open() as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            model = short_model(row["model_id"])
            if model not in MODELS_IN_SCOPE:
                continue
            if row["success"] != "True":
                continue
            if not row["output_first_64"].strip():
                continue
            kv = kv_label(row["quant_signature"])
            backend = row["backend"]
            cell = (model, backend, kv)
            cells[cell].append(
                {
                    "run_id": row["run_id"],
                    "ts": row["timestamp_utc"],
                    "decode_tps": float(row.get("decode_tps") or 0.0),
                    "ttft_ms": float(row.get("ttft_ms") or 0.0),
                    "output": row["output_first_64"],
                    "quant_signature": row["quant_signature"],
                }
            )

    # rMLX has been heavily fixed (Gemma SWA, Bonsai opt, GDN tensorise), so
    # outputs from older commits are stale. Pick the **most recent** run per
    # cell. For upstream backends, latest is also fine (deterministic at temp=0).
    canonical: dict[tuple[str, str, str], dict] = {}
    output_consistency_warnings = []
    for cell, runs in cells.items():
        outs = {r["output"] for r in runs}
        if len(outs) > 1:
            output_consistency_warnings.append(
                {"cell": list(cell), "distinct_outputs": list(outs)}
            )
        # Pick most recent timestamp (latest rMLX fix wins; mlx-lm too).
        best = max(runs, key=lambda r: r["ts"])
        canonical[cell] = best

    # Reference output per model. oMLX is preferred for mxfp8 + Gemma4 since
    # mlx-lm has known Gemma chat-template / SWA divergence; for non-mxfp8 the
    # mlx-lm reference is the de-facto correct decode at temp=0. PARO models
    # use paroquant upstream; ollama for cases where it's the only working backend.
    # Per-model priority list — first match wins.
    PARO = ["paroquant"]
    GEMMA4_MXFP8 = ["omlx", "oMLX", "mlx-lm"]
    DEFAULT = ["mlx-lm", "omlx", "oMLX", "ollama", "mlx-lm-turboquant", "paroquant"]
    per_model_priority = {
        "mlx-community__gemma-4-e2b-it-mxfp8": GEMMA4_MXFP8,
        "mlx-community__gemma-4-e4b-it-mxfp8": GEMMA4_MXFP8,
        "mlx-community__gemma-4-26b-a4b-it-mxfp8": GEMMA4_MXFP8,
        "mlx-community__gemma-4-31b-it-mxfp8": GEMMA4_MXFP8,
        "z-lab__Qwen3.6-27B-PARO": PARO,
        "z-lab__gemma-4-31B-it-PARO": PARO,
    }
    upstream_priority = DEFAULT  # legacy var name preserved for output JSON

    rows = []
    for model in MODELS_IN_SCOPE:
        # Find reference (upstream) using per-model priority.
        priority = per_model_priority.get(model, DEFAULT)
        ref_cell = None
        ref_data = None
        for backend in priority:
            for kv in ("bf16", "k8v8", "k8v4", "planar"):
                cand = (model, backend, kv)
                if cand in canonical:
                    ref_cell = cand
                    ref_data = canonical[cand]
                    break
            if ref_data:
                break

        for (m, backend, kv), data in canonical.items():
            if m != model:
                continue
            ref_output = ref_data["output"] if ref_data else ""
            sim = (
                SequenceMatcher(None, ref_output, data["output"]).ratio()
                if ref_output
                else 0.0
            )
            exact = ref_output == data["output"]
            row = {
                "model": model,
                "arch_class": MODELS_IN_SCOPE[model],
                "backend": backend,
                "kv_quant": kv,
                "weight_quant": data["quant_signature"],
                "decode_tps": data["decode_tps"],
                "ttft_ms": data["ttft_ms"],
                "ref_backend": ref_cell[1] if ref_cell else None,
                "ref_kv": ref_cell[2] if ref_cell else None,
                "ref_output": ref_output,
                "output": data["output"],
                "similarity_to_ref": sim,
                "exact_match": exact,
                "is_reference": (m, backend, kv) == ref_cell,
                "run_id": data["run_id"],
            }
            rows.append(row)

    # Group by model for easy reading.
    rows.sort(
        key=lambda r: (
            list(MODELS_IN_SCOPE).index(r["model"]),
            0 if r["backend"] == "rmlx" else 1,
            r["kv_quant"],
        )
    )

    out = {
        "rows": rows,
        "consistency_warnings": output_consistency_warnings,
        "ref_priority": upstream_priority,
        "models_in_scope": MODELS_IN_SCOPE,
    }

    json_path = OUT_DIR / "drift_table.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {json_path} ({len(rows)} rows)")

    # Pretty-print.
    print("\nDrift × TPS table:\n")
    print(
        f"{'model':<48} {'backend':<10} {'kv':<8} {'decode':>7} {'sim':>5} {'exact':>5}"
    )
    print("-" * 90)
    for r in rows:
        marker = "*" if r["is_reference"] else " "
        print(
            f"{r['model']:<48} {r['backend']:<10} {r['kv_quant']:<8} "
            f"{r['decode_tps']:>7.2f} {r['similarity_to_ref']:>5.2f} "
            f"{'Y' if r['exact_match'] else 'N':>5}{marker}"
        )

    if output_consistency_warnings:
        print(f"\n{len(output_consistency_warnings)} cells had inconsistent decoder outputs across runs:")
        for w in output_consistency_warnings:
            print(f"  {w['cell']}: {len(w['distinct_outputs'])} distinct outputs")


if __name__ == "__main__":
    main()
