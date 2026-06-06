"""Run mlx_lm.evaluate wikitext-PPL across the 9-model in-scope set.

Captures upstream weight-quant PPL as a baseline. Cannot vary KV-quant (mlx-lm
loads native weight-quant only and uses bf16 KV).

Usage (driven from a wrapper bash script that activates the right venv):
    python -m scripts.bench.run_upstream_ppl

Writes:
    metrics/ppl_drift/ppl_upstream.json
    metrics/ppl_drift/ppl_<model>.txt  per-model raw output
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(
    os.environ.get("RMLX_ROOT")
    or Path(__file__).resolve().parents[2]
)
O_MODELS = Path(
    os.environ.get("RMLX_O_MODELS_ROOT")
    or ROOT.parents[1] / "open-models"
)

# Limit to 16 wikitext samples per model — keeps runtime bounded (~2 min/model).
LIMIT = 16
TASKS = ["wikitext"]
OUTPUT_DIR = ROOT / "metrics" / "ppl_drift"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "mlx-community__Qwen3.6-35B-A3B-8bit": str(O_MODELS / "mlx-community__Qwen3.6-35B-A3B-8bit"),
    "z-lab__Qwen3.6-27B-PARO": str(O_MODELS / "z-lab__Qwen3.6-27B-PARO") if (O_MODELS / "z-lab__Qwen3.6-27B-PARO").exists() else None,
    "prism-ml__Ternary-Bonsai-8B-mlx-2bit": str(O_MODELS / "prism-ml__Ternary-Bonsai-8B-mlx-2bit"),
    "mlx-community__medgemma-1.5-4b-it-8bit": str(O_MODELS / "mlx-community__medgemma-1.5-4b-it-8bit"),
    "mlx-community__gemma-4-e2b-it-mxfp8": str(O_MODELS / "mlx-community__gemma-4-e2b-it-mxfp8"),
    "mlx-community__gemma-4-e4b-it-mxfp8": str(O_MODELS / "mlx-community__gemma-4-e4b-it-mxfp8"),
    "mlx-community__gemma-4-26b-a4b-it-mxfp8": str(O_MODELS / "mlx-community__gemma-4-26b-a4b-it-mxfp8"),
    "mlx-community__gemma-4-31b-it-mxfp8": str(O_MODELS / "mlx-community__gemma-4-31b-it-mxfp8"),
    "z-lab__gemma-4-31B-it-PARO": str(O_MODELS / "z-lab__gemma-4-31B-it-PARO") if (O_MODELS / "z-lab__gemma-4-31B-it-PARO").exists() else None,
}


def run_one(name: str, path: str) -> dict:
    """Invoke mlx_lm.evaluate.main() in-process. Returns parsed metrics dict."""
    import mlx.core as mx
    import mlx_lm.evaluate as ev

    out_subdir = OUTPUT_DIR / "raw" / name
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Reset argv for argparse.
    sys.argv = [
        "mlx_lm.evaluate",
        "--model", path,
        "--tasks", *TASKS,
        "--limit", str(LIMIT),
        "--batch-size", "1",
        "--output-dir", str(out_subdir),
        "--no-apply-chat-template",
    ]

    t0 = time.monotonic()
    try:
        ev.main()
        success = True
        err = None
    except Exception as e:
        success = False
        err = repr(e)
    dur = time.monotonic() - t0

    # Parse output file written by main().
    result = {"model": name, "duration_s": dur, "success": success, "error": err}
    if success:
        # mlx_lm.evaluate writes file `eval_<modelpath>_<lmeval-version>_wikitext`.
        # Find the most recent file in out_subdir.
        files = sorted(out_subdir.glob("eval_*"), key=lambda p: p.stat().st_mtime)
        if files:
            try:
                payload = json.loads(files[-1].read_text())
                # Output is `{ "wikitext": { "alias": ..., "word_perplexity,none": float, ... } }`.
                if isinstance(payload, dict):
                    for task, metrics in payload.items():
                        if not isinstance(metrics, dict):
                            continue
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                result[f"{task}.{k}"] = v
            except Exception as e:
                result["parse_error"] = repr(e)

    # Free memory between runs.
    try:
        mx.clear_cache()
    except Exception:
        pass
    return result


def main() -> int:
    out = []
    for name, path in MODEL_PATHS.items():
        if path is None or not Path(path).exists():
            print(f"[skip] {name}: path missing", flush=True)
            out.append({"model": name, "skipped": True, "reason": "path missing"})
            continue
        # Skip ones that exceed budget if env says so.
        if name in os.environ.get("PPL_DRIFT_SKIP", "").split(","):
            out.append({"model": name, "skipped": True, "reason": "PERF51_SKIP"})
            continue
        print(f"[ppl] {name}: starting (limit={LIMIT})", flush=True)
        rec = run_one(name, path)
        print(
            f"[ppl] {name}: done in {rec['duration_s']:.1f}s "
            f"word_ppl={rec.get('word_perplexity,none', 'n/a')} "
            f"byte_ppl={rec.get('byte_perplexity,none', 'n/a')}",
            flush=True,
        )
        out.append(rec)

    (OUTPUT_DIR / "ppl_upstream.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUTPUT_DIR / 'ppl_upstream.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
