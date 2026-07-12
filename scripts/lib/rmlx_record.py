"""Shared §8.5 record emitter for the rMLX bench scripts.

Every rMLX shell bench used to inline its own `python3 -c 'rec = {...}'`
heredoc. Twelve copies of the same dict is how `backend_version` rotted into
seven `'0.0.1'` literals, two guessed `"release-perf"` strings and a handful of
missing keys. This module is the one place the script side builds a record.

Identity is never guessed here: `rmlx metrics identity --json` asks the binary
that is actually being measured, and the answer is merged verbatim. It is
stamped at EMIT time, into the buffer file — a record replayed later by a newer
binary keeps the identity of the build that produced it.

Usage from a bench script:

    python3 - <<'PY'
    import os, sys
    sys.path.insert(0, os.environ["RMLX_DIR"] + "/scripts/lib")
    from rmlx_record import emit
    emit(
        rmlx_bin=os.environ["RMLX_BIN"],
        model_id=os.environ["MODEL_ID"],
        kv_quant="k8v8",
        ctx_max=8192,
        prompt_name="longctx_4k",
        prompt_body=json.load(open(prompt_file)),
        metrics={"decode_tps_warm": 119.1},
        stddev={"decode_tps_warm": 0.6},
        notes="...",
    )
    PY

Non-rMLX emitters (llama_bench_ingest.py, run_mlx-lm*.sh, run_oMLX.sh) describe
a different backend and must NOT use this module.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

# Bump only alongside rmlx_metrics::ingest::RECORD_SCHEMA_VERSION.
RECORD_SCHEMA_VERSION = 1

_WEIGHT_QUANT_TOKENS = (
    "mxfp8", "mxfp4", "nvfp4", "q4_k_m", "q8_0",
    "8bit", "4bit", "2bit", "3bit", "5bit", "6bit",
    "fp16", "bf16", "paro",
)

_identity_cache: dict[str, Any] | None = None


def identity(rmlx_bin: str) -> dict[str, Any]:
    """Ask the measured binary who it is. Cached per process.

    Returns the §8.5 identity block: backend, backend_version, git_sha,
    build_profile, hardware_tag. Never fabricated, never hard-coded.
    """
    global _identity_cache
    if _identity_cache is None:
        out = subprocess.run(
            [rmlx_bin, "metrics", "identity", "--json"],
            check=True, capture_output=True, text=True,
        ).stdout
        _identity_cache = json.loads(out)
    return dict(_identity_cache)


def split_model_id(model_id: str) -> tuple[str, str]:
    """`<ns>__<model>` -> (ns, model). Bare names fall back to the `local` ns."""
    base = os.path.basename(model_id.rstrip("/"))
    if "__" in base:
        ns, _, model = base.partition("__")
        return ns, model
    return "local", base


def infer_weight_quant(model_id: str) -> str:
    """Weight quant from the snapshot name; `bf16` when nothing matches."""
    lower = model_id.lower()
    for token in _WEIGHT_QUANT_TOKENS:
        if token in lower:
            return token
    return "bf16"


def build(
    *,
    rmlx_bin: str,
    model_id: str,
    kv_quant: str,
    ctx_max: int,
    prompt_name: str,
    prompt_body: Any,
    metrics: dict[str, float | None],
    stddev: dict[str, float] | None = None,
    weight_quant: str | None = None,
    ts_utc: str | None = None,
    prompt_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    n_warmups: int | None = None,
    n_measure: int | None = None,
    output_first_64: str | None = None,
    notes: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Assemble one §8.5 record. Callers supply measurements, not identity."""
    ns, model = split_model_id(model_id)
    stddev = stddev or {}

    rec: dict[str, Any] = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "model_namespace": ns,
        "model": model,
        "weight_quant": weight_quant or infer_weight_quant(model_id),
        "kv_quant": kv_quant,
        "ctx_max": int(ctx_max),
        "prompt": {"name": prompt_name, "body": prompt_body},
        "ts_utc": ts_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": [
            {"name": k, "value": None if v is None else float(v),
             **({"stddev": float(stddev[k])} if k in stddev else {})}
            for k, v in metrics.items()
        ],
    }
    for key, val in (
        ("prompt_tokens", prompt_tokens),
        ("max_tokens", max_tokens),
        ("temperature", temperature),
        ("seed", seed),
        ("n_warmups", n_warmups),
        ("n_measure", n_measure),
        ("output_first_64", output_first_64),
        ("notes", notes),
        ("description", description),
    ):
        if val is not None:
            rec[key] = val

    # Identity last: it overwrites, so a caller cannot smuggle its own in.
    rec.update(identity(rmlx_bin))
    return rec


def buffer_dir(rmlx_dir: str) -> Path:
    d = Path(rmlx_dir) / ".rmlx" / "metrics" / "buffer" / "pending"
    d.mkdir(parents=True, exist_ok=True)
    return d


def emit(*, rmlx_dir: str, rmlx_bin: str, db: str | None = None, **kwargs: Any) -> Path:
    """Build the record, write the buffer file, ingest it.

    On success the recorder deletes the buffer file. On rejection the file is
    kept and the error surfaces — a bad record is loud, never a silent NULL.
    """
    rec = build(rmlx_bin=rmlx_bin, **kwargs)

    path = buffer_dir(rmlx_dir) / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(rec, indent=2))

    cmd = [rmlx_bin, "metrics"]
    if db:
        cmd += ["--db", db]
    cmd += ["record", "--file", str(path)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"record rejected: {res.stderr.strip()}\n  buffer kept at {path}")
    return path
