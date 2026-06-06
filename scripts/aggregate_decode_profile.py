#!/usr/bin/env python3
"""Decode-profile aggregator.

Reads per-model decode_profile lines from `profile_<MODEL>.txt` files and emits
one CSV row per model with:

    model, arch, quant, n_steps, prefill_ms, forward_total_ms, eval_total_ms,
    step_total_ms, forward_per_step_ms, eval_per_step_ms, decode_tps

`decode_tps` is computed as `n_steps * 1000 / step_total_ms` (the closed-form
inverse of the per-step wall-clock measured inside generate_greedy). It will
match within rounding the runner's external bench TPS, which is the test
of self-consistency between Rust-side timers and Python-side stream timing.

Quant signature is read from the corresponding `bench_<MODEL>.log` file
(grep `quant_signature`).

Usage:
    python3 aggregate_decode_profile.py <profile_decode_dir>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Map model id -> (arch_class, quant_short).
ARCH_QUANT = {
    "prism-ml__Ternary-Bonsai-8B-mlx-2bit":            ("Qwen3 dense",     "affine g128 b2"),
    "mlx-community__DR-Venus-4B-RL-mlx-8Bit":          ("Qwen3 dense",     "affine g64 b8"),
    "mlx-community__medgemma-1.5-4b-it-8bit":          ("Gemma3",          "affine g64 b8"),
    "mlx-community__gemma-4-e2b-it-mxfp8":             ("Gemma4 small",    "mxfp8 g32 b8"),
    "mlx-community__gemma-4-e4b-it-mxfp8":             ("Gemma4 small",    "mxfp8 g32 b8"),
    "z-lab__Qwen3.6-27B-PARO":                         ("Qwen3.5 MoE PARO","paroquant int4"),
    "z-lab__gemma-4-31B-it-PARO":                      ("Gemma4 dense PARO","paroquant int4"),
    "mlx-community__gemma-4-26b-a4b-it-mxfp8":         ("Gemma4 MoE",      "mxfp8 g32 b8"),
    "mlx-community__gemma-4-31b-it-mxfp8":             ("Gemma4 dense",    "mxfp8 g32 b8"),
    "mlx-community__Laguna-XS.2-mxfp8":                ("Laguna",          "mxfp8 g32 b8"),
    "mlx-community__Qwen3.6-35B-A3B-8bit":             ("Qwen3.5 MoE",     "affine 8b"),
}


# Compact line emitted by tracing-subscriber:
#   ...INFO decode_profile: decode_profile arch="..." n_steps=31 prefill_ms=5031.7 ...
# stdout-fmt layer emits ANSI colour codes around the field markers; strip
# them before parsing.
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
KEY_RE = re.compile(r'(\w+)=("([^"]*)"|([0-9eE.+\-]+))')


def parse_profile_line(line: str) -> dict[str, str | float] | None:
    if "decode_profile" not in line:
        return None
    line = ANSI_RE.sub("", line)
    out: dict[str, str | float] = {}
    for m in KEY_RE.finditer(line):
        key = m.group(1)
        if m.group(3) is not None:
            out[key] = m.group(3)
        else:
            try:
                out[key] = float(m.group(4))
            except ValueError:
                out[key] = m.group(4)
    # Sanity: must have at least n_steps + step_total_ms.
    if "n_steps" not in out or "step_total_ms" not in out:
        return None
    return out


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: aggregate_decode_profile.py <profile_decode_dir>",
              file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        print(f"not a directory: {out_dir}", file=sys.stderr)
        sys.exit(2)

    cols = [
        "model", "arch", "quant", "n_steps",
        "prefill_ms", "forward_total_ms", "eval_total_ms", "step_total_ms",
        "forward_per_step_ms", "eval_per_step_ms", "decode_tps",
    ]
    print(",".join(cols))

    for model, (arch, quant) in ARCH_QUANT.items():
        path = out_dir / f"profile_{model}.txt"
        if not path.exists() or path.stat().st_size == 0:
            print(f"{model},{arch},{quant},,,,,,,,",
                  file=sys.stdout)
            continue
        rec = parse_profile_line(path.read_text())
        if rec is None:
            print(f"{model},{arch},{quant},,,,,,,,",
                  file=sys.stdout)
            continue

        try:
            n = int(rec["n_steps"])
            step_ms = float(rec["step_total_ms"])
            decode_tps = (n * 1000.0 / step_ms) if step_ms > 0 else 0.0
        except (KeyError, TypeError, ValueError):
            decode_tps = 0.0
            n = 0

        n_steps_val = rec.get("n_steps", "")
        if isinstance(n_steps_val, float):
            n_steps_val = int(n_steps_val)

        row = [
            model,
            arch,
            quant,
            str(n_steps_val),
            f'{rec.get("prefill_ms", ""):.3f}' if isinstance(rec.get("prefill_ms"), float) else "",
            f'{rec.get("forward_total_ms", ""):.3f}' if isinstance(rec.get("forward_total_ms"), float) else "",
            f'{rec.get("eval_total_ms", ""):.3f}' if isinstance(rec.get("eval_total_ms"), float) else "",
            f'{rec.get("step_total_ms", ""):.3f}' if isinstance(rec.get("step_total_ms"), float) else "",
            f'{rec.get("forward_per_step_ms", ""):.4f}' if isinstance(rec.get("forward_per_step_ms"), float) else "",
            f'{rec.get("eval_per_step_ms", ""):.4f}' if isinstance(rec.get("eval_per_step_ms"), float) else "",
            f"{decode_tps:.2f}",
        ]
        print(",".join(row))


if __name__ == "__main__":
    main()
