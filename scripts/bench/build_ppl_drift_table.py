"""Render PPL × TPS × similarity-drift table for the report.

Inputs:
    metrics/ppl_drift/drift_table.json    (output similarity proxy from CBB summary.csv)
    metrics/ppl_drift/ppl_upstream.json   (mlx_lm.evaluate wikitext PPL, where supported)

Output:
    metrics/ppl_drift/final_table.md      Markdown table for the report
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(
    os.environ.get("RMLX_ROOT")
    or Path(__file__).resolve().parents[2]
)
PERF51 = ROOT / "metrics" / "ppl_drift"
drift = json.loads((PERF51 / "drift_table.json").read_text())
ppl = json.loads((PERF51 / "ppl_upstream.json").read_text())
ppl_by_model = {r["model"]: r for r in ppl}

ARCH_GROUP = {
    "mlx-community__Qwen3.6-35B-A3B-8bit": ("Qwen3.5MoE / affine8", "Q35MoE"),
    "z-lab__Qwen3.6-27B-PARO":              ("Qwen3.5MoE / paro4",   "Q35MoE"),
    "prism-ml__Ternary-Bonsai-8B-mlx-2bit": ("Qwen3 dense / 2bit",   "Q3D"),
    "mlx-community__medgemma-1.5-4b-it-8bit": ("Gemma3 / affine8",   "Gemma3"),
    "mlx-community__gemma-4-e2b-it-mxfp8":  ("Gemma4 small / mxfp8",  "Gemma4S"),
    "mlx-community__gemma-4-e4b-it-mxfp8":  ("Gemma4 small / mxfp8",  "Gemma4S"),
    "mlx-community__gemma-4-26b-a4b-it-mxfp8": ("Gemma4 MoE / mxfp8", "Gemma4MoE"),
    "mlx-community__gemma-4-31b-it-mxfp8":  ("Gemma4 dense / mxfp8",  "Gemma4D"),
    "z-lab__gemma-4-31B-it-PARO":           ("Gemma4 dense / paro4",   "Gemma4D"),
}


def main() -> None:
    rows = drift["rows"]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out = []
    out.append("# PPL × TPS × drift table\n")

    for model, arch in ARCH_GROUP.items():
        if model not in by_model:
            continue
        rows_m = by_model[model]
        ppl_rec = ppl_by_model.get(model, {})
        word_ppl = ppl_rec.get("wikitext.word_perplexity,none")
        bpb = ppl_rec.get("wikitext.bits_per_byte,none")
        ppl_status = ppl_rec.get("status", "n/a")

        out.append(f"## `{model}` — {arch[0]}")
        out.append(
            f"\n**Upstream wikitext PPL (mlx_lm.evaluate, limit=16)**: "
            f"word_ppl={word_ppl}  bpb={bpb}  ({ppl_status})\n"
        )

        out.append("| Backend | KV-quant | Decode TPS | Sim vs ref | Output (first 64) |")
        out.append("|---|---|---:|---:|---|")
        for r in rows_m:
            mark = " *(ref)*" if r["is_reference"] else ""
            esc_out = r["output"].replace("|", "\\|")[:60]
            out.append(
                f"| {r['backend']} | {r['kv_quant']} | "
                f"{r['decode_tps']:.2f} | {r['similarity_to_ref']:.2f}{mark} | "
                f"`{esc_out}...` |"
            )
        out.append("")

    Path(PERF51 / "final_table.md").write_text("\n".join(out))
    print(f"Wrote {PERF51 / 'final_table.md'}")


if __name__ == "__main__":
    main()
