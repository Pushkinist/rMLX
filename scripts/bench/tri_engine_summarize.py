#!/usr/bin/env python3
"""Normalize and tabulate the tri-engine same-model bench.

Two jobs, one file so the record shape has a single definition:

  --ingest-*  read one engine's raw artifact and print ONE normalized JSON
              record on stdout (the harness appends it to cells.jsonl),
  --table     read cells.jsonl and print the comparison table.

Every metric carries `n` and an observed range, and every field is labelled
`measured` or `derived`. A derived field is one no engine reported: llama.cpp's
TTFT (prompt tokens / prefill t/s) and every bytes-per-token figure.

llama.cpp reports the KV cache it ALLOCATED for n_ctx cells; rMLX and mlx-lm
report the bytes their filled prefix actually occupies. Those are only
comparable per-token, which is why bytes/token is the column that carries the
comparison and the raw byte count is reported beside it with its own basis.
"""

import argparse
import json
import os
import re
import statistics
import sys


def spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    med = statistics.median(vals)
    lo, hi = min(vals), max(vals)
    return {
        "median": med,
        "min": lo,
        "max": hi,
        "range_pct": (hi - lo) / med * 100 if med else 0.0,
        "n": len(vals),
    }


def rec(**kw):
    return json.dumps(kw)


# ------------------------------------------------------------------ llama.cpp

KV_RE = re.compile(
    r"llama_kv_cache: size =\s*([0-9.]+) MiB \(\s*(\d+) cells"
)


def kv_from_err(path):
    """Allocated KV bytes and the cell count they cover."""
    with open(path, errors="replace") as f:
        for line in f:
            m = KV_RE.search(line)
            if m:
                return int(float(m.group(1)) * 1024 * 1024), int(m.group(2))
    return None, None


def read_bench_jsonl(path):
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    return rows


def ingest_llama(a):
    tg = read_bench_jsonl(os.path.join(a.raw, f"{a.tag}.tg.jsonl"))
    pp = read_bench_jsonl(os.path.join(a.raw, f"{a.tag}.pp.jsonl"))
    kv_bytes, cells = kv_from_err(os.path.join(a.raw, f"{a.tag}.tg.err"))
    if not tg or not pp:
        print(rec(engine="llama.cpp", tag=a.tag, status="missing_raw"))
        return
    dec = spread(tg[0].get("samples_ts") or [])
    pre = spread(pp[0].get("samples_ts") or [])
    if dec is None or pre is None or not pre["median"]:
        # llama-bench wrote a row with no timing samples, or a zero prefill
        # rate. Deriving TTFT from that would print an infinity.
        print(rec(engine="llama.cpp", tag=a.tag, status="no_samples"))
        return
    n = int(a.ntok)
    print(rec(
        engine="llama.cpp",
        kv_option=f"ctk={a.ctk}/ctv={a.ctv}/fa={a.fa}",
        ctx=int(a.ctx), prompt_tokens=n, gen_tokens=int(a.gen),
        decode_tps=dec, decode_tps_basis="measured",
        prefill_tps=pre, prefill_tps_basis="measured",
        ttft_ms={"median": n / pre["median"] * 1000.0, "n": pre["n"]},
        ttft_basis="derived (prompt_tokens / prefill t/s)",
        kv_bytes=kv_bytes, kv_cells=cells,
        kv_basis="measured (allocated for n_ctx cells, load log)",
        kv_bytes_per_token=(kv_bytes / cells) if kv_bytes else None,
        build=tg[0].get("build_commit"), tag=a.tag, status="ok",
    ))


# ----------------------------------------------------------------------- rMLX

def ingest_rmlx(a):
    with open(os.path.join(a.raw, f"{a.tag}.json")) as f:
        d = json.load(f)
    m = d["metrics"]
    n = int(a.ntok)
    total = n + int(a.gen)
    kv = m["kv_cache_bytes"]["median"]
    digests = {r.get("token_digest") for r in d.get("runs_detail", [])}
    print(rec(
        engine="rMLX", kv_option=a.codec,
        ctx=int(a.ctx), prompt_tokens=n, gen_tokens=int(a.gen),
        decode_tps=m["decode_tps"], decode_tps_basis="measured",
        prefill_tps=m["prefill_tps"], prefill_tps_basis="measured",
        ttft_ms=m["ttft_ms"], ttft_basis="measured",
        kv_bytes=kv, kv_cells=total,
        kv_basis="measured (filled prefix, rmlx bench)",
        kv_bytes_per_token=kv / total,
        token_digests=sorted(d for d in digests if d),
        tag=a.tag, status="ok",
    ))


# --------------------------------------------------------------------- mlx-lm

def ingest_mlxlm(a):
    path = os.path.join(a.raw, f"mlxlm_{a.ctx}.jsonl")
    by_mode = {}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            # the file opens with a `meta` record describing the arm
            if r.get("record") != "cell":
                continue
            by_mode.setdefault(r["mode"], []).append(r)
    for mode, rows in by_mode.items():
        # A cell the probe refused (host memory) or that raised still carries
        # `record: "cell"` and none of the measured fields. Dropping it here is
        # the difference between a missing row and a KeyError that takes the
        # whole ingest down; the refusal itself is a result and stays in the raw
        # artifact.
        rows = [r for r in rows if "decode_tps" in r]
        if not rows:
            print(rec(engine="mlx-lm", kv_option=mode, ctx=int(a.ctx),
                      tag=f"mlxlm_{mode}_{a.ctx}", status="no_measured_cells"))
            continue
        n = rows[0]["prompt_tokens"]
        # mlx-lm's KVCache grows in 256-token steps; the allocation, not the
        # live length, is what its arrays hold.
        total = -(-(n + int(a.gen)) // 256) * 256
        kv_true = statistics.median([r["kv_bytes_true"] for r in rows])
        # A stock `KVCache` reports no `nbytes`, and the probe returns None
        # rather than inventing one. Median over a list holding None raises.
        claims = [r["kv_bytes_claimed"] for r in rows
                  if r.get("kv_bytes_claimed") is not None]
        kv_claim = statistics.median(claims) if claims else None
        print(rec(
            engine="mlx-lm", kv_option=mode,
            ctx=int(a.ctx), prompt_tokens=n, gen_tokens=int(a.gen),
            decode_tps=spread([r["decode_tps"] for r in rows]),
            decode_tps_basis="measured",
            prefill_tps=spread([r["prefill_tps"] for r in rows]),
            prefill_tps_basis="measured",
            ttft_ms=spread([r["ttft_s"] * 1000 for r in rows]),
            ttft_basis="measured",
            kv_bytes=kv_true, kv_cells=total,
            kv_basis="measured (every mx.array reachable from the cache)",
            kv_bytes_per_token=kv_true / total,
            kv_bytes_self_reported=kv_claim,
            out_hashes=sorted({r["out_hash"] for r in rows}),
            tag=f"mlxlm_{mode}_{a.ctx}", status="ok",
        ))


# ---------------------------------------------------------------------- table

def fmt(s, prec=1):
    if s is None:
        return "-"
    if isinstance(s, (int, float)):
        return f"{s:.{prec}f}"
    return f"{s['median']:.{prec}f} ±{s.get('range_pct', 0)/2:.1f}%"


# K/V values one cell (one token) holds on the fixed checkpoint this harness
# benches: 36 layers x 2 axes x 8 KV heads x 128 head_dim. The harness pins the
# checkpoint on purpose -- an engine-level difference is only readable when the
# weights do not move -- so this is a property of that choice, not a default.
CHECKPOINT_KV_VALUES_PER_CELL = 36 * 2 * 8 * 128


def table(a):
    if not a.cells:
        raise SystemExit("--table needs --cells <cells.jsonl>")
    rows = []
    with open(a.cells) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                rows.append(json.loads(line))
    rows = [r for r in rows if r.get("status") == "ok"]
    order = {"llama.cpp": 0, "rMLX": 1, "mlx-lm": 2}
    rows.sort(key=lambda r: (r["ctx"], order.get(r["engine"], 9), r["kv_option"]))
    hdr = ("| ctx | engine | KV option | decode TPS | TTFT ms | prefill tok/s "
           "| KV MiB | KV bits/value | n |")
    print(hdr)
    print("|" + "---|" * 9)
    for r in rows:
        kv_mib = r["kv_bytes"] / 1048576 if r.get("kv_bytes") else None
        bpv = None
        if r.get("kv_bytes") and r.get("kv_cells"):
            bpv = (r["kv_bytes"] * 8) / (r["kv_cells"] * CHECKPOINT_KV_VALUES_PER_CELL)
        n = r["decode_tps"]["n"] if r.get("decode_tps") else 0
        print("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["ctx"], r["engine"], r["kv_option"],
            fmt(r.get("decode_tps"), 2), fmt(r.get("ttft_ms"), 0),
            fmt(r.get("prefill_tps"), 0),
            fmt(kv_mib, 1), fmt(bpv, 2), n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ingest-llama", action="store_true")
    p.add_argument("--ingest-rmlx", action="store_true")
    p.add_argument("--ingest-mlxlm", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--cells")
    p.add_argument("--raw")
    p.add_argument("--tag")
    p.add_argument("--ctx")
    p.add_argument("--ntok")
    p.add_argument("--gen")
    p.add_argument("--ctk")
    p.add_argument("--ctv")
    p.add_argument("--fa")
    p.add_argument("--codec")
    a = p.parse_args()
    if a.ingest_llama:
        ingest_llama(a)
    elif a.ingest_rmlx:
        ingest_rmlx(a)
    elif a.ingest_mlxlm:
        ingest_mlxlm(a)
    elif a.table:
        table(a)
    else:
        p.error("pick a mode")


if __name__ == "__main__":
    sys.exit(main())
