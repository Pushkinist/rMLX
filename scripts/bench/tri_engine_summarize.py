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


# ------------------------------------------------------------------- geometry

def read_geometry(mlxdir):
    """KV geometry of the benched checkpoint, read from its own config.json.

    Three unlinked copies of this used to be typed in by hand -- a
    values-per-cell constant here, a bytes-per-token constant in the harness,
    and the harness's memory guard -- while `GGUF=` / `MLXDIR=` are documented
    overrides. Pointing the harness at a second checkpoint therefore produced a
    plausible bits/value column and a mis-sized memory guard, both silently.
    """
    with open(os.path.join(mlxdir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    n_layers = int(tc["num_hidden_layers"])
    n_q = int(tc.get("num_attention_heads", 0))
    kv_heads = int(tc.get("num_key_value_heads", n_q))
    head_dim = int(tc.get("head_dim") or (int(tc["hidden_size"]) // n_q))
    if not (n_layers and kv_heads and head_dim):
        raise SystemExit(f"incomplete KV geometry in {mlxdir}/config.json")
    # K and V values one cell (one token) holds across the whole stack.
    values_per_cell = n_layers * 2 * kv_heads * head_dim
    return {
        "n_layers": n_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "values_per_cell": values_per_cell,
        # f16/bf16 bytes per token: one byte pair per value.
        "kv_bytes_per_token": values_per_cell * 2,
    }


def geometry_args(a):
    """Geometry the caller passed through, or None. Ingests carry it onto the
    record so the table never has to guess which checkpoint a row came from."""
    if not a.mlxdir:
        return None
    return read_geometry(a.mlxdir)


# ------------------------------------------------------------------ llama.cpp

# llama.cpp's KV allocation line. It has been renamed across releases, and this
# is the only place the llama.cpp KV column comes from -- so a miss is reported
# as a miss, never as a row that merely has nothing in two columns. A scan that
# reports success while matching nothing is the same defect class as a gate that
# cannot fail.
KV_RE = re.compile(
    r"(?:llama_kv_cache\w*|llama_kv_self)\D*size\s*=\s*([0-9.]+)\s*MiB.*?"
    r"([0-9]+)\s*cells"
)


def kv_from_err(path):
    """Allocated KV bytes and the cell count they cover.

    Returns `(bytes, cells)` on a match and raises `KvLineNotFound` otherwise.
    Splitting "no such line" from "line parsed to zero" on purpose: collapsing
    them into a `None` the caller shrugs at is how a scan comes to report
    `status="ok"` for a row it could not read.
    """
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = KV_RE.search(line)
                if m:
                    mib, cells = float(m.group(1)), int(m.group(2))
                    if cells <= 0:
                        raise KvLineNotFound(
                            f"{os.path.basename(path)}: KV line reports {cells} cells")
                    return int(mib * 1024 * 1024), cells
    except OSError as e:
        raise KvLineNotFound(f"{os.path.basename(path)}: {e}") from e
    raise KvLineNotFound(
        f"{os.path.basename(path)}: no llama.cpp KV allocation line matched "
        f"/{KV_RE.pattern}/ -- llama.cpp renames this line across releases, and "
        "it is the only source of the KV column")


class KvLineNotFound(Exception):
    pass


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
    try:
        kv_bytes, cells = kv_from_err(os.path.join(a.raw, f"{a.tag}.tg.err"))
    except KvLineNotFound as e:
        print(rec(engine="llama.cpp", tag=a.tag, status="kv_line_not_found",
                  error=str(e)))
        return
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
    geom = geometry_args(a)
    print(rec(
        engine="llama.cpp",
        geometry=geom,
        kv_option=f"ctk={a.ctk}/ctv={a.ctv}/fa={a.fa}",
        ctx=int(a.ctx), prompt_tokens=n, gen_tokens=int(a.gen),
        decode_tps=dec, decode_tps_basis="measured",
        prefill_tps=pre, prefill_tps_basis="measured",
        ttft_ms={"median": n / pre["median"] * 1000.0, "n": pre["n"]},
        ttft_basis="derived (prompt_tokens / prefill t/s)",
        kv_bytes=kv_bytes, kv_cells=cells,
        kv_basis="measured (allocated for n_ctx cells, load log)",
        kv_bytes_per_token=(kv_bytes / cells) if kv_bytes and cells else None,
        build=tg[0].get("build_commit"), tag=a.tag, status="ok",
    ))


# ----------------------------------------------------------------------- rMLX

def ingest_rmlx(a):
    with open(os.path.join(a.raw, f"{a.tag}.json")) as f:
        d = json.load(f)
    m = d["metrics"]
    # The measured count, from the artifact just opened -- not a number the
    # caller typed. `--prompt-tokens` names a context bucket; what the fixture
    # tokenizes to through THIS checkpoint's chat template is smaller, and
    # `kv_bytes_per_token` is the column the whole comparison rests on. The
    # mlx-lm ingest beside this one has always read the measured value.
    n = d.get("prompt_tokens")
    if n is None:
        n = m.get("prompt_tokens", {}).get("median")
    if n is None:
        print(rec(engine="rMLX", kv_option=a.codec, tag=a.tag,
                  status="no_measured_prompt_tokens"))
        return
    n = int(n)
    # Same reasoning for the generation length: the artifact reports what was
    # generated, `--gen` reports what was asked for.
    gen = int(d.get("gen_tokens") or a.gen)
    total = n + gen
    kv = m["kv_cache_bytes"]["median"]
    digests = {r.get("token_digest") for r in d.get("runs_detail", [])}
    geom = geometry_args(a)
    print(rec(
        engine="rMLX", kv_option=a.codec,
        geometry=geom, binary_sha256=a.binary_sha256,
        ctx=int(a.ctx), prompt_tokens=n, gen_tokens=gen,
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
        # Every field the body below indexes, not just the first one. A probe
        # record that failed partway through KV accounting carries some of them
        # and would otherwise take the whole ingest down with a KeyError, after
        # earlier modes had already been appended to cells.jsonl.
        needed = ("decode_tps", "prefill_tps", "ttft_s", "kv_bytes_true",
                  "prompt_tokens", "out_hash")
        rows = [r for r in rows if all(k in r for k in needed)]
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
            geometry=geometry_args(a),
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
        # Values-per-cell is the benched checkpoint's own geometry, carried on
        # the record by whichever ingest wrote it. A row without it prints "-"
        # rather than borrowing another checkpoint's shape.
        vpc = (r.get("geometry") or {}).get("values_per_cell")
        bpv = None
        if r.get("kv_bytes") and r.get("kv_cells") and vpc:
            bpv = (r["kv_bytes"] * 8) / (r["kv_cells"] * vpc)
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
    p.add_argument("--ntok",
                   help="prompt token count. llama-bench is TOLD this number "
                        "rather than measuring it, so it is an input there; the "
                        "MLX engines report their own and this is ignored")
    p.add_argument("--gen")
    p.add_argument("--ctk")
    p.add_argument("--ctv")
    p.add_argument("--fa")
    p.add_argument("--codec")
    p.add_argument("--mlxdir",
                   help="checkpoint whose config.json the KV geometry is read "
                        "from; carried onto every record")
    p.add_argument("--binary-sha256",
                   help="sha256 of the engine binary that produced the raw "
                        "artifact, recorded so a row names the build it came from")
    p.add_argument("--geometry", metavar="MLXDIR",
                   help="print this checkpoint's KV geometry and exit")
    a = p.parse_args()
    if a.geometry:
        g = read_geometry(a.geometry)
        # Shell-consumable, one line, fixed field order.
        print(f"{g['n_layers']} {g['kv_heads']} {g['head_dim']} "
              f"{g['values_per_cell']} {g['kv_bytes_per_token']}")
        return
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
