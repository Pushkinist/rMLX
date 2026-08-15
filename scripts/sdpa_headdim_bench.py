#!/usr/bin/env python3
"""What MLX's SDPA dispatch costs as a function of `head_dim`.

Reproduces the measurement block in `docs/FFI.md`
§`scaled_dot_product_attention` → "Head-dim dispatch and the unfused fallback".
Re-run this whenever the pinned MLX pair moves: the numbers in that doc are
valid only for the metallib this script reports at startup.

MLX ships a fused prefill kernel (`steel_attention`) only at `head_dim`
64 / 80 / 128, and a fused decode kernel (`sdpa_vector`) only at 64 / 96 / 128 /
256. Everything else silently falls back to the composite graph in
`mlx/fast.cpp` — `matmul(q, k^T)` -> mask -> softmax -> matmul, with the
`[B, n_heads, L_q, L_k]` score tensor materialised. This measures the gap.

Three sections, each answering one question:

  prefill  How much does the fallback cost at prefill shapes (q_seq == kv_seq)?
           Criterion, fixed before the first run: at matched (B, H, L), going
           128 -> 256 doubles the FLOPs, so a fused kernel present at both
           widths lands near 2.0x. <= 2.4x means the gap costs <= ~20% and is a
           curiosity; >= 4.0x means >= 2x overhead and upstream work pays.
           `head_dim` 512 is the control: it is unfused like 256, so 512 / 256
           near 2.0x would mean "wider is slower" rather than "the kernel is
           missing".

  decode   Is decode affected? At `q_seq = 1` the score tensor is [H, 1, kL],
           so the fallback has no O(L^2) term. Cells below this host's dispatch
           floor (~200 us) cannot resolve a kernel-level difference and are
           reported as such rather than quoted.

  causal   How much work does each path actually perform? A fused kernel can
           skip fully-masked tiles; the composite path computes the whole
           rectangle and then masks it. `causal / unmasked` near 0.5 means the
           tiles are skipped, near or above 1.0 means they are not. This is
           what decides whether a throughput figure should be normalised by
           2*H*L^2*D or by 4*H*L^2*D.

Preconditions:

  - Python MLX whose bundled metallib matches the pinned Homebrew bottle. The
    startup banner prints both the version and the `steel_attention` /
    `sdpa_vector` inventory so a run records the toolchain it measured. Verify
    against the bottle with:
        xcrun metal-nm --defined-only "$(brew --prefix mlx)/lib/mlx.metallib"
  - Exclusive GPU. Hold the rMLX claim file or make sure nothing else is on the
    device; a competing process invalidates every cell.
  - Peak allocation reaches ~72 GB at the largest cell (the materialised score
    tensor). Drop 32768 from `--lengths` on a smaller machine.

Usage:

    python3 scripts/sdpa_headdim_bench.py                 # all three sections
    python3 scripts/sdpa_headdim_bench.py --json out.json
    python3 scripts/sdpa_headdim_bench.py --sections prefill --lengths 2048,8192
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import mlx.core as mx

# (label, n_q_heads, n_kv_heads) — the real GQA ratios of our test targets.
SHAPES = (
    ("8:1", 8, 1),  # gemma-4-e2b
    ("32:8", 32, 8),  # Ternary-Bonsai-8B
)
HEAD_DIMS = (128, 256, 512)
WARMUP = 2
ITERS = 5


def banner():
    """Print the toolchain a run measured, so its numbers stay attributable."""
    print(f"mlx {mx.__version__}  device {mx.default_device()}")
    libs = glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(mx.__file__)), "**", "*.metallib"),
        recursive=True,
    )
    if not libs:
        print("metallib: not found next to the mlx package — inventory unknown")
        return
    lib = libs[0]
    try:
        syms = subprocess.run(
            ["xcrun", "metal-nm", "--defined-only", lib],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"metallib {lib}: inventory unavailable ({exc})")
        return
    full = sorted({m for m in re.findall(r"steel_attention\w*_(bd\d+)_", syms)})
    vec = sorted({m for m in re.findall(r"sdpa_vector\w*?_(\d+)_\1", syms)}, key=int)
    print(f"metallib {lib}")
    print(f"  steel_attention (fused prefill): {', '.join(full) or 'none'}")
    print(f"  sdpa_vector (fused decode):      {', '.join(vec) or 'none'}")


def prewarm():
    """Force pipeline creation for every kernel this run will time.

    Metal builds a compute pipeline the first time a kernel shape is
    dispatched, and that one-time cost lands on whichever cell happens to be
    measured first — inflating it by ~2x at the small end, where it is the same
    order as the cell itself. Paying it up front on throwaway shapes keeps the
    first row comparable with the rest.
    """
    for d in HEAD_DIMS:
        for mask in ("causal", None):
            run(8, 1, 128, 128, d, mask)
            run(8, 1, 1, 128, d, mask)


def _best_ms(q, k, v, scale, mask):
    def once():
        mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask))

    for _ in range(WARMUP):
        once()
    mx.synchronize()
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        once()
        mx.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def run(hq, hkv, q_len, kv_len, d, mask, track_peak=False):
    """Time one SDPA shape. Returns (ms, peak_bytes|None)."""
    scale = d**-0.5
    q = mx.random.normal((1, hq, q_len, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hkv, kv_len, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hkv, kv_len, d)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    if track_peak:
        mx.reset_peak_memory()
    best = _best_ms(q, k, v, scale, mask)
    peak = mx.get_peak_memory() if track_peak else None
    del q, k, v
    mx.clear_cache()
    return best, peak


def prefill(lengths, rows):
    print("\n## prefill — q_seq == kv_seq == L, causal")
    print("\n| q:kv | L | head_dim | ms | peak MB | vs head_dim 128 |")
    print("|---|---|---|---|---|---|")
    for label, hq, hkv in SHAPES:
        for length in lengths:
            base = None
            for d in HEAD_DIMS:
                ms, peak = run(hq, hkv, length, length, d, "causal", track_peak=True)
                base = ms if d == 128 else base
                rows.append(
                    {
                        "section": "prefill",
                        "shape": label,
                        "L": length,
                        "D": d,
                        "ms": ms * 1e3,
                        "peak_mb": peak / 1e6,
                        "vs_d128": ms / base,
                    }
                )
                print(
                    f"| {label} | {length} | {d} | {ms * 1e3:.3f} | "
                    f"{peak / 1e6:.1f} | {ms / base:.2f}x |",
                    flush=True,
                )


def decode(lengths, rows):
    print("\n## decode — q_seq == 1, unmasked")
    print("\n| q:kv | kL | head_dim | us | GB/s KV read |")
    print("|---|---|---|---|---|")
    for label, hq, hkv in SHAPES:
        for kv_len in lengths:
            for d in HEAD_DIMS:
                ms, _ = run(hq, hkv, 1, kv_len, d, None)
                gbps = (2.0 * hkv * kv_len * d * 2) / ms / 1e9
                rows.append(
                    {
                        "section": "decode",
                        "shape": label,
                        "kL": kv_len,
                        "D": d,
                        "us": ms * 1e6,
                        "gbps": gbps,
                    }
                )
                print(
                    f"| {label} | {kv_len} | {d} | {ms * 1e6:.1f} | {gbps:.1f} |",
                    flush=True,
                )


def causal(lengths, rows):
    print("\n## causal ablation — does the path skip fully-masked tiles?")
    print("\n| q:kv | L | head_dim | causal ms | unmasked ms | ratio |")
    print("|---|---|---|---|---|---|")
    for label, hq, hkv in SHAPES:
        for length in lengths:
            for d in HEAD_DIMS:
                c, _ = run(hq, hkv, length, length, d, "causal")
                u, _ = run(hq, hkv, length, length, d, None)
                rows.append(
                    {
                        "section": "causal",
                        "shape": label,
                        "L": length,
                        "D": d,
                        "causal_ms": c * 1e3,
                        "unmasked_ms": u * 1e3,
                        "ratio": c / u,
                    }
                )
                print(
                    f"| {label} | {length} | {d} | {c * 1e3:.3f} | "
                    f"{u * 1e3:.3f} | {c / u:.3f} |",
                    flush=True,
                )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", help="also write every row to this path")
    ap.add_argument(
        "--sections",
        default="prefill,decode,causal",
        help="comma-separated subset of prefill,decode,causal",
    )
    ap.add_argument(
        "--lengths",
        default="2048,8192,32768",
        help="comma-separated prefill lengths; decode uses the last two",
    )
    args = ap.parse_args(argv)

    lengths = [int(x) for x in args.lengths.split(",")]
    wanted = [s.strip() for s in args.sections.split(",")]
    banner()
    prewarm()

    rows = []
    if "prefill" in wanted:
        prefill(lengths, rows)
    if "decode" in wanted:
        decode(lengths[-2:], rows)
    if "causal" in wanted:
        # One length is enough: the question is which side of 0.5 the ratio
        # falls on, and that is a property of the dispatch, not of L.
        causal(lengths[len(lengths) // 2 : len(lengths) // 2 + 1], rows)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwrote {len(rows)} rows to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
