#!/usr/bin/env python3
"""TurboQuant cross-arm decode + KV-residency probe.

Runs an identical decode loop under either the stock `mlx-lm` venv or the
`mlx-lm-turboquant` fork venv, so the only difference between arms is the
model implementation and the KV cache object -- NOT `generate.py`, whose
sampling / streaming code differs between the two trees and would otherwise
contaminate any A/B.

For every measured cell it reports:

  * decode TPS, measured over the generation loop only (prefill excluded),
  * TTFT (prefill wall time),
  * `kv_bytes_true`   -- every mx.array reachable from the cache objects,
                         i.e. packed store PLUS any dense dequant mirror,
  * `kv_bytes_claimed`-- what the cache's own `nbytes` property reports,
  * peak / active MLX memory around the cell,
  * a hash of the generated token ids, for output-identity checks.

`kv_bytes_true` vs `kv_bytes_claimed` is the whole point: a cache that keeps a
dense fp16 mirror alongside a packed store reports the store and resides the
sum.

Arm ordering is caller-controlled (`--seq`), so a palindromic sequence gives an
ABBA schedule inside one process with one model load.

Usage:
  <venv>/bin/python turbo_probe.py --model DIR --prompt-tokens 8192 \
      --seq fp16,turbo3,turbo3,fp16 --reps 3 --gen 128 --out results.jsonl
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load
from mlx_lm.models import cache as cache_mod


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_prompt(tokenizer, prompt_tokens):
    """Load prompts/longctx_<N>k.json and render it through the chat template."""
    n_k = prompt_tokens // 1024
    path = os.path.join(repo_root(), "prompts", f"longctx_{n_k}k.json")
    with open(path) as f:
        fixture = json.load(f)
    text = tokenizer.apply_chat_template(
        fixture["messages"], add_generation_prompt=True, tokenize=False
    )
    return mx.array(tokenizer.encode(text)), path


# ---------------------------------------------------------------- cache modes

def make_cache(model, mode):
    """Build a fresh cache for one arm. Unknown modes raise, never silently
    fall back -- a silent fallback would report the baseline under the
    quantized label."""
    if mode == "fp16":
        return cache_mod.make_prompt_cache(model)
    if mode.startswith("turbo"):
        # turbo3 / turbo4 / turbo3v4  (K bits, optional affine V bits)
        rest = mode[len("turbo"):]
        if "v" in rest:
            k_bits, v_bits = rest.split("v")
            v_bits = int(v_bits)
        else:
            k_bits, v_bits = rest, None
        k_bits = int(k_bits)
        if "turbo_kv_bits" not in cache_mod.make_prompt_cache.__code__.co_varnames:
            raise RuntimeError(
                "this mlx_lm has no turbo_kv_bits support -- wrong venv for "
                f"mode {mode}"
            )
        return cache_mod.make_prompt_cache(
            model, turbo_kv_bits=k_bits, turbo_v_bits=v_bits
        )
    if mode.startswith("mlxq"):
        # mlxq8 / mlxq4 -- stock mlx-lm QuantizedKVCache, the honest
        # in-ecosystem comparison point for "a KV cache that really is smaller".
        #
        # Built the way mlx-lm itself builds it: start from the model's own
        # cache and convert the entries that support conversion. Constructing
        # QuantizedKVCache for every layer instead would be wrong on a hybrid
        # attention/linear-attention model, whose non-attention layers hold an
        # ArraysCache that has no quantized form.
        return cache_mod.make_prompt_cache(model)
    raise ValueError(f"unknown cache mode: {mode}")


def maybe_quantize(cache, mode):
    """Convert convertible cache entries in place, mirroring mlx-lm's
    `maybe_quantize_kv_cache`. A no-op for every non-mlxq mode."""
    if not mode.startswith("mlxq"):
        return
    bits = int(mode[len("mlxq"):])
    for i, c in enumerate(cache):
        if hasattr(c, "to_quantized"):
            cache[i] = c.to_quantized(group_size=64, bits=bits)


# ------------------------------------------------------------ KV accounting

def _arrays_in(obj, seen):
    """Yield every mx.array held directly by obj's __dict__ / list slots."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, mx.array):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _arrays_in(v, seen)
        return
    d = getattr(obj, "__dict__", None)
    if d:
        for v in d.values():
            yield from _arrays_in(v, seen)


def kv_bytes_true(cache):
    """Sum nbytes over every distinct mx.array the cache list holds.

    Deduplicated by buffer identity is not possible through the public API, so
    this deduplicates by python object identity -- which is what we want: the
    packed store and the dense dequant mirror are distinct arrays and both
    count, while a returned view of a buffer already counted is the same
    object and is not double counted."""
    seen = set()
    total = 0
    for c in cache:
        for a in _arrays_in(c, seen):
            total += a.nbytes
    return total


def kv_bytes_claimed(cache):
    """What the caches report about themselves, when they report anything."""
    total = 0
    any_reported = False
    for c in cache:
        nb = getattr(c, "nbytes", None)
        if nb is None:
            # stock KVCache has no nbytes; derive from its keys/values
            k, v = getattr(c, "keys", None), getattr(c, "values", None)
            off = getattr(c, "offset", 0)
            if isinstance(k, mx.array):
                any_reported = True
                total += k[..., :off, :].nbytes
                if isinstance(v, mx.array):
                    total += v[..., :off, :].nbytes
            continue
        any_reported = True
        total += int(nb)
    return total if any_reported else None


def cache_kinds(cache):
    kinds = {}
    for c in cache:
        kinds[type(c).__name__] = kinds.get(type(c).__name__, 0) + 1
    return kinds


# ------------------------------------------------------- host memory guard

def system_state():
    """Available physical memory and free disk, read from the OS.

    `available` follows the same definition Activity Monitor uses for
    "memory available": free + inactive + purgeable + speculative. Pages the
    compressor holds are NOT available -- a page in the compressor has to be
    decompressed into a physical page before anyone can use it, which is exactly
    the situation that livelocks the machine when free pages run out.
    """
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = 16384
    m = re.search(r"page size of (\d+) bytes", out)
    if m:
        page = int(m.group(1))
    vals = {}
    for line in out.splitlines():
        mm = re.match(r'"?([^":]+)"?:\s+(\d+)', line)
        if mm:
            vals[mm.group(1).strip()] = int(mm.group(2))
    avail = page * (
        vals.get("Pages free", 0)
        + vals.get("Pages inactive", 0)
        + vals.get("Pages purgeable", 0)
        + vals.get("Pages speculative", 0)
    )
    st = os.statvfs(".")
    return {
        "avail_mem_bytes": avail,
        "free_pages_bytes": page * vals.get("Pages free", 0),
        "compressor_bytes": page * vals.get("Pages occupied by compressor", 0),
        "free_disk_bytes": st.f_bavail * st.f_frsize,
    }


# Refuse a cell unless this much is free beyond the cell's own estimated need.
HEADROOM_BYTES = 12 * 1024**3
# Below this, stop the whole run rather than start anything.
FLOOR_BYTES = 8 * 1024**3


class InsufficientMemory(Exception):
    pass


def guard_memory(est_bytes, label):
    st = system_state()
    need = 2 * est_bytes + HEADROOM_BYTES
    if st["avail_mem_bytes"] < FLOOR_BYTES:
        raise InsufficientMemory(
            f"host below floor: {st['avail_mem_bytes'] / 1024**3:.1f} GB available "
            f"< {FLOOR_BYTES / 1024**3:.0f} GB floor ({label})")
    if st["avail_mem_bytes"] < need:
        raise InsufficientMemory(
            f"cell would not fit: needs ~{need / 1024**3:.1f} GB "
            f"(2x est KV {est_bytes / 1024**3:.1f} GB + "
            f"{HEADROOM_BYTES / 1024**3:.0f} GB headroom), "
            f"host has {st['avail_mem_bytes'] / 1024**3:.1f} GB available ({label})")
    return st


# ----------------------------------------------------------------- decode

def prefill(model, cache, tokens, chunk, mode):
    """Chunked prefill; returns logits for the final position.

    Quantized modes convert after every chunk, as mlx-lm does, so the packed
    store -- not a full-length fp16 cache -- is what the prefill peak has to
    hold."""
    i = 0
    n = tokens.size
    while i < n - 1:
        end = min(i + chunk, n - 1)
        model(tokens[None, i:end], cache=cache)
        maybe_quantize(cache, mode)
        mx.eval([c.state for c in cache])
        i = end
    logits = model(tokens[None, n - 1: n], cache=cache)
    maybe_quantize(cache, mode)
    mx.eval(logits)
    return logits


def run_cell(model, tokens, mode, gen, chunk, est_kv_bytes=0):
    # Refuse the cell rather than push the host into a reclaim livelock. A cell
    # that cannot run inside the machine's free memory is a reportable result:
    # it is the context ceiling this campaign exists to measure.
    st_before = guard_memory(est_kv_bytes, f"{mode} @{tokens.size}")
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    active_before = mx.get_active_memory()

    cache = make_cache(model, mode)

    t0 = time.perf_counter()
    logits = prefill(model, cache, tokens, chunk, mode)
    t_prefill = time.perf_counter() - t0

    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)

    out = []
    t1 = time.perf_counter()
    for _ in range(gen):
        out.append(y)
        logits = model(y[None], cache=cache)
        maybe_quantize(cache, mode)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
    t_decode = time.perf_counter() - t1

    ids = [int(t.item()) for t in out]
    kv_true = kv_bytes_true(cache)
    kv_claim = kv_bytes_claimed(cache)
    kinds = cache_kinds(cache)
    peak = mx.get_peak_memory()
    active_after = mx.get_active_memory()

    del cache
    gc.collect()
    mx.clear_cache()

    st_after = system_state()
    return {
        "mode": mode,
        "avail_mem_before": st_before["avail_mem_bytes"],
        "avail_mem_after": st_after["avail_mem_bytes"],
        "compressor_after": st_after["compressor_bytes"],
        "free_disk_after": st_after["free_disk_bytes"],
        "prompt_tokens": int(tokens.size),
        "gen_tokens": gen,
        "ttft_s": round(t_prefill, 4),
        "prefill_tps": round((tokens.size - 1) / t_prefill, 2) if t_prefill else 0,
        "decode_tps": round(gen / t_decode, 4) if t_decode else 0,
        "decode_s": round(t_decode, 4),
        "kv_bytes_true": kv_true,
        "kv_bytes_claimed": kv_claim,
        "cache_kinds": kinds,
        "peak_mem_bytes": peak,
        "active_mem_before": active_before,
        "active_mem_after": active_after,
        "out_hash": hashlib.sha256(
            ",".join(map(str, ids)).encode()
        ).hexdigest()[:16],
        "out_head": ids[:8],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--seq", required=True,
                    help="comma-separated cache modes, one measured cell each; "
                         "use a palindrome for ABBA")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--warmup-tokens", type=int, default=4096)
    ap.add_argument("--warmup-gen", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True, help="label for this venv/arm")
    args = ap.parse_args()

    seq = args.seq.split(",")

    model, tokenizer = load(args.model)
    tokens, prompt_path = load_prompt(tokenizer, args.prompt_tokens)

    meta = {
        "record": "meta",
        "arm": args.arm,
        "mlx_version": mx.__version__,
        "mlx_lm_file": mlx_lm.__file__,
        "python": sys.executable,
        "model": args.model,
        "prompt_path": prompt_path,
        "prompt_tokens": int(tokens.size),
        "gen": args.gen,
        "seq": seq,
        "reps": args.reps,
    }
    with open(args.out, "a") as f:
        f.write(json.dumps(meta) + "\n")
    print(json.dumps(meta), flush=True)

    # Warm every distinct mode once on a short prompt: Metal kernel JIT is a
    # one-time per-process tax and would otherwise land entirely on whichever
    # arm happens to run first.
    warm = tokens[: args.warmup_tokens]
    # The warmup doubles as the sizing pass: measure this mode's KV bytes per
    # token on a short prompt, then scale linearly to the real prompt to decide
    # whether the real cell fits in the host's free memory.
    est = {}
    unrunnable = {}
    for mode in dict.fromkeys(seq):
        try:
            w = run_cell(model, warm, mode, args.warmup_gen, args.chunk)
            est[mode] = int(w["kv_bytes_true"] / w["prompt_tokens"] * tokens.size)
            print(f"warmed {mode}  est KV @{tokens.size} = "
                  f"{est[mode] / 1024**3:.2f} GB", flush=True)
        except Exception as e:
            # The warmup is also the sizing pass, and its result is the ONLY
            # input to the guard that keeps a full-length cell from pushing this
            # host into a reclaim livelock. Continuing would run that cell with
            # an estimate of zero, and an OOM on the short prompt is precisely
            # the failure that makes the long one unsafe.
            unrunnable[mode] = f"{type(e).__name__}: {e}"
            print(f"WARMUP FAILED {mode}: {unrunnable[mode]} -- "
                  "its cells will be skipped, not run unsized", flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps({"record": "error", "arm": args.arm,
                                    "mode": mode, "phase": "warmup",
                                    "error": unrunnable[mode]}) + "\n")

    for rep in range(args.reps):
        for pos, mode in enumerate(seq):
            if mode in unrunnable:
                r = {"mode": mode, "skipped": "warmup_failed",
                     "error": unrunnable[mode]}
                print(f"SKIP {mode}: warmup failed, no size estimate to guard on",
                      flush=True)
                r.update({"record": "cell", "arm": args.arm, "rep": rep,
                          "pos": pos, "t_wall": time.time()})
                with open(args.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
                continue
            try:
                r = run_cell(model, tokens, mode, args.gen, args.chunk,
                             est_kv_bytes=est[mode])
            except InsufficientMemory as e:
                # Not a failure of the codec -- a measurement of the ceiling.
                r = {"mode": mode, "skipped": "insufficient_memory",
                     "error": str(e), "est_kv_bytes": est[mode],
                     **system_state()}
                print(f"SKIP {mode}: {e}", flush=True)
            except Exception as e:
                r = {"mode": mode, "error": f"{type(e).__name__}: {e}"}
            r.update({"record": "cell", "arm": args.arm, "rep": rep,
                      "pos": pos, "t_wall": time.time()})
            with open(args.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
