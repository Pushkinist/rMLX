#!/usr/bin/env python3
"""5-model auto-resolution + regression smoke, over both `auto` surfaces.

For each of the 5 regression models, and for each of `--kv-quant auto` and
`--kv-preset auto`:
  1. Start `rmlx serve` with that flag.
  2. Wait for /v1/models to respond.
  3. On the `--kv-quant auto` pass only, send one /v1/chat/completions request
     and measure decode TPS.
  4. Grep the rMLX log for "resolved KV cache quant" and assert KV.
  5. Print one CSV row: surface, model, expected_kv, resolved_kv, baseline_tps,
     observed_tps, delta_pct, ok.

Both surfaces are swept because there is no useful sense in which they may
differ: they are two spellings of "you pick". `--kv-preset auto` used to run its
own resolver — a unified-memory decision tree that returned a "compressing"
preset under pressure — so the two could disagree, and on this hardware they
did. A unit test pins that they read the same constant; this is the check that
the constant is what a served model actually gets, on every architecture.

Single MLX server at a time — strict serial.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(
    os.environ.get("RMLX_ROOT")
    or Path(__file__).resolve().parents[2]
)
O_MODELS = Path(
    os.environ.get("RMLX_O_MODELS_ROOT")
    or ROOT.parents[1] / "open-models"
)
RMLX = str(ROOT / "target" / "release" / "rmlx")
PORT = 62265
HOST = "127.0.0.1"

# What `auto` must resolve to, on every architecture and on every surface. It is
# a single constant on purpose: a per-model column here is what let this smoke
# carry a wrong expectation for Bonsai (recorded K8V8, actually Mixed)
# unnoticed.
EXPECTED_KV = "None"

# The flag pair each `auto` surface is spelled with. Both must land on
# EXPECTED_KV; only the first is timed, because the TPS check is a
# "did serving collapse" floor and running it twice per model doubles the
# GPU time without adding a signal.
AUTO_SURFACES = [
    ("kv-quant", ["--kv-quant", "auto"]),
    ("kv-preset", ["--kv-preset", "auto"]),
]

# TPS anchors below were recorded while `auto` still resolved through the
# retired per-arch table, i.e. at a different codec for four of the five rows.
# They are kept as a coarse "did serving collapse" floor (the check is
# >= 0.95x), not as a codec-comparable baseline.
MODELS = [
    # (path_basename, baseline_tps, baseline_label)
    ("mlx-community__Qwen3.6-35B-A3B-8bit",    94.88, "Qwen3.5MoE 8b"),
    ("z-lab__Qwen3.6-27B-PARO",                26.66, "Qwen3.5MoE PARO"),
    ("prism-ml__Ternary-Bonsai-8B-mlx-2bit",   96.50, "Qwen3 dense 2bit"),
    ("mlx-community__gemma-4-e2b-it-mxfp8",   103.04, "Gemma4 small mxfp8"),
    ("mlx-community__medgemma-1.5-4b-it-8bit", 72.89, "Gemma3 affine8"),
]

PROMPT = (
    "Write a one-paragraph description of a llama. Be concise."
)


def url_get(path: str, timeout: float = 5.0):
    req = urllib.request.Request(f"http://{HOST}:{PORT}{path}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def url_post(path: str, body: dict, timeout: float = 120.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}{path}",
        data=data,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def wait_ready(timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            status, _ = url_get("/v1/models", timeout=2.0)
            if status == 200:
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(1.0)
    return False


def measure_tps(model_id: str, max_tokens: int = 64) -> float:
    """Send chat completions, measure decode TPS via SSE token-time deltas."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{PORT}/v1/chat/completions",
        data=data,
        headers={"content-type": "application/json"},
    )
    t_start = time.monotonic()
    token_times: List[float] = []
    with urllib.request.urlopen(req, timeout=180.0) as r:
        for line in r:
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = (obj.get("choices") or [None])[0]
            if not ch:
                continue
            delta = ch.get("delta") or {}
            if "content" in delta and delta["content"]:
                token_times.append(time.monotonic())

    if len(token_times) < 5:
        return 0.0
    # Drop first token from decode TPS (TTFT)
    decode_window = token_times[-1] - token_times[0]
    decode_tokens = len(token_times) - 1
    if decode_window <= 0:
        return 0.0
    return decode_tokens / decode_window


def grep_resolved_kv(log_path: str) -> Optional[str]:
    """Find the resolved KV from the rMLX log."""
    if not os.path.isfile(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "kv-quant=auto" in line and "resolved" in line:
                m = re.search(r"resolved=([A-Za-z0-9]+)", line)
                if m:
                    return m.group(1)
                m = re.search(r'"resolved":"([^"]+)"', line)
                if m:
                    return m.group(1)
    # Fallback: look for "resolved KV cache quant" lines elsewhere
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            if "resolved KV cache quant" in line:
                m = re.search(r"kv_quant=([A-Za-z0-9]+)", line)
                if m:
                    return m.group(1)
                m = re.search(r'"kv_quant":"([^"]+)"', line)
                if m:
                    return m.group(1)
    return None


def find_latest_log() -> Optional[str]:
    log_dir = f"{ROOT}/logs"
    if not os.path.isdir(log_dir):
        return None
    candidates = sorted(
        (
            os.path.join(log_dir, f)
            for f in os.listdir(log_dir)
            if f.endswith(".jsonl")
        ),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def serve_one(model_path: str, auto_flags: List[str]) -> Tuple[subprocess.Popen, str]:
    """Start rMLX serve under one `auto` surface, return (proc, log_path)."""
    # Discover the log path by snapshotting before/after.
    pre = set(os.listdir(f"{ROOT}/logs")) if os.path.isdir(f"{ROOT}/logs") else set()
    proc = subprocess.Popen(
        [
            RMLX, "serve",
            "--model", model_path,
            "--port", str(PORT),
            "--device", "gpu",
            *auto_flags,
            "--max-ctx", "8192",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=ROOT,
    )
    # Wait for log file to appear (it appears at start)
    log_path = None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if os.path.isdir(f"{ROOT}/logs"):
            now = set(os.listdir(f"{ROOT}/logs"))
            new = now - pre
            new_jsonl = [n for n in new if n.endswith(".jsonl")]
            if new_jsonl:
                log_path = os.path.join(f"{ROOT}/logs", sorted(new_jsonl)[-1])
                break
        time.sleep(0.2)
    if log_path is None:
        log_path = find_latest_log() or ""
    return proc, log_path


def stop_proc(proc: subprocess.Popen):
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    # Clean up claim file (the SIGINT handler may not always remove it).
    try:
        os.remove(f"/tmp/rmlx.{PORT}.claim")
    except FileNotFoundError:
        pass


def main():
    print("surface,model,expected_kv,resolved_kv,baseline_tps,observed_tps,delta_pct,ok")
    all_ok = True
    for basename, baseline_tps, label in MODELS:
        expected_kv = EXPECTED_KV
        path = str(O_MODELS / basename)
        if not os.path.isdir(path):
            for surface, _ in AUTO_SURFACES:
                print(f"{surface},{basename},{expected_kv},MISSING,{baseline_tps},0.0,nan,FAIL")
            all_ok = False
            continue

        for idx, (surface, auto_flags) in enumerate(AUTO_SURFACES):
            timed = idx == 0
            proc, log_path = serve_one(path, auto_flags)
            observed_tps = 0.0
            try:
                if not wait_ready(timeout_s=300):
                    print(
                        f"{surface},{basename},{expected_kv},NOT_READY,{baseline_tps},0.0,nan,FAIL",
                        flush=True,
                    )
                    all_ok = False
                    continue
                if timed:
                    # warm-up + measure: a longer warm primes the prompt cache
                    # + JIT (otherwise first-decode is dominated by setup, not
                    # decode steady-state).
                    try:
                        _ = measure_tps(basename, max_tokens=64)   # warm 1 (load + JIT)
                        _ = measure_tps(basename, max_tokens=128)  # warm 2
                        # Take the best of two measurement runs.
                        t1 = measure_tps(basename, max_tokens=128)
                        t2 = measure_tps(basename, max_tokens=128)
                        observed_tps = max(t1, t2)
                    except Exception as e:
                        print(f"# {basename} measure error: {e}", file=sys.stderr)
                        observed_tps = 0.0
            finally:
                stop_proc(proc)
                time.sleep(2)  # let the claim file release

            # The sentinel must NOT be spellable as a real codec: `auto` now
            # resolves to `None`, so a "NONE" fallback would make a failed log
            # grep compare equal to the expectation and pass this check
            # vacuously.
            resolved = grep_resolved_kv(log_path) or "NO_LOG_MATCH"
            kv_match = (resolved.lower() == expected_kv.lower())
            if timed:
                delta_pct = (
                    (observed_tps - baseline_tps) / baseline_tps * 100.0
                ) if baseline_tps else 0.0
                tps_ok = observed_tps >= baseline_tps * 0.95
                ok = kv_match and tps_ok
                tps_cols = f"{baseline_tps:.2f},{observed_tps:.2f},{delta_pct:+.2f}"
            else:
                ok = kv_match
                tps_cols = "n/a,n/a,n/a"
            if not ok:
                all_ok = False
            print(
                f"{surface},{basename},{expected_kv},{resolved},{tps_cols},"
                f"{'PASS' if ok else 'FAIL'}",
                flush=True,
            )

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
