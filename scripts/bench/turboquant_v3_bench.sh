#!/usr/bin/env bash
# TurboQuant V3 (Lloyd-Max 3-bit) vs Mixed{v_bits:3} affine vs k8v8 baseline.
# Delegates all logic to the embedded Python orchestrator below.
# Requires Python 3.9+ (for subprocess, json, statistics).
#
# Usage:
#   ./scripts/bench/turboquant_v3_bench.sh
#   MODEL_FILTER=e4b ./scripts/bench/turboquant_v3_bench.sh
#
# Required env:
#   RMLX_OMODELS_DIR  — path to your Open Models snapshots root (see .env.example)
#
# Optional env (all have defaults):
#   RMLX_BIN         — path to rmlx binary (default: <repo>/target/release-perf/rmlx)
#   LOG_DIR          — bench log directory (default: <repo>/.rmlx/bench/turboquant_v3)
#   PROMPT_FILE      — JSON prompt file (default: <repo>/prompts/longctx_16k.json)
#   PORT, WARMUP_RUNS, MEASURE_RUNS, MAX_TOKENS, CTX, MODEL_FILTER

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# RMLX binary
: "${RMLX_BIN:=$REPO_ROOT/target/release-perf/rmlx}"

# Log directory
: "${LOG_DIR:=$REPO_ROOT/.rmlx/bench/turboquant_v3}"

# Model paths — required, fail loudly if unset
: "${RMLX_OMODELS_DIR:?set RMLX_OMODELS_DIR to your Open Models snapshots root (see .env.example)}"

# Prompt file
: "${PROMPT_FILE:=$REPO_ROOT/prompts/longctx_16k.json}"

MODEL_FILTER="${MODEL_FILTER:-all}"
PORT="${PORT:-62285}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURE_RUNS="${MEASURE_RUNS:-3}"
MAX_TOKENS="${MAX_TOKENS:-100}"
# longctx_16k.json tokenizes to ~17148 tokens on Gemma4; use 17500 max-ctx with 100 gen headroom.
CTX="${CTX:-17500}"

export RMLX_BIN MODEL_FILTER PORT WARMUP_RUNS MEASURE_RUNS MAX_TOKENS CTX PROMPT_FILE
export LOG_DIR RMLX_OMODELS_DIR REPO_ROOT

exec python3 - "$@" <<'PYEOF'
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
import statistics
import tempfile
import pathlib
import datetime
import signal

RMLX_BIN        = os.environ["RMLX_BIN"]
MODEL_FILTER     = os.environ["MODEL_FILTER"]
PORT             = int(os.environ["PORT"])
WARMUP_RUNS      = int(os.environ["WARMUP_RUNS"])
MEASURE_RUNS     = int(os.environ["MEASURE_RUNS"])
MAX_TOKENS       = int(os.environ["MAX_TOKENS"])
CTX              = int(os.environ["CTX"])
PROMPT_FILE      = os.environ["PROMPT_FILE"]
LOG_BASE         = pathlib.Path(os.environ["LOG_DIR"])
OMODELS_DIR      = os.environ["RMLX_OMODELS_DIR"]
REPO_ROOT        = pathlib.Path(os.environ["REPO_ROOT"])

GIT_SHA = subprocess.run(
    ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True).stdout.strip() or "unknown"
RUN_TS = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
LOG_DIR = LOG_BASE / f"turboquant_v3_bench_{RUN_TS}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("e4b",  f"{OMODELS_DIR}/mlx-community__gemma-4-e4b-it-mxfp8"),
    ("26b",  f"{OMODELS_DIR}/mlx-community__gemma-4-26b-a4b-it-mxfp8"),
]
# kv_label, cli_extra_args (as list)
KV_MODES = [
    ("k8v8",   ["--kv-quant", "k8v8"]),
    ("mixed3", ["--kv-bits", "3"]),
    ("turbo3", ["--kv-quant", "k8vturbo3"]),
]

def log(msg):
    print(f"[turboquant-v3-bench] {msg}", flush=True, file=sys.stderr)

def load_prompt():
    pf = pathlib.Path(PROMPT_FILE)
    if pf.exists():
        return json.loads(pf.read_text())
    log(f"Prompt file not found ({PROMPT_FILE}), generating synthetic...")
    text = ("Explain the theory of large language models in detail. " * (CTX // 10))[:CTX * 4]
    return {"messages": [{"role": "user", "content": text}]}

PROMPT_DATA = load_prompt()

def preflight():
    log("Preflight: killing stale MLX processes...")
    for pat in ["rmlx serve", "mlx_lm", "paroquant", "omlx"]:
        subprocess.run(["pkill", "-f", pat], capture_output=True)
    time.sleep(5)
    claim = pathlib.Path(f"/tmp/rmlx.{PORT}.claim")
    claim.unlink(missing_ok=True)

def wait_health(pid, serve_log_path, timeout=300):
    url = f"http://127.0.0.1:{PORT}/health"
    for elapsed in range(0, timeout, 3):
        time.sleep(3)
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if b'"ok"' in r.read():
                    log(f"Server ready in {elapsed+3}s")
                    return True
        except Exception:
            pass
        # Check if process still alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            log(f"ERROR: server died — see {serve_log_path}")
            return False
    log(f"ERROR: health timeout after {timeout}s")
    return False

def run_one_request(model_name):
    payload = {
        "model": model_name,
        "messages": PROMPT_DATA["messages"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
    except Exception as e:
        log(f"Request error: {e}")
        return 0.0, 0, 0.0
    elapsed = time.time() - t0
    usage = body.get("usage", {})
    completion = usage.get("completion_tokens", 0)
    if completion == 0:
        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        completion = max(1, len(text) // 4)
    tps = completion / elapsed if elapsed > 0 else 0.0
    return tps, completion, elapsed

def read_kv_bytes(model_name, serve_log_path):
    # Tracing log uses ANSI-escaped format: kv_cache_bytes\x1b[0m\x1b[2m=\x1b[0m<N>
    # Strip ANSI first, then match plain kv_cache_bytes=<N>.
    try:
        import re
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        content = ansi_escape.sub('', pathlib.Path(serve_log_path).read_text())
        matches = re.findall(r'kv_cache_bytes=(\d+)', content)
        if matches:
            return int(matches[-1])
    except Exception:
        pass
    return 0

def run_cell(model_label, model_path, kv_label, kv_args):
    model_name = pathlib.Path(model_path).name
    log(f"=== Cell: model={model_label} kv={kv_label} ===")
    preflight()

    serve_log = LOG_DIR / f"serve_{model_label}_{kv_label}.log"
    cmd = [
        RMLX_BIN, "serve",
        "--model", model_path,
        "--port", str(PORT),
        "--host", "127.0.0.1",
        "--device", "gpu",
        "--max-ctx", str(CTX),
    ] + kv_args

    log(f"Starting server: {' '.join(kv_args)}, max-ctx={CTX}")
    with open(serve_log, "w") as flog:
        proc = subprocess.Popen(cmd, stdout=flog, stderr=flog)

    if not wait_health(proc.pid, serve_log):
        proc.kill()
        proc.wait()
        return None, None

    # Warmup
    for i in range(WARMUP_RUNS):
        log(f"  warmup {i+1}/{WARMUP_RUNS}...")
        run_one_request(model_name)

    # Measure
    tps_list = []
    for i in range(MEASURE_RUNS):
        tps, toks, secs = run_one_request(model_name)
        log(f"  measure {i+1}/{MEASURE_RUNS}: tps={tps:.2f} ({toks} tok in {secs:.2f}s)")
        tps_list.append(tps)

    median_tps = statistics.median(tps_list)
    log(f"  median TPS: {median_tps:.2f}")

    kv_bytes = read_kv_bytes(model_name, serve_log)
    kv_mb = f"{kv_bytes/1024/1024:.1f}" if kv_bytes > 0 else "n/a"
    log(f"  KV-cache bytes: {kv_bytes} ({kv_mb} MB)")

    proc.kill()
    proc.wait()

    return median_tps, kv_mb

# === Main ===
results = {}  # (model_label, kv_label) -> (tps, kv_mb)

for model_label, model_path in MODELS:
    if MODEL_FILTER != "all" and MODEL_FILTER not in model_label:
        log(f"Skipping {model_label} (MODEL_FILTER={MODEL_FILTER})")
        continue

    if not pathlib.Path(model_path).is_dir():
        log(f"WARNING: model path not found, skipping: {model_path}")
        continue

    for kv_label, kv_args in KV_MODES:
        tps, kv_mb = run_cell(model_label, model_path, kv_label, kv_args)
        results[(model_label, kv_label)] = (tps, kv_mb)
        log(f"Cell done: {model_label}/{kv_label} TPS={tps} KV={kv_mb}MB")
        time.sleep(30)

# === Summary ===
print()
print("=" * 80)
print(f"TurboQuant V3 (Lloyd-Max 3-bit) vs affine 3-bit vs k8v8")
print(f"git_sha={GIT_SHA}, ts={RUN_TS}, ctx={CTX}, max_tokens={MAX_TOKENS}")
print("=" * 80)
print(f"{'model':<10} | {'kv_mode':<8} | {'median_tps':>12} | {'kv_mb':>8} | {'vs_mixed3':>10} | decision")
print("-" * 80)

for model_label, model_path in MODELS:
    if MODEL_FILTER != "all" and MODEL_FILTER not in model_label:
        continue
    if not pathlib.Path(model_path).is_dir():
        continue

    tps_mixed3, _ = results.get((model_label, "mixed3"), (None, None))

    for kv_label, _ in KV_MODES:
        tps, kv_mb = results.get((model_label, kv_label), (None, "n/a"))
        tps_str = f"{tps:.2f}" if tps is not None else "n/a"

        vs_mixed3 = "—"
        decision  = "—"
        if kv_label == "turbo3" and tps is not None and tps_mixed3 is not None and tps_mixed3 > 0:
            delta_pct = (tps - tps_mixed3) / tps_mixed3 * 100
            within_2pct = tps >= tps_mixed3 * 0.98
            vs_mixed3 = f"{delta_pct:+.1f}%"
            tps_gate = "TPS_OK" if within_2pct else "TPS_FAIL"
            decision = f"{tps_gate} PPL=DEFER"

        print(f"{model_label:<10} | {kv_label:<8} | {tps_str:>12} | {kv_mb or 'n/a':>8} | {vs_mixed3:>10} | {decision}")

    print("-" * 80)

print()
print("Decision criteria:")
print("  Promote k8vturbo3 iff PPL(turbo3) <= PPL(mixed3) - 0.3 AND TPS(turbo3) >= TPS(mixed3) * 0.98")
print("  PPL bench: DEFERRED — no native rMLX PPL harness (see docs/research/turboquant_v3_vs_affine_v3.md)")
print()
print(f"Logs: {LOG_DIR}/")
print("=" * 80)
PYEOF
