#!/usr/bin/env bash
# rmlx-vs-fork.sh — G5 parity gate: does rMLX agree with the mlx-lm-turboquant
# fork on the first 32 tokens for real qwen-pm-style prompts at temperature=0?
#
# This is a GO/NO-GO harness. It does NOT flip any pi config. It launches one
# MLX backend at a time (Apple Silicon Metal is exclusive per process), drives
# the same prompts through it, captures completions, and (in --compare mode)
# tokenizes fork vs rMLX outputs with the model's own tokenizer to compute
# first-token / first-32-token agreement.
#
# Modes:
#   collect <arm>   Launch the backend for <arm>, run all prompts, append rows
#                   to metrics/parity_rmlx_vs_fork.jsonl, then kill + free Metal.
#                   <arm> = fork-fp16 | fork-turbo4 | rmlx-k8v4 | rmlx-k8v8
#
# DEVIATION (2026-05-18): the installed mlx-lm-turboquant fork venv (HEAD
# ee35b61) does NOT accept `--kv-cache-quantization 8,4` / `--quantized-kv-start`
# (the flags pi-tq-server hardcodes — that script is stale vs this venv). This
# fork only exposes TurboQuant KV via `--turbo-kv-bits` (PolarQuant K, max 4
# bits) + `--turbo-v-bits` (affine V). There is NO K8 path in this fork at all,
# confirming the CLAUDE.md "fork's 8,4 is a lie" landmine. The brief's "fork
# k8v4 baseline" is therefore unachievable as written. We use the fork's
# unambiguous deterministic default — fp16 KV (no turbo flags) — as the PRIMARY
# parity reference (no fabricated/guessed turbo mapping), and additionally
# capture the closest real turbo config (turbo-kv-bits 4 + turbo-v-bits 4) as
# informational divergence context. See the G5 report.
#   compare         Pure-Python: read the jsonl, tokenize, emit per-prompt and
#                   mean match rates + verdict to stdout. No backend launched.
#
# HARD rule: never run two MLX backends at once. Each `collect` fully tears the
# backend down (pkill + claim-file rm + sleep) before returning.
set -euo pipefail

# ----- constants ------------------------------------------------------------
REPO="${RMLX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RMLX_BIN="$REPO/target/release/rmlx"
FORK_PY="${MLX_LM_TURBOQUANT_ROOT:-../mlx-lm-turboquant}/.venv/bin/python"
MODEL_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
PORT_FILE="$HOME/.pi/agent/.fork-port"
PORT="$(cat "$PORT_FILE" 2>/dev/null || echo 62260)"
STATS_JSONL="$HOME/.qwen-team/stats.jsonl"
DEV_SYS_PROMPT="$HOME/.qwen-team/dev-system-prompt.md"
OUT_JSONL="$REPO/metrics/parity_rmlx_vs_fork.jsonl"
PROMPTS_JSON="$REPO/metrics/.parity_prompts.json"   # transient prompt set (rebuilt each collect)
LOGDIR="$REPO/logs/parity"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-240}"
MAX_TOKENS=32

mkdir -p "$REPO/metrics" "$LOGDIR" "$(dirname "$PROMPTS_JSON")"

free_metal() {
  pkill -f "rmlx serve" 2>/dev/null || true
  pkill -f mlx_lm 2>/dev/null || true
  pkill -f paroquant 2>/dev/null || true
  pkill -f omlx 2>/dev/null || true
  sleep 5
  rm -f "/tmp/rmlx.${PORT}.claim"
}

# ----- prompt set -----------------------------------------------------------
# stats.jsonl is task TELEMETRY (ts/feature_id/decision/...), it carries NO
# system/user/messages payload. So we fall back to synthetic prompts built from
# the REAL dev-system-prompt.md plus representative atomic code-task user msgs.
# The Python builder records which path was taken into the prompt set.
build_prompts() {
  "$FORK_PY" - "$STATS_JSONL" "$DEV_SYS_PROMPT" "$PROMPTS_JSON" <<'PY'
import json, os, sys
stats_path, sys_prompt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

real = []
if os.path.isfile(stats_path) and os.path.getsize(stats_path) > 0:
    with open(stats_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            # A real dispatch would carry messages / system / user. Telemetry does not.
            msgs = None
            if isinstance(d.get("messages"), list):
                msgs = d["messages"]
            elif d.get("system") or d.get("user"):
                msgs = []
                if d.get("system"):
                    msgs.append({"role": "system", "content": d["system"]})
                if d.get("user"):
                    msgs.append({"role": "user", "content": d["user"]})
            if msgs:
                # strip assistant/thinking — keep only system+user
                kept = [m for m in msgs if m.get("role") in ("system", "user")]
                if kept:
                    real.append(kept)

source = "real-stats.jsonl"
prompts = real[:5]

if not prompts:
    source = "fallback-synthetic(dev-system-prompt.md + synthetic code tasks)"
    sysmsg = ""
    if os.path.isfile(sys_prompt_path):
        sysmsg = open(sys_prompt_path).read()
    else:
        sysmsg = "You are Local Dev. You execute ONE atomic coding task per invocation."
    user_tasks = [
        # Representative atomic Rust coding task (matches qwen-pm dev workload).
        ("Task spec\n\n"
         "Add a function `clamp_temperature(t: f32) -> f32` to `src/sampler.rs` that "
         "returns 0.0 if t is negative, 2.0 if t > 2.0, else t unchanged.\n\n"
         "Files you may modify: src/sampler.rs\n"
         "Test command: cargo test sampler\n\n"
         "Output the DEV REPORT block per the format rules."),
        # Bug-fix style task.
        ("Task spec\n\n"
         "In `src/parser.rs`, function `parse_port` currently panics on empty input. "
         "Change it to return `Result<u16, ParseError>` and return "
         "`Err(ParseError::Empty)` for an empty string. Do not change other functions.\n\n"
         "Files you may modify: src/parser.rs\n"
         "Test command: cargo test parser\n\n"
         "Output the DEV REPORT block per the format rules."),
        # Ambiguity-trigger task (exercises the STATUS=ambiguous control path).
        ("Task spec\n\n"
         "Wire the new `RetryPolicy` into the dispatch loop in `src/dispatch.rs` so "
         "failed tasks are retried per policy.\n\n"
         "Files you may modify: src/dispatch.rs\n"
         "Test command: cargo test dispatch\n\n"
         "(Note: RetryPolicy is not defined in the read-only context.)\n\n"
         "Output the DEV REPORT block per the format rules."),
    ]
    prompts = [
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content": u}]
        for u in user_tasks
    ]

json.dump({"source": source, "prompts": prompts}, open(out_path, "w"))
print(f"[prompts] source={source} count={len(prompts)}", file=sys.stderr)
PY
}

# ----- wait for backend -----------------------------------------------------
wait_ready() {
  local pid="$1" deadline
  deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
  while (( $(date +%s) < deadline )); do
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    if ! ps -p "$pid" >/dev/null 2>&1; then
      echo "[parity] backend pid $pid died during startup" >&2
      return 1
    fi
    sleep 5
  done
  echo "[parity] backend not ready after ${READY_TIMEOUT_S}s" >&2
  return 1
}

# ----- one completion -------------------------------------------------------
# Posts prompt #idx, captures content+reasoning_content, decode TPS if exposed.
run_prompts_against_backend() {
  local arm="$1" kv_quant="$2"
  "$FORK_PY" - "$PROMPTS_JSON" "$OUT_JSONL" "$arm" "$kv_quant" \
      "$MODEL_PATH" "$PORT" "$MAX_TOKENS" <<'PY'
import json, sys, time, urllib.request

prompts_path, out_path, arm, kv_quant, model, port, max_tokens = (
    sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
    sys.argv[6], int(sys.argv[7]))
pset = json.load(open(prompts_path))
source, prompts = pset["source"], pset["prompts"]
url = f"http://localhost:{port}/v1/chat/completions"

with open(out_path, "a") as out:
    for i, messages in enumerate(prompts):
        body = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        dt = time.time() - t0
        choice = resp["choices"][0]["message"]
        # Backend-agnostic capture. At max_tokens=32 with enable_thinking the
        # think block never closes, so different backends stash the realized
        # token stream under different keys: the fork uses `reasoning`, OpenAI-
        # style uses `reasoning_content`, plain text uses `content`. Capture all
        # three; the comparator concatenates whatever is present.
        content = choice.get("content") or ""
        reasoning = (choice.get("reasoning")
                     or choice.get("reasoning_content") or "")
        usage = resp.get("usage") or {}
        ctok = usage.get("completion_tokens")
        tps = (ctok / dt) if (ctok and dt > 0) else None
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "arm": arm,
            "kv_quant": kv_quant,
            "prompt_source": source,
            "prompt_idx": i,
            "content": content,
            "reasoning_content": reasoning,
            "completion_tokens": ctok,
            "elapsed_s": round(dt, 3),
            "decode_tps": round(tps, 2) if tps else None,
        }
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(f"[parity] {arm} prompt#{i} done "
              f"({ctok} tok, {dt:.1f}s, tps={row['decode_tps']})",
              file=sys.stderr)
PY
}

# ----- collect: launch arm, run, tear down ----------------------------------
collect() {
  local arm="$1"
  build_prompts
  free_metal

  local ts run_log kv server_pid
  ts="$(date +%Y%m%d-%H%M%S)"
  run_log="$LOGDIR/${arm}-${ts}.log"

  case "$arm" in
    fork-fp16)
      # Fork PRIMARY reference: fp16 KV (no turbo flags). Deterministic, the
      # fork's true default — see DEVIATION header. This is the parity baseline.
      kv="fp16"
      echo "[parity] launching FORK mlx_lm.server (fp16 KV, no turbo) port $PORT" >&2
      nohup "$FORK_PY" -m mlx_lm.server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 32000 \
        --temp 0 \
        --chat-template-args '{"enable_thinking":true}' \
        --log-level WARNING \
        > "$run_log" 2>&1 &
      server_pid=$!
      ;;
    fork-turbo4)
      # Informational: closest REAL fork TurboQuant config (K PolarQuant 4-bit,
      # V affine 4-bit). Not the parity gate baseline — divergence context only.
      kv="turbo4"
      echo "[parity] launching FORK mlx_lm.server (turbo-kv-bits 4 / turbo-v-bits 4) port $PORT" >&2
      nohup "$FORK_PY" -m mlx_lm.server \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --turbo-kv-bits 4 \
        --turbo-v-bits 4 \
        --max-tokens 32000 \
        --temp 0 \
        --chat-template-args '{"enable_thinking":true}' \
        --log-level WARNING \
        > "$run_log" 2>&1 &
      server_pid=$!
      ;;
    rmlx-k8v4|rmlx-k8v8)
      kv="${arm#rmlx-}"
      local reg="/tmp/parity-rmlx-registry.json"
      cat > "$reg" <<EOF
{"models":[{"id":"$MODEL_PATH","path":"$MODEL_PATH"}]}
EOF
      echo "[parity] launching rMLX serve (--kv-quant $kv) port $PORT" >&2
      nohup "$RMLX_BIN" serve \
        --registry "$reg" \
        --port "$PORT" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant "$kv" \
        --max-ctx 32768 \
        --idle-timeout-secs 0 \
        --prompt-cache-slots 4 \
        --default-temperature 0.0 \
        > "$run_log" 2>&1 &
      server_pid=$!
      ;;
    *)
      echo "[parity] unknown arm: $arm" >&2; exit 2 ;;
  esac
  disown "$server_pid" 2>/dev/null || true

  if ! wait_ready "$server_pid"; then
    echo "[parity] $arm failed to become ready. Last log lines:" >&2
    tail -25 "$run_log" >&2
    free_metal
    exit 1
  fi
  echo "[parity] $arm ready (pid $server_pid). Driving prompts." >&2

  run_prompts_against_backend "$arm" "$kv"

  echo "[parity] $arm done — tearing down backend, freeing Metal." >&2
  free_metal
}

# ----- compare: tokenize jsonl, verdict -------------------------------------
compare() {
  "$FORK_PY" - "$OUT_JSONL" "$MODEL_PATH" <<'PY'
import json, sys
from collections import defaultdict

jsonl_path, model_path = sys.argv[1], sys.argv[2]

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)

# Latest row per (arm, prompt_idx) — re-runs append; newest wins.
rows = {}
src = None
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        src = d.get("prompt_source", src)
        rows[(d["arm"], d["prompt_idx"])] = d

prompt_idxs = sorted({k[1] for k in rows})


def stream(row):
    # Qwen3 emits thinking in reasoning_content. The fork inlines <think>..</think>
    # into content. To compare the same realized token stream we concatenate
    # reasoning then content (visible). Documented in the report.
    r = row.get("reasoning_content") or ""
    c = row.get("content") or ""
    return (r + c) if r else c


def toks(s):
    return tok.encode(s, add_special_tokens=False)


print(f"prompt_source: {src}")
print(f"prompts: {len(prompt_idxs)}  arms present: "
      f"{sorted({k[0] for k in rows})}\n")

baseline = "fork-fp16"  # see DEVIATION header — fork's true deterministic default
summary = {}
divergences = []

for arm in ("rmlx-k8v4", "rmlx-k8v8"):
    per_first = []
    per_frac = []
    print(f"=== {arm} vs {baseline} ===")
    for pi in prompt_idxs:
        b = rows.get((baseline, pi))
        a = rows.get((arm, pi))
        if not b or not a:
            print(f"  prompt#{pi}: MISSING ({'baseline' if not b else arm})")
            per_first.append(0.0)
            per_frac.append(0.0)
            continue
        bt = toks(stream(b))[:32]
        at = toks(stream(a))[:32]
        n = min(len(bt), len(at), 32)
        first = 1.0 if (n > 0 and bt[0] == at[0]) else 0.0
        match = sum(1 for x, y in zip(bt, at) if x == y)
        denom = max(len(bt), len(at), 1)
        frac = match / denom
        per_first.append(first)
        per_frac.append(frac)
        print(f"  prompt#{pi}: first_tok={'Y' if first else 'N'} "
              f"first32_match={frac:.3f} ({match}/{denom})")
        if frac < 1.0:
            di = next((j for j in range(min(len(bt), len(at)))
                       if bt[j] != at[j]), min(len(bt), len(at)))
            lo, hi = max(0, di - 5), di + 6
            divergences.append({
                "arm": arm, "prompt": pi, "div_idx": di,
                "fork_tok": (tok.decode([bt[di]]) if di < len(bt) else "<end>"),
                "rmlx_tok": (tok.decode([at[di]]) if di < len(at) else "<end>"),
                "fork_ctx": tok.decode(bt[lo:hi]),
                "rmlx_ctx": tok.decode(at[lo:hi]),
            })
    mf = sum(per_first) / len(per_first) if per_first else 0.0
    mp = sum(per_frac) / len(per_frac) if per_frac else 0.0
    summary[arm] = (mf, mp)
    print(f"  MEAN first_tok={mf:.3f}  MEAN first32={mp:.3f}\n")

# Informational: fork's own turbo4 KV vs fork fp16 (how much the fork itself
# moves under KV quant — a yardstick for interpreting rMLX divergence).
if any(k[0] == "fork-turbo4" for k in rows):
    pf, pp = [], []
    print(f"=== [info] fork-turbo4 vs {baseline} ===")
    for pi in prompt_idxs:
        b = rows.get((baseline, pi))
        a = rows.get(("fork-turbo4", pi))
        if not b or not a:
            print(f"  prompt#{pi}: MISSING")
            continue
        bt = toks(stream(b))[:32]
        at = toks(stream(a))[:32]
        first = 1.0 if (bt and at and bt[0] == at[0]) else 0.0
        match = sum(1 for x, y in zip(bt, at) if x == y)
        denom = max(len(bt), len(at), 1)
        pf.append(first)
        pp.append(match / denom)
        print(f"  prompt#{pi}: first_tok={'Y' if first else 'N'} "
              f"first32_match={match/denom:.3f} ({match}/{denom})")
    if pf:
        print(f"  MEAN first_tok={sum(pf)/len(pf):.3f}  "
              f"MEAN first32={sum(pp)/len(pp):.3f}\n")

verdict = "FAIL"
winner = None
for arm in ("rmlx-k8v4", "rmlx-k8v8"):
    mf, mp = summary.get(arm, (0.0, 0.0))
    if mf >= 1.0 and mp >= 0.95:
        verdict = "PASS"
        # prefer k8v8 (rMLX default / faster) if it also passes
        if winner is None or arm == "rmlx-k8v8":
            winner = arm

print("================ VERDICT ================")
print(f"VERDICT: {verdict}")
if verdict == "PASS":
    mf, mp = summary[winner]
    print(f"WINNING_KV: {winner}  (mean first_tok={mf:.3f} "
          f"mean first32={mp:.3f})")
else:
    print("Neither kv-quant reached 100% first-tok + >=95% first-32.")
    print("\n--- DIVERGENCE TABLE ---")
    for d in divergences:
        print(f"[{d['arm']} p#{d['prompt']}] idx={d['div_idx']} "
              f"fork={d['fork_tok']!r} rmlx={d['rmlx_tok']!r}")
        print(f"   fork ctx: {d['fork_ctx']!r}")
        print(f"   rmlx ctx: {d['rmlx_ctx']!r}")
PY
}

# ----- dispatch -------------------------------------------------------------
case "${1:-}" in
  collect) shift; collect "${1:?usage: collect <fork-fp16|fork-turbo4|rmlx-k8v4|rmlx-k8v8>}" ;;
  compare) compare ;;
  *) echo "usage: $0 {collect <arm>|compare}" >&2; exit 2 ;;
esac
