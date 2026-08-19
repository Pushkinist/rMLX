#!/usr/bin/env bash
# Per-cell driver for cross-backend bench.
#
# ---------------------------------------------------------------------------
# Positional interface (unchanged, backward compatible):
#
#   .bench_cell.sh TAG MODEL_PATH BACKEND KV_QUANT BUDGET_MIN MODEL_DISK_GB QUANT_SIG
#
# Drives the Python CBB harness against an HTTP backend, parses one cold +
# one warm run, appends a single line to BENCH_PROGRESS.md, and (for rmlx)
# routes records into the canonical metrics DB via RMLX_METRICS_DB.
#
# ---------------------------------------------------------------------------
# Env-var override contract (cache-type bench matrix):
#
# When ANY of the following env vars is set non-empty, the script enters
# "cache-type mode". The rmlx backend then bypasses the HTTP serve + Python
# CBB harness path and instead invokes `rmlx baseline` directly N=
# (WARMUP_RUNS + MEASURED_RUNS) times, computes the median and sample-stddev
# of decode TPS over the last MEASURED_RUNS, then writes ONE §8.5 universal
# RunRecord JSON to <RMLX_HOME>/metrics/buffer/pending/ and ingests it via
# `rmlx metrics record --file`. Non-rmlx backends with these env vars set
# error out (out of scope for cache-type mode).
#
# When no env var is set, the script behaves identically to the pre-existing
# form (Python CBB harness path).
#
#   CTK            — --cache-type-k value (e.g. q8_g128). Treated as
#                    unset when empty or literal "auto".
#   CTV            — --cache-type-v value (e.g. q4_g64). Same auto handling.
#                    Collides with KV_QUANT (positional arg 4) iff it is
#                    not one of the placeholder presets — see code below.
#   PROMPT_ID      — overrides the default longctx_8k bench prompt. Mapped
#                    to --prompt-tokens N: "longctx_4k" → 4096, etc.
#   MAX_TOKENS     — overrides --max-tokens (decode budget). Default 32.
#   CTX_MAX        — overrides --ctx-max. Default 8192 in cache-type mode.
#   WARMUP_RUNS    — discarded runs before measurement. Default 2.
#   MEASURED_RUNS  — measured runs over which median + sample-stddev are
#                    computed. Default 3. If < 2, decode_stddev is 0.0.
#   LABEL_PREFIX   — overrides notes label so a campaign can group cells.
#                    Default "ctype-bench".
#   KV_CANON_OVERRIDE — when set, replaces the kv_quant string computed by
#                    the static CTK/CTV analysis below. Use this when the
#                    static logic cannot determine the canonical string (e.g.
#                    asymmetric-auto combos whose resolver result depends on
#                    the model). Example: KV_CANON_OVERRIDE=k8v4.
#
# Hard rule: if BOTH KV_QUANT (positional, not equal to "kv-flag-replaced")
# AND any of CTK/CTV are set, the script errors out with exit 4 — the
# canonical Mixed/Q8V4-via-flags collision policy mirrors the CLI's
# clap-level conflicts_with.
# ---------------------------------------------------------------------------
set -uo pipefail

TAG="$1"            # e2b/e4b/26b/31b/31b-paro
MODEL_PATH="$2"     # abs path to snapshot
BACKEND="$3"        # rmlx | mlx-lm-turboquant | omlx | paroquant
KV_QUANT="$4"       # k8v4|k8v8|planar|mixed | turboquant | native
BUDGET_MIN="$5"     # int minutes
MODEL_DISK_GB="$6"  # for harness tps-per-gb
WEIGHT_QUANT="$7"   # mxfp8 | paro | ... (whitelist)

PROGRESS=${RMLX_ROOT}/BENCH_PROGRESS.md
CBB=${CROSS_BENCH_ROOT:-../Cross-Backend-Bench}
RMLX_DIR=${RMLX_ROOT}

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/lib/identity.sh"
rmlx_export_identity "$RMLX_DIR/target/release/rmlx"
# git_sha is caller-supplied provenance, not part of RunIdentity — anchored
# to RMLX_DIR (never the process cwd) so a run from another checkout cannot
# stamp that repo's SHA into this one's observations.git_sha.
GIT_SHA="$(git -C "${RMLX_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
PORT=62265
CLAIM=/tmp/rmlx.${PORT}.claim
BUDGET_S=$((BUDGET_MIN * 60))

export RMLX_METRICS_DB="${RMLX_DIR}/.rmlx/metrics/runs.db"
export RMLX_REPO_ROOT="${RMLX_DIR}"

# --- env-var override resolution -------------------------------------------
# Detect cache-type mode: any of the new env vars set?
_CT_MODE=0
for _v in CTK CTV PROMPT_ID MAX_TOKENS CTX_MAX WARMUP_RUNS MEASURED_RUNS LABEL_PREFIX; do
  if [ -n "${!_v:-}" ]; then _CT_MODE=1; fi
done

# Normalize CTK/CTV "auto" → unset (no per-side override).
_CTK_EFF="${CTK:-}"
_CTV_EFF="${CTV:-}"
[ "$_CTK_EFF" = "auto" ] && _CTK_EFF=""
[ "$_CTV_EFF" = "auto" ] && _CTV_EFF=""

# Collision check (per D2): if user sets CTK/CTV AND KV_QUANT positional is
# a real preset (not the "kv-flag-replaced" placeholder), refuse.
if { [ -n "$_CTK_EFF" ] || [ -n "$_CTV_EFF" ]; } \
   && [ "$KV_QUANT" != "kv-flag-replaced" ] \
   && [ "$KV_QUANT" != "auto" ]; then
  echo "[$(date +%T)] CELL FAIL collision tag=$TAG: KV_QUANT='$KV_QUANT' AND CTK='$_CTK_EFF' CTV='$_CTV_EFF' both set" >&2
  exit 4
fi

# Effective values with defaults (cache-type mode defaults vs legacy defaults).
if [ "$_CT_MODE" -eq 1 ]; then
  _PROMPT_ID_EFF="${PROMPT_ID:-longctx_4k}"
  _MAX_TOKENS_EFF="${MAX_TOKENS:-32}"
  _CTX_MAX_EFF="${CTX_MAX:-8192}"
  _WARMUP_EFF="${WARMUP_RUNS:-2}"
  _MEASURED_EFF="${MEASURED_RUNS:-3}"
  _LABEL_PREFIX_EFF="${LABEL_PREFIX:-ctype-bench}"
else
  # Legacy mode preserves original hard-coded values for the Python harness.
  _PROMPT_ID_EFF="longctx_8k"
  _CTX_MAX_EFF=16384
fi

# Legacy env vars for the Python harness (preserved when not in CT mode).
export CBB_CTX_MAX="${_CTX_MAX_EFF}"
export CBB_PROMPT_NAME="${_PROMPT_ID_EFF}"

# Map KV alias → whitelist value used in §8.5 record / DB (legacy path).
case "$KV_QUANT" in
  mixed)      KV_FOR_DB=k8v4 ;;   # rMLX Mixed{K=8,V=4} → catalog k8v4
  turboquant) KV_FOR_DB=turbo4 ;; # mlx-lm-tq 8,4
  native)     KV_FOR_DB=none ;;   # paroquant native (no extra KV quant)
  *)          KV_FOR_DB="$KV_QUANT" ;;
esac

QUANT_SIG="${WEIGHT_QUANT} / ${KV_FOR_DB}"

MODEL_ID=$(basename "$MODEL_PATH")
TS_START=$(date +%s)
TS_HMS=$(date +%H:%M:%S)
LOG=/tmp/bench_${TAG}_${BACKEND}_${KV_QUANT}.log
HARNESS_LOG=/tmp/bench_harness_${TAG}_${BACKEND}_${KV_QUANT}.log

echo "[$(date +%T)] CELL START tag=$TAG backend=$BACKEND kv=$KV_QUANT model_id=$MODEL_ID ct_mode=$_CT_MODE" >&2

# --- Preflight kill ---
pkill -f "rmlx serve" 2>/dev/null
pkill -f mlx_lm 2>/dev/null
pkill -f paroquant 2>/dev/null
pkill -f omlx 2>/dev/null
sleep 5
rm -f /tmp/rmlx.*.claim 2>/dev/null

# ---------------------------------------------------------------------------
# CACHE-TYPE MODE: direct `rmlx baseline` loop, median + stddev,
# single §8.5 record. Only the rmlx backend is supported in this mode.
# ---------------------------------------------------------------------------
if [ "$_CT_MODE" -eq 1 ]; then
  if [ "$BACKEND" != "rmlx" ]; then
    echo "[$(date +%T)] CELL FAIL ct-mode-non-rmlx tag=$TAG backend=$BACKEND" >&2
    echo "[$TS_HMS] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | CT_MODE_NON_RMLX" >>"$PROGRESS"
    exit 5
  fi

  # Resolve prompt-tokens N from PROMPT_ID = longctx_<N/1024>k.
  case "$_PROMPT_ID_EFF" in
    longctx_*k)
      _PROMPT_K="${_PROMPT_ID_EFF#longctx_}"
      _PROMPT_K="${_PROMPT_K%k}"
      _PROMPT_TOKENS=$((_PROMPT_K * 1024))
      ;;
    *)
      echo "[$(date +%T)] CELL FAIL bad-prompt-id tag=$TAG PROMPT_ID=$_PROMPT_ID_EFF" >&2
      exit 6
      ;;
  esac

  TOTAL_RUNS=$((_WARMUP_EFF + _MEASURED_EFF))
  _MEASURED_TPS=()
  _LAST_LOAD_MS=0
  _LAST_TTFT_MS=0
  _LAST_RSS=0

  # Build the baseline invocation argv (no --record on per-iter; we emit ONE
  # synthesized record manually after the loop).
  _BASELINE_ARGS=(
    --model "$MODEL_PATH"
    --prompt-tokens "$_PROMPT_TOKENS"
    --max-tokens "$_MAX_TOKENS_EFF"
    --ctx-max "$_CTX_MAX_EFF"
    --device gpu
  )
  if [ -n "$_CTK_EFF" ] && [ -n "$_CTV_EFF" ]; then
    _BASELINE_ARGS+=(--cache-type-k "$_CTK_EFF" --cache-type-v "$_CTV_EFF")
  elif [ -n "$_CTK_EFF" ]; then
    _BASELINE_ARGS+=(--cache-type-k "$_CTK_EFF" --cache-type-v auto)
  elif [ -n "$_CTV_EFF" ]; then
    _BASELINE_ARGS+=(--cache-type-k auto --cache-type-v "$_CTV_EFF")
  else
    # No per-side override; use legacy --kv-quant if KV_QUANT is set non-default.
    if [ "$KV_QUANT" != "auto" ] && [ "$KV_QUANT" != "" ] && [ "$KV_QUANT" != "kv-flag-replaced" ]; then
      _BASELINE_ARGS+=(--kv-quant "$KV_QUANT")
    fi
  fi

  echo "[$(date +%T)] CT loop: total=$TOTAL_RUNS warmup=$_WARMUP_EFF measured=$_MEASURED_EFF args=${_BASELINE_ARGS[*]}" >&2

  for _i in $(seq 1 "$TOTAL_RUNS"); do
    # Each iteration writes its own log fragment.
    _ITER_LOG="${LOG}.iter${_i}"
    ( cd "$RMLX_DIR" && ./target/release/rmlx baseline "${_BASELINE_ARGS[@]}" ) >"$_ITER_LOG" 2>&1
    _EXIT=$?
    if [ "$_EXIT" -ne 0 ]; then
      # exit 78 = unsupported combo → record skip and bail with status info.
      echo "[$(date +%T)] CELL FAIL baseline-exit=$_EXIT iter=$_i tag=$TAG" >&2
      echo "[$TS_HMS] $TAG | $BACKEND | ctk=$_CTK_EFF ctv=$_CTV_EFF | success=false | coh=false | BASELINE_EXIT_$_EXIT" >>"$PROGRESS"
      # Cleanup before exit.
      rm -f /tmp/rmlx.*.claim 2>/dev/null
      exit "$_EXIT"
    fi

    # Parse the "baseline: model=... load=...ms TTFT=...ms TPS=... ..." line.
    _LINE=$(grep -E "^baseline: " "$_ITER_LOG" | tail -1 || true)
    if [ -z "$_LINE" ]; then
      echo "[$(date +%T)] CELL FAIL no-baseline-line iter=$_i log=$_ITER_LOG" >&2
      exit 7
    fi
    _LOAD_MS=$(echo "$_LINE" | grep -oE "load=[0-9.]+ms" | head -1 | sed -E 's/load=([0-9.]+)ms/\1/')
    _TTFT_MS=$(echo "$_LINE" | grep -oE "TTFT=[0-9.]+ms" | head -1 | sed -E 's/TTFT=([0-9.]+)ms/\1/')
    _TPS=$(echo "$_LINE" | grep -oE "TPS=[0-9.]+" | head -1 | sed -E 's/TPS=([0-9.]+)/\1/')
    _RSS=$(echo "$_LINE" | grep -oE "peak_rss=[0-9.]+MB" | head -1 | sed -E 's/peak_rss=([0-9.]+)MB/\1/')

    if [ "$_i" -le "$_WARMUP_EFF" ]; then
      echo "[$(date +%T)] CT iter=$_i (warmup) tps=$_TPS" >&2
    else
      echo "[$(date +%T)] CT iter=$_i (measured) tps=$_TPS" >&2
      _MEASURED_TPS+=("$_TPS")
      _LAST_LOAD_MS="$_LOAD_MS"
      _LAST_TTFT_MS="$_TTFT_MS"
      _LAST_RSS="$_RSS"
    fi
  done

  # --- Compute median + sample-stddev over measured set --------------------
  _MEDIAN_TPS=$(printf "%s\n" "${_MEASURED_TPS[@]}" | sort -n | awk '
    { a[NR]=$1 }
    END {
      n=NR;
      if (n==0) { print 0; exit }
      if (n%2==1) { print a[(n+1)/2] }
      else { printf "%.6f", (a[n/2]+a[n/2+1])/2.0 }
    }')

  if [ "${#_MEASURED_TPS[@]}" -lt 2 ]; then
    _STDDEV_TPS=0.0
  else
    _STDDEV_TPS=$(printf "%s\n" "${_MEASURED_TPS[@]}" | awk '
      { v[NR]=$1; s+=$1 }
      END {
        n=NR; if (n<2) { print 0; exit }
        m=s/n; ss=0;
        for (i=1;i<=n;i++) ss += (v[i]-m)*(v[i]-m);
        printf "%.6f", sqrt(ss/(n-1));
      }')
  fi

  # --- Resolve weight_quant via `rmlx info` (best-effort; fall back to arg) ---
  _WQ_EFF="$WEIGHT_QUANT"

  # --- Resolve canonical kv_quant string ------------------------------------
  # rmlx baseline --record (when used) writes the canonical KvQuant Display.
  # Since we synthesize the record ourselves, compute the same string here.
  # Resolution order (first match wins):
  #   1. KV_CANON_OVERRIDE env var — operator escape hatch.
  #   2. bf16/bf16 (both sides float)       → KvQuant::None  → "none".
  #   3. q8_g128 + tq4 (either side order)  → KvQuant::K8V4  → "k8v4".
  #   4. q8_g128 + planar4 (K must be q8)   → KvQuant::Planar → "planar".
  #   5. Affine q<bits>_g<group> on both sides → "mixed_k<kb>g<kg>_v<vb>g<vg>".
  #   6. Fallback to legacy KV_QUANT positional name.
  # Note: for asymmetric-auto combos (one side set, the other "auto") where
  # the resolver result depends on the model, set KV_CANON_OVERRIDE explicitly.
  _KV_CANON=""

  if [ -n "${KV_CANON_OVERRIDE:-}" ]; then
    _KV_CANON="$KV_CANON_OVERRIDE"
  elif [ -n "$_CTK_EFF" ] || [ -n "$_CTV_EFF" ]; then
    _CTK_FOR_KV="${_CTK_EFF:-bf16}"
    _CTV_FOR_KV="${_CTV_EFF:-bf16}"

    # Rule 2: both sides are bf16 (unset = model default = bf16 float cache).
    if [ "$_CTK_FOR_KV" = "bf16" ] && [ "$_CTV_FOR_KV" = "bf16" ]; then
      _KV_CANON="none"

    # Rule 3: TurboQuant 4-bit V — K=q8_g128, V=tq4 (or reverse, defensively).
    elif { [ "$_CTK_FOR_KV" = "q8_g128" ] && [ "$_CTV_FOR_KV" = "tq4" ]; } \
      || { [ "$_CTK_FOR_KV" = "tq4" ] && [ "$_CTV_FOR_KV" = "q8_g128" ]; }; then
      _KV_CANON="k8v4"

    # Rule 4: PlanarQuant — K=q8_g128, V=planar4.
    elif [ "$_CTK_FOR_KV" = "q8_g128" ] && [ "$_CTV_FOR_KV" = "planar4" ]; then
      _KV_CANON="planar"

    # Rule 5: pure affine q<bits>_g<group> on both sides.
    else
      _KB=$(echo "$_CTK_FOR_KV" | sed -nE 's/^q([0-9]+)_g[0-9]+$/\1/p')
      _KG=$(echo "$_CTK_FOR_KV" | sed -nE 's/^q[0-9]+_g([0-9]+)$/\1/p')
      _VB=$(echo "$_CTV_FOR_KV" | sed -nE 's/^q([0-9]+)_g[0-9]+$/\1/p')
      _VG=$(echo "$_CTV_FOR_KV" | sed -nE 's/^q[0-9]+_g([0-9]+)$/\1/p')
      if [ -n "$_KB" ] && [ -n "$_KG" ] && [ -n "$_VB" ] && [ -n "$_VG" ]; then
        _KV_CANON="mixed_k${_KB}g${_KG}_v${_VB}g${_VG}"
      fi
    fi
  fi

  # Rule 6: fallback — pass through the legacy KV_QUANT positional name.
  if [ -z "$_KV_CANON" ]; then
    case "$KV_QUANT" in
      mixed)      _KV_CANON="k8v4" ;;
      turboquant) _KV_CANON="k8v4" ;;
      native)     _KV_CANON="none" ;;
      "" | auto | "kv-flag-replaced") _KV_CANON="k8v8" ;;
      *)          _KV_CANON="$KV_QUANT" ;;
    esac
  fi

  # --- Identity: split snapshot path into namespace + model name ------------
  _ABS_MODEL=$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")
  _MODEL_BASENAME=$(basename "$MODEL_PATH")
  case "$_MODEL_BASENAME" in
    *__*)
      _NS="${_MODEL_BASENAME%%__*}"
      _MODEL_NAME="${_MODEL_BASENAME#*__}"
      ;;
    *)
      _NS="local"
      _MODEL_NAME="$_MODEL_BASENAME"
      ;;
  esac

  _TS_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  _HW_TAG="${RMLX_HARDWARE_TAG:-m5_max_128gb}"
  _NOTES="label=${_LABEL_PREFIX_EFF} ctk=${_CTK_EFF:-auto} ctv=${_CTV_EFF:-auto} warmup=${_WARMUP_EFF} measured=${_MEASURED_EFF}"

  # Read the prompt body (first ~256KB) from prompts/longctx_<N>k.json.
  _PROMPT_FILE="${RMLX_DIR}/prompts/${_PROMPT_ID_EFF}.json"
  if [ ! -f "$_PROMPT_FILE" ]; then
    echo "[$(date +%T)] CELL FAIL prompt-file-missing $_PROMPT_FILE" >&2
    exit 8
  fi

  _BUFFER_DIR="${RMLX_DIR}/.rmlx/metrics/buffer/pending"
  mkdir -p "$_BUFFER_DIR"

  # Bug-1: if median TPS < 0.5 the baseline run succeeded syntactically but
  # produced no decode output (e.g. Gemma4 Shared-KV rejects Mixed codec and
  # emits TPS=0.000). Emit a §8.5 record with value=0.0 and
  # notes containing status=runtime_fail, then exit 10 so the outer matrix
  # loop can distinguish this from a clean skip (78) or content failure.
  _TPS_FAIL=$(awk -v t="${_MEDIAN_TPS:-0}" 'BEGIN { print (t+0 < 0.5) ? "1" : "0" }')
  if [ "$_TPS_FAIL" = "1" ]; then
    echo "[$(date +%T)] CELL FAIL tps-below-threshold median=${_MEDIAN_TPS:-0} tag=$TAG" >&2
    echo "[$TS_HMS] $TAG | $BACKEND | kv=$_KV_CANON | success=false | coh=false | TPS_BELOW_THRESHOLD" >>"$PROGRESS"
    _FAIL_UNIQ=$(date -u +"%Y%m%dT%H%M%S%3N")-bench-$$-rtfail
    _FAIL_JSON="${_BUFFER_DIR}/${_FAIL_UNIQ}.json"
    PROMPT_FILE="$_PROMPT_FILE" \
    BACKEND_VAL="rmlx" \
    NS_VAL="$_NS" \
    MODEL_VAL="$_MODEL_NAME" \
    WQ_VAL="$_WQ_EFF" \
    KV_VAL="$_KV_CANON" \
    CTX_VAL="$_CTX_MAX_EFF" \
    PROMPT_NAME_VAL="$_PROMPT_ID_EFF" \
    TS_VAL="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
    HW_VAL="$_HW_TAG" \
    PROMPT_TOK_VAL="$_PROMPT_TOKENS" \
    MAX_TOK_VAL="$_MAX_TOKENS_EFF" \
    N_WARM_VAL="$_WARMUP_EFF" \
    N_MEAS_VAL="$_MEASURED_EFF" \
    NOTES_VAL="${_NOTES} status=runtime_fail tps=${_MEDIAN_TPS:-0}" \
    DESC_VAL="bench_cell ct ${TAG} ${_LABEL_PREFIX_EFF} runtime_fail" \
    GIT_SHA_VAL="$GIT_SHA" \
    python3 -c '
import json, os, sys
with open(os.environ["PROMPT_FILE"], "r") as f:
    body = json.load(f)
rec = {
    **json.loads(os.environ["RMLX_IDENTITY_JSON"]),
  # "unknown" is a fallback, never provenance — a checkout without .git
  # must not stamp git_sha at all.
  **({"git_sha": os.environ["GIT_SHA_VAL"]} if not os.environ["GIT_SHA_VAL"].startswith("unknown") else {}),
  "model_namespace": os.environ["NS_VAL"],
  "model": os.environ["MODEL_VAL"],
  "weight_quant": os.environ["WQ_VAL"],
  "kv_quant": os.environ["KV_VAL"],
  "ctx_max": int(os.environ["CTX_VAL"]),
  "prompt": {"name": os.environ["PROMPT_NAME_VAL"], "body": body},
  "ts_utc": os.environ["TS_VAL"],
  "prompt_tokens": int(os.environ["PROMPT_TOK_VAL"]),
  "max_tokens": int(os.environ["MAX_TOK_VAL"]),
  "temperature": 0.0,
  "n_warmups": int(os.environ["N_WARM_VAL"]),
  "n_measure": int(os.environ["N_MEAS_VAL"]),
  "notes": os.environ["NOTES_VAL"],
  "description": os.environ["DESC_VAL"],
  # A run that failed its throughput threshold measured no throughput. `null`
  # records the attempt (notes carry status=runtime_fail and the observed tps)
  # without filing a fabricated 0.0 that would rank as a measurement.
  "metrics": [
    {"name": "decode_tps_warm", "value": None, "stddev": None},
  ],
}
json.dump(rec, sys.stdout, indent=2)
' >"$_FAIL_JSON"
    if ! ( cd "$RMLX_DIR" && ./target/release/rmlx metrics record --file "$_FAIL_JSON" >/dev/null 2>&1 ); then
      echo "[$(date +%T)] WARN: failed to ingest runtime_fail record $_FAIL_JSON" >&2
    fi
    rm -f /tmp/rmlx.*.claim 2>/dev/null
    exit 10
  fi

  _UNIQ=$(date -u +"%Y%m%dT%H%M%S%3N")-bench-$$
  _OUT_JSON="${_BUFFER_DIR}/${_UNIQ}.json"

  # Build the §8.5 RunRecord JSON. Use python3 to embed the prompt body
  # safely (escapes nested JSON). decode_stddev is carried as MetricEntry's
  # native `stddev` field on decode_tps_warm (no new metric-registry entry
  # required — see cache-type deviation note in REPORT).
  PROMPT_FILE="$_PROMPT_FILE" \
  BACKEND_VAL="rmlx" \
  NS_VAL="$_NS" \
  MODEL_VAL="$_MODEL_NAME" \
  WQ_VAL="$_WQ_EFF" \
  KV_VAL="$_KV_CANON" \
  CTX_VAL="$_CTX_MAX_EFF" \
  PROMPT_NAME_VAL="$_PROMPT_ID_EFF" \
  TS_VAL="$_TS_UTC" \
  HW_VAL="$_HW_TAG" \
  PROMPT_TOK_VAL="$_PROMPT_TOKENS" \
  MAX_TOK_VAL="$_MAX_TOKENS_EFF" \
  N_WARM_VAL="$_WARMUP_EFF" \
  N_MEAS_VAL="$_MEASURED_EFF" \
  NOTES_VAL="$_NOTES" \
  DESC_VAL="bench_cell ct ${TAG} ${_LABEL_PREFIX_EFF}" \
  TPS_VAL="$_MEDIAN_TPS" \
  STDDEV_VAL="$_STDDEV_TPS" \
  TTFT_VAL="$_LAST_TTFT_MS" \
  LOAD_VAL="$_LAST_LOAD_MS" \
  RSS_VAL="$_LAST_RSS" \
  GIT_SHA_VAL="$GIT_SHA" \
  python3 -c '
import json, os, sys

def _num(raw):
    """An empty scrape is a missing measurement, not a zero."""
    raw = (raw or "").strip()
    return float(raw) if raw else None

with open(os.environ["PROMPT_FILE"], "r") as f:
    body = json.load(f)
rec = {
    **json.loads(os.environ["RMLX_IDENTITY_JSON"]),
  # "unknown" is a fallback, never provenance — a checkout without .git
  # must not stamp git_sha at all.
  **({"git_sha": os.environ["GIT_SHA_VAL"]} if not os.environ["GIT_SHA_VAL"].startswith("unknown") else {}),
  "model_namespace": os.environ["NS_VAL"],
  "model": os.environ["MODEL_VAL"],
  "weight_quant": os.environ["WQ_VAL"],
  "kv_quant": os.environ["KV_VAL"],
  "ctx_max": int(os.environ["CTX_VAL"]),
  "prompt": {"name": os.environ["PROMPT_NAME_VAL"], "body": body},
  "ts_utc": os.environ["TS_VAL"],
  "prompt_tokens": int(os.environ["PROMPT_TOK_VAL"]),
  "max_tokens": int(os.environ["MAX_TOK_VAL"]),
  "temperature": 0.0,
  "n_warmups": int(os.environ["N_WARM_VAL"]),
  "n_measure": int(os.environ["N_MEAS_VAL"]),
  "notes": os.environ["NOTES_VAL"],
  "description": os.environ["DESC_VAL"],
  # `or 0` coerced a scrape that came back empty into a measurement of zero.
  # A missed RSS scrape is not a 0 MB process, and a zero rate is not a rate:
  # send null and let the recorder write no row for that metric.
  "metrics": [
    {"name": "decode_tps_warm", "value": _num(os.environ["TPS_VAL"]),
     "stddev": _num(os.environ["STDDEV_VAL"])},
    {"name": "ttft_warm_ms",    "value": _num(os.environ["TTFT_VAL"])},
    {"name": "model_load_ms",   "value": _num(os.environ["LOAD_VAL"])},
    {"name": "peak_rss_mb",     "value": _num(os.environ["RSS_VAL"])},
  ],
}
json.dump(rec, sys.stdout, indent=2)
' >"$_OUT_JSON"

  # Ingest the record into runs.db via the canonical CLI path.
  if ! ( cd "$RMLX_DIR" && ./target/release/rmlx metrics record --file "$_OUT_JSON" ); then
    echo "[$(date +%T)] CELL FAIL metrics-record" >&2
    exit 9
  fi

  echo "[$TS_HMS] $TAG | $BACKEND | kv=$_KV_CANON ct=${_CTK_EFF:-auto}/${_CTV_EFF:-auto} | success=true | coh=true | median=${_MEDIAN_TPS} stddev=${_STDDEV_TPS} (n=${_MEASURED_EFF})" >>"$PROGRESS"

  # --- Release ---
  pkill -f "rmlx serve" 2>/dev/null
  sleep 1
  rm -f /tmp/rmlx.*.claim 2>/dev/null
  echo "[$(date +%T)] CELL DONE tag=$TAG backend=$BACKEND kv=$_KV_CANON median=${_MEDIAN_TPS} stddev=${_STDDEV_TPS}" >&2
  exit 0
fi

# ---------------------------------------------------------------------------
# LEGACY MODE (pre-existing behavior): unchanged HTTP-serve + Python CBB harness.
# ---------------------------------------------------------------------------

# --- Launch backend ---
SERVER_PID=0
WARMUP_MODEL=""

case "$BACKEND" in
  rmlx)
    ( cd "$RMLX_DIR" && ./target/release/rmlx serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --device gpu \
        --max-ctx 16384 \
        --kv-quant "$KV_QUANT" ) >"$LOG" 2>&1 &
    SERVER_PID=$!
    WARMUP_MODEL="$MODEL_ID"
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_ID"
    ;;
  mlx-lm-turboquant)
    "${MLX_LM_TURBOQUANT_ROOT:-../mlx-lm-turboquant}/.venv/bin/mlx_lm.server" \
        --model "$MODEL_PATH" \
        --kv-cache-quantization 8,4 \
        --quantized-kv-start 0 \
        --port "$PORT" \
        --max-tokens 8192 \
        --log-level WARNING >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_PATH"
    ;;
  omlx)
    mkdir -p /tmp/omlx_models
    find /tmp/omlx_models -mindepth 1 -maxdepth 1 -delete 2>/dev/null
    OMLX_MODEL_NAME="gemma4-${TAG}"
    ln -sfn "$MODEL_PATH" "/tmp/omlx_models/$OMLX_MODEL_NAME"
    "${OMLX_ROOT:-../oMLX}/.venv/bin/omlx" serve \
        --model-dir /tmp/omlx_models \
        --port "$PORT" \
        --api-key 1234 >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY="1234"
    HARNESS_MODEL="$OMLX_MODEL_NAME"
    ;;
  paroquant)
    ${PAROQUANT_ROOT:-../paroquant}/.venv/bin/python -m paroquant.cli.serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --log-level WARNING >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_PATH"
    ;;
  *)
    echo "unknown backend $BACKEND" >&2
    echo "[$TS_HMS] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | UNKNOWN_BACKEND" >>"$PROGRESS"
    exit 1
    ;;
esac

# --- Wait for /v1/models within budget ---
READY=0
DEADLINE=$((TS_START + BUDGET_S))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 2
  # Check process alive.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if [ -n "$API_KEY" ]; then
    curl -fsS -m 3 -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/models" >/dev/null 2>&1 && READY=1 && break
  else
    curl -fsS -m 3 "$BASE_URL/v1/models" >/dev/null 2>&1 && READY=1 && break
  fi
done

if [ "$READY" -ne 1 ]; then
  echo "[$(date +%T)] CELL FAIL server-not-ready tag=$TAG backend=$BACKEND kv=$KV_QUANT" >&2
  echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | SERVER_NOT_READY" >>"$PROGRESS"
  kill "$SERVER_PID" 2>/dev/null
  pkill -f "rmlx serve" 2>/dev/null; pkill -f mlx_lm 2>/dev/null; pkill -f omlx 2>/dev/null; pkill -f paroquant 2>/dev/null
  rm -f /tmp/rmlx.*.claim
  exit 2
fi

# --- Warmup (1 token) ---
if [ -n "$API_KEY" ]; then
  curl -fsS -m 30 -X POST -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
    "$BASE_URL/v1/chat/completions" \
    -d "{\"model\":\"$HARNESS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
    >/dev/null 2>&1 || true
else
  curl -fsS -m 30 -X POST -H 'content-type: application/json' \
    "$BASE_URL/v1/chat/completions" \
    -d "{\"model\":\"$HARNESS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
    >/dev/null 2>&1 || true
fi

# --- Run harness ---
REMAINING=$((DEADLINE - $(date +%s)))
if [ "$REMAINING" -lt 60 ]; then
  echo "[$(date +%T)] CELL FAIL pre-bench-out-of-budget tag=$TAG" >&2
  echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | OUT_OF_BUDGET" >>"$PROGRESS"
  kill "$SERVER_PID" 2>/dev/null
  pkill -f "rmlx serve" 2>/dev/null; pkill -f mlx_lm 2>/dev/null; pkill -f omlx 2>/dev/null; pkill -f paroquant 2>/dev/null
  rm -f /tmp/rmlx.*.claim
  exit 3
fi

cd "$CBB"
HARNESS_ARGS=(
  --backend "$BACKEND"
  --backend-version "head"
  --base-url "$BASE_URL"
  --model "$HARNESS_MODEL"
  --model-path "$MODEL_PATH"
  --quant "$QUANT_SIG"
  --device gpu
  --max-tokens 8192
  --runs 2
  --backend-pid "$SERVER_PID"
  --model-disk-gb "$MODEL_DISK_GB"
  --prompt-file "$RMLX_DIR/prompts/longctx_8k.json"
  --request-timeout 1800
)
if [ -n "$API_KEY" ]; then
  HARNESS_ARGS+=(--api-key "$API_KEY")
fi

timeout "${REMAINING}s" uv run python -m runners.run_one "${HARNESS_ARGS[@]}" >"$HARNESS_LOG" 2>&1
HRC=$?

# --- Parse harness output ---
COLD_MS=0; WARM_MS=0; DECODE_TPS=0; SUCCESS=true; COH=true
# Two lines from harness: [1/2] ttft=Xms decode=Y.YY ...
LINES=$(grep -E "^\[[0-9]+/[0-9]+\]" "$HARNESS_LOG" || true)
LINE_COUNT=$(echo "$LINES" | grep -c "^\[" || true)
if [ "$LINE_COUNT" -lt 1 ]; then
  SUCCESS=false; COH=false
fi
COLD_LINE=$(echo "$LINES" | sed -n '1p')
WARM_LINE=$(echo "$LINES" | sed -n '2p')
if [ -n "$COLD_LINE" ]; then
  COLD_MS=$(echo "$COLD_LINE" | grep -oE "ttft=[0-9.]+ms" | head -1 | sed -E 's/ttft=([0-9.]+)ms/\1/')
fi
if [ -n "$WARM_LINE" ]; then
  WARM_MS=$(echo "$WARM_LINE" | grep -oE "ttft=[0-9.]+ms" | head -1 | sed -E 's/ttft=([0-9.]+)ms/\1/')
  DECODE_TPS=$(echo "$WARM_LINE" | grep -oE "decode=[0-9.]+tps" | head -1 | sed -E 's/decode=([0-9.]+)tps/\1/')
fi
# Compute prefill tps = 8192 / (warm_ms/1000) = 8192*1000/warm_ms.
PREFILL_TPS=0
if [ -n "$WARM_MS" ] && [ "$(echo "$WARM_MS" | grep -c '^[0-9.]\+$')" -eq 1 ]; then
  PREFILL_TPS=$(awk -v t="$WARM_MS" 'BEGIN { if (t>0) printf "%.2f", 8192000.0/t; else print "0" }')
fi
# Success check per harness output.
if echo "$LINES" | grep -q "success=False"; then SUCCESS=false; fi
if echo "$LINES" | grep -q "err=other"; then SUCCESS=false; fi

# Coherence: harness output_first_64 is in jsonl, examine.
LATEST_JSONL=$(ls -t "$CBB"/metrics/runs/${BACKEND}_*.jsonl 2>/dev/null | head -1)
if [ -n "$LATEST_JSONL" ]; then
  FIRST64=$(python3 -c "import json; lines=open('$LATEST_JSONL').readlines(); rec=json.loads(lines[-1]); print((rec.get('output_first_64') or '')[:64])" 2>/dev/null)
  if [ -z "$FIRST64" ]; then COH=false; fi
fi

if [ "$HRC" -ne 0 ]; then SUCCESS=false; fi
# Zero-decode = backend bug (e.g. Mixed cache unsupported on this arch).
if [ "$(echo "$DECODE_TPS" | grep -c '^0\(\.0*\)\?$')" -eq 1 ]; then
  SUCCESS=false; COH=false
fi

# --- Append progress line ---
echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=$SUCCESS | coh=$COH | cold=${COLD_MS} warm=${WARM_MS} prefill=${PREFILL_TPS} decode=${DECODE_TPS}" >>"$PROGRESS"

# --- Release ---
kill "$SERVER_PID" 2>/dev/null
pkill -f "rmlx serve" 2>/dev/null
pkill -f mlx_lm 2>/dev/null
pkill -f omlx 2>/dev/null
pkill -f paroquant 2>/dev/null
sleep 3
rm -f /tmp/rmlx.*.claim

echo "[$(date +%T)] CELL DONE tag=$TAG backend=$BACKEND kv=$KV_QUANT success=$SUCCESS coh=$COH decode=$DECODE_TPS" >&2
exit 0
