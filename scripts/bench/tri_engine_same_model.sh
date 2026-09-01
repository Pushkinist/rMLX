#!/usr/bin/env bash
# Engine-vs-engine bench on ONE model checkpoint: llama.cpp (GGUF) vs rMLX (MLX)
# vs stock mlx-lm (MLX), across each engine's KV-cache options.
#
# Every published cross-engine number this repo held before this script compared
# different models or different weight formats. This one holds the checkpoint
# fixed (Qwen/Qwen3-8B) and varies only the engine and its KV option, so an
# engine-level difference is not confounded by the weights.
#
# It is NOT a controlled quantization experiment: GGUF Q8_0 (symmetric, block
# 32, scale-only) and MLX 8-bit affine (asymmetric, group 64, scale+bias) are
# different schemes that happen to land on the same 8.50 bits/param. The
# preflight prints both file sizes so the reader can see how close they are.
#
# Prefill numbers from a llama.cpp build whose Metal tensor API is inert are
# worthless on M5+ (~3x low). This script refuses to emit a llama.cpp row unless
# the binary reports `has tensor = true`, and records the on/off control pair.
# ref: https://github.com/ggml-org/llama.cpp/issues/27473
#
# Usage:
#   tri_engine_same_model.sh preflight
#   tri_engine_same_model.sh llama  <ctx_tokens> [reps]
#   tri_engine_same_model.sh rmlx   <ctx_tokens> [runs]
#   tri_engine_same_model.sh mlxlm  <ctx_tokens> [reps]
#
# Results append to $OUT_DIR/cells.jsonl (default .rmlx/analysis/tri-engine/).
# Summarize with scripts/bench/tri_engine_summarize.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT/.rmlx/analysis/tri-engine}"
CELLS="$OUT_DIR/cells.jsonl"
RAW="$OUT_DIR/raw"
mkdir -p "$RAW"

# Snapshot roots resolve under RMLX_O_MODELS_ROOT (see docs/TESTING.md); the
# dev checkout points it at the sibling Open Models tree. Override GGUF/MLXDIR
# directly for a checkpoint that lives elsewhere.
MODELS_ROOT="${RMLX_O_MODELS_ROOT:-$ROOT/../../O-Models}"
GGUF="${GGUF:-$MODELS_ROOT/gguf/Qwen3-8B-Q8_0.gguf}"
MLXDIR="${MLXDIR:-$MODELS_ROOT/mlx-community__Qwen3-8B-8bit}"
LBENCH="${LBENCH:-}"
RMLX="${RMLX:-$ROOT/target/release-perf/rmlx}"
MLXLM_PY="${MLXLM_PY:-$ROOT/../mlx-lm/.venv/bin/python}"
PROBE="$ROOT/scripts/baseline/turbo_probe.py"
GEN="${GEN:-128}"

SUMMARIZE="$ROOT/scripts/bench/tri_engine_summarize.py"

# KV geometry, read once from the benched checkpoint's own config.json rather
# than typed in three places. `MLXDIR=` is a documented override, so a
# hand-written layers/kv_heads/head_dim would silently mis-size the memory guard
# and mis-scale the bits/value column the moment anyone used it.
read_geometry() {
    [ -f "$MLXDIR/config.json" ] || {
        echo "no config.json under $MLXDIR -- set MLXDIR" >&2; exit 2; }
    python3 "$SUMMARIZE" --geometry "$MLXDIR"
}
read -r GEOM_LAYERS GEOM_KV_HEADS GEOM_HEAD_DIM GEOM_VALUES_PER_CELL KV_B_PER_TOK \
    <<<"$(read_geometry)"
[ -n "${KV_B_PER_TOK:-}" ] || { echo "could not read KV geometry from $MLXDIR" >&2; exit 2; }
echo "geometry: ${GEOM_LAYERS}L x ${GEOM_KV_HEADS}kvh x ${GEOM_HEAD_DIM}d" \
     "= ${GEOM_VALUES_PER_CELL} values/token, ${KV_B_PER_TOK} f16 B/token" >&2

# Prompt-token count per context bucket, MEASURED through one checkpoint's chat
# template. Unlike the geometry above this is not derivable without running that
# tokenizer, so it is pinned to the checkpoint it was measured on and the run is
# refused for any other. rMLX and mlx-lm read the same prompts/longctx_<N>k.json
# fixture and report what they actually tokenized; llama-bench synthesizes
# tokens and is TOLD this integer, which is why it has to be right.
TOKENS_FOR_CHECKPOINT="mlx-community__Qwen3-8B-8bit"
tokens_for() {
    local have; have="$(basename "$MLXDIR")"
    if [ "$have" != "$TOKENS_FOR_CHECKPOINT" ]; then
        cat >&2 <<MSG
REFUSING: the prompt-token counts below were measured through
$TOKENS_FOR_CHECKPOINT's chat template, and MLXDIR is $have.
llama-bench is told this integer rather than measuring it, so a mismatch
produces a full, plausible row set at the wrong prompt length. Re-measure the
counts through the new checkpoint's tokenizer and update tokens_for() and
TOKENS_FOR_CHECKPOINT together.
MSG
        exit 7
    fi
    case "$1" in
        4096)   echo 3766   ;;
        32768)  echo 31549  ;;
        131072) echo 126752 ;;
        *) echo "unknown context bucket: $1" >&2; exit 2 ;;
    esac
}

# --- memory guard -----------------------------------------------------------
# This host kernel-panicked earlier in the campaign on a memory-reclaim livelock.
# A cell that cannot run without pushing the host into swap is a reportable
# result, not something to force.
avail_gb() {
    vm_stat | awk '
        /page size of/ { for (i=1;i<=NF;i++) if ($i=="of") { ps=$(i+1); break } }
        /Pages free/        { f=$3 }
        /Pages inactive/    { i=$3 }
        /Pages speculative/ { s=$3 }
        /Pages purgeable/   { p=$3 }
        END { gsub(/\./,"",f); gsub(/\./,"",i); gsub(/\./,"",s); gsub(/\./,"",p);
              printf "%.1f", (f+i+s+p) * ps / 1073741824 }'
}

require_mem() {  # require_mem <needed_gb> <label>
    local need="$1" label="$2" have
    have="$(avail_gb)"
    if awk -v h="$have" -v n="$need" 'BEGIN{exit !(h < n)}'; then
        echo "REFUSED $label: ${have} GiB reclaimable < ${need} GiB required" >&2
        emit "$(printf '{"tag":"%s","status":"refused_memory","avail_gb":%s,"need_gb":%s}' \
                "$label" "$have" "$need")"
        return 1
    fi
    echo "mem ok for $label: ${have} GiB reclaimable >= ${need} GiB" >&2
}

# `KV_B_PER_TOK` is the f16 bytes-per-token figure read out of config.json above.
est_gb() {  # est_gb <cells> <bytes_per_value_ratio>
    awk -v c="$1" -v r="${2:-1}" -v b=$KV_B_PER_TOK \
        'BEGIN{ printf "%.0f", (c*b*r)/1073741824 + 12 }'
}

emit() { printf '%s\n' "$1" >> "$CELLS"; }

# Two guards, because one is not enough. A snapshotted rMLX binary reports its
# process name as `rmlx_main`, so a list holding only `rmlx` misses it and both
# arms of a comparison silently become one contended process. `ps aux | grep`
# under this host's
# command hook has also been observed returning EMPTY while two llama-bench processes
# were live, so every check here uses pgrep directly with no shell pipeline, and
# an exclusive directory lock backs it up. A contended bench does not fail --
# it prints a plausible wrong number (an uncaught pair read 5.64 TPS where the
# uncontended figure was 37.0).
acquire_lock() {
    LOCK="${TMPDIR:-/tmp}/tri_engine_same_model.lock"
    if ! mkdir "$LOCK" 2>/dev/null; then
        echo "another tri_engine run holds $LOCK (pid $(cat "$LOCK/pid" 2>/dev/null)) -- refusing" >&2
        exit 5
    fi
    echo $$ > "$LOCK/pid"
    trap 'rm -rf "$LOCK"' EXIT INT TERM
}

no_other_mlx() {
    acquire_lock
    local p
    for p in llama-bench llama-cli llama-server rmlx rmlx_main; do
        if pgrep -x "$p" >/dev/null 2>&1; then
            echo "REFUSING: '$p' is already running (pids: $(pgrep -x "$p" | tr '\n' ' '))" >&2
            exit 3
        fi
    done
    for p in turbo_probe.py "mlx_lm."; do
        if pgrep -f "$p" >/dev/null 2>&1; then
            echo "REFUSING: a process matching '$p' is already running" >&2
            exit 3
        fi
    done
}

# --- preflight --------------------------------------------------------------
phase_preflight() {
    echo "=== weights ==="
    local gb mb
    gb=$(stat -f '%z' "$GGUF")
    mb=$(find "$MLXDIR" -name '*.safetensors' -not -path '*/.cache/*' \
         -exec stat -f '%z' {} \; | awk '{s+=$1} END{print s}')
    awk -v g="$gb" -v m="$mb" 'BEGIN{
        printf "gguf  %d bytes (%.4f GiB)\nmlx   %d bytes (%.4f GiB)\ndelta %.3f%%\n",
               g, g/1073741824, m, m/1073741824, 100*(g-m)/m }'

    echo "=== llama.cpp tensor API ==="
    local on off
    on=$("$LBENCH" --list-devices 2>&1 | awk '/has tensor/{print $NF}')
    off=$(GGML_METAL_TENSOR_DISABLE=1 "$LBENCH" --list-devices 2>&1 | awk '/has tensor/{print $NF}')
    echo "default=$on  with GGML_METAL_TENSOR_DISABLE=1=$off"
    [ "$on" = "true" ] || { echo "REFUSING: tensor API inert, every prefill number would be ~3x low" >&2; exit 4; }
    [ "$off" = "false" ] || { echo "REFUSING: GGML_METAL_TENSOR_DISABLE is not a working toggle" >&2; exit 4; }
    echo "=== memory ==="; echo "reclaimable: $(avail_gb) GiB"
}

# --- llama.cpp --------------------------------------------------------------
# KV combos. llama.cpp needs flash-attention for a quantized V cache, so the
# fa=off arm is f16-only by construction, not by choice.
llama_combos() {
    # COMBOS overrides the default list: newline-separated "ctk ctv fa" triples.
    # The asymmetric q8_0/q4_0 pair has no Metal flash-attn kernel and falls to
    # CPU, so it is worth measuring once and then excluding -- at 81 tok/s a
    # 32k prefill costs ~6.5 min per repetition.
    if [ -n "${COMBOS:-}" ]; then printf '%s\n' "$COMBOS"; return; fi
    cat <<'EOF'
f16 f16 on
q8_0 q8_0 on
q8_0 q4_0 on
q4_0 q4_0 on
f16 f16 off
EOF
}

phase_llama() {
    [ -x "$LBENCH" ] || { echo "set LBENCH to a verified llama-bench" >&2; exit 2; }
    local ctx="$1" reps="${2:-3}" n; n=$(tokens_for "$ctx")
    no_other_mlx
    llama_combos | while read -r ctk ctv fa; do
        local tag="llama_${ctk}_${ctv}_fa${fa}_${ctx}"
        local ratio=1
        case "$ctk" in q8_0) ratio=0.53 ;; q4_0) ratio=0.28 ;; esac
        require_mem "$(est_gb $((n+GEN)) $ratio)" "$tag" || continue

        # Every llama-bench repetition at depth re-runs the whole prefill, so a
        # 128k cell costs (reps + warmup) x a multi-minute prefill. LLAMA_EXTRA
        # (e.g. --no-warmup) and PP_REPS exist to buy that back deliberately
        # rather than by quietly dropping repetitions.
        # decode at depth (and the KV allocation for that depth)
        # shellcheck disable=SC2086
        "$LBENCH" -m "$GGUF" -p 0 -n "$GEN" -d "$n" -r "$reps" -fa "$fa" \
            -ctk "$ctk" -ctv "$ctv" -o jsonl -v ${LLAMA_EXTRA:-} \
            > "$RAW/$tag.tg.jsonl" 2> "$RAW/$tag.tg.err" || { echo "FAILED $tag tg" >&2; continue; }
        # prefill
        # shellcheck disable=SC2086
        "$LBENCH" -m "$GGUF" -p "$n" -n 0 -r "${PP_REPS:-$reps}" -fa "$fa" \
            -ctk "$ctk" -ctv "$ctv" -o jsonl -v ${LLAMA_EXTRA:-} \
            > "$RAW/$tag.pp.jsonl" 2> "$RAW/$tag.pp.err" || { echo "FAILED $tag pp" >&2; continue; }

        python3 "$SUMMARIZE" --ingest-llama \
            --tag "$tag" --ctx "$ctx" --ntok "$n" --gen "$GEN" \
            --ctk "$ctk" --ctv "$ctv" --fa "$fa" \
            --mlxdir "$MLXDIR" --raw "$RAW" >> "$CELLS"
        echo "done $tag" >&2
    done
}

# --- rMLX -------------------------------------------------------------------
phase_rmlx() {
    # Same check the LBENCH sibling carries. A stale release-perf binary left by
    # another branch yields a complete, plausible row set attributed to this one.
    [ -x "$RMLX" ] || { echo "set RMLX to a built rmlx binary (missing: $RMLX)" >&2; exit 2; }
    local ctx="$1" runs="${2:-3}" n; n=$(tokens_for "$ctx")
    no_other_mlx
    local home; home="$OUT_DIR/rmlx_home"; mkdir -p "$home"
    local rmlx_sha; rmlx_sha="$(shasum -a 256 "$RMLX" | cut -d' ' -f1)"
    echo "rmlx binary $RMLX sha256=$rmlx_sha" >&2
    for codec in none k8v8 mixed_k8g64_v4g64; do
        local tag="rmlx_${codec}_${ctx}"
        local ratio=1
        case "$codec" in mixed_k8g64_v4g64) ratio=0.6 ;; esac
        require_mem "$(est_gb $((n+GEN)) $ratio)" "$tag" || continue
        # --max-ctx is sized from the MEASURED prompt length plus the
        # generation, not from the bucket name: an over-long prompt is rejected
        # and reads TTFT and decode as 0 with no error at all.
        RMLX_HOME="$home" "$RMLX" bench --model "$MLXDIR" \
            --kv-quant "$codec" --prompt-tokens "$ctx" \
            --max-ctx $((n+GEN+64)) --max-prompt-tokens 200000 \
            --max-tokens "$GEN" --runs "$runs" --warmup 1 \
            --metrics off --json > "$RAW/$tag.json" 2> "$RAW/$tag.err" \
            || { echo "FAILED $tag" >&2; continue; }
        python3 "$SUMMARIZE" --ingest-rmlx \
            --tag "$tag" --ctx "$ctx" --gen "$GEN" --codec "$codec" \
            --mlxdir "$MLXDIR" --binary-sha256 "$rmlx_sha" --raw "$RAW" >> "$CELLS"
        echo "done $tag" >&2
    done
}

# --- stock mlx-lm -----------------------------------------------------------
phase_mlxlm() {
    local ctx="$1" reps="${2:-3}" n; n=$(tokens_for "$ctx")
    no_other_mlx
    require_mem "$(est_gb $((n+GEN)) 1)" "mlxlm_${ctx}" || return 0
    local out="$RAW/mlxlm_${ctx}.jsonl"
    "$MLXLM_PY" "$PROBE" --model "$MLXDIR" --prompt-tokens "$ctx" \
        --seq "${MLXLM_SEQ:-fp16,mlxq8,mlxq4,mlxq4,mlxq8,fp16}" --reps "$reps" --gen "$GEN" \
        --arm stock --out "$out" > "$RAW/mlxlm_${ctx}.log" 2>&1 \
        || { echo "FAILED mlxlm $ctx" >&2; return 0; }
    python3 "$SUMMARIZE" --ingest-mlxlm \
        --ctx "$ctx" --gen "$GEN" --mlxdir "$MLXDIR" --raw "$RAW" >> "$CELLS"
    echo "done mlxlm_$ctx" >&2
}

case "${1:?phase required}" in
    preflight) phase_preflight ;;
    llama)     phase_llama "${2:?ctx}" "${3:-3}" ;;
    rmlx)      phase_rmlx  "${2:?ctx}" "${3:-3}" ;;
    mlxlm)     phase_mlxlm "${2:?ctx}" "${3:-3}" ;;
    *) echo "unknown phase: $1" >&2; exit 2 ;;
esac
