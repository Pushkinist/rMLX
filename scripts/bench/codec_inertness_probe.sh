#!/usr/bin/env bash
# codec_inertness_probe.sh — does a KV codec change anything a caller can see?
#
# For one model and one prompt length, runs `rmlx baseline` once per codec at
# temperature 0 and records the three quantities a codec disposition rests on:
#
#   * `kv_cache_bytes`   — resident KV. A codec that saves no bytes is not a
#                          memory lever, whatever its bits/value says.
#   * sha256(token_ids)  — the exact greedy id sequence. Identical to the
#                          `none` arm means the codec is not on the decode read
#                          path at all; decoded text cannot show this, because
#                          different id sequences decode to the same string.
#   * `store_skipped`    — whether the run logged `exit_prefill: packed store
#                          skipped`, i.e. the codec encoded nothing at all.
#
# The three together separate "dominated" from "merely unused": a codec whose
# bytes and ids both equal `none`'s, and which skipped its store, is inert —
# selecting it is a no-op with a name.
#
# `ttft_ms`, `decode_tps` and `prefill_tps` are recorded for context only and
# are NOT comparable across rows: these are single unpaired runs on a shared
# host, where the same binary and flags have read 11% apart thirty minutes
# apart. Use `scripts/perf_ab.sh` for any throughput claim. The two columns the
# probe exists for — bytes and the id digest — do not depend on host load.
#
# Usage:
#   scripts/bench/codec_inertness_probe.sh --model <snapshot-abs-path> \
#       --prompt-tokens 4096 [--max-tokens 100] [--max-ctx N] [--out CSV] \
#       [--codecs "none k8v8 ..."]
#
# Output CSV (appended, header written once):
#   timestamp,model,prompt_tokens,prompt_tokens_measured,max_tokens,codec,
#   exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,
#   prefill_tps,binary_sha
#
# `prompt_tokens` is the fixture NAME (`--prompt-tokens 4096` selects
# `prompts/longctx_4k.json`); `prompt_tokens_measured` is what the run actually
# tokenized, which the chat template makes ~5% larger. They are different
# numbers and only the second is a token count.
#
# Default CSV: <RMLX_HOME>/bench/codec_inertness.csv
#
# Hard rule 8: kills competing MLX processes and clears the claim file before
# every run. A stale claim turns every later run into an empty-output exit 11
# that reads like "the model emitted nothing", so the claim is re-checked per
# codec, not once per sweep.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -x "${RMLX_BINARY:-}" ]]; then
	BINARY="${RMLX_BINARY}"
elif [[ -x "${REPO_ROOT}/target/release-perf/rmlx" ]]; then
	BINARY="${REPO_ROOT}/target/release-perf/rmlx"
else
	BINARY="${REPO_ROOT}/target/release/rmlx"
fi

RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
OUT="${RMLX_HOME}/bench/codec_inertness.csv"

MODEL=""
PROMPT_TOKENS=4096
MAX_TOKENS=100
MAX_CTX=""
PORT="${RMLX_PORT:-62265}"

# Every `KvQuant` the enum can spell, one representative per parameterised
# family. Kept in the order of `ALL_KV_QUANTS` so a reader can diff the two.
CODECS_DEFAULT=(
	none
	k8v8 k8v4 planar planar3 planar_k
	k8vturbo3 k8vturbo3tcq k8vturbo2 k8vturbo2tcq
	tsym3 tsym4
	iso3 iso4 rotor3 rotor4
	rotor_k_4_asym_v4_g64
	mixed_k8g64_v4g64 rot_k_v8g64
	iso3_sym iso4_sym k_iso3 k_iso4
	rotor3_sym rotor4_sym k_rotor3 k_rotor4
)
CODECS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		MODEL="$2"
		shift 2
		;;
	--prompt-tokens)
		PROMPT_TOKENS="$2"
		shift 2
		;;
	--max-tokens)
		MAX_TOKENS="$2"
		shift 2
		;;
	--max-ctx)
		MAX_CTX="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	--codecs)
		read -r -a CODECS <<<"$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,46p' "$0"
		exit 0
		;;
	*)
		echo "ERROR: unknown argument '$1'" >&2
		exit 2
		;;
	esac
done

if [[ -z "$MODEL" ]]; then
	echo "ERROR: --model is required" >&2
	exit 2
fi
if [[ ! -d "$MODEL" ]]; then
	echo "ERROR: model snapshot not found: $MODEL" >&2
	exit 2
fi
if [[ ! -x "$BINARY" ]]; then
	echo "ERROR: rmlx binary not found or not executable: $BINARY" >&2
	exit 2
fi
if [[ ${#CODECS[@]} -eq 0 ]]; then
	CODECS=("${CODECS_DEFAULT[@]}")
fi
if [[ -z "$MAX_CTX" ]]; then
	# `--prompt-tokens N` names a fixture, not a token count: the chat template
	# wraps it, so the served prompt runs ~5% long (a 32 768 fixture prefills
	# 34 355 on gemma-4), and the KV ring rounds its grow steps up on top of
	# that. A ceiling of `N + gen` is therefore refused mid-prefill with
	# "kv request exceeds max-ctx ceiling" and every cell of the sweep comes
	# back empty — which reads like the model emitted nothing. Leave room.
	MAX_CTX=$((PROMPT_TOKENS * 5 / 4 + MAX_TOKENS + 2048))
fi

BINARY_SHA="$(shasum -a 256 "$BINARY" | cut -c1-12)"
MODEL_NAME="$(basename "$MODEL")"

mkdir -p "$(dirname "$OUT")"
if [[ ! -f "$OUT" ]]; then
	echo "timestamp,model,prompt_tokens,prompt_tokens_measured,max_tokens,codec,exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,prefill_tps,binary_sha" >"$OUT"
fi

preflight() {
	pkill -f "rmlx serve" 2>/dev/null
	pkill -f "rmlx_main serve" 2>/dev/null
	pkill -f mlx_lm 2>/dev/null
	rm -f "/tmp/rmlx.${PORT}.claim"
	sleep 1
}

echo "probe: binary=$BINARY ($BINARY_SHA)"
echo "probe: model=$MODEL_NAME prompt_tokens=$PROMPT_TOKENS max_tokens=$MAX_TOKENS max_ctx=$MAX_CTX"
echo "probe: ${#CODECS[@]} codecs -> $OUT"

for codec in "${CODECS[@]}"; do
	preflight
	raw="$(mktemp -t codec_inertness)"
	RMLX_HOME="$RMLX_HOME" "$BINARY" baseline \
		--model "$MODEL" \
		--kv-quant "$codec" \
		--prompt-tokens "$PROMPT_TOKENS" \
		--max-tokens "$MAX_TOKENS" \
		--max-ctx "$MAX_CTX" \
		--emit-token-ids \
		--metrics off \
		--log debug >"$raw" 2>&1
	rc=$?

	# `token_ids=` is emitted only on a completed generation. An empty digest
	# here is the sha256 of the empty string and must never be read as a
	# result — the exit code column is what tells the two apart.
	ids_sha=""
	if grep -q '^baseline: token_ids=' "$raw"; then
		ids_sha="$(sed -n 's/^baseline: token_ids=//p' "$raw" | head -1 | shasum -a 256 | cut -c1-16)"
	fi

	summary="$(grep -m1 '^baseline: model=' "$raw" || true)"
	kv_bytes="$(sed -n 's/.*kv_cache_bytes=\([0-9]*\).*/\1/p' <<<"$summary" | head -1)"
	measured="$(sed -n 's/.*prompt_tokens=\([0-9]*\).*/\1/p' <<<"$summary" | head -1)"
	ttft="$(sed -n 's/.*ttft_ms=\([0-9.]*\).*/\1/p' <<<"$summary" | head -1)"
	dtps="$(sed -n 's/.*decode_tps=\([0-9.]*\).*/\1/p' <<<"$summary" | head -1)"
	ptps="$(sed -n 's/.*prefill_tps=\([0-9.]*\).*/\1/p' <<<"$summary" | head -1)"

	skipped=0
	if grep -q 'packed store skipped' "$raw"; then
		skipped=1
	fi

	printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
		"$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$MODEL_NAME" "$PROMPT_TOKENS" "${measured:-}" \
		"$MAX_TOKENS" "$codec" "$rc" "${kv_bytes:-}" "${ids_sha:-}" "$skipped" \
		"${ttft:-}" "${dtps:-}" "${ptps:-}" "$BINARY_SHA" >>"$OUT"

	printf '  %-24s rc=%-3s bytes=%-12s ids=%-16s skipped=%s\n' \
		"$codec" "$rc" "${kv_bytes:-NONE}" "${ids_sha:-NONE}" "$skipped"

	rm -f "$raw"
done

preflight
echo "probe: done -> $OUT"
