#!/usr/bin/env bash
# KV-codec smoke + NIAH gate matrix runner.
#
# Reads `kv_codec_matrix.toml` + `smoke_prompts.toml`, executes per-row:
#
#   1. Skip if `skip_reason != ""` (structured log line).
#   2. The Metal claim (CLAUDE.md hard rule 8: single MLX process): each
#      `rmlx baseline` takes it, and the NIAH pass runs under `rmlx claim run`.
#      A claim another process holds fails the row with exit 11; nothing is
#      stopped for it.
#   3. Smoke probe: one `rmlx baseline` call per `smoke_probe_prompts[i]`,
#      validating the decoded continuation against the regex in
#      `smoke_prompts.toml`. Any prompt fail => row fail.
#   4. NIAH harness at `context_length`. Re-uses the
#      `niah_long_context` cargo-test harness, scoped to the row's
#      `niah_filter`. Retrieval-pct is parsed from the test stdout.
#   5. Compare retrieval to `expected_retrieval_pct`:
#       * `--record-baseline` AND expected == 0.0 -> measured value is
#         written back into the manifest (Exec B path).
#       * expected == 0.0 without `--record-baseline` -> the row has no
#         recorded baseline, so nothing can fail it: its verdict is
#         `unrecorded` and the run ends INCOMPLETE.
#       * otherwise -> measured >= expected - 2pp passes; else fails.
#
# Aggregated rows are emitted to
# `scripts/release_e2e/stage6_perf/last_run.json` with `ran`, `skipped_known`,
# `skipped_absent`, `unrecorded` and `incomplete`.
#
# Exit codes:
#   0  at least one row ran, every row that ran passed, no row skipped for an
#      absent snapshot or an unset root, no row without a recorded baseline
#   1  a row failed, or no row ran
#   3  matrix INCOMPLETE: every row that ran passed, but a row skipped for an
#      absent snapshot or an unset root, or ran against an unrecorded baseline
#   2  usage or setup error
# `matrix_done` prints `ran=N skipped=M`. A row with a manifest `skip_reason`
# is a known disposition and does not make the run INCOMPLETE.
#
# TOML parsing dependency: pure-python3 + `tomllib` (stdlib, Python 3.11+).
# `dasel` and `yq` are NOT installed on the dev host; python3 is the
# unconditional choice. Fallback: `python3 -c "import tomli"` if 3.10 or
# older (script will error and print install hint).
#
# Usage:
#   bash scripts/release_e2e/stage6_perf/codec_smoke_runner.sh \
#       [--manifest <path>] \
#       [--filter codec_name=<x>] [--filter model=<m>] \
#       [--record-baseline] \
#       [--dry-run] [--out <path>]
#
# Each row's `model` names a snapshot slug under $RMLX_O_MODELS_ROOT (see
# `model_slug` below; the NIAH harness resolves the same slug). A row whose
# slug is absent from the root is skipped with a named reason.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

MANIFEST="$REPO_ROOT/scripts/release_e2e/stage6_perf/kv_codec_matrix.toml"
PROMPTS="$REPO_ROOT/scripts/release_e2e/stage6_perf/smoke_prompts.toml"
LAST_RUN="$REPO_ROOT/scripts/release_e2e/stage6_perf/last_run.json"

FILTER_CODEC=""
FILTER_MODEL=""
RECORD_BASELINE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            [[ $# -lt 2 ]] && { printf 'ERROR: --manifest requires a value\n' >&2; exit 2; }
            MANIFEST="$2"; shift 2 ;;
        --filter)
            [[ $# -lt 2 ]] && { printf 'ERROR: --filter requires a value\n' >&2; exit 2; }
            case "$2" in
                codec_name=*) FILTER_CODEC="${2#codec_name=}" ;;
                model=*)      FILTER_MODEL="${2#model=}" ;;
                *) printf 'ERROR: --filter expects codec_name=<x> or model=<m>, got: %s\n' "$2" >&2; exit 2 ;;
            esac
            shift 2 ;;
        --out)
            [[ $# -lt 2 ]] && { printf 'ERROR: --out requires a value\n' >&2; exit 2; }
            LAST_RUN="$2"; shift 2 ;;
        --record-baseline) RECORD_BASELINE=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,56p' "$0"; exit 0 ;;
        *)
            printf 'ERROR: unknown flag: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# Pick a python that has `tomllib` (stdlib, 3.11+). Fall back to whichever
# python3 has `tomli` installed. The dev host's `/usr/bin/python3` is 3.9;
# Homebrew's `python3.13` / `python3.14` ship `tomllib` directly.
PYTHON3=""
for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import tomllib' >/dev/null 2>&1 \
            || "$candidate" -c 'import tomli' >/dev/null 2>&1; then
            PYTHON3="$candidate"
            break
        fi
    fi
done
if [[ -z "$PYTHON3" ]]; then
    printf 'ERROR: no python3 with `tomllib` (>=3.11) or `tomli` found.\n' >&2
    printf '       brew install python@3.13 OR pip3 install tomli.\n' >&2
    exit 2
fi

# Every process a pass starts under the claim inherits it, so everything the
# passes run is compiled here, outside it; a pass that still compiles fails.
if [[ $DRY_RUN -eq 0 ]]; then
    cargo build --profile release-perf --bin rmlx || exit 2
    cargo test --no-run --profile release-perf -p rmlx-models --test niah_long_context || exit 2
fi
CLAIM_RMLX="$REPO_ROOT/target/release-perf/rmlx"

# Map manifest `model` -> snapshot slug under $RMLX_O_MODELS_ROOT. The
# `GoldenModel` consts in crates/rmlx-models/tests/niah_long_context.rs list the
# same slugs; change both together.
model_slug() {
    case "$1" in
        bonsai-8b)        printf 'prism-ml__Ternary-Bonsai-8B-mlx-2bit' ;;
        gemma4-e4b)       printf 'mlx-community__gemma-4-e4b-it-mxfp8' ;;
        qwen3.6-moe-8bit) printf 'mlx-community__Qwen3.6-35B-A3B-8bit' ;;
        qwen3.8-27b-4bit) printf 'mlx-community__Qwen3.8-27B-4bit' ;;
        bonsai2-27b)      printf 'prism-ml__Ternary-Bonsai-2-27B-mlx-2bit' ;;
        *) printf '' ;;
    esac
}

# Print the first file a snapshot directory lacks; nothing when it can be
# served. The same files the NIAH harness requires of a snapshot.
snapshot_gap() {
    local dir="$1" f
    for f in config.json tokenizer.json; do
        [[ -f "$dir/$f" ]] || { printf '%s' "$f"; return; }
    done
    if [[ ! -e "$dir/model.safetensors.index.json" && ! -e "$dir/model.safetensors" ]]; then
        printf 'model.safetensors[.index.json]'
    elif ! compgen -G "$dir/*.safetensors" >/dev/null; then
        printf 'any *.safetensors shard'
    fi
}

# Emit a single structured line to stderr (tracing-style key=value).
log_evt() {
    local level="$1"; shift
    printf 'evt level=%s ts=%s %s\n' "$level" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >&2
}

# ── manifest enumerate (python3 → bash tsv) ─────────────────────────────────
ROWS_TSV="$("$PYTHON3" - "$MANIFEST" <<'PY'
import sys
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

with open(sys.argv[1], 'rb') as f:
    data = toml.load(f)

for e in data.get('entries', []):
    # Use the sentinel "-" for empty string fields so bash `read -r` with
    # IFS=$'\t' does not collapse consecutive empty tabs (it would shift
    # downstream fields into the wrong slots otherwise).
    def s(v):
        v = str(v)
        return v if v != '' else '-'
    fields = [
        s(e.get('codec_name', '')),
        s(e.get('model', '')),
        str(e.get('context_length', 0)),
        repr(float(e.get('expected_retrieval_pct', 0.0))),
        s(','.join(e.get('smoke_probe_prompts', []))),
        s(e.get('skip_reason', '')),
        s(e.get('cli_args', '')),
        s(e.get('niah_filter', '')),
    ]
    # TSV — TOML strings cannot contain literal \t (forbidden in our keys).
    print('\t'.join(fields))
PY
)"

# ── smoke prompts: emit "name\tprompt\tmax_tokens\tregex\tmin_ratio" rows ───
PROMPTS_TSV="$("$PYTHON3" - "$PROMPTS" <<'PY'
import sys
try:
    import tomllib as toml
except ModuleNotFoundError:
    import tomli as toml

with open(sys.argv[1], 'rb') as f:
    data = toml.load(f)

for name, body in data.get('prompts', {}).items():
    # Replace literal tabs / newlines in prompt content with sentinels we
    # decode in bash. Newlines are preserved because the multi_turn prompt
    # needs them.
    p = body.get('prompt', '').replace('\t', ' ').replace('\n', '\\n')
    print('\t'.join([
        str(name),
        p,
        str(body.get('max_tokens', 32)),
        str(body.get('validate_regex', '.')),
        str(body.get('min_printable_ratio', 0.0)),
    ]))
PY
)"

# Build a python helper invocation for regex validation that returns 0/1.
validate_output() {
    local regex="$1"
    local min_ratio="$2"
    local outfile="$3"
    "$PYTHON3" - "$regex" "$min_ratio" "$outfile" <<'PY'
import re, sys
regex = sys.argv[1]
min_ratio = float(sys.argv[2])
path = sys.argv[3]
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    txt = f.read()
if not txt.strip():
    print('fail: empty output')
    sys.exit(1)
m = re.search(regex, txt)
if not m:
    print(f'fail: regex did not match ({regex!r})')
    sys.exit(1)
printable = sum(1 for c in txt if c.isprintable() or c in ('\n', '\t'))
ratio = printable / max(1, len(txt))
if ratio < min_ratio:
    print(f'fail: printable ratio {ratio:.3f} < {min_ratio}')
    sys.exit(1)
print('pass')
sys.exit(0)
PY
}

# Lookup helpers over PROMPTS_TSV.
prompt_field() {
    local name="$1"; local field_idx="$2"
    awk -F'\t' -v n="$name" -v i="$field_idx" '$1==n {print $i; exit}' <<<"$PROMPTS_TSV"
}

# Parse retrieval-pct from cargo-test stdout. Convention: the NIAH harness
# prints lines containing `needle_found=<bool>` (one per cell — see
# `run_cell` in `crates/rmlx-models/tests/niah_long_context.rs`). The
# aggregate retrieval-pct is `hits / total` across all cells in the filter
# scope.
parse_retrieval_pct() {
    local log="$1"
    "$PYTHON3" - "$log" <<'PY'
import re, sys
with open(sys.argv[1], 'r', encoding='utf-8', errors='replace') as f:
    txt = f.read()
matches = re.findall(r'needle_found=(true|false)', txt)
if not matches:
    print('NA')
else:
    total = len(matches)
    hits = sum(1 for m in matches if m == 'true')
    print(f'{hits / total:.4f}')
PY
}

# Write a measured retrieval_pct back into the manifest TOML, in place.
# Only mutates the (codec_name, model) row whose value is currently 0.0.
manifest_record() {
    local codec="$1"; local model="$2"; local measured="$3"; local path="$4"
    "$PYTHON3" - "$path" "$codec" "$model" "$measured" <<'PY'
import os, re, sys, pathlib
path = pathlib.Path(sys.argv[1])
codec = sys.argv[2]
model = sys.argv[3]
measured = float(sys.argv[4])
src = path.read_text()
# Walk [[entries]] blocks; rewrite the matching row.
blocks = re.split(r'(?m)^\[\[entries\]\]\s*$', src)
out = [blocks[0]]
for block in blocks[1:]:
    has_codec = re.search(r'^codec_name\s*=\s*"' + re.escape(codec) + r'"\s*$', block, re.M)
    has_model = re.search(r'^model\s*=\s*"' + re.escape(model) + r'"\s*$', block, re.M)
    if has_codec and has_model:
        block = re.sub(
            r'^expected_retrieval_pct\s*=\s*[0-9.]+\s*$',
            f'expected_retrieval_pct = {measured:.4f}',
            block,
            count=1,
            flags=re.M,
        )
    out.append('[[entries]]\n' + block.lstrip('\n'))
# Atomic write: tmp + rename so a SIGKILL mid-write does not corrupt the
# manifest (review HIGH-2).
tmp = path.with_suffix(path.suffix + '.tmp')
tmp.write_text(''.join(out))
os.replace(tmp, path)
PY
}

# Map manifest `codec_name` -> KvQuant string accepted by `KvQuant::from_str`
# (see crates/rmlx-kv-quant/src/quant.rs). The NIAH harness reads
# `RMLX_NIAH_KV_QUANT` and parses via `FromStr`; unmapped names fall through
# to the harness's existing per-FlashKind default (review HIGH-1).
codec_to_kv_quant() {
    case "$1" in
        bf16)        printf 'bf16' ;;
        k8v4)        printf 'k8v4' ;;
        k8v8)        printf 'k8v8' ;;
        TurboSym3)   printf 'tsym3' ;;
        TurboSym4)   printf 'tsym4' ;;
        Iso3Sym)     printf 'iso3_sym' ;;
        Iso4Sym)     printf 'iso4_sym' ;;
        Rotor3Sym)   printf 'rotor3_sym' ;;
        Rotor4Sym)   printf 'rotor4_sym' ;;
        PlanarK)     printf 'planar_k' ;;
        planar)      printf 'planar' ;;
        Mixed)       printf 'mixed_k8g64_v4g64' ;;
        *)           printf '' ;;
    esac
}

# Extract decoded text from the rmlx baseline tracing stream. The baseline
# emits a `decoded=<text>` field on a tracing line; the rest of the stream
# carries log noise that can accidentally satisfy a validate_regex (the
# input prompt itself is echoed by tracing into stderr — review MED-2).
# Replaces $1 in place with only the joined decoded fragments.
extract_decoded_from_trace() {
    local outfile="$1"
    "$PYTHON3" - "$outfile" <<'PY'
import re, sys, pathlib
path = pathlib.Path(sys.argv[1])
txt = path.read_text(encoding='utf-8', errors='replace')
# Strip ANSI escape codes before matching: the tracing subscriber wraps
# field keys and values in ANSI CSI sequences (\x1b[...m) which break the
# `decoded="..."` regex if applied to raw output (review MED-2 path (b)).
txt = re.sub(r'\x1b\[[0-9;]*m', '', txt)
# Match `decoded=` followed by either a quoted string or a tail-of-line.
# tracing's default formatter prints `decoded="..."` (Debug); we accept both.
parts = re.findall(r'decoded="([^"]*)"', txt)
if not parts:
    parts = [m.group(1).strip() for m in re.finditer(r'decoded=(.*)$', txt, re.M)]
if parts:
    path.write_text(''.join(parts), encoding='utf-8')
PY
}

# ── per-row execution ───────────────────────────────────────────────────────
ROW_JSONS=()
AGG_RC=0
RAN=0
SKIPPED_KNOWN=0
SKIPPED_ABSENT=0
UNRECORDED=0
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD 2>/dev/null || printf nogit)"

while IFS=$'\t' read -r CODEC MODEL CTX EXPECTED PROMPTS_CSV SKIP CLI_ARGS NIAH_FILTER; do
    [[ -z "$CODEC" ]] && continue
    # Decode the python emitter's "-" sentinel for empty fields.
    [[ "$SKIP"        == "-" ]] && SKIP=""
    [[ "$CLI_ARGS"    == "-" ]] && CLI_ARGS=""
    [[ "$NIAH_FILTER" == "-" ]] && NIAH_FILTER=""
    [[ "$PROMPTS_CSV" == "-" ]] && PROMPTS_CSV=""

    # Filters.
    if [[ -n "$FILTER_CODEC" && "$CODEC" != "$FILTER_CODEC" ]]; then continue; fi
    if [[ -n "$FILTER_MODEL" && "$MODEL" != "$FILTER_MODEL" ]]; then continue; fi

    if [[ -n "$SKIP" ]]; then
        log_evt info "skip codec=$CODEC model=$MODEL reason=\"$SKIP\""
        SKIPPED_KNOWN=$((SKIPPED_KNOWN + 1))
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"skip","reason":"%s"}' \
            "$CODEC" "$MODEL" "$SKIP")")
        continue
    fi

    SLUG="$(model_slug "$MODEL")"
    if [[ -z "$SLUG" ]]; then
        log_evt error "unknown_model codec=$CODEC model=$MODEL"
        AGG_RC=1
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"fail","reason":"unknown_model"}' \
            "$CODEC" "$MODEL")")
        continue
    fi
    if [[ $DRY_RUN -eq 1 ]]; then
        SNAPSHOT_STATE=absent
        if [[ -n "${RMLX_O_MODELS_ROOT:-}" && -d "$RMLX_O_MODELS_ROOT" \
            && -z "$(snapshot_gap "$RMLX_O_MODELS_ROOT/$SLUG")" ]]; then
            SNAPSHOT_STATE=present
        fi
        log_evt info "dry-run codec=$CODEC model=$MODEL slug=$SLUG ctx=$CTX cli_args=\"$CLI_ARGS\" niah_filter=$NIAH_FILTER snapshot=$SNAPSHOT_STATE"
    fi
    if [[ -z "${RMLX_O_MODELS_ROOT:-}" ]]; then
        log_evt warn "skip_no_snapshot codec=$CODEC model=$MODEL slug=$SLUG missing=RMLX_O_MODELS_ROOT"
        SKIPPED_ABSENT=$((SKIPPED_ABSENT + 1))
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"skip","reason":"snapshot_absent:%s:RMLX_O_MODELS_ROOT unset"}' \
            "$CODEC" "$MODEL" "$SLUG")")
        continue
    fi
    if [[ ! -d "$RMLX_O_MODELS_ROOT" ]]; then
        log_evt error "models_root_not_a_directory codec=$CODEC model=$MODEL root=$RMLX_O_MODELS_ROOT"
        AGG_RC=1
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"fail","reason":"models_root_not_a_directory"}' \
            "$CODEC" "$MODEL")")
        continue
    fi
    MODEL_PATH="$RMLX_O_MODELS_ROOT/$SLUG"
    MISSING="$(snapshot_gap "$MODEL_PATH")"
    if [[ -n "$MISSING" ]]; then
        log_evt warn "skip_no_snapshot codec=$CODEC model=$MODEL slug=$SLUG missing=\"$MISSING\""
        SKIPPED_ABSENT=$((SKIPPED_ABSENT + 1))
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"skip","reason":"snapshot_absent:%s:%s"}' \
            "$CODEC" "$MODEL" "$SLUG" "$MISSING")")
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        RAN=$((RAN + 1))
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","verdict":"dry_run"}' "$CODEC" "$MODEL")")
        continue
    fi

    log_evt info "row_start codec=$CODEC model=$MODEL ctx=$CTX"
    RAN=$((RAN + 1))

    # ── smoke probes ────────────────────────────────────────────────────────
    SMOKE_VERDICT="pass"
    SMOKE_FAIL_PROMPT=""
    IFS=',' read -r -a PROMPT_NAMES <<<"$PROMPTS_CSV"
    for PNAME in ${PROMPT_NAMES[@]+"${PROMPT_NAMES[@]}"}; do
        [[ -z "$PNAME" ]] && continue
        PCONTENT_RAW="$(prompt_field "$PNAME" 2)"
        # Decode \n sentinel into real newlines.
        PCONTENT="$(printf '%b' "${PCONTENT_RAW//\\n/$'\n'}")"
        PMAX="$(prompt_field "$PNAME" 3)"
        PREGEX="$(prompt_field "$PNAME" 4)"
        PMIN="$(prompt_field "$PNAME" 5)"

        PROMPT_FILE="$(mktemp -t rmlx-smoke-prompt.XXXXXX)"
        OUTPUT_FILE="$(mktemp -t rmlx-smoke-output.XXXXXX)"
        printf '%s' "$PCONTENT" >"$PROMPT_FILE"

        # `rmlx baseline` writes a one-line summary to stdout; the actual
        # decoded text is emitted to a tracing log. We grep the tracing
        # output for the `decoded=` field that baseline emits.
        #
        # The `||` arm keeps errexit from ending the run and records the
        # failed command's status: PIPESTATUS read after `|| true` would
        # be the `true`'s, and a refused Metal claim would look like a
        # clean run with empty output.
        BASELINE_RC=0
        # shellcheck disable=SC2086  # CLI_ARGS deliberately word-split
        RUST_LOG=info cargo run --quiet --profile release-perf --bin rmlx -- \
            baseline \
            --model "$MODEL_PATH" \
            --prompt "$PROMPT_FILE" \
            --max-tokens "$PMAX" \
            --prompt-label "$PNAME" \
            --record --run-class diagnostic --label "codec_smoke:$CODEC:$PNAME" \
            $CLI_ARGS 2>&1 | tee "$OUTPUT_FILE" >/dev/null || BASELINE_RC=${PIPESTATUS[0]}

        # Extract `decoded=` so validate_regex sees only model output, not the
        # tracing-echoed prompt (review MED-2 path (a)).
        extract_decoded_from_trace "$OUTPUT_FILE"

        if [[ $BASELINE_RC -ne 0 ]]; then
            SMOKE_VERDICT="fail"
            SMOKE_FAIL_PROMPT="$PNAME:baseline_rc=$BASELINE_RC"
            log_evt error "smoke_fail codec=$CODEC model=$MODEL prompt=$PNAME rc=$BASELINE_RC"
            rm -f "$PROMPT_FILE" "$OUTPUT_FILE"
            break
        fi

        if ! validate_output "$PREGEX" "$PMIN" "$OUTPUT_FILE" >/dev/null; then
            SMOKE_VERDICT="fail"
            SMOKE_FAIL_PROMPT="$PNAME:validate"
            log_evt error "smoke_fail codec=$CODEC model=$MODEL prompt=$PNAME reason=validate"
            rm -f "$PROMPT_FILE" "$OUTPUT_FILE"
            break
        fi

        rm -f "$PROMPT_FILE" "$OUTPUT_FILE"
    done

    if [[ "$SMOKE_VERDICT" != "pass" ]]; then
        AGG_RC=1
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","smoke":"fail","fail_prompt":"%s","verdict":"fail"}' \
            "$CODEC" "$MODEL" "$SMOKE_FAIL_PROMPT")")
        continue
    fi

    # ── NIAH ────────────────────────────────────────────────────────────────
    NIAH_LOG="$(mktemp -t rmlx-niah.XXXXXX.log)"

    # The NIAH harness wraps `cargo test -p rmlx-models --test niah_long_context`
    # at the row's filter scope and resolves the row's slug under
    # `RMLX_O_MODELS_ROOT` itself.
    #
    # Forward the row's codec selection to the
    # harness via `RMLX_NIAH_KV_QUANT`. If the codec is unmapped, leave the
    # env unset and the harness falls back to its per-FlashKind default.
    # The `||` arm records the failed command's status, as for the baseline.
    NIAH_KV_QUANT="$(codec_to_kv_quant "$CODEC")"
    NIAH_ENV=(env)
    if [[ -n "$NIAH_KV_QUANT" ]]; then
        NIAH_ENV+=("RMLX_NIAH_KV_QUANT=$NIAH_KV_QUANT")
    else
        log_evt warn "niah_codec_unmapped codec=$CODEC — harness uses default"
    fi
    NIAH_RC=0
    "${NIAH_ENV[@]}" \
        timeout 3600 "$CLAIM_RMLX" claim run -- cargo test --profile release-perf \
            -p rmlx-models --test niah_long_context \
            -- --ignored --test-threads=1 --nocapture "$NIAH_FILTER" \
            2>&1 | tee "$NIAH_LOG" >/dev/null || NIAH_RC=${PIPESTATUS[0]}
    if grep -Eq '^[[:space:]]*Compiling [A-Za-z0-9_-]+ v' "$NIAH_LOG"; then
        log_evt error "niah_compiled_under_claim codec=$CODEC model=$MODEL log=$NIAH_LOG"
        NIAH_RC=1
    fi

    if [[ $NIAH_RC -ne 0 ]]; then
        log_evt error "niah_fail codec=$CODEC model=$MODEL rc=$NIAH_RC log=$NIAH_LOG"
        AGG_RC=1
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","smoke":"pass","niah":"fail","verdict":"fail","log":"%s"}' \
            "$CODEC" "$MODEL" "$NIAH_LOG")")
        continue
    fi

    MEASURED="$(parse_retrieval_pct "$NIAH_LOG")"
    if [[ "$MEASURED" == "NA" ]]; then
        log_evt error "niah_no_pct codec=$CODEC model=$MODEL log=$NIAH_LOG"
        AGG_RC=1
        ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","smoke":"pass","niah":"no_pct","verdict":"fail"}' \
            "$CODEC" "$MODEL")")
        continue
    fi

    # Gate or record.
    EXP_FLOAT="$(printf '%s' "$EXPECTED" | "$PYTHON3" -c 'import sys; print(float(sys.stdin.read().strip()))')"
    VERDICT="pass"
    REASON=""

    if [[ $RECORD_BASELINE -eq 1 ]] && awk "BEGIN{exit !($EXP_FLOAT == 0.0)}"; then
        # Refuse to record a zero baseline: a zero would permanently disable
        # the gate (measured >= 0 - 0.02 is always true). Fail the row so
        # the operator can investigate (review MED-3).
        if awk -v m="$MEASURED" 'BEGIN{exit !(m == 0.0)}'; then
            VERDICT="fail"
            REASON="baseline_would_be_zero"
            log_evt error "baseline_zero_refused codec=$CODEC model=$MODEL measured=$MEASURED"
            AGG_RC=1
        else
            log_evt info "record_baseline codec=$CODEC model=$MODEL measured=$MEASURED"
            manifest_record "$CODEC" "$MODEL" "$MEASURED" "$MANIFEST"
        fi
    elif awk "BEGIN{exit !($EXP_FLOAT == 0.0)}"; then
        # A zero expectation passes any measurement (measured >= 0 - 0.02).
        VERDICT="unrecorded"
        REASON="no recorded baseline"
        UNRECORDED=$((UNRECORDED + 1))
        log_evt warn "niah_unrecorded codec=$CODEC model=$MODEL measured=$MEASURED"
    else
        # Gate: measured >= expected - 0.02
        if ! awk -v m="$MEASURED" -v e="$EXP_FLOAT" 'BEGIN{exit !(m >= e - 0.02)}'; then
            VERDICT="fail"
            REASON="$(printf 'measured %.4f < expected %.4f - 0.02' "$MEASURED" "$EXP_FLOAT")"
            log_evt error "niah_regress codec=$CODEC model=$MODEL measured=$MEASURED expected=$EXP_FLOAT"
            AGG_RC=1
        else
            log_evt info "niah_pass codec=$CODEC model=$MODEL measured=$MEASURED expected=$EXP_FLOAT"
        fi
    fi

    ROW_JSONS+=("$(printf '{"codec":"%s","model":"%s","smoke":"pass","niah_pct":%s,"expected_pct":%s,"verdict":"%s","reason":"%s"}' \
        "$CODEC" "$MODEL" "$MEASURED" "$EXP_FLOAT" "$VERDICT" "$REASON")")

done <<<"$ROWS_TSV"

# ── aggregate output ────────────────────────────────────────────────────────
SKIPPED=$((SKIPPED_KNOWN + SKIPPED_ABSENT))
INCOMPLETE=false
[[ $SKIPPED_ABSENT -gt 0 || $UNRECORDED -gt 0 ]] && INCOMPLETE=true
{
    printf '{"run_id":"%s","ran":%s,"skipped_known":%s,"skipped_absent":%s,"unrecorded":%s,"incomplete":%s,"rows":[' \
        "$RUN_ID" "$RAN" "$SKIPPED_KNOWN" "$SKIPPED_ABSENT" "$UNRECORDED" "$INCOMPLETE"
    sep=""
    for r in ${ROW_JSONS[@]+"${ROW_JSONS[@]}"}; do
        printf '%s%s' "$sep" "$r"
        sep=","
    done
    printf ']}\n'
} >"$LAST_RUN"

log_evt info "matrix_done agg_rc=$AGG_RC ran=$RAN skipped=$SKIPPED rows=${#ROW_JSONS[@]} out=$LAST_RUN"
if [[ $INCOMPLETE == true ]]; then
    log_evt warn "matrix INCOMPLETE skipped_absent=$SKIPPED_ABSENT unrecorded=$UNRECORDED: rows skipped for an absent snapshot or an unset RMLX_O_MODELS_ROOT, or run without a recorded baseline"
fi
if [[ $AGG_RC -ne 0 ]]; then
    exit "$AGG_RC"
fi
if [[ $RAN -eq 0 ]]; then
    log_evt error "matrix ran no row"
    exit 1
fi
if [[ $INCOMPLETE == true ]]; then
    exit 3
fi
exit 0
