#!/usr/bin/env bash
# NIAH (Needle-In-A-Haystack) validation driver.
#
# Two flash families share this driver:
#
#   * TurboFlash — gated on KvQuant::K8V4 + kv_seq > 4096.
#     Env: RMLX_TURBO_FLASH={0|1}. Tests:  niah_<model>_*
#
#   * planar_flash_decode — gated on KvStorage::PlanarK + pow-2
#     head_dim. Env: RMLX_PLANAR_FLASH_DECODE={0|1}. Tests: niah_pflash_<model>_*
#
# Runs the relevant test family in two passes (OFF then ON) per `--mode`. Each
# pass is a fresh `cargo test` process because the kernel gates are `OnceLock`
# values latched on first read — flipping mid-process has no effect.
#
# The test bodies themselves are `#[ignore]` so plain `cargo test` skips
# them; this driver passes `-- --ignored --test-threads=1` to opt in
# (serial: one model at a time, to honour the single-MLX-process rule).
#
# Single-MLX-process discipline (CLAUDE.md hard rule 8): each cargo invocation
# runs under `rmlx claim run`, which holds the Metal claim for the whole pass.
# A claim another process holds refuses the pass with exit 11 and names the
# holder; nothing is stopped for it.
#
# Usage:
#   bash scripts/release_e2e/stage6_perf/niah_long_context.sh \
#     [--mode {turbo|pflash|both}] [--off-only|--on-only] [--filter <substr>]
#
#   --mode MODE   `turbo` (default) — TurboFlash cells; sets RMLX_TURBO_FLASH.
#                 `pflash`         — planar_flash_decode cells; sets
#                                    RMLX_PLANAR_FLASH_DECODE. Filters to
#                                    `niah_pflash_` by default unless
#                                    `--filter` is passed.
#                 `both`           — runs turbo then pflash.
#   --off-only    Run only the baseline (kernel OFF) sweep.
#   --on-only     Run only the kernel-ON sweep.
#   --filter STR  Pass `STR` as the cargo-test test-name filter (e.g.
#                 `niah_pflash_bonsai_16k` to limit scope).
#
# Env vars (required, per docs/TESTING.md):
#   RMLX_TEST_MODEL_GEMMA4_E4B
#   RMLX_TEST_MODEL_QWEN36
#   RMLX_TEST_MODEL_BONSAI
#
# Skips gracefully when an env var is unset (printed by the test binary).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

FILTER=""
MODE="turbo"
RUN_OFF=true
RUN_ON=true

# Manual flag parse (preserve back-compat: bare `--filter STR` and positional STR).
while [[ $# -gt 0 ]]; do
    case "$1" in
        --off-only) RUN_ON=false; shift ;;
        --on-only)  RUN_OFF=false; shift ;;
        --mode)
            if [[ $# -lt 2 ]]; then echo "ERROR: --mode requires a value" >&2; exit 2; fi
            MODE="$2"; shift 2 ;;
        --filter)
            if [[ $# -lt 2 ]]; then echo "ERROR: --filter requires a value" >&2; exit 2; fi
            FILTER="$2"; shift 2 ;;
        --*)
            echo "ERROR: unknown flag: $1" >&2
            exit 2 ;;
        *)
            # Back-compat: bare positional becomes the filter.
            if [[ -z "$FILTER" ]]; then FILTER="$1"; fi
            shift ;;
    esac
done

case "$MODE" in
    turbo|pflash|both) ;;
    *) echo "ERROR: --mode must be one of {turbo,pflash,both}, got: $MODE" >&2; exit 2 ;;
esac

# The binary the passes take the Metal claim with is this checkout's.
CLAIM_RMLX="${REPO_ROOT}/target/debug/rmlx"
# Every process a pass starts under the claim inherits it, so everything the
# passes run is compiled here, outside it; a pass that still compiles fails.
cargo build --manifest-path "${REPO_ROOT}/Cargo.toml" -p rmlx-cli --bin rmlx || exit 1
cargo test --manifest-path "${REPO_ROOT}/Cargo.toml" --no-run --profile release-perf \
    -p rmlx-models --test niah_long_context || exit 1

# Run a single pass for one family (turbo|pflash) × one mode label (off|on).
# Sets the corresponding env var and applies a default filter when none is
# supplied so the pflash pass does not also run all the turbo cells.
run_pass() {
    local family="$1"   # turbo | pflash
    local label="$2"    # off | on
    local val           # 0 or 1
    case "$label" in
        off) val=0 ;;
        on)  val=1 ;;
        *) echo "ERROR: run_pass label must be off|on" >&2; return 2 ;;
    esac

    local env_var
    local default_filter
    case "$family" in
        turbo)
            env_var="RMLX_TURBO_FLASH"
            default_filter=""
            ;;
        pflash)
            env_var="RMLX_PLANAR_FLASH_DECODE"
            # Default scope: pflash cells only. User can override with --filter.
            default_filter="niah_pflash_"
            ;;
        *) echo "ERROR: run_pass family must be turbo|pflash" >&2; return 2 ;;
    esac

    local effective_filter="${FILTER:-$default_filter}"

    echo ""
    echo "================================================================"
    echo "NIAH pass: family=$family label=$label ($env_var=$val) filter=${effective_filter:-<all>}"
    echo "================================================================"

    # `--test-threads=1`: only one MLX context at a time.
    # `--ignored`: cells are `#[ignore]` by default.
    # `--nocapture`: surface the per-cell prompt_len/decoded lines.
    local args=(test --manifest-path "${REPO_ROOT}/Cargo.toml" --profile release-perf
        -p rmlx-models --test niah_long_context)
    if [[ -n "$effective_filter" ]]; then
        args+=("--" "--ignored" "--test-threads=1" "--nocapture" "$effective_filter")
    else
        args+=("--" "--ignored" "--test-threads=1" "--nocapture")
    fi

    # Per-invocation env: declare via `env <NAME>=<VAL>` so $env_var expands
    # correctly. Direct `"$env_var"="$val" cmd` would not be a valid env-assignment.
    env "${env_var}=${val}" timeout 1800 "${CLAIM_RMLX}" claim run -- cargo "${args[@]}" 2>&1 \
        | tee "/tmp/niah-${family}-${label}.log"
    local rc=${PIPESTATUS[0]}
    if grep -Eq '^[[:space:]]*Compiling [A-Za-z0-9_-]+ v' "/tmp/niah-${family}-${label}.log"; then
        echo "ERROR: the pass compiled under the Metal claim." >&2
        rc=1
    fi
    echo ""
    echo "exit=$rc  log=/tmp/niah-${family}-${label}.log"
    return $rc
}

# Run a family in OFF / ON / both orientations per the user flags.
run_family() {
    local family="$1"
    local rc_off=0
    local rc_on=0
    if $RUN_OFF; then run_pass "$family" "off" || rc_off=$?; fi
    if $RUN_ON;  then run_pass "$family" "on"  || rc_on=$?;  fi
    echo "  family=$family OFF=$rc_off ON=$rc_on"
    if (( rc_off != 0 )) || (( rc_on != 0 )); then return 1; fi
    return 0
}

agg_rc=0
case "$MODE" in
    turbo)  run_family turbo  || agg_rc=$? ;;
    pflash) run_family pflash || agg_rc=$? ;;
    both)
        run_family turbo  || agg_rc=$?
        run_family pflash || agg_rc=$?
        ;;
esac

echo ""
echo "================================================================"
echo "NIAH summary mode=$MODE  agg_rc=$agg_rc"
echo "================================================================"
exit $agg_rc
