#!/usr/bin/env bash
# scripts/codec_smoke_runner_selftest.sh — recall test for the verdict of
# `scripts/release_e2e/stage6_perf/codec_smoke_runner.sh`.
#
# doc-refs: fixture
#
# Every case runs the runner with --dry-run over a throwaway manifest and a
# throwaway models root, so nothing builds, no GPU is touched and no model is
# loaded. Each case asserts the literal exit code and the text the run prints.
# The gate cases run a real pass over a copy of the runner with a stub `cargo` and
# a stub `rmlx` that print a chosen NIAH reading, so the verdict path runs.
# One more case holds the runner's slug table equal to the NIAH cells' slugs.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO_ROOT/scripts/release_e2e/stage6_perf/codec_smoke_runner.sh"
NIAH="$REPO_ROOT/crates/rmlx-models/tests/niah_long_context.rs"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

GEMMA_SLUG="mlx-community__gemma-4-e4b-it-mxfp8"
QWEN38_SLUG="mlx-community__Qwen3.8-27B-4bit"

FAILED=0
PASSED=0

# Models root holding a runnable stub of each slug named.
make_root() {
    local root="$WORK/root_$1" slug
    shift
    mkdir -p "$root"
    for slug in "$@"; do
        mkdir -p "$root/$slug"
        : >"$root/$slug/config.json"
        : >"$root/$slug/tokenizer.json"
        : >"$root/$slug/model.safetensors"
    done
    printf '%s' "$root"
}

# row <model> <codec> [skip_reason] [expected]: one manifest entry.
row() {
    printf '[[entries]]\ncodec_name = "%s"\nmodel = "%s"\ncontext_length = 32768\n' "$2" "$1"
    printf 'expected_retrieval_pct = %s\nsmoke_probe_prompts = []\nskip_reason = "%s"\n' "${4:-0.0}" "${3:-}"
    printf 'cli_args = "--kv-quant %s"\nniah_filter = "niah_%s"\n\n' "$2" "$2"
}

# run_case <name> <want-exit> <what> <root|UNSET> <manifest> <needle>...
# A needle starting with ! must be absent from the output.
run_case() {
    local name="$1" want="$2" what="$3" root="$4" manifest="$5"
    shift 5
    local out="$WORK/$name.log" got ok=1 needle
    if [ "$root" = UNSET ]; then
        env -u RMLX_O_MODELS_ROOT bash "$RUNNER" --dry-run --manifest "$manifest" \
            --out "$WORK/$name.json" >"$out" 2>&1
    else
        RMLX_O_MODELS_ROOT="$root" bash "$RUNNER" --dry-run --manifest "$manifest" \
            --out "$WORK/$name.json" >"$out" 2>&1
    fi
    got=$?
    [ "$got" = "$want" ] || ok=0
    for needle in "$@"; do
        if [ "${needle#!}" != "$needle" ]; then
            ! grep -qF -- "${needle#!}" "$out" || { ok=0; needle="unexpected: ${needle#!}"; }
        else
            grep -qF -- "$needle" "$out" || { ok=0; needle="missing: $needle"; }
        fi
    done
    if [ "$ok" = 1 ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-30s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-30s exit=%s (want %s) — %s\n' "$name" "$got" "$want" "$what"
        sed 's/^/       | /' "$out" | tail -15
    fi
}

# json_has <name> <what> <needle>: the result file of a finished case.
json_has() {
    if grep -qF -- "$3" "$WORK/$1.json"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-30s — %s\n' "$1.json" "$2"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-30s — %s (missing %s)\n' "$1.json" "$2" "$3"
    fi
}

ROOT_GEMMA="$(make_root gemma "$GEMMA_SLUG")"
NOT_A_DIR="$WORK/not_a_directory"
: >"$NOT_A_DIR"

{ row gemma4-e4b k8v4; } >"$WORK/m_present.toml"
{ row gemma4-e4b k8v4; row qwen3.8-27b-4bit k8v4; } >"$WORK/m_absent.toml"
{ row gemma4-e4b k8v4; row gemma4-e4b planar "known-disposition"; } >"$WORK/m_known.toml"
{ row gemma4-e4b planar "known-disposition"; } >"$WORK/m_only_known.toml"

echo "codec_smoke_runner selftest"

run_case present 0 "a present slug runs and the verdict is clean" \
    "$ROOT_GEMMA" "$WORK/m_present.toml" "ran=1 skipped=0" "snapshot=present" "!INCOMPLETE"
json_has present "the result file records ran=1 and not incomplete" '"ran":1,"skipped_known":0,"skipped_absent":0,"unrecorded":0,"incomplete":false'

run_case absent_slug 3 "an absent slug beside a run row is INCOMPLETE with its own exit code" \
    "$ROOT_GEMMA" "$WORK/m_absent.toml" "ran=1 skipped=1" "snapshot=absent" "matrix INCOMPLETE"
json_has absent_slug "the result file records the incompleteness" '"skipped_absent":1,"unrecorded":0,"incomplete":true'

run_case skip_reason_row 0 "a manifest skip_reason row is a known disposition, not INCOMPLETE" \
    "$ROOT_GEMMA" "$WORK/m_known.toml" "ran=1 skipped=1" "!INCOMPLETE"
json_has skip_reason_row "the result file counts it as skipped_known" '"skipped_known":1,"skipped_absent":0,"unrecorded":0,"incomplete":false'

run_case unset_root 1 "an unset root runs no row: non-zero, and INCOMPLETE is said" \
    UNSET "$WORK/m_absent.toml" "ran=0 skipped=2" "matrix INCOMPLETE" "matrix ran no row"

run_case only_known 1 "a manifest of known dispositions alone ran nothing: non-zero" \
    "$ROOT_GEMMA" "$WORK/m_only_known.toml" "ran=0 skipped=1" "matrix ran no row" "!INCOMPLETE"

run_case root_not_a_directory 1 "a root that is not a directory fails the row" \
    "$NOT_A_DIR" "$WORK/m_present.toml" "models_root_not_a_directory"

# Gate cases: a real pass over a copy of the runner. The stub `rmlx` prints
# $STUB_FOUND as every needle reading; the stub `cargo` builds nothing.
FAKE="$WORK/fake"
mkdir -p "$FAKE/scripts/release_e2e/stage6_perf" "$FAKE/target/release-perf" "$FAKE/bin"
cp "$RUNNER" "$REPO_ROOT/scripts/release_e2e/stage6_perf/smoke_prompts.toml" \
    "$FAKE/scripts/release_e2e/stage6_perf/"
printf '#!/bin/sh\nfor i in 1 2 3 4 5; do echo "needle_found=$STUB_FOUND"; done\nexit ${STUB_RC:-0}\n' >"$FAKE/target/release-perf/rmlx"
printf '#!/bin/sh\nexit 0\n' >"$FAKE/bin/cargo"
chmod +x "$FAKE/target/release-perf/rmlx" "$FAKE/bin/cargo"
# A `cargo` whose `run` fails, as `rmlx baseline` does when the Metal claim is held.
mkdir -p "$FAKE/bin_run_fails"
printf '#!/bin/sh\n[ "$1" = run ] && exit 7\nexit 0\n' >"$FAKE/bin_run_fails/cargo"
chmod +x "$FAKE/bin_run_fails/cargo"

# gate_case <name> <want-exit> <what> <found> <manifest> <flag> <needle>...
# GATE_BIN names the stub `cargo` directory (default: one that always succeeds).
gate_case() {
    local name="$1" want="$2" what="$3" found="$4" manifest="$5" flag="$6" got ok=1 needle
    shift 6
    local out="$WORK/$name.log"
    PATH="${GATE_BIN:-$FAKE/bin}:$PATH" STUB_FOUND="$found" RMLX_O_MODELS_ROOT="$ROOT_GEMMA" \
        bash "$FAKE/scripts/release_e2e/stage6_perf/codec_smoke_runner.sh" \
        --manifest "$manifest" --out "$WORK/$name.json" $flag >"$out" 2>&1
    got=$?
    [ "$got" = "$want" ] || ok=0
    for needle in "$@"; do
        if [ "${needle#!}" != "$needle" ]; then
            ! grep -qF -- "${needle#!}" "$out" || { ok=0; needle="unexpected: ${needle#!}"; }
        else
            grep -qF -- "$needle" "$out" || { ok=0; needle="missing: $needle"; }
        fi
    done
    if [ "$ok" = 1 ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-30s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-30s exit=%s (want %s) — %s\n' "$name" "$got" "$want" "$what"
        sed 's/^/       | /' "$out" | tail -15
    fi
}

{ row gemma4-e4b k8v4 "" 0.0; } >"$WORK/g_zero.toml"
{ row gemma4-e4b k8v4 "" 1.0; } >"$WORK/g_one.toml"
cp "$WORK/g_zero.toml" "$WORK/g_record.toml"

gate_case gate_zero_unrecorded 3 "a 0.0 expectation outside a record run is unrecorded and INCOMPLETE" \
    true "$WORK/g_zero.toml" "" "niah_unrecorded" "matrix INCOMPLETE" "!niah_pass"
json_has gate_zero_unrecorded "the result file marks the row unrecorded" '"verdict":"unrecorded"'
gate_case gate_recorded_pass 0 "a recorded baseline that is met passes" \
    true "$WORK/g_one.toml" "" "niah_pass" "!INCOMPLETE"
gate_case gate_recorded_regress 1 "a recorded baseline that is missed fails" \
    false "$WORK/g_one.toml" "" "niah_regress"
gate_case gate_record_run 0 "a record run still records a zero expectation" \
    true "$WORK/g_record.toml" "--record-baseline" "record_baseline" "!niah_unrecorded"

sed 's/^smoke_probe_prompts = \[\]/smoke_probe_prompts = ["coherence"]/' "$WORK/g_one.toml" >"$WORK/g_smoke.toml"
GATE_BIN="$FAKE/bin_run_fails" gate_case gate_baseline_fails 1 "a baseline that exits non-zero fails the row with its own status" \
    true "$WORK/g_smoke.toml" "" "smoke_fail" "rc=7" "!reason=validate"

STUB_RC=7 gate_case gate_niah_fails 1 "a NIAH pass that exits non-zero fails the row with its own status" \
    true "$WORK/g_one.toml" "" "niah_fail" "rc=7" "!niah_pass" "!niah_no_pct"

# The runner and the NIAH cells each list the slugs; they must be the same set.
runner_slugs="$(sed -n '/^model_slug()/,/^}/p' "$RUNNER" | grep -o "printf '[^']*'" | sed "s/printf '//; s/'$//" | grep -v '^$' | sort)"
niah_slugs="$(grep -o 'slug: "[^"]*"' "$NIAH" | sed 's/slug: "//; s/"$//' | sort)"
if [ -n "$runner_slugs" ] && [ "$runner_slugs" = "$niah_slugs" ]; then
    PASSED=$((PASSED + 1))
    printf '  ok   %-30s — the runner and the NIAH cells list the same slugs\n' twin_slug_lists
else
    FAILED=$((FAILED + 1))
    printf '  FAIL %-30s — the runner and the NIAH cells list different slugs\n       runner: %s\n       niah:   %s\n' \
        twin_slug_lists "$(echo "$runner_slugs" | tr '\n' ' ')" "$(echo "$niah_slugs" | tr '\n' ' ')"
fi

echo
if [ "$FAILED" -eq 0 ]; then
    echo "codec_smoke_runner_selftest: OK ($PASSED checks)"
    exit 0
fi
echo "codec_smoke_runner_selftest: FAIL ($FAILED of $((FAILED + PASSED)))"
exit 1
