#!/usr/bin/env bash
# scripts/bench/codec_matrix/runner_selftest.sh — recall test for the verdict of
# `scripts/bench/codec_matrix/runner.sh`.
#
# doc-refs: fixture
#
# Every case runs the runner with --dry-run over a throwaway manifest and a
# throwaway models root, so nothing builds, no GPU is touched and no model is
# loaded. Each case asserts the literal exit code and the text the run prints.
# The gate cases run a real pass over a copy of the runner with a stub `cargo` and
# a stub `rmlx` that print a chosen NIAH reading, so the verdict path runs.
# Cases over a stub `cargo` that prints a chosen log hold the smoke verdict: no decoded field, an escaped
# quote, an empty text. The decoded-text reader (decoded_field.py over lib/decoded_text.py) is run on logs
# of its own. One more case holds the runner's slug table equal to the NIAH cells' slugs.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNNER="$REPO_ROOT/scripts/bench/codec_matrix/runner.sh"
FIELD_CLI="$REPO_ROOT/scripts/bench/codec_matrix/decoded_field.py"
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
json_has present "the result file records ran=1 and not incomplete" '"ran":1,"skipped_known":0,"skipped_absent":0,"unrecorded":0,"records_pending":0,"incomplete":false'

run_case absent_slug 3 "an absent slug beside a run row is INCOMPLETE with its own exit code" \
    "$ROOT_GEMMA" "$WORK/m_absent.toml" "ran=1 skipped=1" "snapshot=absent" "matrix INCOMPLETE"
json_has absent_slug "the result file records the incompleteness" '"skipped_absent":1,"unrecorded":0,"records_pending":0,"incomplete":true'

run_case skip_reason_row 0 "a manifest skip_reason row is a known disposition, not INCOMPLETE" \
    "$ROOT_GEMMA" "$WORK/m_known.toml" "ran=1 skipped=1" "!INCOMPLETE"
json_has skip_reason_row "the result file counts it as skipped_known" '"skipped_known":1,"skipped_absent":0,"unrecorded":0,"records_pending":0,"incomplete":false'

run_case unset_root 1 "an unset root runs no row: non-zero, and INCOMPLETE is said" \
    UNSET "$WORK/m_absent.toml" "ran=0 skipped=2" "matrix INCOMPLETE" "matrix ran no row"

run_case only_known 1 "a manifest of known dispositions alone ran nothing: non-zero" \
    "$ROOT_GEMMA" "$WORK/m_only_known.toml" "ran=0 skipped=1" "matrix ran no row" "!INCOMPLETE"

run_case root_not_a_directory 1 "a root that is not a directory fails the row" \
    "$NOT_A_DIR" "$WORK/m_present.toml" "models_root_not_a_directory"

RMLX_HOME=rel/home run_case relative_rmlx_home 2 "a relative RMLX_HOME is a setup error and the matrix does not start" \
    "$ROOT_GEMMA" "$WORK/m_present.toml" "RMLX_HOME=rel/home must be an absolute path" "!ran="

# Gate cases: a real pass over a copy of the runner. The stub `rmlx` prints
# $STUB_FOUND as every needle reading; the stub `cargo` builds nothing.
FAKE="$WORK/fake"
. "$REPO_ROOT/scripts/lib/identity.sh"
FAKE_RMLX="$(rmlx_default_binary "$FAKE" release-perf)"
mkdir -p "$FAKE/scripts/bench/codec_matrix" "$(dirname "$FAKE_RMLX")" "$FAKE/bin"
cp "$RUNNER" "$REPO_ROOT/scripts/bench/codec_matrix/smoke_prompts.toml" "$FIELD_CLI" \
    "$FAKE/scripts/bench/codec_matrix/"
mkdir -p "$FAKE/scripts/lib"
cp "$REPO_ROOT/scripts/lib/record_pending.sh" "$REPO_ROOT/scripts/lib/identity.sh" "$REPO_ROOT/scripts/lib/rmlx_home.sh" \
    "$REPO_ROOT/scripts/lib/decoded_text.py" "$FAKE/scripts/lib/"
printf '#!/bin/sh\nfor i in 1 2 3 4 5; do echo "needle_found=$STUB_FOUND"; done\nexit ${STUB_RC:-0}\n' >"$FAKE_RMLX"
printf '#!/bin/sh\nexit 0\n' >"$FAKE/bin/cargo"
chmod +x "$FAKE_RMLX" "$FAKE/bin/cargo"
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
        bash "$FAKE/scripts/bench/codec_matrix/runner.sh" \
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

# A `cargo` whose baseline measured fluent text and whose record did not land.
mkdir -p "$FAKE/bin_pending"
cat >"$FAKE/bin_pending/cargo" <<'CARGO'
#!/bin/sh
[ "$1" = run ] || exit 0
echo 'decoded="The sun rises slowly over the quiet hills, and the light spreads gold across every field."'
echo "error: baseline: measured, but the record did not reach runs.db; it is kept at /x/pending/smoke.json for \`rmlx metrics record --replay-pending\`" >&2
exit 75
CARGO
chmod +x "$FAKE/bin_pending/cargo"
GATE_BIN="$FAKE/bin_pending" gate_case gate_smoke_record_pending 3 "a smoke run whose record did not land is judged on its output, named, and the matrix is INCOMPLETE" \
    true "$WORK/g_smoke.toml" "" "record_pending" "record pending: /x/pending/smoke.json" "niah_pass" "!smoke_fail"
json_has gate_smoke_record_pending "the result file counts the pending record" '"records_pending":1'

# trace_cargo <case> <trace>: a `cargo` whose baseline run prints <trace> and exits 0.
trace_cargo() {
    mkdir -p "$FAKE/bin_$1"
    printf '%s\n' "$2" >"$WORK/$1.trace"
    printf '#!/bin/sh\n[ "$1" = run ] || exit 0\ncat "%s"\n' "$WORK/$1.trace" >"$FAKE/bin_$1/cargo"
    chmod +x "$FAKE/bin_$1/cargo"
}
sed 's/\["coherence"\]/["multi_turn"]/' "$WORK/g_smoke.toml" >"$WORK/g_smoke_seed.toml"

trace_cargo no_field 'INFO baseline: prompt echo: The coastal lighthouse was painted red and white.'
GATE_BIN="$FAKE/bin_no_field" gate_case gate_smoke_no_decoded_field 1 "a log with no decoded field fails the smoke, though its prompt echo matches the row's regex" \
    true "$WORK/g_smoke_seed.toml" "" "smoke_fail" "reason=no_decoded_field" "no decoded field" "!niah_pass"
json_has gate_smoke_no_decoded_field "the result file names the prompt and the reason" '"fail_prompt":"multi_turn:no_decoded_field"'

trace_cargo escaped_quote 'INFO baseline: decoded preview decoded="it was \"red\" indeed"'
GATE_BIN="$FAKE/bin_escaped_quote" gate_case gate_smoke_escaped_quote 0 "a decoded text is read past an escaped quote" \
    true "$WORK/g_smoke_seed.toml" "" "niah_pass" "!smoke_fail"

trace_cargo empty_text 'INFO baseline: decoded preview decoded=""'
GATE_BIN="$FAKE/bin_empty_text" gate_case gate_smoke_empty_text 1 "an empty decoded text fails the smoke as empty output" \
    true "$WORK/g_smoke_seed.toml" "" "smoke_fail" "reason=validate" "fail: empty output" "!niah_pass"

# field_case <name> <want-exit> <what> <log> <want-stdout>: the decoded-text reader over a log on stdin.
field_case() {
    local name="$1" want="$2" what="$3" out got err
    out="$(printf '%s' "$4" | python3 "$FIELD_CLI" 2>"$WORK/$name.err")"
    got=$?
    err="$(cat "$WORK/$name.err")"
    if [ "$got" = "$want" ] && [ "$out" = "$5" ] && { [ "$want" = 0 ] || [ "$err" = "no decoded field in the log" ]; }; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-30s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-30s exit=%s (want %s) — %s\n       | stdout [%s] want [%s]\n       | stderr [%s]\n' \
            "$name" "$got" "$want" "$what" "$out" "$5" "$err"
    fi
}

field_case field_none 1 "a log with no decoded field exits 1 and names it" 'INFO baseline: prompt: red and white' ''
field_case field_empty 0 "an empty decoded text is printed as nothing" 'decoded=""' ''
field_case field_escaped_quote 0 "a decoded text runs past an escaped quote" 'decoded="say \"hi\" now"' 'say "hi" now'
field_case field_backslash 0 "an escaped backslash is one backslash" 'decoded="C:\\temp"' 'C:\temp'
field_case field_newline_tab 0 "escaped newline and tab are the characters" 'decoded="a\nb\tc"' "$(printf 'a\nb\tc')"
field_case field_ansi 0 "the colour codes the subscriber wraps around the field are stripped" \
    "$(printf '\033[2m12:00\033[0m INFO baseline: decoded preview \033[3mdecoded\033[0m\033[2m=\033[0m"x y"')" 'x y'
field_case field_last_wins 0 "of two fields, the last is the text" 'decoded="first" and later decoded="second"' 'second'

# A `cargo` that logs the argv of each `run` and builds nothing.
mkdir -p "$FAKE/bin_argv"
printf '#!/bin/sh\n[ "$1" = run ] && printf "%%s\\n" "$*" >>"$ARGV_LOG"\nexit 0\n' >"$FAKE/bin_argv/cargo"
chmod +x "$FAKE/bin_argv/cargo"
ARGV_LOG="$WORK/smoke.argv"
: >"$ARGV_LOG"
PATH="$FAKE/bin_argv:$PATH" ARGV_LOG="$ARGV_LOG" RMLX_O_MODELS_ROOT="$ROOT_GEMMA" RMLX_HOME="$WORK/home_smoke" \
    bash "$FAKE/scripts/bench/codec_matrix/runner.sh" --manifest "$WORK/g_smoke.toml" \
    --out "$WORK/smoke_argv.json" >"$WORK/smoke_argv.log" 2>&1
if grep -q -- 'baseline .*--prompt-label coherence .*--record --run-class diagnostic --label codec_smoke:k8v4:coherence' "$ARGV_LOG"; then
    PASSED=$((PASSED + 1))
    printf '  ok   %-30s — the baseline of each smoke prompt records as diagnostic\n' codec_smoke_records_diagnostic
else
    FAILED=$((FAILED + 1))
    printf '  FAIL %-30s — the baseline of a smoke prompt does not record as diagnostic\n' codec_smoke_records_diagnostic
    printf '       | baseline argv: %s\n' "$(grep baseline "$ARGV_LOG" | head -1)"
    sed 's/^/       | /' "$WORK/smoke_argv.log" | tail -2
fi

STUB_RC=7 gate_case gate_niah_fails 1 "a NIAH pass that exits non-zero fails the row with its own status" \
    true "$WORK/g_one.toml" "" "niah_fail" "rc=7" "!niah_pass" "!niah_no_pct"

# Without --out the result file lands under RMLX_HOME and nowhere under the runner's tree.
HOME_CASE="$WORK/home_case"
PATH="$FAKE/bin:$PATH" STUB_FOUND=true RMLX_HOME="$HOME_CASE" RMLX_O_MODELS_ROOT="$ROOT_GEMMA" \
    bash "$FAKE/scripts/bench/codec_matrix/runner.sh" --manifest "$WORK/g_one.toml" \
    >"$WORK/last_run_home.log" 2>&1
if [ -s "$HOME_CASE/bench/codec_matrix/last_run.json" ] && [ -z "$(find "$FAKE/scripts" -name last_run.json)" ] \
    && [ ! -e "$FAKE/.rmlx" ]; then
    PASSED=$((PASSED + 1))
    printf '  ok   %-30s — the result file is under RMLX_HOME and not under scripts/\n' last_run_under_rmlx_home
else
    FAILED=$((FAILED + 1))
    printf '  FAIL %-30s — the result file is not under RMLX_HOME alone\n' last_run_under_rmlx_home
    sed 's/^/       | /' "$WORK/last_run_home.log" | tail -15
fi

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
