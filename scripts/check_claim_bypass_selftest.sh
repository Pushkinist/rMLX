#!/usr/bin/env bash
# scripts/check_claim_bypass_selftest.sh — recall test for check_claim_bypass.sh.
#
# Each case builds a throwaway scan root with one planted line, runs the gate
# and asserts the literal exit code and, for every failure, the rule and the
# file:line it names. Every scan root also holds one clean in-scope file, so a
# must-fail case cannot pass through the "empty scope" branch.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$REPO_ROOT/scripts/check_claim_bypass.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0
PASSED=0

# fresh <name>: a scan root with one clean script in scope.
fresh() {
    local root="$WORK/$1"
    mkdir -p "$root/scripts"
    # shellcheck disable=SC2016 # the planted script must hold the literal $SERVER_PID
    printf '#!/bin/sh\nkill "$SERVER_PID"\nwait "$SERVER_PID"\n' >"$root/scripts/clean.sh"
    printf '%s' "$root"
}

# plant <root> <relpath> <line>: write <line> as line 2 of <relpath>.
plant() {
    mkdir -p "$(dirname "$1/$2")"
    printf '# header\n%s\n' "$3" >"$1/$2"
}

# case_run <name> <what> <want-exit> <needle|-> <root>
# <needle> is the "<rule>: <file>:2:" prefix a failing case must print.
case_run() {
    local name="$1" what="$2" want="$3" needle="$4" root="$5"
    local out status
    out=$(bash "$GATE" "$root" 2>&1)
    status=$?
    if [ "$status" -ne "$want" ]; then
        echo "FAIL $name ($what): exit $status, want $want"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        return
    fi
    if [ "$needle" != "-" ] && ! grep -qF -- "$needle" <<<"$out"; then
        echo "FAIL $name ($what): output does not name '$needle'"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        return
    fi
    PASSED=$((PASSED + 1))
}

# name | relpath | planted line | want-exit | needle ("-" = must pass)
CASES=(
    "rm_glob|scripts/a.sh|rm -f /tmp/rmlx.*.claim|1|claim-delete: scripts/a.sh:2:"
    "rm_fixed_port|scripts/bench/b.sh|rm -f /tmp/rmlx.62265.claim 2>/dev/null \|\| true|1|claim-delete: scripts/bench/b.sh:2:"
    "rm_machine_claim|scripts/c.sh|rm /tmp/rmlx.claim|1|claim-delete: scripts/c.sh:2:"
    "rm_variable|scripts/d.sh|rm -f \"\$CLAIM_FILE\"  # the Metal claim|1|claim-delete: scripts/d.sh:2:"
    "unlink|scripts/e.sh|unlink /tmp/rmlx.claim|1|claim-delete: scripts/e.sh:2:"
    "makefile_recipe|Makefile|	@pkill -f \"rmlx serve\" \|\| true; rm -f /tmp/rmlx.*.claim|1|claim-delete: Makefile:2:"
    "rust_remove_file|crates/x/tests/t.rs|    let _ = std::fs::remove_file(format!(\"/tmp/rmlx.{port}.claim\"));|1|claim-delete: crates/x/tests/t.rs:2:"
    "rust_unit_tests_file|crates/x/src/a/b_tests.rs|    let _ = std::fs::remove_file(format!(\"/tmp/rmlx.{port}.claim\"));|1|claim-delete: crates/x/src/a/b_tests.rs:2:"
    "hint_in_comment|crates/x/tests/t.rs|// Preflight: rm -f /tmp/rmlx.*.claim|1|claim-delete: crates/x/tests/t.rs:2:"
    "doc_hint|docs/X.md|Recover with \`rm -f /tmp/rmlx.claim\`.|1|claim-delete: docs/X.md:2:"
    "claude_md_hint|CLAUDE.md|run rm -f /tmp/rmlx.claim first|1|claim-delete: CLAUDE.md:2:"

    "glob_match|crates/x/tests/t.rs|            if name.ends_with(\".claim\") {|1|claim-path: crates/x/tests/t.rs:2:"
    "path_in_script|scripts/f.sh|CLAIM=/tmp/rmlx.claim|1|claim-path: scripts/f.sh:2:"
    "path_in_example|crates/x/examples/p.rs|// holds /tmp/rmlx.{port}.claim|1|claim-path: crates/x/examples/p.rs:2:"
    "path_in_bin|crates/x/src/bin/d.rs|let p = \"/tmp/rmlx.claim\";|1|claim-path: crates/x/src/bin/d.rs:2:"

    "pkill_shell|scripts/g.sh|pkill -f \"rmlx serve\" 2>/dev/null \|\| true|1|process-kill: scripts/g.sh:2:"
    "pkill_other_server|scripts/h.sh|pkill -f \"llama-server .*--port \${PORT}\"|1|process-kill: scripts/h.sh:2:"
    "killall|scripts/i.sh|killall rmlx|1|process-kill: scripts/i.sh:2:"
    "pkill_rust|crates/x/tests/t.rs|        let _ = Command::new(\"pkill\").args([\"-f\", pat]).output();|1|process-kill: crates/x/tests/t.rs:2:"
    "pkill_python|scripts/bench/j.sh|        subprocess.run([\"pkill\", \"-f\", pat])|1|process-kill: scripts/bench/j.sh:2:"
    "pkill_hint_echo|scripts/k.sh|echo \"Stop it first: pkill -f 'rmlx serve'\"|1|process-kill: scripts/k.sh:2:"
    "pkill_doc|docs/Y.md|The preflight runs \`pkill -f\` on \`rmlx serve\`.|1|process-kill: docs/Y.md:2:"

    "kill_own_pid|scripts/l.sh|kill \"\$SERVER_PID\"; wait \"\$SERVER_PID\"|0|-"
    "doc_names_path|docs/Z.md|The claim file is \`/tmp/rmlx.claim\`; a dead holder releases it.|0|-"
    "rm_other_file|scripts/m.sh|rm -f \"\$WORK/out.json\"|0|-"
    "engine_source_out_of_scope|crates/x/src/claim.rs|    let _ = std::fs::remove_file(format!(\"/tmp/rmlx.{port}.claim\"));|0|-"
    "changelog_out_of_scope|CHANGELOG.md|- scripts no longer run rm -f /tmp/rmlx.*.claim|0|-"
    "word_inside_identifier|scripts/n.sh|no_pkill_here=1; format_claim_report|0|-"
    "word_prefix|scripts/o.sh|echo \"confirm the claim\"; call_nopkill now|0|-"
)

for spec in "${CASES[@]}"; do
    IFS='|' read -r name rel line want needle <<<"${spec//\\|/$'\x1f'}"
    line="${line//$'\x1f'/|}"
    root="$(fresh "$name")"
    plant "$root" "$rel" "$line"
    case_run "$name" "$rel: $line" "$want" "$needle" "$root"
done

# The gate's own files are outside its scope.
root="$(fresh self_exclude)"
plant "$root" scripts/check_claim_bypass.sh "rm -f /tmp/rmlx.claim; pkill -f rmlx"
plant "$root" scripts/check_claim_bypass_selftest.sh "rm -f /tmp/rmlx.claim; pkill -f rmlx"
case_run self_exclude "the gate and its selftest are not scanned" 0 - "$root"

# A scope with no file cannot pass.
mkdir -p "$WORK/empty_scope/crates/x/src"
printf 'rm -f /tmp/rmlx.claim\n' >"$WORK/empty_scope/crates/x/src/lib.rs"
case_run empty_scope "no in-scope file is exit 2, not a pass" 2 "unavailable: no in-scope file" "$WORK/empty_scope"

case_run missing_root "a missing scan root is exit 2" 2 "is not a directory" "$WORK/does-not-exist"

echo "check-claim-bypass selftest: $PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
