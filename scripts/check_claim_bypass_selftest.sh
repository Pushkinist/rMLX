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
# A literal `|` inside the planted line is written `\|`.
CASES=(
    "rm_glob|scripts/a.sh|rm -f /tmp/rmlx.*.claim|1|claim-delete: scripts/a.sh:2:"
    "rm_fixed_port|scripts/bench/b.sh|rm -f /tmp/rmlx.62265.claim 2>/dev/null \|\| true|1|claim-delete: scripts/bench/b.sh:2:"
    "rm_machine_claim|scripts/c.sh|rm /var/tmp/rmlx.claim|1|claim-delete: scripts/c.sh:2:"
    "rm_variable|scripts/d.sh|rm -f \"\$CLAIM\"|1|claim-delete: scripts/d.sh:2:"
    "rm_variable_mixed_case|scripts/d2.sh|rm -f \"\$Metal_Claim_File\"|1|claim-delete: scripts/d2.sh:2:"
    "rm_prefix_glob|scripts/d3.sh|rm -f /tmp/rmlx*|1|claim-delete: scripts/d3.sh:2:"
    "rm_var_tmp_prefix_glob|scripts/d4.sh|rm -f /var/tmp/rmlx*|1|claim-delete: scripts/d4.sh:2:"
    "unlink|scripts/e.sh|unlink /tmp/rmlx.claim|1|claim-delete: scripts/e.sh:2:"
    "find_delete|scripts/e2.sh|find /tmp -name 'rmlx*' -delete|1|claim-delete: scripts/e2.sh:2:"
    "find_exec_rm|scripts/e3.sh|find /var/tmp -maxdepth 1 -name 'rmlx*' -exec rm {} +|1|claim-delete: scripts/e3.sh:2:"
    "ls_xargs_rm|scripts/e4.sh|ls /tmp/rmlx.*.claim \| xargs rm -f|1|claim-delete: scripts/e4.sh:2:"
    "makefile_recipe|Makefile|	@pkill -f \"rmlx serve\" \|\| true; rm -f /tmp/rmlx.*.claim|1|claim-delete: Makefile:2:"
    "rust_remove_file|crates/x/tests/t.rs|    let _ = std::fs::remove_file(format!(\"/tmp/rmlx.{port}.claim\"));|1|claim-delete: crates/x/tests/t.rs:2:"
    "rust_unit_tests_file|crates/x/src/a/b_tests.rs|    let _ = std::fs::remove_file(format!(\"/tmp/rmlx.{port}.claim\"));|1|claim-delete: crates/x/src/a/b_tests.rs:2:"
    "rust_engine_source|crates/x/src/lib.rs|        let _ = std::fs::remove_file(&self.claim_path);|1|claim-delete: crates/x/src/lib.rs:2:"
    "claim_owner_unlinks|crates/rmlx-server/src/claim.rs|        let _ = std::fs::remove_file(&self.claim_path);|1|claim-delete: crates/rmlx-server/src/claim.rs:2:"
    "python_os_remove_doc|docs/P.md|    os.remove(\"/var/tmp/rmlx.claim\")|1|claim-delete: docs/P.md:2:"
    "python_path_unlink|scripts/q.py|Path(\"/var/tmp/rmlx.claim\").unlink(missing_ok=True)|1|claim-delete: scripts/q.py:2:"
    "hint_in_comment|crates/x/tests/t.rs|// Preflight: rm -f /tmp/rmlx.*.claim|1|claim-delete: crates/x/tests/t.rs:2:"
    "doc_hint|docs/X.md|Recover with \`rm -f /var/tmp/rmlx.claim\`.|1|claim-delete: docs/X.md:2:"
    "claude_md_hint|CLAUDE.md|run rm -f /tmp/rmlx.claim first|1|claim-delete: CLAUDE.md:2:"
    "github_workflow|.github/workflows/ci.yml|        run: rm -f /var/tmp/rmlx.claim|1|claim-delete: .github/workflows/ci.yml:2:"

    "glob_match|crates/x/tests/t.rs|            if name.ends_with(\".claim\") {|1|claim-path: crates/x/tests/t.rs:2:"
    "path_in_script|scripts/f.sh|CLAIM=/var/tmp/rmlx.claim|1|claim-path: scripts/f.sh:2:"
    "path_in_example|crates/x/examples/p.rs|// holds /tmp/rmlx.{port}.claim|1|claim-path: crates/x/examples/p.rs:2:"
    "path_in_bin|crates/x/src/bin/d.rs|let p = \"/var/tmp/rmlx.claim\";|1|claim-path: crates/x/src/bin/d.rs:2:"
    "path_in_engine_source|crates/x/src/lib.rs|const P: &str = \"/var/tmp/rmlx.claim\";|1|claim-path: crates/x/src/lib.rs:2:"
    "path_in_other_claim_rs|crates/x/src/claim.rs|const P: &str = \"/var/tmp/rmlx.claim\";|1|claim-path: crates/x/src/claim.rs:2:"
    "path_in_workflow|.github/workflows/ci.yml|        run: test -e /var/tmp/rmlx.claim|1|claim-path: .github/workflows/ci.yml:2:"

    "pkill_shell|scripts/g.sh|pkill -f \"rmlx serve\" 2>/dev/null \|\| true|1|process-kill: scripts/g.sh:2:"
    "pkill_other_server|scripts/h.sh|pkill -f \"llama-server .*--port \${PORT}\"|1|process-kill: scripts/h.sh:2:"
    "killall|scripts/i.sh|killall rmlx|1|process-kill: scripts/i.sh:2:"
    "kill_pgrep_subst|scripts/i2.sh|kill \$(pgrep -f 'rmlx serve')|1|process-kill: scripts/i2.sh:2:"
    "kill_pgrep_backtick|scripts/i3.sh|kill -9 \`pgrep mlx_lm\`|1|process-kill: scripts/i3.sh:2:"
    "kill_lsof_subst|scripts/i4.sh|kill \$(lsof -ti :8080)|1|process-kill: scripts/i4.sh:2:"
    "pgrep_xargs_kill|scripts/i5.sh|pgrep -f rmlx \| xargs kill|1|process-kill: scripts/i5.sh:2:"
    "lsof_xargs_kill|scripts/i6.sh|lsof -ti :8080 \| xargs kill -9|1|process-kill: scripts/i6.sh:2:"
    "pkill_rust_test|crates/x/tests/t.rs|        let _ = Command::new(\"pkill\").args([\"-f\", pat]).output();|1|process-kill: crates/x/tests/t.rs:2:"
    "pkill_rust_src|crates/x/src/run.rs|        let _ = Command::new(\"pkill\").arg(pat).status();|1|process-kill: crates/x/src/run.rs:2:"
    "pkill_python|scripts/bench/j.sh|        subprocess.run([\"pkill\", \"-f\", pat])|1|process-kill: scripts/bench/j.sh:2:"
    "pkill_hint_echo|scripts/k.sh|echo \"Stop it first: pkill -f 'rmlx serve'\"|1|process-kill: scripts/k.sh:2:"
    "pkill_doc|docs/Y.md|The preflight runs \`pkill -f\` on \`rmlx serve\`.|1|process-kill: docs/Y.md:2:"
    "pkill_workflow|.github/workflows/ci.yml|        run: pkill -f rmlx \|\| true|1|process-kill: .github/workflows/ci.yml:2:"

    "kill_own_pid|scripts/l.sh|kill \"\$SERVER_PID\"; wait \"\$SERVER_PID\"|0|-"
    "doc_names_path|docs/Z.md|The claim file is \`/var/tmp/rmlx.claim\`; a dead holder releases it.|0|-"
    "doc_never_unlinked|docs/Z2.md|The claim file is never unlinked, and no script deletes it.|0|-"
    "rule_row_wording|CLAUDE.md|\| \`make check-claim-bypass\` \| CI gate: nothing deletes the claim or kills by process-name pattern. \||0|-"
    "claim_owner_names_path|crates/rmlx-server/src/claim.rs|const CLAIM_PATH: &str = \"/var/tmp/rmlx.claim\";|0|-"
    "rm_other_file|scripts/m.sh|rm -f \"\$WORK/out.json\"|0|-"
    "rm_arg_stops_at_hash|scripts/m2.sh|rm -f log.txt  # keep the claim log apart|0|-"
    "git_rm_claim_named_file|scripts/m3.sh|git rm scripts/old_claim_probe.sh|0|-"
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
mkdir -p "$WORK/empty_scope/other"
printf 'rm -f /tmp/rmlx.claim\n' >"$WORK/empty_scope/other/notes.txt"
case_run empty_scope "no in-scope file is exit 2, not a pass" 2 "unavailable: no in-scope file" "$WORK/empty_scope"

case_run missing_root "a missing scan root is exit 2" 2 "is not a directory" "$WORK/does-not-exist"

echo "check-claim-bypass selftest: $PASSED passed, $FAILED failed"
[ "$FAILED" -eq 0 ]
