#!/usr/bin/env bash
# Recall test for check_claim_bypass.sh: planted git work trees, asserting exit code and file:line.
#
#
# doc-refs: fixture — the docs/ paths below belong to the synthetic scan roots.
#
# Recall test for: check_claim_bypass.sh
#
# Each case builds a throwaway git work tree with one planted line, runs the
# gate and asserts the literal exit code and, for every failure, the rule and
# the file:line it names. Every scan root also holds one clean in-scope file,
# so a must-fail case cannot pass through the "empty scope" branch. A few
# cases plant a `.gitignore` to prove the gate scans exactly the files git
# does not ignore, and one runs outside any git work tree.
#
# Exit 0 = every case held. Exit 1 = at least one did not. Exit 2 = a scan root
# could not be built.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$REPO_ROOT/scripts/check_claim_bypass.sh"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init check-claim-bypass-selftest

# fresh <name>: a git work tree with one clean script in scope.
fresh() {
    local root
    root="$(st_git_root "$1")"
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
    "contributing_hint|CONTRIBUTING.md|run rm -f /tmp/rmlx.claim first|1|claim-delete: CONTRIBUTING.md:2:"
    "readme_hint|README.md|run rm -f /tmp/rmlx.claim first|1|claim-delete: README.md:2:"
    "rust_bench_file|crates/x/benches/b.rs|    let _ = std::fs::remove_file(&claim_path);|1|claim-delete: crates/x/benches/b.rs:2:"
    "nested_doc_hint|docs/sub/deeper/N.md|run rm -f /tmp/rmlx.claim first|1|claim-delete: docs/sub/deeper/N.md:2:"
    "doc_non_md_out_of_scope|docs/notes.txt|run rm -f /tmp/rmlx.claim first|0|-"
    "python_object_remove|scripts/q2.py|claim_file.remove()|1|claim-delete: scripts/q2.py:2:"
    "python_os_unlink|scripts/q3.py|os.unlink(\"/tmp/rmlx.claim\")|1|claim-delete: scripts/q3.py:2:"
    "xargs_unlink|scripts/e5.sh|ls /tmp/rmlx.*.claim \| xargs unlink|1|claim-delete: scripts/e5.sh:2:"
)

for spec in "${CASES[@]}"; do
    IFS='|' read -r name rel line want needle <<<"${spec//\\|/$'\x1f'}"
    line="${line//$'\x1f'/|}"
    root="$(fresh "$name")"
    plant "$root" "$rel" "$line"
    if [ "$needle" = - ]; then
        st_case "$name" "$want" --what "$rel: $line" -- bash "$GATE" "$root"
    else
        st_case "$name" "$want" --what "$rel: $line" --has "$needle" -- bash "$GATE" "$root"
    fi
done

# The gate's own files are outside its scope.
root="$(fresh self_exclude)"
plant "$root" scripts/check_claim_bypass.sh "rm -f /tmp/rmlx.claim; pkill -f rmlx"
plant "$root" scripts/check_claim_bypass_selftest.sh "rm -f /tmp/rmlx.claim; pkill -f rmlx"
st_case self_exclude 0 --what "the gate and its selftest are not scanned" -- bash "$GATE" "$root"

# doc_rows_plant <docs root> <scan root>: copy the lines of CLAUDE.md and
# scripts/INDEX.md that name the gate into the scan root; set why when a doc
# has fewer than two (why is empty otherwise).
doc_rows_plant() {
    local docs="$1" root="$2" doc rows
    why=""
    for doc in CLAUDE.md scripts/INDEX.md; do
        rows="$(grep -F -e 'check-claim-bypass' -e 'check_claim_bypass' "$docs/$doc" || true)"
        if [ "$(grep -c '' <<<"$rows")" -lt 2 ]; then
            why="$doc has fewer than two rows naming the gate"
            return 0
        fi
        mkdir -p "$(dirname "$root/$doc")"
        printf '%s\n' "$rows" >"$root/$doc"
    done
}

# The rows that document the gate say what it refuses without spelling a
# refused command. They are planted as the tree has them, so a row reworded into
# a violation fails here, and a lost row fails as a missing row.
root="$(fresh doc_rows)"
doc_rows_plant "$REPO_ROOT" "$root"
if [ -n "$why" ]; then
    st_verdict doc_rows "$why"
else
    st_case doc_rows 0 --what "the gate's rows in CLAUDE.md and scripts/INDEX.md pass it" -- bash "$GATE" "$root"
fi

# A doc with no row is named, and the verdict is reached.
docs="$ST_WORK/docs_without_rows"
mkdir -p "$docs/scripts"
printf 'no row here\n' >"$docs/CLAUDE.md"
printf 'no row here\n' >"$docs/scripts/INDEX.md"
root="$(fresh doc_rows_lost)"
doc_rows_plant "$docs" "$root"
st_verdict doc_rows_lost "$([ "$why" = "CLAUDE.md has fewer than two rows naming the gate" ] || echo "verdict is '$why'")" \
    "a doc with no row is named, not a silent abort"

# A violation in a file that is tracked, not merely untracked, is reported.
root="$(fresh tracked_violation)"
plant "$root" scripts/staged.sh 'rm -f /tmp/rmlx.claim'
git -C "$root" add scripts/staged.sh
st_case tracked_violation 1 --what "a staged file is scanned" \
    --has "claim-delete: scripts/staged.sh:2:" -- bash "$GATE" "$root"

# The summary counts every violation, not the files that hold one.
root="$(fresh two_violations)"
plant "$root" docs/v1.md 'rm -f /tmp/rmlx.claim'
plant "$root" docs/v2.md 'rm -f /tmp/rmlx.claim'
st_case two_violations_counted 1 --what "two planted lines are two violations" \
    --has "FAIL: 2 violation(s)" -- bash "$GATE" "$root"

# A scope with no file cannot pass.
root="$(st_git_root empty_scope)"
mkdir -p "$root/other"
printf 'rm -f /tmp/rmlx.claim\n' >"$root/other/notes.txt"
st_case empty_scope 2 --what "no in-scope file is exit 2, not a pass" \
    --has "unavailable: no in-scope file" -- bash "$GATE" "$root"

st_case missing_root 2 --what "a missing scan root is exit 2" \
    --has "is not a directory" -- bash "$GATE" "$ST_WORK/does-not-exist"

# A violation inside a git-ignored file is invisible to the gate; the same
# violation in an untracked, not-ignored file is reported.
root="$(fresh ignored_violation)"
printf 'scripts/hidden.sh\n' >"$root/.gitignore"
plant "$root" scripts/hidden.sh 'rm -f /tmp/rmlx.claim'
st_case ignored_violation 0 --what "a violation in a git-ignored file is not reported" -- bash "$GATE" "$root"

root="$(fresh untracked_violation)"
plant "$root" scripts/visible.sh 'rm -f /tmp/rmlx.claim'
st_case untracked_violation 1 --what "the same violation, untracked but not git-ignored, is reported" \
    --has "claim-delete: scripts/visible.sh:2:" -- bash "$GATE" "$root"

# Outside any git work tree the scope is unknown: fail closed, exit 2.
root="$ST_WORK/not_a_git_tree"
mkdir -p "$root/scripts"
printf '#!/bin/sh\nrm -f /tmp/rmlx.claim\n' >"$root/scripts/x.sh"
st_case not_a_git_tree 2 --what "outside a git work tree the gate cannot scan: exit 2" \
    --has "unavailable: git ls-files exit 128" --lacks "no in-scope file" -- bash "$GATE" "$root"

st_done
