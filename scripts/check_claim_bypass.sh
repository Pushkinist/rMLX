#!/usr/bin/env bash
# scripts/check_claim_bypass.sh — no script, Makefile recipe, test or doc
# deletes the Metal claim, names the claim file, or kills a process by pattern.
#
# The Metal claim is a flock on one file. A second process that deletes the
# file creates a new file with a free lock, so every deletion is a bypass of the
# single-MLX-process rule. A dead holder releases its flock when it exits, so no
# caller needs to delete the file to recover. `pkill -f <pattern>` and `killall`
# cannot prove that the process they kill is one the caller started; a harness
# stops its own children by PID.
#
# Rules (comment lines are scanned too: an operator hint is an instruction):
#   claim-delete  `rm` / `unlink` whose arguments name a claim, or a Rust
#                 `remove_file(` on a line that names a claim.            all files
#   claim-path    a claim file name: `rmlx<anything>.claim`, or the literal
#                 `".claim"`.                                  all files except docs
#   process-kill  the word `pkill` or `killall`.                             all files
#
# Scope, under the scan root:
#   Makefile, scripts/**, crates/*/tests/**, crates/*/src/**/{*_tests.rs,tests.rs},
#   crates/*/examples/**, crates/*/benches/**, crates/*/src/bin/**,
#   docs/**/*.md, CLAUDE.md, CONTRIBUTING.md, README.md.
# This script and its selftest are outside the scope.
#
# Usage: check_claim_bypass.sh [<scan-root>]   (default: the repo root)
# Exit 0 = no violation. 1 = at least one violation. 2 = cannot scan (the root
# is not a directory, or the scope holds no file).

set -uo pipefail
export LC_ALL=C

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -d "$ROOT" ]; then
    echo "check-claim-bypass: unavailable: scan root '$ROOT' is not a directory" >&2
    exit 2
fi

SELF_EXCLUDE='^scripts/check_claim_bypass(_selftest)?\.sh$'

in_scope_files() {
    (
        cd "$ROOT" || exit 2
        find . \( -name .git -o -name target -o -name .rmlx -o -name node_modules \) -prune \
            -o -type f -print |
            sed 's|^\./||' |
            grep -E '^(Makefile|CLAUDE\.md|CONTRIBUTING\.md|README\.md)$|^scripts/|^docs/.*\.md$|^crates/[^/]+/(tests|examples|benches|src/bin)/|^crates/[^/]+/src/(.*/)?([^/]*_tests|tests)\.rs$' |
            grep -Ev "$SELF_EXCLUDE" |
            sort
    )
}

RE_DELETE='(^|[^[:alnum:]_-])(rm|unlink)[[:space:]][^;|&]*claim|remove_file\(.*claim'
RE_PATH='rmlx[^[:space:]/"'"'"'`]*\.claim|"\.claim"'
RE_KILL='(^|[^[:alnum:]_-])(pkill|killall)([^[:alnum:]_-]|$)'

files="$(in_scope_files)"
if [ -z "$files" ]; then
    echo "check-claim-bypass: unavailable: no in-scope file under '$ROOT'" >&2
    exit 2
fi

violations=0
report() {
    local rule="$1" file="$2" hits="$3"
    local line
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        echo "$rule: $file:$line"
        violations=$((violations + 1))
    done <<<"$hits"
}

while IFS= read -r f; do
    report claim-delete "$f" "$(grep -nE "$RE_DELETE" "$ROOT/$f")"
    case "$f" in
        docs/* | *.md) ;;
        *) report claim-path "$f" "$(grep -nE "$RE_PATH" "$ROOT/$f")" ;;
    esac
    report process-kill "$f" "$(grep -nE "$RE_KILL" "$ROOT/$f")"
done <<<"$files"

if [ "$violations" -gt 0 ]; then
    echo "check-claim-bypass: FAIL: $violations violation(s). Stop a server by the PID the claim refusal names; a dead holder's claim is reclaimed without deletion." >&2
    exit 1
fi
echo "check-claim-bypass: ok ($(wc -l <<<"$files" | tr -d ' ') file(s) scanned)"
