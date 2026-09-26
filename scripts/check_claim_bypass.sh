#!/usr/bin/env bash
# scripts/check_claim_bypass.sh — nothing in the tree deletes the Metal claim,
# names the claim file outside its owner, or kills a process by pattern.
#
# The Metal claim is a flock on one file. A process that deletes the file lets
# the next process create a new file with a free lock, so every deletion is a
# bypass of the single-MLX-process rule. A dead holder releases its flock when
# it exits, so no caller needs to delete the file to recover. A kill by name
# pattern (pkill, killall, pgrep / lsof -t output into kill) cannot prove that
# the caller started the process it stops; a harness stops its own children by
# PID.
#
# Rules. Matching ignores case. Comment lines are scanned too: an operator hint
# is an instruction. A shell argument list stops at `#`, `;`, `|` or `&`.
#   claim-delete  a delete whose argument names a claim or /tmp/rmlx (a
#                 substring of /var/tmp/rmlx): `rm` / `unlink` (not `git rm`), a
#                 `remove(` / `remove_file(` / `unlink(` call, a `.unlink()` /
#                 `.remove()` on such a path, `find … -delete` / `-exec rm`
#                 under /tmp or /var/tmp, or such a path piped into `xargs rm`.
#   claim-path    a claim file name: `rmlx<anything>.claim`, or the literal
#                 `".claim"`. Docs (`*.md`) and the claim module
#                 (crates/rmlx-server/src/claim.rs) are exempt.
#   process-kill  `pkill` or `killall`; `kill` of `$(pgrep …)` / `$(lsof …)`
#                 or the backtick form; `xargs … kill`.
#
# Scope, under the scan root: Makefile, scripts/**, .github/**,
# crates/*/{src,tests,examples,benches}/**, docs/**/*.md, CLAUDE.md,
# CONTRIBUTING.md, README.md. This script and its selftest are outside it.
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

SCOPE='^(Makefile|CLAUDE\.md|CONTRIBUTING\.md|README\.md)$|^scripts/|^\.github/|^docs/.*\.md$|^crates/[^/]+/(src|tests|examples|benches)/'
SELF_EXCLUDE='^scripts/check_claim_bypass(_selftest)?\.sh$'
CLAIM_OWNER='crates/rmlx-server/src/claim.rs'

in_scope_files() {
    (
        cd "$ROOT" || exit 2
        find . \( -name .git -o -name target -o -name .rmlx -o -name node_modules \) -prune \
            -o -type f -print |
            sed 's|^\./||' |
            grep -E "$SCOPE" |
            grep -Ev "$SELF_EXCLUDE" |
            sort
    )
}

# `read -d ''` and not `$(cat <<EOF)`: bash 3.2 misparses the single quote
# inside a heredoc inside a command substitution.
IFS= read -r -d '' AWK_RULES <<'EOF'
{
    l = tolower($0)
    d = l
    gsub(/git[ \t]+rm[ \t]/, "git_rm ", d)
    if (d ~ /(^|[^a-z0-9_-])(rm|unlink)[ \t][^#;|&]*(claim|\/tmp\/rmlx)/ ||
        d ~ /(^|[^a-z0-9_])(remove|remove_file|unlink)[ \t]*[(][^#;]*(claim|\/tmp\/rmlx)/ ||
        d ~ /(claim|\/tmp\/rmlx)[^#;]*[.](unlink|remove)[ \t]*[(]/ ||
        d ~ /find[ \t][^#;|&]*\/tmp[^#;|&]*rmlx[^#;|&]*(-delete|-exec[ \t]+rm)/ ||
        d ~ /(claim|\/tmp\/rmlx)[^#;]*[|][ \t]*xargs[^#;|]*(rm|unlink)([^a-z0-9_-]|$)/)
        print "claim-delete: " file ":" FNR ": " $0
    if (pathrule && l ~ /rmlx[^ \t\/"'`]*[.]claim|"[.]claim"/)
        print "claim-path: " file ":" FNR ": " $0
    if (l ~ /(^|[^a-z0-9_-])(pkill|killall)([^a-z0-9_-]|$)/ ||
        l ~ /(^|[^a-z0-9_-])kill[^#;|]*([$][(]|`)[ \t]*(pgrep|lsof)/ ||
        l ~ /xargs[ \t][^#;|]*kill([^a-z0-9_-]|$)/)
        print "process-kill: " file ":" FNR ": " $0
}
EOF

files="$(in_scope_files)"
if [ -z "$files" ]; then
    echo "check-claim-bypass: unavailable: no in-scope file under '$ROOT'" >&2
    exit 2
fi

violations=0
while IFS= read -r f; do
    pathrule=1
    case "$f" in
        *.md | "$CLAIM_OWNER") pathrule=0 ;;
    esac
    hits="$(awk -v file="$f" -v pathrule="$pathrule" "$AWK_RULES" "$ROOT/$f")"
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits"
        violations=$((violations + $(wc -l <<<"$hits")))
    fi
done <<<"$files"

if [ "$violations" -gt 0 ]; then
    echo "check-claim-bypass: FAIL: $violations violation(s). Stop a server by the PID the claim refusal names; a dead holder's claim is reclaimed without deletion." >&2
    exit 1
fi
echo "check-claim-bypass: ok ($(wc -l <<<"$files" | tr -d ' ') file(s) scanned)"
