#!/usr/bin/env bash
# CI gate: forbidden lines in the files git does not ignore, one rule set per mode.
#
# Both modes scan the files git does not ignore (tracked plus untracked-not-ignored), by the
# names git lists them under, whatever characters a name holds: a git-ignored file cannot
# hold a violation the gate reports. Matching ignores case. A file git lists but the work
# tree lacks is skipped; a file in scope that cannot be read ends the run.
#
# The tool's own files are left out by identity (the same file, whatever path or symlink
# reaches it), never by name: claim-bypass leaves out this tool and its recall test (the
# file beside it with `_selftest` before `.sh`); personal-data leaves out the recall test
# only, and scans the tool. A file of the same name in another tree is scanned.
#
# claim-bypass: nothing deletes the Metal claim, names its file outside its owner, or
# kills by name.
#
#   The Metal claim is a flock on one file. A process that deletes the file lets the next
#   process create a new file with a free lock, so every deletion is a bypass of the
#   single-MLX-process rule. A dead holder releases its flock when it exits, so no caller
#   needs to delete the file to recover. A kill by name pattern (pkill, killall, pgrep /
#   lsof -t output into kill) cannot prove that the caller started the process it stops;
#   a harness stops its own children by PID.
#
#   Comment lines are scanned too: an operator hint is an instruction. No marker exempts
#   a line. A shell argument list stops at `#`, `;`, `|` or `&`.
#     claim-delete  a delete whose argument names a claim or /tmp/rmlx (a substring of
#                   /var/tmp/rmlx): `rm` / `unlink` (not `git rm`), a `remove(` /
#                   `remove_file(` / `unlink(` call, a `.unlink()` / `.remove()` on such
#                   a path, `find … -delete` / `-exec rm` under /tmp or /var/tmp, or such
#                   a path piped into `xargs rm`.
#     claim-path    a claim file name: `rmlx<anything>.claim`, or the literal `".claim"`.
#                   Docs (`*.md`) and the claim module (crates/rmlx-server/src/claim.rs)
#                   are exempt.
#     process-kill  `pkill` or `killall`; `kill` of `$(pgrep …)` / `$(lsof …)` or the
#                   backtick form; `xargs … kill`.
#   Scope, under the scan root: Makefile, scripts/**, .github/**,
#   crates/*/{src,tests,examples,benches}/**, docs/**/*.md, CLAUDE.md, CONTRIBUTING.md,
#   README.md.
#
# personal-data: no personal email, macOS home or temp path, or assistant session link
# in published files.
#
#   The repository is public, and a line published once is cached and indexed even after
#   it is edited out.
#     email         an address (a local part, an at sign, a dotted domain), unless its
#                   domain is a placeholder or a GitHub noreply address: `example.invalid`,
#                   `example.com`, `example.org`, `example.net`,
#                   `users.noreply.github.com`, `noreply.github.com`.
#     home-path     a macOS home directory (the `Users` directory at the root, then a
#                   name) or a per-user temp directory (the per-user `folders` directory
#                   under `/var`, or an assistant scratch directory under `/private/tmp`).
#                   Linux `/home/<name>` is not a rule: the tree uses it for placeholders
#                   such as `/home/user`.
#     session-link  a link to an assistant session on the assistant's web host.
#   Scope: every text file under the scan root. Binary files are skipped.
#
# Usage:  bash scripts/check/tree_lines.sh <claim-bypass|personal-data> [<scan-root>]
#         scan-root: default the repo root
# Exit:   0 no violation; 1 at least one violation; 2 cannot scan: no mode or an unknown
#         one, a root that is not a directory or not inside a git work tree, a scope that
#         holds no file, a file that cannot be read, or no file left to scan because every
#         one in scope is missing, empty or binary
# Not a mode of: check_no_inline_tests.sh — that gate reads Rust items (attributes, braces)
#                under crates/; this one matches raw lines, comments included, in any file

set -euo pipefail
export LC_ALL=C

SELF="${BASH_SOURCE[0]}"
while [ -L "$SELF" ]; do
    link="$(readlink "$SELF")"
    case "$link" in
    /*) SELF="$link" ;;
    *) SELF="$(dirname "$SELF")/$link" ;;
    esac
done
SELF_DIR="$(cd "$(dirname "$SELF")" && pwd -P)"
SELF_PATH="$SELF_DIR/$(basename "$SELF")"
SELFTEST_PATH="${SELF_PATH%.sh}_selftest.sh"
REPO_ROOT="$(cd "$SELF_DIR/../.." && pwd -P)"

# `read -d ''` and not `$(cat <<EOF)`: bash 3.2 misparses the single quote
# inside a heredoc inside a command substitution. read returns 1 at the end of input.
IFS= read -r -d '' CLAIM_BYPASS_AWK <<'EOF' || true
BEGIN {
    pathrule = (file !~ /[.]md$/ && file != "crates/rmlx-server/src/claim.rs")
}
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

IFS= read -r -d '' PERSONAL_DATA_AWK <<'EOF' || true
{
    l = tolower($0)
    rest = l
    while (match(rest, /[a-z0-9._%+-]+@[a-z0-9-]+([.][a-z0-9-]+)*[.][a-z][a-z]+/)) {
        addr = substr(rest, RSTART, RLENGTH)
        rest = substr(rest, RSTART + RLENGTH)
        domain = addr
        sub(/^[^@]*@/, "", domain)
        if (domain !~ /^(example[.](invalid|com|org|net)|users[.]noreply[.]github[.]com|noreply[.]github[.]com)$/) {
            print "email: " file ":" FNR ": " $0
            break
        }
    }
    if (l ~ /\/users\/[a-z0-9._-]+/ || l ~ /\/var\/folders\// || l ~ /\/private\/tmp\/claude/)
        print "home-path: " file ":" FNR ": " $0
    if (l ~ /claude[.]ai\/code\/session/)
        print "session-link: " file ":" FNR ": " $0
}
EOF

MODE="${1:-}"
case "$MODE" in
claim-bypass)
    LABEL=check-claim-bypass
    SCOPE='^(Makefile|CLAUDE\.md|CONTRIBUTING\.md|README\.md)$|^scripts/|^\.github/|^docs/.*\.md$|^crates/[^/]+/(src|tests|examples|benches)/'
    AWK_RULES="$CLAIM_BYPASS_AWK"
    SKIP_BINARY=0
    SKIP_TOOL=1
    HINT="Stop a server by the PID the claim refusal names; a dead holder's claim is reclaimed without deletion."
    ;;
personal-data)
    LABEL=check-personal-data
    SCOPE='.'
    AWK_RULES="$PERSONAL_DATA_AWK"
    SKIP_BINARY=1
    SKIP_TOOL=0
    HINT="Replace an address with a placeholder such as contact@example.invalid, and a home path with a repo-relative path or an env-var name."
    ;;
*)
    echo "usage: bash scripts/check/tree_lines.sh <claim-bypass|personal-data> [<scan-root>]" >&2
    exit 2
    ;;
esac

ROOT="${2:-$REPO_ROOT}"
if [ ! -d "$ROOT" ]; then
    echo "$LABEL: unavailable: scan root '$ROOT' is not a directory" >&2
    exit 2
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/tree_lines.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Names arrive NUL-terminated: git quotes a name with a tab, a quote or a non-ASCII
# character in newline-terminated output.
in_scope_files() {
    local f
    while IFS= read -r -d '' f; do
        [[ $f =~ $SCOPE ]] || continue
        if [ "$ROOT/$f" -ef "$SELFTEST_PATH" ]; then
            continue
        fi
        if [ "$SKIP_TOOL" = 1 ] && [ "$ROOT/$f" -ef "$SELF_PATH" ]; then
            continue
        fi
        printf '%s\0' "$f"
    done
}

rc=0
git -C "$ROOT" ls-files -z --cached --others --exclude-standard 2>"$WORK/err" |
    in_scope_files | sort -z >"$WORK/files" || rc=$?
if [ "$rc" -ne 0 ]; then
    echo "$LABEL: unavailable: git ls-files exit $rc: $(cat "$WORK/err")" >&2
    exit 2
fi
if [ ! -s "$WORK/files" ]; then
    echo "$LABEL: unavailable: no in-scope file under '$ROOT'" >&2
    exit 2
fi

violations=0
scanned=0
while IFS= read -r -d '' f; do
    [ -f "$ROOT/$f" ] || continue
    if [ ! -r "$ROOT/$f" ]; then
        echo "$LABEL: unavailable: cannot read $f" >&2
        exit 2
    fi
    if [ "$SKIP_BINARY" = 1 ] && ! grep -Iq . "$ROOT/$f" 2>/dev/null; then
        continue
    fi
    scanned=$((scanned + 1))
    hits="$(awk -v file="$f" "$AWK_RULES" "$ROOT/$f")"
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits"
        violations=$((violations + $(wc -l <<<"$hits")))
    fi
done <"$WORK/files"

if [ "$scanned" -eq 0 ]; then
    echo "$LABEL: unavailable: no file to scan under '$ROOT': every in-scope file is missing, empty or binary" >&2
    exit 2
fi
if [ "$violations" -gt 0 ]; then
    echo "$LABEL: FAIL: $violations violation(s). $HINT" >&2
    exit 1
fi
echo "$LABEL: ok ($scanned file(s) scanned)"
