#!/usr/bin/env bash
# scripts/check_personal_data.sh — no personal data in the files the public
# repository publishes.
#
# The repository is public, and a line published once is cached and indexed
# even after it is edited out. This gate keeps three kinds of line out of every
# file git does not ignore.
#
# Rules. Matching ignores case.
#   email         an address `local@domain.tld`, unless its domain is a
#                 placeholder or a GitHub noreply address: `example.invalid`,
#                 `example.com`, `example.org`, `example.net`,
#                 `users.noreply.github.com`, `noreply.github.com`.
#   home-path     a macOS home directory (`/Users/<name>`) or a per-user temp
#                 directory (`/var/folders/`, `/private/tmp/claude`). Linux
#                 `/home/<name>` is not a rule: the tree uses it for
#                 placeholders such as `/home/user`.
#   session-link  a link to an assistant session (`claude.ai/code/session`).
#
# Scope: every text file under the scan root that git does not ignore
# (tracked plus untracked-not-ignored). Binary files are skipped. This script
# and its selftest are outside the scope.
#
# Usage: check_personal_data.sh [<scan-root>]   (default: the repo root)
# Exit 0 = no violation. 1 = at least one violation. 2 = cannot scan (the root
# is not a directory, the root is not inside a git work tree, or the scope
# holds no file).

set -uo pipefail
export LC_ALL=C

ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -d "$ROOT" ]; then
    echo "check-personal-data: unavailable: scan root '$ROOT' is not a directory" >&2
    exit 2
fi

SELF_EXCLUDE='^scripts/check_personal_data(_selftest)?\.sh$'

in_scope_files() {
    local tracked rc
    tracked="$(git -C "$ROOT" ls-files --cached --others --exclude-standard 2>&1)"
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "check-personal-data: unavailable: git ls-files exit $rc: $tracked" >&2
        return 2
    fi
    printf '%s\n' "$tracked" | grep -Ev "$SELF_EXCLUDE" | grep -v '^$' | sort
}

# `read -d ''` and not `$(cat <<EOF)`: bash 3.2 misparses the single quote
# inside a heredoc inside a command substitution.
IFS= read -r -d '' AWK_RULES <<'EOF'
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

files="$(in_scope_files)"
if [ $? -eq 2 ]; then
    exit 2
fi
if [ -z "$files" ]; then
    echo "check-personal-data: unavailable: no in-scope file under '$ROOT'" >&2
    exit 2
fi

violations=0
scanned=0
while IFS= read -r f; do
    [ -f "$ROOT/$f" ] || continue
    # Skip binary files: grep -I treats a file with a NUL byte as binary.
    grep -Iq . "$ROOT/$f" 2>/dev/null || continue
    scanned=$((scanned + 1))
    hits="$(awk -v file="$f" "$AWK_RULES" "$ROOT/$f")"
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits"
        violations=$((violations + $(wc -l <<<"$hits")))
    fi
done <<<"$files"

if [ "$violations" -gt 0 ]; then
    echo "check-personal-data: FAIL: $violations violation(s). Replace an address with a placeholder such as contact@example.invalid, and a home path with a repo-relative path or an env-var name." >&2
    exit 1
fi
echo "check-personal-data: ok ($scanned file(s) scanned)"
