#!/usr/bin/env bash
# scripts/check_doc_source_citations.sh — CI gate: every `crates/...` source
# path cited in a tracked docs/*.md file must exist in the tree.
#
# Why this exists: the subsystem docs replace figures with citations — "pinned
# by X (`crates/.../y_tests.rs`)", "the predicate lives in `crates/.../z.rs`".
# That trade is only honest while the citations resolve. A crate rename or a
# `foo.rs` -> `foo/mod.rs` split leaves the prose reading correctly and pointing
# at nothing, which is indistinguishable from a stale figure and harder to spot.
#
# Scope: backticked paths beginning `crates/` and ending in a source extension,
# in files `git ls-files docs` reports. Fenced code blocks are included on
# purpose — a shell snippet naming a moved file is as wrong as prose naming it.
#
# Not in scope: identifiers (a test or function name), which need a language
# parser to tell from a variable, and non-crate paths.
#
# Exit 0 = every cited path resolves. Exit 1 = at least one does not.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC2016
pattern='`(crates/[A-Za-z0-9_][A-Za-z0-9_./-]*\.(rs|metal|sql|toml|txt))`'

violations=0
cited=0

while IFS= read -r doc; do
    [ -f "$doc" ] || continue
    while IFS=: read -r lineno path; do
        cited=$((cited + 1))
        if [ ! -e "$path" ]; then
            if [ "$violations" -eq 0 ]; then
                echo "ERROR: docs cite source paths that do not exist:" >&2
            fi
            echo "  ${doc}:${lineno}: ${path}" >&2
            violations=$((violations + 1))
        fi
    done < <(grep -nEo "$pattern" "$doc" | sed 's/`//g')
done < <(git ls-files 'docs/*.md' 'docs/**/*.md')

if [ "$violations" -gt 0 ]; then
    echo >&2
    echo "Repoint each citation at the file that holds the item today, or drop it." >&2
    echo "A citation that resolves to nothing is the same defect as a stale figure." >&2
    exit 1
fi

echo "OK: ${cited} cited source paths in docs/ all resolve."
