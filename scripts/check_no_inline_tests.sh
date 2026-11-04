#!/usr/bin/env bash
# CI gate: no inline #[cfg(test)] module body in a non-test source file.
#
# Usage:  bash scripts/check_no_inline_tests.sh [<root>]
# Exit:   0 clean; 1 a violation; 2 nothing to scan under <root>/crates
#
# <root> is the directory that holds crates/; it defaults to the repository.
#
# ALLOWED (sibling-pointer pattern):
#   #[cfg(test)]
#   #[path = "foo_tests.rs"]
#   mod foo_tests;
#
#   #[cfg(test)]
#   mod tests;        <- semicolon form (points at sibling tests.rs)
#
# BANNED (inline block):
#   #[cfg(test)]
#   mod tests {       <- brace form — test body lives inside this file
#
# A module of any name, with any visibility, with the brace on the line or the
# next one, under any cfg attribute that names `test` (`cfg(all(test, ..))`,
# `cfg(any(test, ..))`), on the same line as the attribute or after it.
# `cfg(not(test))` and a `test` inside a string literal do not name it.
#
# Known gap: a comment line between the cfg attribute and the module hides it.
#
# The check scans crates/**/src/**/*.rs excluding:
#   - files under tests/ directories (integration tests)
#   - tests.rs files
#   - *_tests.rs files

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="${1:-${REPO_ROOT}}"

if [ ! -d "${SCAN_ROOT}/crates" ]; then
    echo "check_no_inline_tests: no crates directory at ${SCAN_ROOT}/crates, nothing to scan." >&2
    exit 2
fi

violators=()
scanned=0

while IFS= read -r -d '' f; do
    scanned=$((scanned + 1))
    # Use awk to detect the banned pattern: a cfg attribute that names `test`,
    # then (after blank lines and other attributes) `mod <name> {`. The `;` form
    # is explicitly allowed — it points at a sibling file.
    #
    # The found flag avoids macOS awk's END-block exit-code clobber issue.
    if awk '
        function names_test(attr) {
            gsub(/"[^"]*"/, "", attr)
            gsub(/not[[:space:]]*\([[:space:]]*test[[:space:]]*\)/, "", attr)
            return attr ~ /(^|[^A-Za-z0-9_])test([^A-Za-z0-9_]|$)/
        }
        BEGIN {
            cfg = 0; brace_next = 0; found = 0
            mod = "^[[:space:]]*(pub([[:space:]]*\\([^)]*\\))?[[:space:]]+)?mod[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*"
            mod_brace = mod "\\{"
            mod_bare = mod "$"
        }
        {
            line = $0
            if (brace_next && line ~ /^[[:space:]]*$/) next
            if (brace_next && line ~ /^[[:space:]]*\{/) found = 1
            brace_next = 0
            hit = cfg
            while (match(line, /^[[:space:]]*#\[[^]]*\]/)) {
                attr = substr(line, 1, RLENGTH)
                line = substr(line, RLENGTH + 1)
                if (attr ~ /^[[:space:]]*#\[cfg\(/ && names_test(attr)) hit = 1
            }
            if (line ~ /^[[:space:]]*$/) { cfg = hit; next }
            if (hit && line ~ mod_brace) found = 1
            if (hit && line ~ mod_bare) brace_next = 1
            cfg = 0
        }
        END { exit !found }
    ' "$f" 2>/dev/null; then
        violators+=("$f")
    fi
done < <(
    find "${SCAN_ROOT}/crates" -name "*.rs" \
        -not -path "*/target/*" \
        -not -path "*/tests/*" \
        -not -name "tests.rs" \
        -not -name "*_tests.rs" \
        -print0
)

if [ "${scanned}" -eq 0 ]; then
    echo "check_no_inline_tests: no .rs file under ${SCAN_ROOT}/crates, nothing to scan." >&2
    exit 2
fi

if [ ${#violators[@]} -gt 0 ]; then
    echo "ERROR: inline #[cfg(test)] mod <name> { ... } blocks found outside tests.rs / *_tests.rs:" >&2
    for f in "${violators[@]}"; do
        echo "  $f" >&2
    done
    echo >&2
    echo "Move test bodies to a sibling *_tests.rs file and reference via:" >&2
    echo "  #[cfg(test)]" >&2
    echo "  #[path = \"<name>_tests.rs\"]" >&2
    echo "  mod <name>_tests;" >&2
    exit 1
fi

echo "OK: no inline test modules outside sibling test files."
