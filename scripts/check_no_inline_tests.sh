#!/usr/bin/env bash
# scripts/check_no_inline_tests.sh — CI gate: fail if any non-test source file
# contains an inline `#[cfg(test)] mod tests {` or `mod <name>_tests {` block.
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
# The check scans crates/**/src/**/*.rs excluding:
#   - files under tests/ directories (integration tests)
#   - tests.rs files
#   - *_tests.rs files
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

violators=()

while IFS= read -r -d '' f; do
    # Use awk to detect the banned pattern:
    # A line matching `#[cfg(test)]` (possibly with surrounding whitespace)
    # is immediately followed (next non-blank line) by `mod <name> {` (brace).
    # The `;` form is explicitly allowed — it points at a sibling file.
    #
    # The found flag avoids macOS awk's END-block exit-code clobber issue.
    if awk '
        BEGIN { cfg = 0; found = 0 }
        /^[[:space:]]*#\[cfg\(test\)\][[:space:]]*$/ {
            cfg = 1
            next
        }
        cfg && /^[[:space:]]*$/ { next }
        cfg && /^[[:space:]]*#\[/ { next }
        cfg && /^[[:space:]]*mod[[:space:]]+[a-z_]+[[:space:]]*\{/ {
            found = 1
        }
        { cfg = 0 }
        END { exit !found }
    ' "$f" 2>/dev/null; then
        violators+=("$f")
    fi
done < <(
    find "${REPO_ROOT}/crates" -name "*.rs" \
        -not -path "*/target/*" \
        -not -path "*/tests/*" \
        -not -name "tests.rs" \
        -not -name "*_tests.rs" \
        -print0
)

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
