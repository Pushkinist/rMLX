#!/usr/bin/env bash
# Print one version's CHANGELOG.md section, for use as the GitHub release body.
#
# Use:  gh release create v<ver> --notes-file <(scripts/release/changelog_section.sh <ver>)
#
# Defaults to the current [workspace.package].version.
#
# Usage:
#   scripts/release/changelog_section.sh            # current version
#   scripts/release/changelog_section.sh 0.1.0      # explicit version
set -euo pipefail
cd "$(dirname "$0")/../.."

VER="${1:-$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)}"
[ -n "$VER" ] || { echo "error: no version" >&2; exit 1; }

# Extract lines from `## [VER]` up to (but not including) the next `## [` header.
awk -v ver="$VER" '
  $0 ~ "^## \\[" ver "\\]" { grab=1; next }
  grab && /^## \[/ { exit }
  grab { print }
' CHANGELOG.md | sed -e 's/^[[:space:]]*$//' | awk 'NF{blank=0} !NF{blank++} blank<2 || NF'
