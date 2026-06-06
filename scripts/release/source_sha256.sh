#!/usr/bin/env bash
# Compute the sha256 of a GitHub source tarball for the Homebrew formula, and
# (optionally) patch packaging/homebrew/rmlx.rb in place.
#
# GitHub serves the auto-generated source tarball at:
#   https://github.com/<owner>/<repo>/archive/refs/tags/v<ver>.tar.gz
# The tag must already exist (and the repo must be reachable — public, or a gh
# token in the environment for a private repo).
#
# Usage:
#   scripts/release/source_sha256.sh                 # print sha for current version
#   scripts/release/source_sha256.sh --write         # also patch the formula's sha256 line
set -euo pipefail
cd "$(dirname "$0")/../.."

OWNER_REPO="Pushkinist/rMLX"
FORMULA="packaging/homebrew/rmlx.rb"

VER=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
[ -n "$VER" ] || { echo "error: could not read version from Cargo.toml"; exit 1; }

URL="https://github.com/${OWNER_REPO}/archive/refs/tags/v${VER}.tar.gz"
echo "==> fetching $URL" >&2

# gh-authenticated fetch works for private repos; falls back to plain curl.
if command -v gh >/dev/null 2>&1; then
  SHA=$(gh api "repos/${OWNER_REPO}/tarball/v${VER}" 2>/dev/null | shasum -a 256 | awk '{print $1}')
else
  SHA=$(curl -fsSL "$URL" | shasum -a 256 | awk '{print $1}')
fi
[ -n "$SHA" ] || { echo "error: failed to compute sha256 (tag v${VER} reachable?)"; exit 1; }

echo "sha256: $SHA"

if [ "${1:-}" = "--write" ]; then
  sed -i '' -E "s|^(  sha256 )\"[^\"]*\"|\1\"$SHA\"|" "$FORMULA"
  echo "==> patched $FORMULA"
  grep -n "sha256" "$FORMULA"
fi
