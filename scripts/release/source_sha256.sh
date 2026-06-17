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

# MUST hash the exact `archive/refs/tags/...tar.gz` the formula `url` downloads.
# NOTE: `gh api .../tarball/...` returns a DIFFERENT tarball (different prefix /
# packing) with a different sha — do not use it here. The repo must be public
# (Homebrew downloads anonymously), so a plain curl of the archive URL is right.
SHA=$(curl -fsSL "$URL" | shasum -a 256 | awk '{print $1}')
[ -n "$SHA" ] || { echo "error: failed to compute sha256 (tag v${VER} public + reachable?)"; exit 1; }

echo "sha256: $SHA"

if [ "${1:-}" = "--write" ]; then
  # Bump BOTH the url tag version AND the sha256. Patching only the sha leaves
  # the formula pointing at the previous tag's tarball with the new tag's sha,
  # so `brew install` fails on a sha mismatch. The url substitution is anchored
  # to the `  url "...archive/refs/tags/` line so the instructional comment
  # above (which mentions an older tag) is left untouched.
  sed -i '' -E "s|^(  url \"https://github.com/${OWNER_REPO}/archive/refs/tags/)v[^\"]*\.tar\.gz\"|\1v${VER}.tar.gz\"|" "$FORMULA"
  sed -i '' -E "s|^(  sha256 )\"[^\"]*\"|\1\"$SHA\"|" "$FORMULA"
  echo "==> patched $FORMULA (url -> v${VER}, sha256)"
  grep -nE 'archive/refs/tags|sha256 ' "$FORMULA"
fi
