#!/usr/bin/env bash
# Sync the canonical formula (packaging/homebrew/rmlx.rb) into the Homebrew tap
# repo (Pushkinist/homebrew-rmlx) as Formula/rmlx.rb, commit, and push.
#
# The in-repo formula is the source of truth (reviewed with the code); the tap
# is a thin published copy so `brew tap Pushkinist/rmlx` works.
#
# Usage:
#   scripts/release/sync_tap.sh                 # clone tap to a temp dir, sync, push
#   scripts/release/sync_tap.sh /path/to/tap    # use an existing local tap clone
set -euo pipefail
cd "$(dirname "$0")/../.."

TAP_REPO="Pushkinist/homebrew-rmlx"
SRC="packaging/homebrew/rmlx.rb"
VER=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)

[ -f "$SRC" ] || { echo "error: $SRC missing"; exit 1; }

if [ -n "${1:-}" ]; then
  TAP_DIR="$1"
else
  TAP_DIR=$(mktemp -d)
  echo "==> cloning $TAP_REPO -> $TAP_DIR"
  gh repo clone "$TAP_REPO" "$TAP_DIR" >/dev/null
fi

mkdir -p "$TAP_DIR/Formula"
cp "$SRC" "$TAP_DIR/Formula/rmlx.rb"

( cd "$TAP_DIR"
  git add Formula/rmlx.rb
  if git diff --cached --quiet; then
    echo "==> tap already up to date"
  else
    git commit -q -m "rmlx ${VER}"
    git push -q
    echo "==> pushed Formula/rmlx.rb (rmlx ${VER}) to $TAP_REPO"
  fi
)
