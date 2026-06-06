#!/usr/bin/env bash
# Build + package the rMLX release binary for aarch64-apple-darwin.
#
# Produces, under dist/ (gitignored):
#   rmlx-v<ver>-aarch64-apple-darwin.tar.gz        (rmlx + licenses + README)
#   rmlx-v<ver>-aarch64-apple-darwin.tar.gz.sha256 (checksum, `shasum -c` format)
#
# Hosted GitHub macOS runners cannot build rMLX (no usable Metal); run this on a
# real Apple-Silicon machine with `brew install mlx-c` present.
#
# Usage: scripts/release/package_binary.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

# Apple Silicon only — the binary links Metal MLX.
[ "$(uname -m)" = "arm64" ] || { echo "error: must build on Apple Silicon (arm64)"; exit 1; }

# Version: single source of truth = [workspace.package].version in Cargo.toml.
VER=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
[ -n "$VER" ] || { echo "error: could not read version from Cargo.toml"; exit 1; }

TRIPLE="aarch64-apple-darwin"
NAME="rmlx-v${VER}-${TRIPLE}"
DIST="dist"
STAGE="${DIST}/${NAME}"

echo "==> building rmlx v${VER} (release)"
cargo build --release -p rmlx-cli

BIN="target/release/rmlx"
[ -x "$BIN" ] || { echo "error: $BIN not found"; exit 1; }

echo "==> staging ${STAGE}"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$BIN" "$STAGE/"
cp LICENSE-MIT LICENSE-APACHE README.md "$STAGE/"

echo "==> archiving"
tar -C "$DIST" -czf "${DIST}/${NAME}.tar.gz" "$NAME"
( cd "$DIST" && shasum -a 256 "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256" )

echo "==> done"
echo "    ${DIST}/${NAME}.tar.gz"
cat "${DIST}/${NAME}.tar.gz.sha256"
