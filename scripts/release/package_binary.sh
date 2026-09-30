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

# The tarball runs on Macs other than this one, so it is compiled for the oldest
# supported chip rather than this machine's (config.toml's target-cpu=native).
RELEASE_RUSTFLAGS=$(python3 scripts/release/release_cpu.py rustflags)

echo "==> building rmlx v${VER} (release, $(printf '%s' "$RELEASE_RUSTFLAGS" | tr '\037' ' '))"
# A rustc wrapper's arguments never reach cargo's record, which is all the check
# reads. An empty variable also overrides build.rustc-wrapper in any config.
RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= CARGO_ENCODED_RUSTFLAGS="$RELEASE_RUSTFLAGS" \
  cargo build --release -p rmlx-cli

BIN="target/release/rmlx"
[ -x "$BIN" ] || { echo "error: $BIN not found"; exit 1; }

# The repository and its releases are public. A dependency's panic location is
# an absolute source path, and under the build machine's home directory it
# carries the builder's user name; the release rustflags remap it to `~`.
# Check the bytes rather than the flags: a path a remap missed still fails here.
if LC_ALL=C grep -qaF "$HOME/" "$BIN"; then
  echo "error: $BIN contains the build machine's home directory; nothing is packaged"
  exit 1
fi

echo "==> staging ${STAGE}"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp "$BIN" "$STAGE/"
cp LICENSE-MIT LICENSE-APACHE README.md "$STAGE/"

echo "==> archiving"
# The tar headers would otherwise record the builder's user and group names and
# ids, and macOS extended attributes. Neither belongs in a published archive.
COPYFILE_DISABLE=1 tar -C "$DIST" --uid 0 --gid 0 --uname "" --gname "" \
  --no-xattrs --no-mac-metadata -czf "${DIST}/${NAME}.tar.gz" "$NAME"
python3 - "${DIST}/${NAME}.tar.gz" <<'PY' || {
import sys, tarfile
with tarfile.open(sys.argv[1]) as tf:
    bad = [(m.name, m.uname, m.gname, m.uid, m.gid, sorted(m.pax_headers))
           for m in tf.getmembers()
           if m.uname or m.gname or m.uid or m.gid or any("xattr" in k for k in m.pax_headers)]
for b in bad:
    print("error: tar header carries an owner or an xattr:", b, file=sys.stderr)
sys.exit(1 if bad else 0)
PY
  rm -rf "${STAGE:?}" "${DIST:?}/${NAME:?}.tar.gz"
  echo "error: the archive records the builder's identity; tarball and staging removed"
  exit 1
}

echo "==> checking the packaged binary's target CPU"
python3 scripts/release/release_cpu.py check "${DIST}/${NAME}.tar.gz" || {
  rm -rf "$STAGE" "${DIST}/${NAME}.tar.gz" "${DIST}/${NAME}.tar.gz.sha256"
  echo "error: the packaged binary is not built for the release baseline CPU; tarball and staging removed"
  exit 1
}
( cd "$DIST" && shasum -a 256 "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256" )

echo "==> done"
echo "    ${DIST}/${NAME}.tar.gz"
cat "${DIST}/${NAME}.tar.gz.sha256"
