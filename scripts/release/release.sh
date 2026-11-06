#!/usr/bin/env bash
# The release steps in one tool: package, sign, sha, bottle, tap-sync, notes.
#
# Usage:
#   scripts/release/release.sh package               # build + bundle dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz
#   scripts/release/release.sh sign [KEY]            # cosign-sign that tarball (KEY default ~/.config/rmlx/cosign.key)
#   scripts/release/release.sh sha [--write]         # sha256 of the v<ver> source tarball; --write patches the formula
#   scripts/release/release.sh bottle [--root-url U] # Homebrew bottle from the installed rmlx keg
#   scripts/release/release.sh tap-sync [DIR]        # copy the formula into the tap (a temp clone unless DIR) and push
#   scripts/release/release.sh notes [VER]           # print one version's CHANGELOG.md section (default: current)
# Exit:   0 done; 1 a step failed or refused; 2 usage error or the repo root unreachable;
#         passthrough of cargo, curl, cosign, brew and awk
# Not a mode of: release/release_cpu.py — that one only reads the packaged binary's target CPU; these run the external tools
#
# The version is [workspace.package].version in Cargo.toml. Hosted CI cannot build rMLX (no usable Metal):
# package runs on a real Apple-Silicon Mac with `brew install mlx-c` present. Nothing here runs from CI.
set -euo pipefail

TRIPLE="aarch64-apple-darwin"
OWNER_REPO="Pushkinist/rMLX"
FORMULA="packaging/homebrew/rmlx.rb"
DIST="dist"

usage() {
  echo "usage: release.sh {package|sign [KEY]|sha [--write]|bottle [--root-url URL]|tap-sync [DIR]|notes [VER]}" >&2
  exit 2
}

cargo_version() { awk -F'"' '/^version = /{print $2; exit}' Cargo.toml; }
release_name() { echo "rmlx-v$1-${TRIPLE}"; }

# ---- package ------------------------------------------------------------------------------------
# Produces ${DIST}/<name>.tar.gz and <name>.tar.gz.sha256 (`shasum -c` format). The binary is
# compiled for apple-m1, not this machine's CPU, and the tarball is kept only when
# `release_cpu.py check` passes.
package() {
  [ "$(uname -m)" = "arm64" ] || { echo "error: must build on Apple Silicon (arm64)"; exit 1; }

  VER=$(cargo_version)
  [ -n "$VER" ] || { echo "error: could not read version from Cargo.toml"; exit 1; }

  NAME=$(release_name "$VER")
  STAGE="${DIST}/${NAME}"

  # The tarball runs on Macs other than this one, so it is compiled for the oldest
  # supported chip rather than this machine's (config.toml's target-cpu=native).
  RELEASE_RUSTFLAGS=$(python3 scripts/release/release_cpu.py rustflags)

  echo "==> building rmlx v${VER} (release, $(printf '%s' "$RELEASE_RUSTFLAGS" | tr '\037' ' '))"
  # A rustc wrapper's arguments never reach cargo's record, which is all the check
  # reads. An empty variable also overrides build.rustc-wrapper in any config.
  RUSTC_WRAPPER= RUSTC_WORKSPACE_WRAPPER= CARGO_ENCODED_RUSTFLAGS="$RELEASE_RUSTFLAGS" \
    cargo build --release -p rmlx-cli

  BIN="$(rmlx_default_binary . release)"
  [ -x "$BIN" ] || { echo "error: $BIN not found"; exit 1; }

  # A dependency's panic location is an absolute source path, and under the build
  # machine's home directory it carries the builder's user name; the release rustflags
  # remap it to `~`. Check the bytes rather than the flags: a path a remap missed fails here.
  if LC_ALL=C grep -qaF "$HOME/" "$BIN"; then
    echo "error: $BIN contains the build machine's home directory; nothing is packaged"
    exit 1
  fi

  echo "==> staging ${STAGE}"
  rm -rf "${STAGE:?}"
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
    rm -rf "${STAGE:?}" "${DIST:?}/${NAME:?}.tar.gz" "${DIST:?}/${NAME:?}.tar.gz.sha256"
    echo "error: the packaged binary is not built for the release baseline CPU; tarball and staging removed"
    exit 1
  }
  ( cd "$DIST" && shasum -a 256 "${NAME}.tar.gz" > "${NAME}.tar.gz.sha256" )

  echo "==> done"
  echo "    ${DIST}/${NAME}.tar.gz"
  cat "${DIST}/${NAME}.tar.gz.sha256"
}

# ---- sign ---------------------------------------------------------------------------------------
# Writes <tarball>.cosign.bundle (signature + Rekor inclusion proof) beside the tarball. The key
# pair carries no identity: keyless signing with a personal OIDC login would write the maintainer's
# email into the bundle's certificate and the public Rekor log, where it can never be removed. The
# private key stays outside the repository; cosign prompts for its password, so run this in a terminal.
sign() {
  command -v cosign >/dev/null 2>&1 || {
    echo "error: cosign not found. Install it first: brew install cosign" >&2
    exit 1
  }

  KEY="${1:-$HOME/.config/rmlx/cosign.key}"
  [ -f "$KEY" ] || {
    echo "error: private key $KEY not found — pass its path: make release-sign KEY=<path>" >&2
    exit 1
  }
  [ -f cosign.pub ] || {
    echo "error: cosign.pub not found at the repository root" >&2
    exit 1
  }

  VER=$(cargo_version)
  [ -n "$VER" ] || { echo "error: could not read version from Cargo.toml" >&2; exit 1; }

  TARBALL="${DIST}/$(release_name "$VER").tar.gz"
  [ -f "$TARBALL" ] || {
    echo "error: $TARBALL not found — run 'make release-package' first" >&2
    exit 1
  }

  BUNDLE="${TARBALL}.cosign.bundle"
  echo "==> signing ${TARBALL} with the cosign key pair"
  cosign sign-blob --yes --key "$KEY" --bundle "$BUNDLE" "$TARBALL"

  # A key that does not match the published cosign.pub makes a bundle no
  # consumer can verify, so check before reporting success.
  cosign verify-blob --key cosign.pub --bundle "$BUNDLE" "$TARBALL" >/dev/null 2>&1 || {
    echo "error: the bundle does not verify against cosign.pub — the key at $KEY is not the published key" >&2
    rm -f "$BUNDLE"
    exit 1
  }
  echo "==> wrote ${BUNDLE} (verified against cosign.pub)"
  echo
  echo "Upload it to the release:"
  echo "  gh release upload v${VER} ${BUNDLE}"
  echo
  echo "Verify:"
  echo "  cosign verify-blob --key cosign.pub --bundle ${BUNDLE} ${TARBALL}"
}

# ---- sha ----------------------------------------------------------------------------------------
# The tag must already exist and the repo be public: Homebrew downloads anonymously.
sha() {
  local empty_sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

  VER=$(cargo_version)
  [ -n "$VER" ] || { echo "error: could not read version from Cargo.toml"; exit 1; }

  URL="https://github.com/${OWNER_REPO}/archive/refs/tags/v${VER}.tar.gz"
  echo "==> fetching $URL" >&2

  # MUST hash the exact `archive/refs/tags/...tar.gz` the formula `url` downloads.
  # `gh api .../tarball/...` returns a DIFFERENT tarball (different prefix / packing).
  SHA=$(curl -fsSL "$URL" | shasum -a 256 | awk '{print $1}')
  [ "$SHA" != "$empty_sha256" ] || { echo "error: the download was empty (tag v${VER} public + reachable?)"; exit 1; }

  echo "sha256: $SHA"

  if [ "${1:-}" = "--write" ]; then
    # Bump BOTH the url tag version AND the sha256: patching only the sha leaves the formula
    # pointing at the previous tag's tarball, and `brew install` fails on a sha mismatch. The url
    # substitution is anchored to the `  url "...archive/refs/tags/` line so the instructional
    # comment above (which mentions an older tag) is left untouched.
    sed -i '' -E "s|^(  url \"https://github.com/${OWNER_REPO}/archive/refs/tags/)v[^\"]*\.tar\.gz\"|\1v${VER}.tar.gz\"|" "$FORMULA"
    sed -i '' -E "s|^(  sha256 )\"[^\"]*\"|\1\"$SHA\"|" "$FORMULA"
    echo "==> patched $FORMULA (url -> v${VER}, sha256)"
    grep -nE 'archive/refs/tags|sha256 ' "$FORMULA"
  fi
}

# ---- bottle -------------------------------------------------------------------------------------
# Needs rmlx installed from the TAP with `brew install --build-bottle` on the macOS major version to
# bottle, and a published GitHub Release for the tag so the root_url is valid. No release step runs
# this (docs/RELEASING.md § "No Homebrew bottle").
bottle() {
  command -v brew >/dev/null 2>&1 || { echo "error: brew not found" >&2; exit 1; }
  command -v jq  >/dev/null 2>&1 || { echo "error: jq not found (brew install jq)" >&2; exit 1; }

  [ "$(uname -m)" = "arm64" ] || { echo "error: must run on Apple Silicon (arm64)" >&2; exit 1; }

  VER=$(cargo_version)
  [ -n "$VER" ] || { echo "error: could not read version from Cargo.toml" >&2; exit 1; }

  ROOT_URL="https://github.com/${OWNER_REPO}/releases/download/v${VER}"
  if [ "${1:-}" = "--root-url" ]; then
    ROOT_URL="${2:?error: --root-url requires a URL argument}"
  fi

  echo "==> building bottle for rmlx v${VER}"
  echo "    root_url: ${ROOT_URL}"

  brew list rmlx >/dev/null 2>&1 || {
    echo "" >&2
    echo "error: rmlx is not installed in Homebrew." >&2
    echo "" >&2
    echo "Install it from the tap first (Homebrew >=6 rejects loose .rb paths)," >&2
    echo "after the tap formula carries this version's url+sha:" >&2
    echo "  cp packaging/homebrew/rmlx.rb \\" >&2
    echo "    \"\$(brew --repository)/Library/Taps/pushkinist/homebrew-rmlx/Formula/rmlx.rb\"" >&2
    echo "  HOMEBREW_NO_INSTALL_FROM_API=1 brew install --build-bottle pushkinist/rmlx/rmlx" >&2
    echo "" >&2
    echo "Then re-run this script. See docs/RELEASING.md." >&2
    exit 1
  }

  mkdir -p "$DIST"

  # --json writes metadata to a JSON file (not into the formula); --root-url sets the root_url
  # field; --no-rebuild starts the rebuild counter clean; --force-core-tap allows bottling a
  # non-core-tap formula. The files land in the current directory and are moved into dist/ below.
  echo "==> running brew bottle"
  brew bottle \
    --json \
    --root-url="${ROOT_URL}" \
    --no-rebuild \
    --force-core-tap \
    rmlx

  # The local filename uses double-dash (a Homebrew quirk); the remote asset must use single-dash.
  LOCAL_TAR=$(find . -maxdepth 1 -name 'rmlx--*.bottle.tar.gz' -print 2>/dev/null | sort | tail -1 | sed 's|^\./||')
  JSON_FILE=$(find . -maxdepth 1 -name 'rmlx--*.bottle.json'   -print 2>/dev/null | sort | tail -1 | sed 's|^\./||')

  [ -n "$LOCAL_TAR" ] || { echo "error: brew bottle did not produce a .bottle.tar.gz file" >&2; exit 1; }
  [ -n "$JSON_FILE" ] || { echo "error: brew bottle did not produce a .bottle.json file" >&2; exit 1; }

  # The json's top-level key is the formula's full name: bare "rmlx" for a file/core install,
  # "<user>/<tap>/rmlx" when the keg came from a tap. Derive it, or every lookup returns null.
  FKEY=$(jq -r 'keys[0]' "$JSON_FILE")

  REMOTE_TAR=$(jq -r --arg k "$FKEY" '.[$k].bottle.tags | to_entries[0].value.filename' "$JSON_FILE" 2>/dev/null || echo "")
  [ "$REMOTE_TAR" != "null" ] || { echo "error: $JSON_FILE names no filename for the bottle" >&2; exit 1; }
  if [ -z "$REMOTE_TAR" ]; then
    REMOTE_TAR="${LOCAL_TAR/--/-}"
  fi

  mv "$LOCAL_TAR" "${DIST}/${REMOTE_TAR}"
  mv "$JSON_FILE" "${DIST}/${JSON_FILE}"

  echo "==> bottle:  ${DIST}/${REMOTE_TAR}"
  echo "==> json:    ${DIST}/${JSON_FILE}"

  OS_TAG=$(jq -r --arg k "$FKEY" '.[$k].bottle.tags | keys[0]' "${DIST}/${JSON_FILE}" 2>/dev/null || echo "arm64_tahoe")
  SHA256=$(jq -r --arg k "$FKEY" --arg t "$OS_TAG" '.[$k].bottle.tags[$t].sha256' "${DIST}/${JSON_FILE}" 2>/dev/null || echo "")
  # The cellar is a symbol (:any, :any_skip_relocation) unless it is an absolute path, which is quoted.
  CELLAR_RAW=$(jq -r --arg k "$FKEY" '.[$k].bottle.cellar' "${DIST}/${JSON_FILE}" 2>/dev/null || echo "any")
  case "$CELLAR_RAW" in
    /*) CELLAR="\"${CELLAR_RAW}\"" ;;
    *)  CELLAR=":${CELLAR_RAW}" ;;
  esac

  echo ""
  echo "==> paste this bottle do block into packaging/homebrew/rmlx.rb"
  echo "    (replace any existing bottle do ... end block)"
  echo ""
  echo "  bottle do"
  echo "    root_url \"${ROOT_URL}\""
  echo "    sha256 cellar: ${CELLAR}, ${OS_TAG}: \"${SHA256}\""
  echo "  end"
  echo ""

  echo "==> upload the bottle to the GitHub Release:"
  echo "  gh release upload v${VER} ${DIST}/${REMOTE_TAR}"
  echo ""
  echo "==> then commit the updated formula and sync the tap:"
  echo "  make tap-sync"
  echo ""
  echo "The formula has no bottle block; no release step runs this script (docs/RELEASING.md)."
}

# ---- tap-sync -----------------------------------------------------------------------------------
# The in-repo formula is the source of truth; the tap is a thin published copy so
# `brew tap Pushkinist/rmlx` works.
tap_sync() {
  TAP_REPO="Pushkinist/homebrew-rmlx"
  VER=$(cargo_version)

  [ -f "$FORMULA" ] || { echo "error: $FORMULA missing"; exit 1; }

  if [ -n "${1:-}" ]; then
    TAP_DIR="$1"
  else
    TAP_DIR=$(mktemp -d)
    echo "==> cloning $TAP_REPO -> $TAP_DIR"
    gh repo clone "$TAP_REPO" "$TAP_DIR" >/dev/null
  fi

  mkdir -p "$TAP_DIR/Formula"
  cp -f "$FORMULA" "$TAP_DIR/Formula/rmlx.rb"

  ( cd "$TAP_DIR"
    git add Formula/rmlx.rb
    if git diff --cached --quiet; then
      echo "==> tap already up to date"
    else
      # GitHub email-privacy push block: commit as the noreply identity, not the operator's global one.
      git -c user.name="Pushkinist" -c user.email="4850452+Pushkinist@users.noreply.github.com" commit -q -m "rmlx ${VER}"
      git push -q
      echo "==> pushed Formula/rmlx.rb (rmlx ${VER}) to $TAP_REPO"
    fi
  )
}

# ---- notes --------------------------------------------------------------------------------------
# Use: gh release create v<ver> --notes-file <(scripts/release/release.sh notes <ver>)
notes() {
  VER="${1:-}"
  [ -n "$VER" ] || VER=$(cargo_version)
  [ -n "$VER" ] || { echo "error: no version" >&2; exit 1; }

  # The lines from `## [VER]` up to (not including) the next `## [` header. The version is matched
  # as text, not as a pattern.
  local rc=0
  awk -v ver="$VER" '
    index($0, "## [" ver "]") == 1 { grab=1; found=1; next }
    grab && /^## \[/ { exit }
    grab { print }
    END { if (!found) exit 3 }
  ' CHANGELOG.md | sed -e 's/^[[:space:]]*$//' | awk 'NF{blank=0} !NF{blank++} blank<2 || NF' || rc=$?
  case "$rc" in
    0) ;;
    3) echo "error: CHANGELOG.md has no section for version $VER" >&2; exit 1 ;;
    *) exit "$rc" ;;
  esac
}

# ---- dispatch -----------------------------------------------------------------------------------

[ "$#" -ge 1 ] || usage
MODE="$1"
shift

case "$MODE" in
  package) [ "$#" -eq 0 ] || usage ;;
  sign|tap-sync|notes) [ "$#" -le 1 ] || usage ;;
  sha) [ "$#" -eq 0 ] || { [ "$#" -eq 1 ] && [ "$1" = "--write" ]; } || usage ;;
  bottle) [ "$#" -eq 0 ] || { [ "$1" = "--root-url" ] && [ "$#" -le 2 ]; } || usage ;;
  *) usage ;;
esac

cd "$(dirname "$0")/../.." || exit 2
# shellcheck source=scripts/lib/identity.sh
. scripts/lib/identity.sh

case "$MODE" in
  package) package ;;
  sign) sign "$@" ;;
  sha) sha "$@" ;;
  bottle) bottle "$@" ;;
  tap-sync) tap_sync "$@" ;;
  notes) notes "$@" ;;
esac
