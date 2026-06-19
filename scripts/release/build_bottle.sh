#!/usr/bin/env bash
# build_bottle.sh — build a Homebrew bottle from the installed rmlx keg and
# prepare it for upload to the GitHub Release.
#
# What this script does:
#   1. Verifies rmlx is installed in Homebrew (brew install --build-bottle
#      must have been run first — see docs/RELEASING.md).
#   2. Runs `brew bottle --json --root-url=<github-release-root>` to produce:
#        dist/<local-filename>.bottle.tar.gz
#        dist/rmlx--<ver>.<tag>.bottle.json
#   3. Renames the local file to the remote filename (single-dash naming —
#      Homebrew intentionally uses double-dash locally; the remote asset must
#      use single-dash so brew install resolves it correctly).
#   4. Prints the `bottle do ... end` DSL block to paste into
#      packaging/homebrew/rmlx.rb, with root_url already set.
#   5. Prints the exact `gh release upload` command to attach the bottle.
#
# Prerequisites:
#   - rmlx installed via `brew install --build-bottle packaging/homebrew/rmlx.rb`
#     on the same macOS major version you want to bottle.
#   - A published GitHub Release for the current version tag must already
#     exist (step 6 in docs/RELEASING.md) so the root_url is valid at
#     install time.
#   - `brew` and `jq` available in PATH.
#
# Usage:
#   bash scripts/release/build_bottle.sh
#   bash scripts/release/build_bottle.sh --root-url https://github.com/Pushkinist/rMLX/releases/download/v1.2.3
#
# The --root-url flag overrides the default (derived from Cargo.toml version).
# Use it when you need to bottle against a future or patched release URL.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ── prerequisites ──────────────────────────────────────────────────────────────
command -v brew >/dev/null 2>&1 || { echo "error: brew not found" >&2; exit 1; }
command -v jq  >/dev/null 2>&1 || { echo "error: jq not found (brew install jq)" >&2; exit 1; }

# Apple Silicon only — the binary links Metal MLX.
[ "$(uname -m)" = "arm64" ] || { echo "error: must run on Apple Silicon (arm64)" >&2; exit 1; }

# ── version ────────────────────────────────────────────────────────────────────
VER=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
[ -n "$VER" ] || { echo "error: could not read version from Cargo.toml" >&2; exit 1; }

# ── root-url (GitHub Release download root) ────────────────────────────────────
OWNER_REPO="Pushkinist/rMLX"
DEFAULT_ROOT_URL="https://github.com/${OWNER_REPO}/releases/download/v${VER}"

ROOT_URL="${DEFAULT_ROOT_URL}"
if [ "${1:-}" = "--root-url" ]; then
  ROOT_URL="${2:?error: --root-url requires a URL argument}"
fi

echo "==> building bottle for rmlx v${VER}"
echo "    root_url: ${ROOT_URL}"

# ── check keg is installed ─────────────────────────────────────────────────────
brew list rmlx >/dev/null 2>&1 || {
  echo "" >&2
  echo "error: rmlx is not installed in Homebrew." >&2
  echo "" >&2
  echo "Install it first with:" >&2
  echo "  brew install --build-bottle packaging/homebrew/rmlx.rb" >&2
  echo "" >&2
  echo "Then re-run this script." >&2
  exit 1
}

# ── output directory ───────────────────────────────────────────────────────────
mkdir -p dist

# ── brew bottle ────────────────────────────────────────────────────────────────
# --json writes metadata to a JSON file (not into the formula).
# --root-url sets the root_url field in the JSON / bottle do block.
# --no-rebuild suppresses the rebuild counter (starts clean for every release).
# --force-core-tap allows bottling a non-core-tap formula.
# Output files land in the current directory; we move them into dist/ below.
echo "==> running brew bottle"
brew bottle \
  --json \
  --root-url="${ROOT_URL}" \
  --no-rebuild \
  --force-core-tap \
  rmlx

# ── locate generated files ─────────────────────────────────────────────────────
# brew bottle emits files to $PWD.  Local filename uses double-dash (intentional
# Homebrew quirk); the remote asset must use single-dash.
LOCAL_TAR=$(find . -maxdepth 1 -name 'rmlx--*.bottle.tar.gz' -print 2>/dev/null | sort | tail -1 | sed 's|^\./||')
JSON_FILE=$(find . -maxdepth 1 -name 'rmlx--*.bottle.json'   -print 2>/dev/null | sort | tail -1 | sed 's|^\./||')

[ -n "$LOCAL_TAR" ] || { echo "error: brew bottle did not produce a .bottle.tar.gz file" >&2; exit 1; }
[ -n "$JSON_FILE" ] || { echo "error: brew bottle did not produce a .bottle.json file" >&2; exit 1; }

# The brew bottle --json top-level key is the formula's full name — bare "rmlx"
# for a file/core install, but the fully-qualified "<user>/<tap>/rmlx" when the
# keg was installed from a tap. Derive it instead of hard-coding, or every
# lookup below silently returns null.
FKEY=$(jq -r 'keys[0]' "$JSON_FILE")

# ── rename local → remote filename (double-dash → single-dash) ─────────────────
# The JSON tab contains both fields:
#   local_filename  →  rmlx--<ver>.arm64_tahoe.bottle.tar.gz
#   filename        →  rmlx-<ver>.arm64_tahoe.bottle.tar.gz
REMOTE_TAR=$(jq -r --arg k "$FKEY" '.[$k].bottle.tags | to_entries[0].value.filename' "$JSON_FILE" 2>/dev/null || echo "")
if [ -z "$REMOTE_TAR" ]; then
  # Fallback: replace first occurrence of '--' with '-'
  REMOTE_TAR="${LOCAL_TAR/--/-}"
fi

mv "$LOCAL_TAR" "dist/${REMOTE_TAR}"
mv "$JSON_FILE" "dist/${JSON_FILE}"

echo "==> bottle:  dist/${REMOTE_TAR}"
echo "==> json:    dist/${JSON_FILE}"

# ── print bottle do block ──────────────────────────────────────────────────────
OS_TAG=$(jq -r --arg k "$FKEY" '.[$k].bottle.tags | keys[0]' "dist/${JSON_FILE}" 2>/dev/null || echo "arm64_tahoe")
SHA256=$(jq -r --arg k "$FKEY" --arg t "$OS_TAG" '.[$k].bottle.tags[$t].sha256' "dist/${JSON_FILE}" 2>/dev/null || echo "")
# cellar lives at .bottle.cellar (not per-tag); it is a symbol (:any /
# :any_skip_relocation) unless it is an absolute Cellar path, which must be quoted.
CELLAR_RAW=$(jq -r --arg k "$FKEY" '.[$k].bottle.cellar' "dist/${JSON_FILE}" 2>/dev/null || echo "any")
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

# ── upload instructions ────────────────────────────────────────────────────────
echo "==> upload the bottle to the GitHub Release:"
echo "  gh release upload v${VER} dist/${REMOTE_TAR}"
echo ""
echo "==> then commit the updated formula and sync the tap:"
echo "  make tap-sync"
echo ""
echo "See docs/RELEASING.md for the full bottle release flow."
