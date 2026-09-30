#!/usr/bin/env bash
# sign_artifact.sh — cosign key-pair signature for the release tarball.
#
# Produces, alongside dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz:
#   <tarball>.cosign.bundle   signature + Rekor inclusion proof
#
# The signature is made with the maintainer's cosign private key and checks
# against `cosign.pub` at the repository root. A key-pair signature carries no
# identity: keyless signing with a personal OIDC login would write the
# maintainer's email into the bundle's certificate and into the public Rekor
# log, where it can never be removed.
#
# The private key stays outside the repository, encrypted with its password.
# Pass its path as the first argument (`make release-sign KEY=<path>`); the
# default is ~/.config/rmlx/cosign.key. cosign prompts for the password, so run
# this in a terminal.
#
# The release binary is built locally (hosted CI has no Metal — see RELEASING),
# so this is the provenance signal the prebuilt tarball would otherwise lack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

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

VER=$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)
[ -n "$VER" ] || { echo "error: could not read version from Cargo.toml" >&2; exit 1; }

TARBALL="dist/rmlx-v${VER}-aarch64-apple-darwin.tar.gz"
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
