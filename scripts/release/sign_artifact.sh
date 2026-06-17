#!/usr/bin/env bash
# sign_artifact.sh — keyless (sigstore) signature for the release tarball.
#
# Produces, alongside dist/rmlx-v<ver>-aarch64-apple-darwin.tar.gz:
#   <tarball>.cosign.bundle   self-contained cert + signature + Rekor proof
#
# Keyless signing: cosign opens a browser for an OIDC login (GitHub / Google),
# obtains a short-lived Fulcio certificate bound to that identity, signs the
# tarball, and records the signature in the public Rekor transparency log. There
# is no private key to store or leak. Upload the .cosign.bundle as a release
# asset; consumers verify with the command this script prints (also documented
# in docs/RELEASING.md).
#
# The release binary is built locally (hosted CI has no Metal — see RELEASING),
# so this is the provenance signal the prebuilt tarball would otherwise lack:
# the bundle binds the artifact to the maintainer's authenticated identity.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

command -v cosign >/dev/null 2>&1 || {
  echo "error: cosign not found. Install it first: brew install cosign" >&2
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
echo "==> keyless-signing ${TARBALL} (a browser OIDC prompt will open)"
cosign sign-blob --yes --bundle "$BUNDLE" "$TARBALL"
echo "==> wrote ${BUNDLE}"
echo
echo "Upload it to the release:"
echo "  gh release upload v${VER} ${BUNDLE}"
echo
echo "Verify (substitute the identity you authenticated as):"
echo "  cosign verify-blob \\"
echo "    --bundle ${BUNDLE} \\"
echo "    --certificate-identity <your-oidc-email> \\"
echo "    --certificate-oidc-issuer <issuer-url> \\"
echo "    ${TARBALL}"
echo "  (GitHub issuer: https://github.com/login/oauth  ·  Google: https://accounts.google.com)"
