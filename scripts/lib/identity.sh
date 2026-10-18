#!/usr/bin/env bash
# Shared §8.5 run-identity for rMLX bench scripts.
#
# Asks the binary that is actually being measured who it is, and exports the
# answer as RMLX_IDENTITY_JSON. Bench scripts merge that block into their §8.5
# record instead of hard-coding `backend_version: '0.0.1'`, guessing
# `build_profile: 'release-perf'`, or omitting the fields entirely.
#
# Usage (after RMLX_BIN / BINARY is resolved):
#
#     source "$REPO_ROOT/scripts/lib/identity.sh"
#     rmlx_export_identity "$RMLX_BIN"
#
# Then, inside the record-building python:
#
#     import json, os
#     rec = {
#         **json.loads(os.environ["RMLX_IDENTITY_JSON"]),
#         ... measurement fields ...
#     }
#
# Non-rMLX emitters (mlx_lm, oMLX, llama.cpp) describe a different backend and
# must NOT source this.

# Resolve identity once and export it. Idempotent.
rmlx_export_identity() {
  local bin="${1:?rmlx_export_identity: pass the rmlx binary path}"

  if [[ -n "${RMLX_IDENTITY_JSON:-}" ]]; then
    return 0
  fi

  if [[ ! -x "$bin" ]]; then
    echo "rmlx_export_identity: binary not found / not executable: $bin" >&2
    return 1
  fi

  RMLX_IDENTITY_JSON="$("$bin" metrics identity --json)" || {
    echo "rmlx_export_identity: '$bin metrics identity --json' failed" >&2
    return 1
  }
  export RMLX_IDENTITY_JSON
}

# The `git_sha` the identity carries, or nothing. The binary reports none; a
# caller that built it at another commit than the checkout's HEAD supplies it in
# RMLX_IDENTITY_JSON. `unknown...` is the placeholder scripts use for "no
# commit", not a commit, so it counts as absent. Fails with one line when
# RMLX_IDENTITY_JSON is not a JSON object.
rmlx_identity_git_sha() {
  python3 -c 'import json, os, sys
try:
    identity = json.loads(os.environ.get("RMLX_IDENTITY_JSON") or "{}")
except ValueError:
    identity = None
if not isinstance(identity, dict):
    sys.exit("rmlx_identity_git_sha: RMLX_IDENTITY_JSON is not a JSON object")
sha = identity.get("git_sha")
print(sha if isinstance(sha, str) and not sha.startswith("unknown") else "")'
}

# The commit a run is attributed to: the identity's `git_sha` when it carries
# one, else HEAD of the checkout at $1, else `unknown`. A hex sha that resolves
# in the checkout is filed in the short form HEAD is filed in; any other value
# is kept as given.
rmlx_git_sha() {
  local repo="${1:?rmlx_git_sha: pass the checkout root}" sha
  sha="$(rmlx_identity_git_sha)" || return 1
  if [[ -z "$sha" ]]; then
    sha="$(git -C "$repo" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  elif [[ "$sha" =~ ^[0-9a-fA-F]{4,40}$ ]]; then
    sha="$(git -C "$repo" rev-parse --verify --quiet --short "${sha}^{commit}" 2>/dev/null || echo "$sha")"
  fi
  echo "$sha"
}
