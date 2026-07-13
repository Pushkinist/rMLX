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
