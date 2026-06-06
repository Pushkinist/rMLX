#!/usr/bin/env bash
# Shared env bootstrap for rMLX dev/bench scripts.
#
# Loads the repo-root `.env` (gitignored) if present, then validates that the
# model-snapshot root is set. Source it AFTER `set -euo pipefail`, near the top:
#
#     REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # adjust depth per script
#     source "$REPO_ROOT/scripts/lib/env.sh"
#
# Already-exported environment variables win over `.env` values, so a one-off
#   RMLX_O_MODELS_ROOT=/other bash scripts/foo.sh
# overrides the file. Exports: RMLX_O_MODELS_ROOT (validated) + O_MODELS_ROOT.

# Resolve repo root from this file if the caller did not set one.
if [[ -z "${REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

# Load .env without clobbering vars already present in the environment.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS='=' read -r _k _v; do
    [[ "$_k" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue   # skip comments / blanks
    [[ -n "${!_k:-}" ]] && continue                        # environment wins
    export "$_k=$_v"
  done < "$REPO_ROOT/.env"
  unset _k _v
fi

: "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT to your models folder (cp .env.example .env). See README.}"
export RMLX_O_MODELS_ROOT
export O_MODELS_ROOT="$RMLX_O_MODELS_ROOT"
