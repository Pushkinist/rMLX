#!/usr/bin/env bash
# scripts/metal_dirs.sh — the directories holding gated `.metal` kernels.
#
# Sourced by `check_metal_compiles.sh` and `check_metal_format.sh`. Single-sourced
# on purpose: two copies held in step by a comment is how a fourth crate gets
# added to one gate and forgotten by the other, which silently halves the
# coverage while both jobs stay green.
#
# Scope is the DIRECTORY, not the crate: a `.metal` file is gated by where it
# lives, wherever its Rust dispatcher sits. A kernel inside a gated crate but
# outside its `metal/` directory is not gated.
#
# Add a directory here when a crate starts shipping MSL — nothing else discovers
# it. Each one needs a `probes/kernels.manifest` (compile gate) and inherits a
# `.clang-format` (format gate).
#
# Expects REPO_ROOT to be set by the sourcing script. Exports METAL_DIRS.

if [ -z "${REPO_ROOT:-}" ]; then
    echo "metal_dirs.sh: REPO_ROOT must be set before sourcing" >&2
    exit 2
fi

METAL_DIRS=(
    "${REPO_ROOT}/crates/rmlx-kv-quant/src/metal"
    "${REPO_ROOT}/crates/rmlx-models/src/metal"
    "${REPO_ROOT}/crates/rmlx-mlx/src/metal"
)
