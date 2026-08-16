#!/usr/bin/env bash
# scripts/check_metal_format.sh — CI gate: every `.metal` kernel file is
# clang-format clean.
#
# Metal Shading Language is a C++14 dialect, so clang-format's Cpp mode formats
# it. Style is pinned by a `.clang-format` in each kernel directory, which
# clang-format discovers by walking up from each file.
#
# Scope is the same directory list the compile gate uses — directory, not crate:
# a `.metal` file is gated by where it lives, not by whose dispatcher reads it.
#
# clang-format is not on PATH on a stock macOS box — it ships inside the Xcode
# Command Line Tools and is reachable via `xcrun -f clang-format`. Some machines
# instead have it from `brew install clang-format` / `brew install llvm`. Both
# are accepted; when neither is present the gate skips rather than fails, since
# toolchain availability varies across dev machines.
#
# `--strict` turns a missing clang-format into a hard failure. CI passes it: the
# runner has the toolchain, so a skip there would mean the gate silently
# protected nothing. Detection and enforcement stay in one place so the two
# cannot drift.
#
# Exit 0 = clean (or skipped). Exit 1 = a file needs reformatting, or --strict
# and clang-format is missing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Keep in step with METAL_DIRS in scripts/check_metal_compiles.sh.
METAL_DIRS=(
    "${REPO_ROOT}/crates/rmlx-kv-quant/src/metal"
    "${REPO_ROOT}/crates/rmlx-models/src/metal"
    "${REPO_ROOT}/crates/rmlx-mlx/src/metal"
)

STRICT=0
for arg in "$@"; do
    case "${arg}" in
        --strict) STRICT=1 ;;
        *) echo "usage: $(basename "$0") [--strict]" >&2; exit 2 ;;
    esac
done

# Resolve clang-format: PATH first, then the Command Line Tools copy.
CLANG_FORMAT=""
if command -v clang-format >/dev/null 2>&1; then
    CLANG_FORMAT="$(command -v clang-format)"
elif command -v xcrun >/dev/null 2>&1 && xcrun -f clang-format >/dev/null 2>&1; then
    CLANG_FORMAT="$(xcrun -f clang-format)"
fi

if [ -z "${CLANG_FORMAT}" ]; then
    if [ "${STRICT}" = 1 ]; then
        echo "ERROR: --strict: clang-format not found (neither on PATH nor via xcrun -f)." >&2
        echo "       Refusing to pass by skipping. Install it ('brew install clang-format'" >&2
        echo "       or the Xcode Command Line Tools) or drop --strict." >&2
        exit 1
    fi
    echo "SKIP: clang-format not found (PATH or xcrun); MSL format gate not run."
    echo "      Install with 'brew install clang-format' or the Xcode Command Line Tools."
    exit 0
fi

shopt -s nullglob
files=()
for d in "${METAL_DIRS[@]}"; do
    files+=("${d}"/*.metal)
done
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "SKIP: no .metal files under ${METAL_DIRS[*]}."
    exit 0
fi

if ! "${CLANG_FORMAT}" --dry-run -Werror "${files[@]}" 2>&1; then
    echo >&2
    echo "ERROR: the .metal files above are not clang-format clean." >&2
    echo "Fix with:" >&2
    for d in "${METAL_DIRS[@]}"; do
        echo "  xcrun clang-format -i ${d#"${REPO_ROOT}/"}/*.metal" >&2
    done
    exit 1
fi

echo "OK: ${#files[@]} .metal files are clang-format clean ($("${CLANG_FORMAT}" --version))."
