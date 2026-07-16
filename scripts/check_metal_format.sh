#!/usr/bin/env bash
# scripts/check_metal_format.sh — CI gate: every KV `.metal` kernel file is
# clang-format clean.
#
# Metal Shading Language is a C++14 dialect, so clang-format's Cpp mode formats
# it. Style is pinned by `crates/rmlx-kv-quant/src/metal/.clang-format`, which
# clang-format discovers by walking up from each file.
#
# clang-format is not on PATH on a stock macOS box — it ships inside the Xcode
# Command Line Tools and is reachable via `xcrun -f clang-format`. Some machines
# instead have it from `brew install clang-format` / `brew install llvm`. Both
# are accepted; when neither is present the gate skips rather than fails, since
# toolchain availability varies across dev and CI machines.
#
# Exit 0 = clean (or skipped). Exit 1 = a file needs reformatting.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METAL_DIR="${REPO_ROOT}/crates/rmlx-kv-quant/src/metal"

# Resolve clang-format: PATH first, then the Command Line Tools copy.
CLANG_FORMAT=""
if command -v clang-format >/dev/null 2>&1; then
    CLANG_FORMAT="$(command -v clang-format)"
elif command -v xcrun >/dev/null 2>&1 && xcrun -f clang-format >/dev/null 2>&1; then
    CLANG_FORMAT="$(xcrun -f clang-format)"
fi

if [ -z "${CLANG_FORMAT}" ]; then
    echo "SKIP: clang-format not found (PATH or xcrun); MSL format gate not run."
    echo "      Install with 'brew install clang-format' or the Xcode Command Line Tools."
    exit 0
fi

shopt -s nullglob
files=("${METAL_DIR}"/*.metal)
shopt -u nullglob

if [ ${#files[@]} -eq 0 ]; then
    echo "SKIP: no .metal files under ${METAL_DIR}."
    exit 0
fi

if ! "${CLANG_FORMAT}" --dry-run -Werror "${files[@]}" 2>&1; then
    echo >&2
    echo "ERROR: the .metal files above are not clang-format clean." >&2
    echo "Fix with:" >&2
    echo "  xcrun clang-format -i crates/rmlx-kv-quant/src/metal/*.metal" >&2
    exit 1
fi

echo "OK: ${#files[@]} .metal files are clang-format clean ($("${CLANG_FORMAT}" --version))."
