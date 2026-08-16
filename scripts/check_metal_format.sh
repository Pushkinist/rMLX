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
# `--strict` turns a missing clang-format into a hard failure, and likewise an
# empty file set: a renamed or moved kernel directory would otherwise disable the
# whole gate while the CI job stayed green — protecting nothing is the one
# outcome a gate must never report as success. CI passes it. Detection and
# enforcement stay in one place so the two cannot drift.
#
# A missing directory is always an error, strict or not: `METAL_DIRS` names
# directories that are supposed to exist, so one that does not is a stale list,
# not a toolchain difference between dev boxes.
#
# Exit 0 = clean (or skipped for a missing clang-format). Exit 1 = a file needs
# reformatting, a listed directory is missing, or --strict and clang-format is
# missing or the file set is empty.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directories holding gated `.metal` kernels. Single-sourced with the compile
# gate so the two cannot drift apart.
# shellcheck source=scripts/metal_dirs.sh
. "$(dirname "${BASH_SOURCE[0]}")/metal_dirs.sh"

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

missing_dirs=()
shopt -s nullglob
files=()
for d in "${METAL_DIRS[@]}"; do
    if [ ! -d "${d}" ]; then
        missing_dirs+=("${d#"${REPO_ROOT}/"}")
        continue
    fi
    files+=("${d}"/*.metal)
done
shopt -u nullglob

if [ ${#missing_dirs[@]} -gt 0 ]; then
    echo "ERROR: kernel director(ies) listed in scripts/metal_dirs.sh do not exist:" >&2
    for d in "${missing_dirs[@]}"; do echo "  ${d}" >&2; done
    echo "       A stale list silently shrinks this gate. Fix the list or restore the directory." >&2
    exit 1
fi

if [ ${#files[@]} -eq 0 ]; then
    if [ "${STRICT}" = 1 ]; then
        echo "ERROR: --strict: no .metal files found under ${METAL_DIRS[*]}." >&2
        echo "       An empty file set means this gate checked nothing." \
             "Refusing to pass by skipping." >&2
        exit 1
    fi
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
