#!/usr/bin/env bash
# Recall test for the MSL kernel gates: planted kernel trees and stub toolchains.
#
# Recall test for: check_metal_compiles.sh default --strict
#                  check_metal_format.sh default --strict
#
# Each case copies the gates into a throwaway root, plants the gated kernel directories
# beside them with a manifest and some `.metal` files, puts stub `xcrun` and `clang-format`
# on a PATH of their own and asserts the literal exit code and the line or the file the gate
# names. A kernel holding the word PLANTED_FAIL is the one the stubs reject. The stubs record
# their argv, so a case can read which language levels and which files the gate handed them.
# No case runs the real compiler or the real formatter. The gates are named in one place,
# `gate` and `GATE_FILES`.
#
# Usage:  bash scripts/check/metal_kernels_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init metal-kernels-selftest

GATE_FILES=(scripts/check_metal_compiles.sh scripts/check_metal_format.sh scripts/lib/metal_dirs.sh)

# gate <compile|format> <root> [args]: one gate, run from the copy in <root>, under GATE_PATH.
gate() {
    local kind="$1" root="$2"
    shift 2
    if [ "$kind" = compile ]; then
        gate_compile "$root" ${1+"$@"}
    else
        gate_format "$root" ${1+"$@"}
    fi
}

# clean_env: the command prefix that runs a gate with the stubs' variables and nothing else.
clean_env() {
    echo env -i HOME="$ST_WORK" PATH="$GATE_PATH" STUB_LOG="$STUB_LOG" STUB_NO_TOOLCHAIN="$STUB_NO_TOOLCHAIN" \
        STUB_NO_METAL4="$STUB_NO_METAL4" STUB_NO_GUARD="$STUB_NO_GUARD" STUB_CF="$STUB_CF"
}

gate_compile() {
    local root="$1"
    shift
    # shellcheck disable=SC2046 # the prefix is words
    st_bounded 60 $(clean_env) bash "$root/scripts/check_metal_compiles.sh" ${1+"$@"}
}

gate_format() {
    local root="$1"
    shift
    # shellcheck disable=SC2046 # the prefix is words
    st_bounded 60 $(clean_env) bash "$root/scripts/check_metal_format.sh" ${1+"$@"}
}

# ---- stub toolchains ---------------------------------------------------------

TOOLS="$ST_WORK/tools"
mkdir -p "$TOOLS" "$ST_WORK/xcrun" "$ST_WORK/clang-format" "$ST_WORK/cf-only-via-xcrun"
for tool in bash mktemp cut xargs tr grep cat basename dirname head rm sed awk mkdir sort touch; do
    found="$(command -v "$tool")" || st_die "no $tool on this host"
    ln -s "$found" "$TOOLS/$tool"
done

cat >"$ST_WORK/xcrun/xcrun" <<'EOF'
#!/bin/sh
[ -z "$STUB_NO_TOOLCHAIN" ] || { echo "xcrun: error: cannot execute tool 'metal' due to missing Metal Toolchain" >&2; exit 1; }
if [ "$1" = -f ]; then
    [ -n "$STUB_CF" ] || exit 1
    echo "$STUB_CF"
    exit 0
fi
shift 3
case "$1" in --version) echo "stub metal"; exit 0 ;; esac
echo "metal $*" >>"$STUB_LOG"
if [ "$1" = -std=metal4.0 ] && [ -n "$STUB_NO_METAL4" ]; then
    echo "error: unknown value 'metal4.0' for '-std'" >&2
    exit 1
fi
case "$3" in */guard_probe.metal) [ -z "$STUB_NO_GUARD" ] || { echo "error: __HAVE_TENSOR__ is not defined" >&2; exit 1; } ;; esac
if grep -q PLANTED_FAIL "$3"; then
    echo "$3:1:1: error: planted failure" >&2
    exit 1
fi
: >"$5"
EOF

CF_BODY='#!/bin/sh
case "$1" in --version) echo "stub-clang-format 1.0"; exit 0 ;; esac
echo "clang-format $*" >>"$STUB_LOG"
rc=0
for f in "$@"; do
    case "$f" in
    -*) ;;
    *) if grep -q PLANTED_FAIL "$f"; then echo "$f:1:1: error: code should be clang-formatted"; rc=1; fi ;;
    esac
done
exit $rc'
printf '%s\n' "$CF_BODY" >"$ST_WORK/clang-format/clang-format"
printf '%s\n' "$CF_BODY" >"$ST_WORK/cf-only-via-xcrun/clang-format"
chmod +x "$ST_WORK/xcrun/xcrun" "$ST_WORK/clang-format/clang-format" "$ST_WORK/cf-only-via-xcrun/clang-format"

# toolchain <xcrun yes|no> <clang-format path|xcrun|no>: what the next gate run finds.
toolchain() {
    GATE_PATH="$TOOLS"
    STUB_NO_TOOLCHAIN=""
    STUB_NO_METAL4=""
    STUB_NO_GUARD=""
    STUB_CF=""
    [ "$1" != yes ] || GATE_PATH="$ST_WORK/xcrun:$GATE_PATH"
    case "$2" in
    path) GATE_PATH="$ST_WORK/clang-format:$GATE_PATH" ;;
    xcrun) STUB_CF="$ST_WORK/cf-only-via-xcrun/clang-format" ;;
    no) ;;
    *) st_die "toolchain: unknown clang-format '$2'" ;;
    esac
    STUB_LOG="$ST_WORK/log.$((++LOG_N))"
    : >"$STUB_LOG"
}
LOG_N=0

# ---- planted roots -----------------------------------------------------------

# kernel_dirs <root>: the gated directories, as the copy of metal_dirs.sh in <root> names them.
kernel_dirs() {
    (
        REPO_ROOT="$1"
        # shellcheck source=scripts/lib/metal_dirs.sh
        . "$1/scripts/lib/metal_dirs.sh"
        printf '%s\n' "${METAL_DIRS[@]}"
    )
}

# kdir <root> <n>: the nth gated directory, relative to <root>.
kdir() {
    local dir
    dir="$(kernel_dirs "$1" | sed -n "${2}p")"
    printf '%s' "${dir#"$1"/}"
}

# plant <name>: a root holding the gates and three gated directories, each with one manifested kernel.
plant() {
    local root="$ST_WORK/$1" f d i=0 names=(a_one b_one c_one)
    mkdir -p "$root"
    for f in "${GATE_FILES[@]}"; do
        mkdir -p "$root/$(dirname "$f")"
        cp "$REPO_ROOT/$f" "$root/$f" || st_die "cannot copy $f into $root"
    done
    while IFS= read -r d; do
        mkdir -p "$d/probes"
        printf 'buf[0] = 1u;\n' >"$d/${names[$i]}.metal"
        printf '# planted\n%s.metal|-|buf:u|-\n' "${names[$i]}" >"$d/probes/kernels.manifest"
        i=$((i + 1))
    done < <(kernel_dirs "$root")
    printf '%s' "$root"
}

# add_why <reason>: one more line in the variable why, the reasons a case failed.
add_why() {
    why+="${why:+$'\n'}$1"
}

# expect_logged <pattern>: a reason unless the stubs' argv log holds a line matching <pattern>.
expect_logged() {
    grep -Eq -- "$1" "$STUB_LOG" || add_why "no argv line matches '$1'"
    return 0
}

KV=crates/rmlx-kv-quant/src/metal
MODELS=crates/rmlx-models/src/metal
MLX=crates/rmlx-mlx/src/metal

# ---- compile gate ------------------------------------------------------------

toolchain no no
root="$(plant c-no-xcrun)"
st_case "compile: no xcrun is a skip" 0 --has "SKIP: xcrun not found" --lacks "OK:" -- gate compile "$root"
st_case "compile: no xcrun with --strict is a refusal" 1 --has "ERROR: --strict: xcrun not found" -- gate compile "$root" --strict

toolchain yes no
STUB_NO_TOOLCHAIN=1
st_case "compile: a missing Metal Toolchain is a skip" 0 --has "SKIP: Xcode is selected but the Metal Toolchain component is not installed" --lacks "OK:" -- gate compile "$root"
st_case "compile: a missing Metal Toolchain with --strict is a refusal" 1 \
    --has "ERROR: --strict: Xcode is selected but the Metal Toolchain component is not installed" -- gate compile "$root" --strict

toolchain yes no
STUB_NO_METAL4=1
st_case "compile: no metal4.0 is a reduced run" 0 --has "NOTE: this toolchain cannot compile at -std=metal4.0" \
    --has "OK: 3 .metal kernels compile clean at metal3.1." -- gate compile "$root"
st_case "compile: no metal4.0 with --strict is a refusal" 1 --has "ERROR: --strict: this toolchain cannot compile at -std=metal4.0" -- gate compile "$root" --strict

toolchain yes no
STUB_NO_GUARD=1
root="$(plant c-no-guard)"
printf '// __HAVE_TENSOR__\n' >>"$root/$MODELS/b_one.metal"
st_case "compile: no tensor guard skips a guarded body and says so" 0 --has "NOTE: SKIP $MODELS/b_one.metal" \
    --has "OK: 2 .metal kernels compile clean at metal3.1 metal4.0; 1 skipped" -- gate compile "$root"
st_case "compile: no tensor guard with --strict is a refusal" 1 \
    --has "ERROR: --strict: this toolchain compiles at -std=metal4.0 but does not define __HAVE_TENSOR__" -- gate compile "$root" --strict

toolchain yes no
root="$(plant c-clean)"
st_case "compile: a clean set passes at both levels" 0 --has "OK: 3 .metal kernels compile clean at metal3.1 metal4.0." -- gate compile "$root"
st_case "compile: a clean set passes under --strict" 0 --has "OK: 3 .metal kernels compile clean at metal3.1 metal4.0." -- gate compile "$root" --strict

toolchain yes no
st_run gate compile "$root"
why="$(st_check 0)"
for std in metal3.1 metal4.0; do
    for kernel in a_one b_one c_one; do
        expect_logged "-std=$std -c .*/probe_$kernel.metal -o "
    done
done
st_verdict "compile: every kernel is compiled at -std=metal3.1 and -std=metal4.0" "$why"

toolchain yes no
root="$(plant c-fail)"
printf 'PLANTED_FAIL\n' >>"$root/$(kdir "$root" 2)/b_one.metal"
st_case "compile: a kernel the compiler rejects fails the gate and is named" 1 \
    --has "FAIL: $MODELS/b_one.metal at -std=metal3.1" --has "FAIL: $MODELS/b_one.metal at -std=metal4.0" \
    --has "ERROR: 2 .metal kernel check(s) failed:" --has "  $MODELS/b_one.metal (-std=metal3.1)" \
    --has "  $MODELS/b_one.metal (-std=metal4.0)" --lacks "a_one.metal" --lacks "c_one.metal" -- gate compile "$root"

toolchain yes no
root="$(plant c-unmanifested)"
printf 'buf[0] = 2u;\n' >"$root/$MLX/z_orphan.metal"
st_case "compile: a .metal the manifest does not name fails and is named" 1 \
    --has "ERROR: $MLX/z_orphan.metal is not named by $MLX/probes/kernels.manifest" --has "  $MLX/z_orphan.metal (not in manifest)" -- gate compile "$root"

toolchain yes no
root="$(plant c-header)"
printf 'constant uint hdr = 1u;\n' >"$root/$MLX/c_hdr.metal"
printf '# planted\nc_one.metal|../c_hdr.metal|buf:u|-\n' >"$root/$MLX/probes/kernels.manifest"
st_case "compile: a ../ header on a manifest line counts as named" 0 --has "OK: 3 .metal kernels compile clean" --lacks "not named" -- gate compile "$root"

toolchain yes no
root="$(plant c-ghost)"
printf 'ghost.metal|-|buf:u|-\n' >>"$root/$MODELS/probes/kernels.manifest"
st_case "compile: a manifest line for a missing body fails and names it" 1 \
    --has "ERROR: manifest references missing body: $MODELS/ghost.metal" --has "  $MODELS/ghost.metal (missing)" -- gate compile "$root"

toolchain yes no
root="$(plant c-no-dir)"
rm -rf "${root:?}/$MLX"
st_case "compile: a missing kernel directory fails and is named" 1 --has "ERROR: missing kernel directory: $MLX" --has "  $MLX (missing directory)" -- gate compile "$root"

toolchain yes no
root="$(plant c-no-manifest)"
rm -f "$root/$KV/probes/kernels.manifest"
st_case "compile: a missing manifest fails and is named" 1 --has "ERROR: missing $root/$KV/probes/kernels.manifest" --has "  $KV (missing manifest)" -- gate compile "$root"

st_case "compile: an unknown argument is a usage error" 2 --has "usage:" -- gate compile "$root" --loose

# ---- format gate -------------------------------------------------------------

toolchain no no
root="$(plant f-no-tool)"
st_case "format: no clang-format is a skip" 0 --has "SKIP: clang-format not found" --lacks "OK:" -- gate format "$root"
st_case "format: no clang-format with --strict is a refusal" 1 --has "ERROR: --strict: clang-format not found" -- gate format "$root" --strict

toolchain no path
root="$(plant f-clean)"
st_case "format: a clean set passes" 0 --has "OK: 3 .metal files are clang-format clean (stub-clang-format 1.0)" -- gate format "$root"
st_case "format: a clean set passes under --strict" 0 --has "OK: 3 .metal files are clang-format clean" -- gate format "$root" --strict

toolchain no path
st_run gate format "$root"
why="$(st_check 0)"
expect_logged "^clang-format --dry-run -Werror "
for dir in "$KV/a_one" "$MODELS/b_one" "$MLX/c_one"; do
    expect_logged " $root/$dir.metal( |\$)"
done
st_verdict "format: clang-format gets --dry-run -Werror and every kernel of every directory" "$why"

toolchain yes xcrun
root="$(plant f-via-xcrun)"
st_case "format: clang-format is found through xcrun -f when it is not on PATH" 0 --has "OK: 3 .metal files are clang-format clean" -- gate format "$root"

toolchain no path
root="$(plant f-fail)"
printf 'PLANTED_FAIL\n' >>"$root/$MODELS/b_one.metal"
st_case "format: a file that needs formatting fails the gate and is named" 1 \
    --has "$MODELS/b_one.metal:1:1: error" --has "ERROR: the .metal files above are not clang-format clean." --lacks "a_one.metal:" -- gate format "$root"

toolchain no path
root="$(plant f-empty)"
rm -f "$root/$KV/a_one.metal" "$root/$MODELS/b_one.metal" "$root/$MLX/c_one.metal"
st_case "format: an empty kernel set is a skip" 0 --has "SKIP: no .metal files under" --lacks "OK:" -- gate format "$root"
st_case "format: an empty kernel set with --strict is a refusal" 1 --has "ERROR: --strict: no .metal files found under" -- gate format "$root" --strict

toolchain no path
root="$(plant f-no-dir)"
rm -rf "${root:?}/$MODELS"
st_case "format: a missing kernel directory fails and is named" 1 --has "do not exist:" --has "  $MODELS" -- gate format "$root"
st_case "format: a missing kernel directory fails under --strict too" 1 --has "  $MODELS" -- gate format "$root" --strict

st_case "format: an unknown argument is a usage error" 2 --has "usage:" -- gate format "$root" --loose

st_done
