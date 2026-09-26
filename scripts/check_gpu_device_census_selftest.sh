#!/usr/bin/env bash
# scripts/check_gpu_device_census_selftest.sh — recall test for
# check_gpu_device_census.sh.
#
# Each case builds a throwaway scan root holding the one legitimate site, plants
# one file, runs the gate and asserts the literal exit code and, for a failure,
# the rule and the file:line it names (or the reason). The last case runs the
# gate on the real tree.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$REPO_ROOT/scripts/check_gpu_device_census.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0
PASSED=0

# fresh <name> [none]: a scan root holding the one legitimate site, or, with
# `none`, the same file with the site removed.
fresh() {
    local root="$WORK/$1"
    mkdir -p "$root/commands"
    if [ "${2:-}" = none ]; then
        printf 'use rmlx_mlx::Device;\npub fn claim_gpu() {}\n' >"$root/commands/parse.rs"
    else
        printf 'use rmlx_mlx::Device;\npub fn claim_gpu() -> Device {\n    Device::Gpu\n}\n' \
            >"$root/commands/parse.rs"
    fi
    printf '%s' "$root"
}

# plant <root> <relpath> <line>: write <line> as line 2 of <relpath>.
plant() {
    mkdir -p "$(dirname "$1/$2")"
    printf '// header\n%s\n' "$3" >"$1/$2"
}

# case_run <name> <what> <want-exit> <needle|-> <root>
case_run() {
    local name="$1" what="$2" want="$3" needle="$4" root="$5"
    local out status
    out=$(bash "$GATE" "$root" 2>&1)
    status=$?
    if [ "$status" -ne "$want" ]; then
        echo "FAIL $name ($what): exit $status, want $want"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        return
    fi
    if [ "$needle" != "-" ] && ! grep -qF -- "$needle" <<<"$out"; then
        echo "FAIL $name ($what): output does not name '$needle'"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAILED=$((FAILED + 1))
        return
    fi
    PASSED=$((PASSED + 1))
}

# name | relpath | planted line | want-exit | needle
CASES=(
    "second_site|commands/serve.rs|    \"gpu\" => Device::Gpu,|1|gpu-device: commands/serve.rs:2:"
    "comparison|commands/transcribe.rs|    if device == Device::Gpu {|1|gpu-device: commands/transcribe.rs:2:"
    "full_path|main.rs|    let d = rmlx_mlx::Device::Gpu;|1|gpu-device: main.rs:2:"
    "spaced_path|commands/eval.rs|    let d = Device :: Gpu;|1|gpu-device: commands/eval.rs:2:"
    "nested_dir|commands/metrics/x.rs|    let d = Device::Gpu;|1|gpu-device: commands/metrics/x.rs:2:"
    "two_on_one_line|commands/b.rs|    let (a, b) = (Device::Gpu, Device::Gpu);|1|3 Device::Gpu sites"
    "code_before_comment|commands/c.rs|    let d = Device::Gpu; // the GPU|1|gpu-device: commands/c.rs:2:"
    "glob_import|commands/d.rs|use rmlx_mlx::Device::*;|1|device-alias: commands/d.rs:2:"
    "braced_import|commands/e.rs|use rmlx_mlx::Device::{Cpu, Gpu};|1|device-alias: commands/e.rs:2:"
    "rename|commands/f.rs|use rmlx_mlx::Device as Dev;|1|device-alias: commands/f.rs:2:"
    "line_comment|commands/g.rs|// Device::Gpu is named only in claim_gpu|0|ok (1 site"
    "doc_comment|commands/h.rs|/// Returns \`Device::Gpu\` with the claim.|0|ok (1 site"
    "string_literal|commands/i.rs|    let s = \"Device::Gpu\";|0|ok (1 site"
    "longer_ident|commands/j.rs|    let d = Device::GpuLike;|0|ok (1 site"
    "unit_tests_file|commands/k_tests.rs|    let d = Device::Gpu;|0|ok (1 site"
    "tests_rs|commands/tests.rs|    let d = Device::Gpu;|0|ok (1 site"
    "bin_program|bin/diag.rs|    \"gpu\" => Device::Gpu,|0|ok (1 site"
    "alias_in_comment|commands/l.rs|// use rmlx_mlx::Device as Dev;|0|ok (1 site"
)

for c in "${CASES[@]}"; do
    IFS='|' read -r name rel line want needle <<<"$c"
    root="$(fresh "$name")"
    plant "$root" "$rel" "$line"
    case_run "$name" "$rel" "$want" "$needle" "$root"
done

case_run "one_site" "the legitimate site alone passes" 0 \
    "gpu-device: commands/parse.rs:3:" "$(fresh one_site)"
case_run "no_site" "the claimed-GPU helper lost its site" 1 \
    "no Device::Gpu site" "$(fresh no_site none)"
case_run "missing_root" "a scan root that does not exist" 2 \
    "is not a directory" "$WORK/does-not-exist"
mkdir -p "$WORK/empty_root/bin"
printf 'fn main() {}\n' >"$WORK/empty_root/bin/only.rs"
case_run "empty_root" "a root with no in-scope file" 2 \
    "no in-scope .rs file" "$WORK/empty_root"

# The real tree: exactly one site, in the claimed-GPU helper.
out=$(bash "$GATE" 2>&1)
status=$?
if [ "$status" -eq 0 ] && grep -qF 'gpu-device: commands/parse.rs:' <<<"$out"; then
    PASSED=$((PASSED + 1))
else
    echo "FAIL real_tree: exit $status, want 0 with the one site in commands/parse.rs"
    printf '%s\n' "$out" | sed 's/^/    /'
    FAILED=$((FAILED + 1))
fi

if [ "$FAILED" -ne 0 ]; then
    echo "check-gpu-device-census selftest: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi
echo "check-gpu-device-census selftest: ok ($PASSED cases)"
