#!/usr/bin/env bash
# scripts/check_gpu_device_census_selftest.sh — recall test for
# check_gpu_device_census.sh.
#
# Each case builds a throwaway scan root holding the one legitimate site, plants
# one file, runs the gate and asserts the literal exit code and, for a failure,
# the rule and the file:line it names (or the reason). The library-mode cases
# plant one file in a root with no device site. The last case runs the binary
# mode on the real tree; the library mode is red on the real tree until the
# library sites take their device from the caller, so no case runs it there.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$REPO_ROOT/scripts/check_gpu_device_census.sh"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init check-gpu-device-census-selftest

# fresh <name> [none]: a scan root holding the one legitimate site, or, with
# `none`, the same file with the site removed.
fresh() {
    local root="$ST_WORK/$1"
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
    "type_alias|commands/f2.rs|pub type Dev = rmlx_mlx::Device;|1|device-alias: commands/f2.rs:2:"
    "bare_type_alias|commands/f3.rs|type Dev = Device;|1|device-alias: commands/f3.rs:2:"
    "other_type_alias|commands/f4.rs|type DeviceList = Vec<Device>;|0|ok (1 site"
    "rooted_type_alias|commands/f5.rs|type Dev = ::rmlx_mlx::Device;|1|device-alias: commands/f5.rs:2:"
    "generic_type_alias|commands/f6.rs|type Dev<T> = Device;|1|device-alias: commands/f6.rs:2:"
    "drop_parse_device|commands/m.rs|    let d = parse_device(&device)?.device();|1|claim-dropped: commands/m.rs:2:"
    "drop_claim_gpu|commands/n.rs|    let d = claim_gpu()?.device();|1|claim-dropped: commands/n.rs:2:"
    "drop_check_claim|commands/o.rs|    let d = check_claim(claim_gpu())?.device();|1|claim-dropped: commands/o.rs:2:"
    "drop_by_path|commands/p.rs|    let d = parse_device(&s).map(ClaimedDevice::device)?;|1|claim-dropped: commands/p.rs:2:"
    "bound_then_device|commands/r.rs|    let claimed = parse_device(&s)?; let d = claimed.device();|0|ok (1 site"
    "device_in_comment|commands/s.rs|    let claimed = parse_device(&s)?; // not .device()|0|ok (1 site"
    "line_comment|commands/g.rs|// Device::Gpu is named only in claim_gpu|0|ok (1 site"
    "doc_comment|commands/h.rs|/// Returns \`Device::Gpu\` with the claim.|0|ok (1 site"
    "string_literal|commands/i.rs|    let s = \"Device::Gpu\";|0|ok (1 site"
    "longer_ident|commands/j.rs|    let d = Device::GpuLike;|0|ok (1 site"
    "unit_tests_file|commands/k_tests.rs|    let d = Device::Gpu;|0|ok (1 site"
    "tests_rs|commands/tests.rs|    let d = Device::Gpu;|0|ok (1 site"
    "bin_program|bin/diag.rs|    \"gpu\" => Device::Gpu,|0|ok (1 site"
    "alias_in_comment|commands/l.rs|// use rmlx_mlx::Device as Dev;|0|ok (1 site"
    "deep_type_alias|commands/f7.rs|type Dev = rmlx_mlx::device::Device;|1|device-alias: commands/f7.rs:2:"
    "drop_check_claim_alone|commands/o2.rs|    let d = check_claim(res)?.device();|1|claim-dropped: commands/o2.rs:2:"
    "drop_generic_angle|commands/o3.rs|    let d = parse_device<T>(&s)?.device();|1|claim-dropped: commands/o3.rs:2:"
)

for c in "${CASES[@]}"; do
    IFS='|' read -r name rel line want needle <<<"$c"
    root="$(fresh "$name")"
    plant "$root" "$rel" "$line"
    st_case "$name" "$want" --what "$rel" --has "$needle" -- bash "$GATE" "$root"
done

# A chain split over two lines is one statement.
root="$(fresh drop_split_lines)"
mkdir -p "$root/commands"
printf '// header\nlet d = parse_device(&device)?\n    .device();\n' >"$root/commands/t.rs"
st_case "drop_split_lines" 1 --what "commands/t.rs" \
    --has "claim-dropped: commands/t.rs:3:" -- bash "$GATE" "$root"

root="$(fresh drop_in_closure)"
printf '// header\nlet d = parse_device(&s).map(|c| c.device())?;\n' >"$root/commands/q.rs"
st_case "drop_in_closure" 1 --what "commands/q.rs" \
    --has "claim-dropped: commands/q.rs:2:" -- bash "$GATE" "$root"

root="$(fresh drop_in_block_closure)"
printf '// header\nlet d = parse_device(&s).map(|c| { c.device() })?;\n' >"$root/commands/u.rs"
st_case "drop_in_block_closure" 1 --what "commands/u.rs" \
    --has "claim-dropped: commands/u.rs:2:" -- bash "$GATE" "$root"

root="$(fresh drop_in_match)"
printf '// header\nlet d = match parse_device(&s)? {\n    c => c.device(),\n};\n' >"$root/commands/v.rs"
st_case "drop_in_match" 1 --what "commands/v.rs" \
    --has "claim-dropped: commands/v.rs:4:" -- bash "$GATE" "$root"

root="$(fresh drop_by_destructure)"
printf '// header\nlet ClaimedDevice { device, .. } = parse_device(&s)?;\n' >"$root/commands/w.rs"
st_case "drop_by_destructure" 1 --what "commands/w.rs" \
    --has "claim-dropped: commands/w.rs:2:" -- bash "$GATE" "$root"

root="$(fresh tail_expression_then_fn)"
printf '// header\nfn a() -> X {\n    parse_device(s)\n}\nfn b(c: &ClaimedDevice) {\n    run(c.device());\n}\n' >"$root/commands/x.rs"
st_case "tail_expression_then_fn" 0 --what "commands/x.rs" \
    --has "ok (1 site" -- bash "$GATE" "$root"

root="$(fresh definition_is_not_a_call)"
printf '// header\npub(crate) fn claim_gpu() -> Result<ClaimedDevice, E> {\n    let claimed = ClaimedDevice { claim: None };\n    Ok(claimed)\n}\n' >"$root/commands/y.rs"
st_case "definition_is_not_a_call" 0 --what "commands/y.rs" \
    --has "ok (1 site" -- bash "$GATE" "$root"

for fn in parse_device check_claim; do
    root="$(fresh "definition_of_${fn}")"
    printf '// header\npub(crate) fn %s() -> Result<ClaimedDevice, E> {\n    let claimed = ClaimedDevice { claim: None };\n    Ok(claimed)\n}\n' "$fn" >"$root/commands/y.rs"
    st_case "definition_of_${fn}_is_not_a_call" 0 --what "commands/y.rs" \
        --has "ok (1 site" -- bash "$GATE" "$root"
done

st_case "one_site" 0 --what "the legitimate site alone passes" \
    --has "gpu-device: commands/parse.rs:3:" -- bash "$GATE" "$(fresh one_site)"
st_case "no_site" 1 --what "the claimed-GPU helper lost its site" \
    --has "no Device::Gpu site" -- bash "$GATE" "$(fresh no_site none)"
st_case "missing_root" 2 --what "a scan root that does not exist" \
    --has "is not a directory" -- bash "$GATE" "$ST_WORK/does-not-exist"
mkdir -p "$ST_WORK/empty_root/bin"
printf 'fn main() {}\n' >"$ST_WORK/empty_root/bin/only.rs"
st_case "empty_root" 2 --what "a root with no in-scope file" \
    --has "no in-scope .rs file" -- bash "$GATE" "$ST_WORK/empty_root"

# ── Library mode ────────────────────────────────────────────────────────────
# lib_root <name> <file-body>: a library root holding src/lib.rs, which has no
# device, and src/x.rs with <file-body>.
lib_root() {
    local root="$ST_WORK/lib_$1"
    mkdir -p "$root"
    printf 'pub fn run(device: Device) {}\n' >"$root/lib.rs"
    printf '%b' "$2" >"$root/x.rs"
    printf '%s' "$root"
}

# name ~ x.rs body ~ want-exit ~ needle
LIB_CASES=(
    "lib_let_value~use rmlx_mlx::Device;\n    let d = Device::Gpu;\n~1~gpu-value: $ST_WORK/lib_lib_let_value/x.rs:2:"
    "lib_argument~    let w = w.transpose(&[0, 2, 1], Device::Gpu)?;\n~1~gpu-value: $ST_WORK/lib_lib_argument/x.rs:1:"
    "lib_full_path~    let device = rmlx_mlx::Device::Gpu;\n~1~gpu-value: $ST_WORK/lib_lib_full_path/x.rs:1:"
    "lib_field~    Cfg {\n        device: Device::Gpu,\n    }\n~1~gpu-value: $ST_WORK/lib_lib_field/x.rs:2:"
    "lib_argument_list~    run_smoke_probe(\n        path,\n        rmlx_mlx::Device::Gpu,\n    )\n~1~gpu-value: $ST_WORK/lib_lib_argument_list/x.rs:3:"
    "lib_value_beside_compare~    let d = if x == Device::Gpu { Device::Gpu } else { Device::Cpu };\n~1~1 value site(s)"
    "lib_cfg_test_fn_in_source~#[cfg(test)]\nfn gpu() -> Device {\n    Device::Gpu\n}\n~1~gpu-value: $ST_WORK/lib_lib_cfg_test_fn_in_source/x.rs:3:"
    "lib_value_after_closed_matches~    let g = matches!(d, Device::Cpu); run(Device::Gpu);\n~1~1 value site(s)"
    "lib_alias~use rmlx_mlx::Device::*;\n~1~device-alias: $ST_WORK/lib_lib_alias/x.rs:1:"
    "lib_operator_on_line_above~    if device ==\n        Device::Gpu\n    {\n~1~gpu-value: $ST_WORK/lib_lib_operator_on_line_above/x.rs:2:"
    "lib_wrapped_matches~    let g = matches!(\n        d,\n        Device::Gpu\n    );\n~1~gpu-value: $ST_WORK/lib_lib_wrapped_matches/x.rs:3:"
    "lib_split_compare~    if device\n        == Device::Gpu\n    {\n~0~0 value site(s)"
    "lib_eq~    if device == Device::Gpu {\n~0~0 value site(s)"
    "lib_ne~    if d != rmlx_mlx::Device::Gpu {\n~0~0 value site(s)"
    "lib_match_arm~        Device::Gpu => 1,\n~0~0 value site(s)"
    "lib_or_pattern~        Device::Cpu | Device::Gpu => 1,\n~0~0 value site(s)"
    "lib_or_pattern_first~        Device::Gpu | Device::Cpu => 1,\n~0~0 value site(s)"
    "lib_matches~    let g = matches!(self.device(), Device::Gpu);\n~0~0 value site(s)"
    "lib_or_pattern_in_if_let~    if let Device::Cpu | Device::Gpu = d {\n~0~0 value site(s)"
    "lib_if_let~    if let Device::Gpu = d {\n~0~0 value site(s)"
    "lib_while_let~    while let Device::Gpu = d {\n~0~0 value site(s)"
    "lib_comment~    // pass Device::Gpu here\n~0~0 value site(s)"
    "lib_string~    let s = \"Device::Gpu\";\n~0~0 value site(s)"
    "lib_longer_ident~    let d = Device::GpuLike;\n~0~0 value site(s)"
)

for c in "${LIB_CASES[@]}"; do
    IFS='~' read -r name body want needle <<<"$c"
    st_case "$name" "$want" --what "library x.rs" --has "$needle" -- bash "$GATE" --library "$(lib_root "$name" "$body")"
done

# A value in a sibling test file is not library code.
root="$(lib_root tests_file '    if device == Device::Gpu {\n')"
printf 'fn t() { run(Device::Gpu); }\n' >"$root/x_tests.rs"
printf 'fn t() { run(Device::Gpu); }\n' >"$root/tests.rs"
st_case "lib_tests_files" 0 --what "values in *_tests.rs and tests.rs" \
    --has "0 value site(s)" -- bash "$GATE" --library "$root"

# Two roots: a clean one does not hide a value in the other, and each root
# reports its own count.
clean="$(lib_root two_clean '    if device == Device::Gpu {\n')"
dirty="$(lib_root two_dirty '    let d = Device::Gpu;\n')"
st_case "lib_two_roots_first_clean" 1 --what "the value is in the second root" \
    --has "$dirty: 1 value site(s)" -- bash "$GATE" --library "$clean" "$dirty"
st_case "lib_two_roots_count_clean" 1 --what "the clean root reports zero" \
    --has "$clean: 0 value site(s)" -- bash "$GATE" --library "$clean" "$dirty"
st_case "lib_missing_root" 2 --what "one of two roots does not exist" \
    --has "is not a directory" -- bash "$GATE" --library "$clean" "$ST_WORK/does-not-exist"
mkdir -p "$ST_WORK/lib_empty/bin"
printf 'fn main() {}\n' >"$ST_WORK/lib_empty/bin/only.rs"
st_case "lib_empty_root" 2 --what "a root with no in-scope file" \
    --has "no in-scope .rs file" -- bash "$GATE" --library "$ST_WORK/lib_empty"

# The real tree: exactly one site, in the claimed-GPU helper.
st_case real_tree 0 --what "exit 0 with the one site in commands/parse.rs" \
    --has 'gpu-device: commands/parse.rs:' -- bash "$GATE"

st_done
