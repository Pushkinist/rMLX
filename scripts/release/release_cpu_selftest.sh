#!/usr/bin/env bash
# scripts/release/release_cpu_selftest.sh — recall test for
# `scripts/release/release_cpu.py`.
#
# Each case builds a throwaway repo root holding a `.cargo/config.toml`, a
# cargo-shaped `target/release/` (a `deps/rmlx-<hash>` binary and the
# fingerprint record naming its rustflags) and a release tarball, runs the
# check against it and asserts the literal exit code and the line naming why.
# No cargo, no build, no GPU.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/release_cpu.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0
PASSED=0

HASH_A=13f7f9e47a3e69b8
HASH_B=7ee0f132db9aebf3
CONFIG_FLAGS='["-C", "target-cpu=native", "-C", "link-arg=-dead_strip"]'
BASELINE_FLAGS='["-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip"]'

# unit <root> <hash> <bytes> <rustflags-json>: one linked binary and its record.
unit() {
    local root="$1" hash="$2" bytes="$3" flags="$4"
    mkdir -p "$root/target/release/deps" "$root/target/release/.fingerprint/rmlx-cli-$hash"
    printf '%s' "$bytes" >"$root/target/release/deps/rmlx-$hash"
    printf '{"rustc": 1, "features": "[]", "rustflags": %s, "compile_kind": 0}\n' "$flags" \
        >"$root/target/release/.fingerprint/rmlx-cli-$hash/bin-rmlx.json"
}

# package <root> <bytes>: the release tarball, holding `rmlx` with those bytes.
package() {
    local root="$1" bytes="$2" stage="$1/dist/rmlx-v0.0.0-aarch64-apple-darwin"
    mkdir -p "$stage"
    printf '%s' "$bytes" >"$stage/rmlx"
    printf 'readme\n' >"$stage/README.md"
    tar -C "$root/dist" -czf "$root/dist/rmlx.tar.gz" "rmlx-v0.0.0-aarch64-apple-darwin"
}

# fresh <name> <rustflags-json>: config.toml as the tree has it, one unit built
# with <rustflags-json>, and the tarball packaging that unit's binary.
fresh() {
    local root="$WORK/$1"
    mkdir -p "$root/.cargo"
    printf '[build]\nrustflags = %s\n' "$CONFIG_FLAGS" >"$root/.cargo/config.toml"
    unit "$root" "$HASH_A" "binary-a" "$2"
    package "$root" "binary-a"
    printf '%s' "$root"
}

# case_run <name> <what> <want-exit> <needle> <root>
case_run() {
    local name="$1" what="$2" want="$3" needle="$4" root="$5"
    local out status
    out=$(python3 "$TOOL" check "$root/dist/rmlx.tar.gz" --root "$root" 2>&1)
    status=$?
    if [ "$status" != "$want" ]; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (want exit %s, got %s) — %s\n%s\n' "$name" "$want" "$status" "$what" "$out"
    elif ! printf '%s' "$out" | grep -qF -- "$needle"; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (missing %q) — %s\n%s\n' "$name" "$needle" "$what" "$out"
    else
        PASSED=$((PASSED + 1))
        printf '  ok   %-34s — %s\n' "$name" "$what"
    fi
}

root=$(fresh baseline "$BASELINE_FLAGS")
case_run baseline_passes "apple-m1 plus config.toml's other flags passes" \
    0 "release-cpu: ok rmlx.tar.gz built with target-cpu=apple-m1" "$root"

root=$(fresh native "$CONFIG_FLAGS")
case_run native_fails "config.toml's own flags, target-cpu=native, fail" \
    1 "built with target-cpu ['native'], expected exactly ['apple-m1']" "$root"

root=$(fresh newer_core '["-C", "target-cpu=apple-m4", "-C", "link-arg=-dead_strip"]')
case_run newer_core_fails "a pinned core newer than the baseline fails" \
    1 "built with target-cpu ['apple-m4']" "$root"

root=$(fresh no_cpu '["-C", "link-arg=-dead_strip"]')
case_run no_cpu_fails "no target-cpu at all is not an explicit pin" \
    1 "built with target-cpu [], expected exactly ['apple-m1']" "$root"

root=$(fresh two_cpus '["-C", "target-cpu=native", "-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip"]')
case_run two_cpus_fail "native then apple-m1 (the last wins in rustc) still fails: one pin, not two" \
    1 "built with target-cpu ['native', 'apple-m1']" "$root"

root=$(fresh joined_spelling '["-Ctarget-cpu=native", "-C", "link-arg=-dead_strip"]')
case_run joined_spelling_fails "-Ctarget-cpu=native in one token is read as a CPU" \
    1 "built with target-cpu ['native']" "$root"

root=$(fresh long_spelling '["--codegen=target-cpu=native", "--codegen", "link-arg=-dead_strip"]')
case_run long_spelling_fails "--codegen=target-cpu=native is read as a CPU" \
    1 "built with target-cpu ['native']" "$root"

root=$(fresh long_spelling_ok '["--codegen", "target-cpu=apple-m1", "-Clink-arg=-dead_strip"]')
case_run long_spelling_passes "the baseline in the --codegen spelling passes" \
    0 "release-cpu: ok" "$root"

root=$(fresh no_dead_strip '["-C", "target-cpu=apple-m1"]')
case_run dead_strip_dropped_fails "the baseline without config.toml's -dead_strip fails" \
    1 "was built with flags [] besides the target CPU; .cargo/config.toml sets ['-Clink-arg=-dead_strip']" "$root"

root=$(fresh extra_feature '["-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip", "-C", "target-feature=+i8mm"]')
case_run extra_feature_fails "a target feature beyond the baseline fails" \
    1 "'-Ctarget-feature=+i8mm'] besides the target CPU" "$root"

root=$(fresh stale_record "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
rm -rf "$root/dist"
package "$root" "binary-b"
case_run stale_baseline_record_ignored "a baseline record beside the packaged native unit does not pass it" \
    1 "rmlx-cli-$HASH_B/bin-rmlx.json" "$root"

root=$(fresh stale_native "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
case_run stale_native_record_ignored "a native record beside the packaged baseline unit does not fail it" \
    0 "release-cpu: ok" "$root"

root=$(fresh not_linked "$BASELINE_FLAGS")
rm -rf "$root/dist"
package "$root" "binary-from-elsewhere"
case_run unlinked_binary_unavailable "a packaged binary cargo did not link here is unknown, not a pass" \
    2 "found 0" "$root"

root=$(fresh twice_linked "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-a" "$BASELINE_FLAGS"
case_run twice_linked_unavailable "two units with the packaged bytes is ambiguous" \
    2 "found 2" "$root"

root=$(fresh no_record "$BASELINE_FLAGS")
rm -rf "$root/target/release/.fingerprint"
case_run missing_record_unavailable "a linked binary with no fingerprint record is unknown" \
    2 "cargo left no record of how rmlx-$HASH_A was built" "$root"

root=$(fresh record_without_flags "$BASELINE_FLAGS")
printf '{"rustc": 1}\n' >"$root/target/release/.fingerprint/rmlx-cli-$HASH_A/bin-rmlx.json"
case_run record_without_flags_unavailable "a record with no rustflags list is unknown, not an empty list" \
    2 "no rustflags list" "$root"

root=$(fresh record_garbled "$BASELINE_FLAGS")
printf '{"rustc": ' >"$root/target/release/.fingerprint/rmlx-cli-$HASH_A/bin-rmlx.json"
case_run record_garbled_unavailable "an unreadable record is unknown" \
    2 "release-cpu: unavailable" "$root"

root=$(fresh no_config "$BASELINE_FLAGS")
rm -f "$root/.cargo/config.toml"
case_run missing_config_unavailable "no config.toml to hold the other flags to" \
    2 "config.toml not found" "$root"

root=$(fresh multiline_config "$BASELINE_FLAGS")
printf '[build]\nrustflags = [\n  "-C", "link-arg=-dead_strip",\n]\n' >"$root/.cargo/config.toml"
case_run multiline_config_unavailable "a rustflags array over several lines is refused, not misread" \
    2 "the rustflags array is not on one line" "$root"

root=$(fresh two_rustflags "$BASELINE_FLAGS")
printf '[target.x]\nrustflags = []\n' >>"$root/.cargo/config.toml"
case_run two_rustflags_unavailable "two rustflags lines are refused, not the first taken" \
    2 "found 2" "$root"

root=$(fresh no_tarball "$BASELINE_FLAGS")
rm -rf "$root/dist"
case_run missing_tarball_unavailable "no tarball" \
    2 "rmlx.tar.gz not found" "$root"

root=$(fresh no_binary_in_tarball "$BASELINE_FLAGS")
rm -f "$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/rmlx" "$root/dist/rmlx.tar.gz"
tar -C "$root/dist" -czf "$root/dist/rmlx.tar.gz" "rmlx-v0.0.0-aarch64-apple-darwin"
case_run tarball_without_binary_unavailable "a tarball holding no rmlx binary" \
    2 "expected one \`rmlx\` binary, found 0" "$root"

root=$(fresh two_binaries_in_tarball "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
mkdir -p "$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/extra"
printf 'binary-b' >"$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/extra/rmlx"
tar -C "$root/dist" -czf "$root/dist/rmlx.tar.gz" "rmlx-v0.0.0-aarch64-apple-darwin"
case_run tarball_with_two_binaries_unavailable "a tarball holding two rmlx binaries is ambiguous, not the first checked" \
    2 "expected one \`rmlx\` binary, found 2" "$root"

# flags_run <name> <what> <want-exit> <want-output> <root>: the `rustflags`
# producer, its 0x1f separators shown as spaces.
flags_run() {
    local name="$1" what="$2" want="$3" want_out="$4" root="$5"
    local out status
    out=$(python3 "$TOOL" rustflags --root "$root" 2>&1)
    status=$?
    out=$(printf '%s' "$out" | tr '\037' ' ')
    if [ "$status" != "$want" ] || [ "$out" != "$want_out" ]; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (want exit %s, %q; got %s, %q) — %s\n' \
            "$name" "$want" "$want_out" "$status" "$out" "$what"
    else
        PASSED=$((PASSED + 1))
        printf '  ok   %-34s — %s\n' "$name" "$what"
    fi
}

root=$(fresh producer "$BASELINE_FLAGS")
flags_run producer_replaces_cpu "config.toml's flags with native replaced by the baseline" \
    0 "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip" "$root"

root=$(fresh producer_carries_flags "$BASELINE_FLAGS")
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "link-arg=-dead_strip", "-C", "force-frame-pointers=yes"]\n' \
    >"$root/.cargo/config.toml"
flags_run producer_carries_new_flag "a flag added to config.toml reaches the release build" \
    0 "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip -Cforce-frame-pointers=yes" "$root"

root=$(fresh producer_no_cpu "$BASELINE_FLAGS")
printf '[build]\nrustflags = ["-C", "link-arg=-dead_strip"]\n' >"$root/.cargo/config.toml"
flags_run producer_pins_without_config_cpu "config.toml naming no CPU still gets the pin" \
    0 "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip" "$root"

root=$(fresh producer_multiline "$BASELINE_FLAGS")
printf '[build]\nrustflags = [\n  "-C", "link-arg=-dead_strip",\n]\n' >"$root/.cargo/config.toml"
flags_run producer_refuses_unreadable_config "an unreadable config.toml is refused, not an empty flag list" \
    2 "release-cpu: unavailable: $root/.cargo/config.toml: the rustflags array is not on one line: rustflags = [" "$root"

root=$(fresh producer_round_trip "$BASELINE_FLAGS")
produced=$(python3 "$TOOL" rustflags --root "$root" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\x1f")))')
unit "$root" "$HASH_A" "binary-a" "$produced"
case_run producer_output_passes_check "a build with exactly the producer's flags passes the check" \
    0 "release-cpu: ok" "$root"

printf '\nrelease-cpu selftest:%d passed, %d failed\n' "$PASSED" "$FAILED"
[ "$FAILED" -eq 0 ]
