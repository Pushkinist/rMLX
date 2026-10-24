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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$HERE/release_cpu.py"
# shellcheck source=scripts/lib/selftest_support.sh
. "$HERE/../lib/selftest_support.sh"
st_init release-cpu-selftest

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
    local root="$ST_WORK/$1"
    mkdir -p "$root/.cargo"
    printf '[build]\nrustflags = %s\n' "$CONFIG_FLAGS" >"$root/.cargo/config.toml"
    unit "$root" "$HASH_A" "binary-a" "$2"
    package "$root" "binary-a"
    printf '%s' "$root"
}

# release_check <root>: `release_cpu.py check` over the tarball built in <root>.
release_check() {
    python3 "$TOOL" check "$1/dist/rmlx.tar.gz" --root "$1"
}

root=$(fresh baseline "$BASELINE_FLAGS")
st_case baseline_passes 0 --what "apple-m1 plus config.toml's other flags passes" \
    --has "release-cpu: ok rmlx.tar.gz built with target-cpu=apple-m1" -- release_check "$root"

root=$(fresh native "$CONFIG_FLAGS")
st_case native_fails 1 --what "config.toml's own flags, target-cpu=native, fail" \
    --has "built with target-cpu ['native'], expected exactly ['apple-m1']" -- release_check "$root"

root=$(fresh newer_core '["-C", "target-cpu=apple-m4", "-C", "link-arg=-dead_strip"]')
st_case newer_core_fails 1 --what "a pinned core newer than the baseline fails" \
    --has "built with target-cpu ['apple-m4']" -- release_check "$root"

root=$(fresh no_cpu '["-C", "link-arg=-dead_strip"]')
st_case no_cpu_fails 1 --what "no target-cpu at all is not an explicit pin" \
    --has "built with target-cpu [], expected exactly ['apple-m1']" -- release_check "$root"

root=$(fresh two_cpus '["-C", "target-cpu=native", "-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip"]')
st_case two_cpus_fail 1 --what "native then apple-m1 (the last wins in rustc) still fails: one pin, not two" \
    --has "built with target-cpu ['native', 'apple-m1']" -- release_check "$root"

root=$(fresh joined_spelling '["-Ctarget-cpu=native", "-C", "link-arg=-dead_strip"]')
st_case joined_spelling_fails 1 --what "-Ctarget-cpu=native in one token is read as a CPU" \
    --has "built with target-cpu ['native']" -- release_check "$root"

root=$(fresh long_spelling '["--codegen=target-cpu=native", "--codegen", "link-arg=-dead_strip"]')
st_case long_spelling_fails 1 --what "--codegen=target-cpu=native is read as a CPU" \
    --has "built with target-cpu ['native']" -- release_check "$root"

root=$(fresh long_spelling_ok '["--codegen", "target-cpu=apple-m1", "-Clink-arg=-dead_strip"]')
st_case long_spelling_passes 0 --what "the baseline in the --codegen spelling passes" \
    --has "release-cpu: ok" -- release_check "$root"

root=$(fresh no_dead_strip '["-C", "target-cpu=apple-m1"]')
st_case dead_strip_dropped_fails 1 --what "the baseline without config.toml's -dead_strip fails" \
    --has "was built with flags [] besides the target CPU; .cargo/config.toml sets ['-Clink-arg=-dead_strip']" -- release_check "$root"

root=$(fresh extra_feature '["-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip", "-C", "target-feature=+i8mm"]')
st_case extra_feature_fails 1 --what "a target feature beyond the baseline fails" \
    --has "was built with ['-Ctarget-feature=+i8mm']: no feature beyond apple-m1's" -- release_check "$root"

root=$(fresh feature_in_config '["-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip", "-C", "target-feature=+bf16"]')
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "link-arg=-dead_strip", "-C", "target-feature=+bf16"]\n' \
    >"$root/.cargo/config.toml"
st_case feature_in_config_fails 1 --what "a target feature fails even when config.toml sets it too" \
    --has "was built with ['-Ctarget-feature=+bf16']" -- release_check "$root"

root=$(fresh underscore_spelling '["-C", "target_cpu=native", "-C", "link-arg=-dead_strip"]')
st_case underscore_spelling_fails 1 --what "target_cpu (rustc reads _ and - alike) is read as a CPU" \
    --has "built with target-cpu ['native']" -- release_check "$root"

root=$(fresh underscore_feature '["-C", "target-cpu=apple-m1", "-C", "link-arg=-dead_strip", "-Ctarget_feature=+i8mm"]')
st_case underscore_feature_fails 1 --what "target_feature is read as a target feature" \
    --has "was built with ['-Ctarget-feature=+i8mm']" -- release_check "$root"

root=$(fresh order_swapped '["-C", "target-cpu=apple-m1", "-C", "link-arg=-b", "-C", "link-arg=-a"]')
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "link-arg=-a", "-C", "link-arg=-b"]\n' \
    >"$root/.cargo/config.toml"
st_case order_swapped_fails 1 --what "config.toml's flags in another order fail: order reaches the linker" \
    --has "['-Clink-arg=-b', '-Clink-arg=-a'] besides the target CPU" -- release_check "$root"

root=$(fresh stale_record "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
rm -rf "$root/dist"
package "$root" "binary-b"
st_case stale_baseline_record_ignored 1 --what "a baseline record beside the packaged native unit does not pass it" \
    --has "rmlx-cli-$HASH_B/bin-rmlx.json" -- release_check "$root"

root=$(fresh stale_native "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
st_case stale_native_record_ignored 0 --what "a native record beside the packaged baseline unit does not fail it" \
    --has "release-cpu: ok" -- release_check "$root"

root=$(fresh not_linked "$BASELINE_FLAGS")
rm -rf "$root/dist"
package "$root" "binary-from-elsewhere"
st_case unlinked_binary_unavailable 2 --what "a packaged binary cargo did not link here is unknown, not a pass" \
    --has "found 0" -- release_check "$root"

root=$(fresh depinfo_beside "$BASELINE_FLAGS")
printf 'binary-a' >"$root/target/release/deps/rmlx-$HASH_A.d"
st_case depinfo_not_a_binary 0 --what "a deps/rmlx-<hash>.d beside the binary is not a second linked unit" \
    --has "release-cpu: ok" -- release_check "$root"

root=$(fresh twice_linked "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-a" "$BASELINE_FLAGS"
st_case twice_linked_unavailable 2 --what "two units with the packaged bytes is ambiguous" \
    --has "found 2" -- release_check "$root"

root=$(fresh no_record "$BASELINE_FLAGS")
rm -rf "$root/target/release/.fingerprint"
st_case missing_record_unavailable 2 --what "a linked binary with no fingerprint record is unknown" \
    --has "cargo left no record of how rmlx-$HASH_A was built" -- release_check "$root"

root=$(fresh record_without_flags "$BASELINE_FLAGS")
printf '{"rustc": 1}\n' >"$root/target/release/.fingerprint/rmlx-cli-$HASH_A/bin-rmlx.json"
st_case record_without_flags_unavailable 2 --what "a record with no rustflags list is unknown, not an empty list" \
    --has "no rustflags list" -- release_check "$root"

root=$(fresh record_garbled "$BASELINE_FLAGS")
printf '{"rustc": ' >"$root/target/release/.fingerprint/rmlx-cli-$HASH_A/bin-rmlx.json"
st_case record_garbled_unavailable 2 --what "an unreadable record is unknown" \
    --has "bin-rmlx.json: not JSON: Expecting value" -- release_check "$root"

root=$(fresh no_config "$BASELINE_FLAGS")
rm -f "$root/.cargo/config.toml"
st_case missing_config_unavailable 2 --what "no config.toml to hold the other flags to" \
    --has "config.toml not found" -- release_check "$root"

root=$(fresh multiline_config "$BASELINE_FLAGS")
printf '[build]\nrustflags = [\n  "-C", "link-arg=-dead_strip",\n]\n' >"$root/.cargo/config.toml"
st_case multiline_config_unavailable 2 --what "a rustflags array over several lines is refused, not misread" \
    --has "the rustflags array is not on one line" -- release_check "$root"

root=$(fresh two_rustflags "$BASELINE_FLAGS")
printf '[target.x]\nrustflags = []\n' >>"$root/.cargo/config.toml"
st_case two_rustflags_unavailable 2 --what "two rustflags lines are refused, not the first taken" \
    --has "found 2" -- release_check "$root"

root=$(fresh no_tarball "$BASELINE_FLAGS")
rm -rf "$root/dist"
st_case missing_tarball_unavailable 2 --what "no tarball" \
    --has "rmlx.tar.gz not found" -- release_check "$root"

root=$(fresh no_binary_in_tarball "$BASELINE_FLAGS")
rm -f "$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/rmlx" "$root/dist/rmlx.tar.gz"
tar -C "$root/dist" -czf "$root/dist/rmlx.tar.gz" "rmlx-v0.0.0-aarch64-apple-darwin"
st_case tarball_without_binary_unavailable 2 --what "a tarball holding no rmlx binary" \
    --has "expected one \`rmlx\` binary, found 0" -- release_check "$root"

root=$(fresh two_binaries_in_tarball "$BASELINE_FLAGS")
unit "$root" "$HASH_B" "binary-b" "$CONFIG_FLAGS"
mkdir -p "$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/extra"
printf 'binary-b' >"$root/dist/rmlx-v0.0.0-aarch64-apple-darwin/extra/rmlx"
tar -C "$root/dist" -czf "$root/dist/rmlx.tar.gz" "rmlx-v0.0.0-aarch64-apple-darwin"
st_case tarball_with_two_binaries_unavailable 2 --what "a tarball holding two rmlx binaries is ambiguous, not the first checked" \
    --has "expected one \`rmlx\` binary, found 2" -- release_check "$root"

# rustflags_shown <root>: the `rustflags` producer's output, its 0x1f separators
# shown as spaces.
rustflags_shown() {
    python3 "$TOOL" rustflags --root "$1" 2>&1 | tr '\037' ' '
}

root=$(fresh producer "$BASELINE_FLAGS")
st_case producer_replaces_cpu 0 --what "config.toml's flags with native replaced by the baseline" \
    --is "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip --remap-path-prefix=$HOME=~" -- rustflags_shown "$root"

root=$(fresh producer_carries_flags "$BASELINE_FLAGS")
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "link-arg=-dead_strip", "-C", "force-frame-pointers=yes"]\n' \
    >"$root/.cargo/config.toml"
st_case producer_carries_new_flag 0 --what "a flag added to config.toml reaches the release build" \
    --is "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip -Cforce-frame-pointers=yes --remap-path-prefix=$HOME=~" -- rustflags_shown "$root"

root=$(fresh producer_no_cpu "$BASELINE_FLAGS")
printf '[build]\nrustflags = ["-C", "link-arg=-dead_strip"]\n' >"$root/.cargo/config.toml"
st_case producer_pins_without_config_cpu 0 --what "config.toml naming no CPU still gets the pin" \
    --is "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip --remap-path-prefix=$HOME=~" -- rustflags_shown "$root"

root=$(fresh producer_multiline "$BASELINE_FLAGS")
printf '[build]\nrustflags = [\n  "-C", "link-arg=-dead_strip",\n]\n' >"$root/.cargo/config.toml"
st_case producer_refuses_unreadable_config 2 --what "an unreadable config.toml is refused, not an empty flag list" \
    --is "release-cpu: unavailable: $root/.cargo/config.toml: the rustflags array is not on one line: rustflags = [" -- rustflags_shown "$root"

root=$(fresh producer_feature "$BASELINE_FLAGS")
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "target-feature=+i8mm"]\n' >"$root/.cargo/config.toml"
st_case producer_refuses_feature 1 --what "config.toml setting a target feature is refused, not carried into the release" \
    --is "release-cpu: FAIL .cargo/config.toml sets ['-Ctarget-feature=+i8mm']: the release binary enables no feature beyond apple-m1's" -- rustflags_shown "$root"

root=$(fresh producer_round_trip "$BASELINE_FLAGS")
produced=$(python3 "$TOOL" rustflags --root "$root" | python3 -c 'import json, sys; print(json.dumps(sys.stdin.read().split("\x1f")))')
unit "$root" "$HASH_A" "binary-a" "$produced"
st_case producer_output_passes_check 0 --what "a build with exactly the producer's flags passes the check" \
    --has "release-cpu: ok" -- release_check "$root"

# package_root <name> <honors-env: yes|no>: a copy of the files
# package_binary.sh reads, and a stub `cargo` first on PATH. The stub links a
# binary and records rustflags the way cargo does: CARGO_ENCODED_RUSTFLAGS when
# set (unless told to ignore it), else config.toml's. It refuses to build under
# a rustc wrapper, whose arguments no record would show.
package_root() {
    local root="$ST_WORK/$1" honors="$2" release
    release="$(dirname "$TOOL")"
    mkdir -p "$root/scripts/release" "$root/.cargo" "$root/bin"
    cp "$release/package_binary.sh" "$release/release_cpu.py" "$root/scripts/release/"
    printf '[workspace.package]\nversion = "0.0.0"\n' >"$root/Cargo.toml"
    printf '[build]\nrustflags = %s\n' "$CONFIG_FLAGS" >"$root/.cargo/config.toml"
    printf 'licence\n' >"$root/LICENSE-MIT"
    printf 'licence\n' >"$root/LICENSE-APACHE"
    printf 'readme\n' >"$root/README.md"
    cat >"$root/bin/cargo" <<STUB
#!/usr/bin/env bash
if [ -n "\${RUSTC_WRAPPER:-}\${RUSTC_WORKSPACE_WRAPPER:-}" ]; then
    echo "stub cargo: a rustc wrapper is set" >&2
    exit 1
fi
flags=\${CARGO_ENCODED_RUSTFLAGS-unset}
[ "$honors" = yes ] || flags=unset
mkdir -p target/release/deps target/release/.fingerprint/rmlx-cli-$HASH_A
printf 'stub-binary %s' "\$flags" >target/release/deps/rmlx-$HASH_A
[ -z "\${STUB_LEAK_HOME:-}" ] || printf ' panicked at %s/.cargo/registry/src/x.rs' "\$HOME" >>target/release/deps/rmlx-$HASH_A
cp target/release/deps/rmlx-$HASH_A target/release/rmlx
chmod +x target/release/rmlx
FLAGS="\$flags" python3 - <<'PY' >target/release/.fingerprint/rmlx-cli-$HASH_A/bin-rmlx.json
import json, os, re
flags = os.environ["FLAGS"]
if flags == "unset":
    line = re.search(r"^rustflags = (.*)$", open(".cargo/config.toml").read(), re.M).group(1)
    listed = json.loads(line)
else:
    listed = flags.split("\x1f")
print(json.dumps({"rustflags": listed}))
PY
STUB
    chmod +x "$root/bin/cargo"
    printf '%s' "$root"
}

# package_exec <root> [VAR=value...]: package_binary.sh run from <root> with its
# stub cargo first on PATH.
package_exec() {
    local root="$1"
    shift
    (cd "$root" && env PATH="$root/bin:$PATH" "$@" bash scripts/release/package_binary.sh)
}

# package_run <name> <what> <want-exit> <want-kept: yes|no> <root> [VAR=value...]
# "kept" is the tarball, its .sha256 and the staging directory, all three;
# "no" is none of them.
package_run() {
    local name="$1" what="$2" want="$3" kept="$4" root="$5" why have
    shift 5
    local tarball="$root/dist/rmlx-v0.0.0-aarch64-apple-darwin.tar.gz"
    st_run package_exec "$root" "$@"
    if [ -f "$tarball" ] && [ -f "$tarball.sha256" ] && [ -d "${tarball%.tar.gz}" ]; then
        have=yes
    elif [ -e "$tarball" ] || [ -e "$tarball.sha256" ] || [ -e "${tarball%.tar.gz}" ]; then
        have=partial
    else
        have=no
    fi
    why="$(st_check "$want")"
    [ "$have" = "$kept" ] || why="${why:+$why$'\n'}kept $have, want $kept"
    st_verdict "$name" "$why" "$what"
}

root=$(package_root package_pins yes)
package_run package_keeps_checked_tarball "package_binary.sh builds with the pin, checks it, keeps the tarball" \
    0 yes "$root"

root=$(package_root package_clears_wrappers yes)
package_run package_clears_wrappers "a rustc wrapper in the environment does not reach the release build" \
    0 yes "$root" RUSTC_WRAPPER=/wrapper RUSTC_WORKSPACE_WRAPPER=/wrapper

root=$(package_root package_native_build no)
package_run package_removes_unpinned_tarball "a build ignoring the pin fails, leaving no tarball, checksum or staging" \
    1 no "$root"

root=$(package_root package_home_path yes)
package_run package_refuses_home_path "a binary carrying the build machine's home directory is not packaged" \
    1 no "$root" STUB_LEAK_HOME=1

st_done
