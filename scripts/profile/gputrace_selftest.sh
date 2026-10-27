#!/usr/bin/env bash
# Recall test for the .gputrace readers: synthetic bundles, no trace, no Xcode tool, no GPU.
#
# Recall test for: profile/gputrace.sh
# Not a mode of: check/metal_kernels_selftest.sh — that one plants MSL kernel trees; this one builds .gputrace bundles
#
# Each case builds a small bundle in scratch (an MTSP archive of `function` records, and the
# files the summary reads), runs one mode of the reader and asserts the literal exit code and
# the lines or the reason it prints. `plutil` is a stub that reads a one-line `PLIST` marker,
# so the cases run on any host. The reader is named in one place, the function `reader`.
#
# Usage:  bash scripts/profile/gputrace_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init gputrace-selftest

# reader <kernels|diff|summary> [args]: one mode of the reader, run from the repo.
reader() {
    st_bounded 60 bash "$REPO_ROOT/scripts/profile/gputrace.sh" ${1+"$@"}
}

mkdir -p "$ST_WORK/stubs"
cat >"$ST_WORK/stubs/plutil" <<'EOF2'
#!/bin/sh
[ "$1" = -p ] && [ "$(head -n 1 "$2")" = PLIST ] || { echo "$2: Property List error: unexpected character" >&2; exit 1; }
sed 1d "$2"
EOF2
chmod +x "$ST_WORK/stubs/plutil"
export PATH="$ST_WORK/stubs:$PATH"

OPAQUE=0A1B2C3D4E5F6071

# archive <file> <record>...: an MTSP archive. A record is name:<n> (a named function), opaque
# (a function stored by object id), bad (a function whose value is a container tag) or trunc
# (a function marker with nothing after it).
archive() {
    local file="$1" r
    shift
    printf 'MTSP\0' >"$file"
    for r in "$@"; do
        case "$r" in
        name:*) printf 'function\0%s\0' "${r#name:}" ;;
        opaque) printf 'function\0%s\0' "$OPAQUE" ;;
        bad) printf 'function\0CSuwuw\0' ;;
        trunc) printf 'function\0' ;;
        *) st_die "archive: unknown record '$r'" ;;
        esac >>"$file"
    done
}

# bundle <name> <used-record>...: a bundle directory holding one referenced-functions archive.
bundle() {
    local dir="$ST_WORK/$1.gputrace"
    shift
    mkdir -p "$dir"
    archive "$dir/device-resources-0x1" ${1+"$@"}
    printf '%s' "$dir"
}

# unused <bundle> <record>...: add the unused-functions archive.
unused() {
    local dir="$1"
    shift
    archive "$dir/unused-device-resources-0x2" ${1+"$@"}
}

# ---- modes -------------------------------------------------------------------

st_case "no mode is a usage error" 2 --has "usage:" --has "{kernels|diff|summary}" -- reader
st_case "an unknown mode is a usage error" 2 --has "unknown mode: bogus" --has "usage:" -- reader bogus

# ---- kernels -----------------------------------------------------------------

b="$(bundle k-ok name:alpha_kernel name:beta_kernel name:alpha_kernel opaque bad)"
unused "$b" name:gamma_kernel
st_case "kernels: a well-formed bundle lists the named functions and counts the rest" 0 \
    --has "bundle: $b" \
    --has "referenced by the window (device-resources-0x1): 5 function records — 2 named, 1 stored by object id, 1 unreadable" \
    --has "  alpha_kernel" --has "  beta_kernel" \
    --has "recorded unused by the capture layer (unused-device-resources-0x2): 1 function records — 1 named, 0 stored by object id" \
    --has "  gamma_kernel" --lacks "$OPAQUE" -- reader kernels "$b"
st_case "kernels: --set used --names-only prints the sorted unique names and nothing else" 0 \
    --is $'alpha_kernel\nbeta_kernel' -- reader kernels "$b" --set used --names-only
st_case "kernels: --set unused --names-only prints the unused names" 0 --is gamma_kernel -- reader kernels "$b" --set unused --names-only
st_case "kernels: --set unused leaves the referenced set out" 0 --has "  gamma_kernel" --lacks "alpha_kernel" -- reader kernels "$b" --set unused

b="$(bundle k-no-unused name:alpha_kernel)"
st_case "kernels: a bundle with no unused archive says so" 0 \
    --has "recorded unused: no unused-device-resources-0x* file in this bundle" --has "  alpha_kernel" -- reader kernels "$b"
st_case "kernels: --names-only on a missing unused archive is a refusal" 3 \
    --has "has no unused-device-resources-0x* file" -- reader kernels "$b" --set unused --names-only

b="$(bundle k-unused-empty name:alpha_kernel)"
unused "$b"
st_case "kernels: an unused archive with no records is reported, not refused" 0 \
    --has "(unused-device-resources-0x2): 0 function records — 0 named, 0 stored by object id" -- reader kernels "$b"
st_case "kernels: an empty unused archive prints no names" 0 --is "" -- reader kernels "$b" --set unused --names-only

b="$(bundle k-trunc name:alpha_kernel name:beta_kernel trunc)"
st_case "kernels: one truncated record among good ones is counted, not fatal" 0 \
    --has "3 function records — 2 named, 0 stored by object id, 1 unreadable" -- reader kernels "$b" --set used

b="$(bundle k-mid-trunc name:alpha_kernel trunc name:beta_kernel)"
st_case "kernels: a marker with no value before the next record is counted, not fatal" 0 \
    --has "3 function records — 2 named, 0 stored by object id, 1 unreadable" -- reader kernels "$b" --set used

b="$(bundle k-no-res name:alpha_kernel)"
rm -f "$b/device-resources-0x1"
st_case "kernels: a bundle with no device-resources archive is refused" 3 --has "ERROR: no device-resources-0x* file in $b" -- reader kernels "$b"

b="$(bundle k-magic name:alpha_kernel)"
printf 'XXXX\0function\0alpha_kernel\0' >"$b/device-resources-0x1"
st_case "kernels: a wrong container magic is refused" 3 --has "does not start with the expected MTSP magic (got 'XXXX')" -- reader kernels "$b"

b="$(bundle k-magic-unused name:alpha_kernel)"
printf 'XXXX\0' >"$b/unused-device-resources-0x2"
st_case "kernels: a wrong magic on the unused archive is refused" 3 \
    --has "unused-device-resources-0x2 does not start with the expected MTSP magic" -- reader kernels "$b"

b="$(bundle k-no-records)"
st_case "kernels: a referenced archive with no function record is refused" 3 --has "holds no 'function' records at all" -- reader kernels "$b"

b="$(bundle k-mostly-bad name:alpha_kernel bad bad)"
st_case "kernels: most records unreadable is refused" 3 --has "2 of 3 function records unreadable" -- reader kernels "$b"

b="$(bundle k-lone-trunc trunc)"
st_case "kernels: a lone truncated record is refused" 3 --has "1 of 1 function records unreadable" -- reader kernels "$b"

b="$(bundle k-only-opaque opaque opaque)"
st_case "kernels: records that carry no name are refused" 3 --has "2 function records, none with a readable name" -- reader kernels "$b"

b="$(bundle k-bad-unused name:alpha_kernel)"
unused "$b" bad
st_case "kernels: an unreadable unused archive is refused" 3 --has "unused-device-resources-0x2: 1 of 1 function records unreadable" -- reader kernels "$b"

b="$(bundle k-usage name:alpha_kernel)"
st_case "kernels: no argument is a usage error" 2 --has "usage:" -- reader kernels
st_case "kernels: an unknown option is a usage error" 2 --has "unknown argument: --bogus" -- reader kernels "$b" --bogus
st_case "kernels: --set without a value is a usage error" 2 --has "--set needs used|unused|all" -- reader kernels "$b" --set
st_case "kernels: a bad --set value is a usage error" 2 --has "--set must be used, unused or all (got 'some')" -- reader kernels "$b" --set some
st_case "kernels: two bundles are a usage error" 2 --has "only one bundle at a time" -- reader kernels "$b" "$b"
st_case "kernels: --names-only over both sets is a usage error" 2 --has "--names-only needs --set used or --set unused" -- reader kernels "$b" --names-only
st_case "kernels: a path that is not a directory is a usage error" 2 --has "is not a .gputrace bundle directory" -- reader kernels "$ST_WORK/none.gputrace"
st_case "kernels: --help exits 0 with the usage line" 0 --has "usage:" -- reader kernels --help

# ---- diff --------------------------------------------------------------------

a="$(bundle d-a name:alpha_kernel name:beta_kernel)"
b="$(bundle d-b name:beta_kernel name:delta_kernel)"
unused "$a" name:gamma_kernel
unused "$b" name:gamma_kernel name:omega_kernel
st_case "diff: two bundles differ in both directions" 0 --is "set:    used functions
A:      $a (2 named)
B:      $b (2 named)
shared: 1

only in A (1):
  alpha_kernel

only in B (1):
  delta_kernel" -- reader diff "$a" "$b"
st_case "diff: --set unused compares the unused archives" 0 --has "set:    unused functions" --has "shared: 1" \
    --has "only in B (1):" --has "  omega_kernel" --lacks "alpha_kernel" -- reader diff "$a" "$b" --set unused
st_case "diff: a bundle against itself has no difference" 0 --has "shared: 2" --has "only in A (0):" --has "only in B (0):" \
    --has "identical named function sets" -- reader diff "$a" "$a"
st_case "diff: a difference does not print the identical line" 0 --lacks "identical named function sets" -- reader diff "$a" "$b"

m="$(bundle d-bad name:alpha_kernel)"
printf 'XXXX\0' >"$m/device-resources-0x1"
st_case "diff: a malformed first bundle is refused" 3 --has "could not list functions for $m" --has "no diff produced" --lacks "shared:" -- reader diff "$m" "$a"
st_case "diff: a malformed second bundle is refused" 3 --has "could not list functions for $m" --lacks "shared:" -- reader diff "$a" "$m"

st_case "diff: no argument is a usage error" 2 --has "usage:" -- reader diff
st_case "diff: one bundle is a usage error" 2 --has "usage:" -- reader diff "$a"
st_case "diff: a third bundle is a usage error" 2 --has "expected exactly two bundles" -- reader diff "$a" "$b" "$a"
st_case "diff: an unknown option is a usage error" 2 --has "unknown argument: --bogus" -- reader diff "$a" "$b" --bogus
st_case "diff: a bad --set value is a usage error" 2 --has "--set must be used or unused (got 'all')" -- reader diff "$a" "$b" --set all
st_case "diff: --set without a value is a usage error" 2 --has "--set needs used|unused" -- reader diff "$a" "$b" --set
st_case "diff: --help exits 0 with the usage line" 0 --has "usage:" -- reader diff --help

# ---- summary -----------------------------------------------------------------

# traced <name> [metadata-lines]: a bundle the summary can read in full.
traced() {
    local dir="$ST_WORK/$1" meta="${2-  \"captured_frames_count\" => 7}"
    mkdir -p "$dir"
    archive "$dir/device-resources-0x1" name:alpha_kernel
    printf 'stream' >"$dir/capture"
    printf 'PLIST\n%s\n' "$meta" >"$dir/metadata"
    printf '%s' "$dir"
}

s="$(traced gemma4e4b-iso3-128tok-20261027-101500.gputrace)"
st_case "summary: a harness-named bundle prints its fields and is not profiled" 0 \
    --has "bundle:   $s" --has "  model:  gemma4e4b" --has "  codec:  iso3" --has "  prompt: 128 tokens" --has "  when:   20261027 101500" \
    --has "entries" --has "  timing: NOT PROFILED" --has "  frames: 7" --lacks "  timing: PROFILED" -- reader summary "$s"

: >"$s/capture.gpuprofiler_raw"
st_case "summary: a bundle holding a profiler file is profiled" 0 --has "  timing: PROFILED — capture.gpuprofiler_raw" --lacks "NOT PROFILED" -- reader summary "$s"

s="$(traced iso3-128tok-20261027-101500.gputrace)"
st_case "summary: a name with no model says it was not stamped" 0 --has "  model:  <not in name" --has "  codec:  iso3" -- reader summary "$s"

s="$(traced plain.gputrace)"
st_case "summary: a name off the convention is reported and the bundle is still read" 0 \
    --has "  name:   'plain.gputrace' does not follow the harness convention" --has "  frames: 7" -- reader summary "$s"

s="$(traced not-a-trace-name)"
st_case "summary: a bundle name without the suffix is reported" 0 --has "does not end in .gputrace" -- reader summary "$s"

s="$(traced counts-m-c-9tok-20261027-101500.gputrace '  "captured_frames_count" => 2
  "unusedFunctionCount" => 3
  "unusedBufferCount" => 0')"
st_case "summary: nonzero unused counts are listed and zero ones are not" 0 --has "  unused: unusedFunctionCount=3" --lacks "unusedBufferCount" -- reader summary "$s"

s="$(traced nokey-m-c-9tok-20261027-101500.gputrace '  "other" => 1')"
st_case "summary: metadata without the frame count is reported" 0 --has "<no captured_frames_count key" -- reader summary "$s"

s="$(traced badmeta-m-c-9tok-20261027-101500.gputrace)"
printf 'garbage\n' >"$s/metadata"
st_case "summary: metadata that is not a property list is refused" 3 --has "$s/metadata is not a readable property list" -- reader summary "$s"

s="$(traced nocap-m-c-9tok-20261027-101500.gputrace)"
rm -f "$s/capture"
st_case "summary: a bundle with no capture stream is refused" 3 --has "is missing: capture" -- reader summary "$s"
s="$(traced nometa-m-c-9tok-20261027-101500.gputrace)"
rm -f "$s/metadata"
st_case "summary: a bundle with no metadata is refused" 3 --has "is missing: metadata" -- reader summary "$s"
s="$(traced nores-m-c-9tok-20261027-101500.gputrace)"
rm -f "$s/device-resources-0x1"
st_case "summary: a bundle with no device-resources archive is refused" 3 --has "is missing: device-resources-0x*" -- reader summary "$s"

good="$(traced multi-m-c-9tok-20261027-101500.gputrace)"
st_case "summary: a broken bundle does not stop the next one being read" 3 --has "is missing: device-resources-0x*" --has "bundle:   $good" --has "  frames: 7" \
    -- reader summary "$s" "$good"

st_case "summary: a path that is not a directory is a usage error" 2 --has "is not a .gputrace bundle directory" -- reader summary "$ST_WORK/none.gputrace"
st_case "summary: no argument is a usage error" 2 --has "usage:" -- reader summary
st_case "summary: --help exits 0 with the usage line" 0 --has "usage:" -- reader summary --help

st_done
