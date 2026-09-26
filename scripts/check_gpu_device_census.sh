#!/usr/bin/env bash
# scripts/check_gpu_device_census.sh — the rmlx binary names the GPU device in
# exactly one place.
#
# A GPU command must hold the Metal claim from before its first GPU call until
# its last one. In the rmlx binary the GPU device comes from one helper,
# `claim_gpu` in `commands/parse.rs`, which returns it together with the claim.
# A second `Device::Gpu` anywhere in the binary's source is a GPU device that
# did not come with a claim.
#
# Rules. Code only: a `//` comment and the body of a string literal are not
# code (the readers in scripts/lib/awk_text.sh).
#   gpu-device    `Device::Gpu` (spaces around `::` allowed, any path prefix).
#                 Exactly one such site passes; zero or two or more fail.
#   device-alias  a `use` of the `Device` variants (`Device::*`,
#                 `Device::{…}`), a rename (`Device as …`) or a type alias
#                 (`type X[<…>] = [::][path::]Device;`): each lets the GPU
#                 device be named without the token this gate counts.
#   claim-dropped a statement that calls `parse_device`, `claim_gpu` or
#                 `check_claim` and takes `.device()`, `ClaimedDevice::device`
#                 or a `ClaimedDevice { .. }` pattern from the result, in a
#                 closure, a `match` arm or a destructuring `let`. `Device` is
#                 `Copy`, so the claim-carrying value is a temporary and the
#                 claim is released at the end of the statement. A statement is
#                 the code up to a `;`, across lines and braces; an `fn` line
#                 starts a new one, so a tail expression does not run into the
#                 next item. Blind spot: a statement that also names `.device(`
#                 on a value it bound earlier reads as a drop.
#
# Scope: every `.rs` file under the scan root except `*_tests.rs`, `tests.rs`
# and the `bin/` directory. The programs under `bin/` are separate binaries
# that take the claim themselves.
#
# What it cannot see: MLX's default device is the GPU, so a call that passes
# no device, or runs on a default-device stream, reaches Metal with no
# `Device::Gpu` anywhere; and nothing outside the scan root is counted. Library
# code that names the GPU itself is outside it and runs under `--device cpu`
# with no claim: the BitNet loader (crates/rmlx-models/src/bitnet/loader.rs)
# transposes its weights on the GPU whatever device the caller passed; the
# Qwen3-TTS codec and synthesis (crates/rmlx-audio/src/tts.rs) and the server's
# transcription handler (crates/rmlx-server/src/audio.rs) run on the GPU.
#
# Usage: check_gpu_device_census.sh [<rmlx-cli-src-dir>]
#        (default: crates/rmlx-cli/src)
# Exit 0 = exactly one site, no alias, no dropped claim. 1 = a rule failed. 2 = cannot scan (the
# root is not a directory, or it holds no in-scope file).

set -uo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/awk_text.sh
. "$REPO_ROOT/scripts/lib/awk_text.sh"

SRC="${1:-$REPO_ROOT/crates/rmlx-cli/src}"
if [ ! -d "$SRC" ]; then
    echo "check-gpu-device-census: unavailable: scan root '$SRC' is not a directory" >&2
    exit 2
fi

files="$(
    cd "$SRC" || exit 2
    find . -path ./bin -prune -o -type f -name '*.rs' \
        ! -name '*_tests.rs' ! -name 'tests.rs' -print |
        sed 's|^\./||' | sort
)"
if [ -z "$files" ]; then
    echo "check-gpu-device-census: unavailable: no in-scope .rs file under '$SRC'" >&2
    exit 2
fi

IFS= read -r -d '' AWK_RULES <<'EOF'
{
    code = blank_strings(decomment($0))
    rest = code
    while (match(rest, /Device[ \t]*::[ \t]*Gpu([^A-Za-z0-9_]|$)/)) {
        print "gpu-device: " file ":" FNR ": " $0
        rest = substr(rest, RSTART + RLENGTH)
    }
    if (code ~ /Device[ \t]*::[ \t]*([*]|[{])/ || code ~ /(^|[^A-Za-z0-9_])Device[ \t]+as[ \t]/ ||
        code ~ /(^|[^A-Za-z0-9_])type[ \t]+[A-Za-z0-9_]+[ \t]*(<[^=]*>)?[ \t]*=[ \t]*(::[ \t]*)?([A-Za-z0-9_]+[ \t]*::[ \t]*)*Device[ \t]*;/)
        print "device-alias: " file ":" FNR ": " $0
    if (code ~ /(^|[^A-Za-z0-9_])fn[ \t]/)
        stmt = ""
    rest = code
    while (match(rest, /;/)) {
        stmt = stmt " " substr(rest, 1, RSTART - 1)
        check_dropped(stmt)
        stmt = ""
        rest = substr(rest, RSTART + 1)
    }
    stmt = stmt " " rest
}
function check_dropped(s) {
    gsub(/fn[ \t]+(parse_device|claim_gpu|check_claim)/, "fn _", s)
    if (s ~ /(^|[^A-Za-z0-9_])(parse_device|claim_gpu|check_claim)[ \t]*[(<]/ &&
        s ~ /([.][ \t]*device[ \t]*\(|ClaimedDevice[ \t]*(::[ \t]*device([^A-Za-z0-9_]|$)|[{]))/)
        print "claim-dropped: " file ":" FNR ": " $0
}
EOF

hits=""
while IFS= read -r f; do
    out="$(awk -v file="$f" "$AWK_TEXT_FNS $AWK_RULES" "$SRC/$f")"
    [ -n "$out" ] && hits="${hits}${out}"$'\n'
done <<<"$files"

sites="$(grep -c '^gpu-device: ' <<<"$hits")"
aliases="$(grep -c '^device-alias: ' <<<"$hits")"
dropped="$(grep -c '^claim-dropped: ' <<<"$hits")"
nfiles="$(wc -l <<<"$files" | tr -d ' ')"

status=0
if [ "$aliases" -gt 0 ]; then
    grep '^device-alias: ' <<<"$hits"
    echo "check-gpu-device-census: FAIL: $aliases Device alias(es); name the GPU device only as Device::Gpu, in claim_gpu" >&2
    status=1
fi
if [ "$dropped" -gt 0 ]; then
    grep '^claim-dropped: ' <<<"$hits"
    echo "check-gpu-device-census: FAIL: $dropped statement(s) take .device() from a fresh claim, which is released at the semicolon; bind the ClaimedDevice and pass it by reference" >&2
    status=1
fi
if [ "$sites" -eq 0 ]; then
    echo "check-gpu-device-census: FAIL: no Device::Gpu site in $nfiles file(s); claim_gpu must be the one place that names the GPU device" >&2
    status=1
elif [ "$sites" -gt 1 ]; then
    grep '^gpu-device: ' <<<"$hits"
    echo "check-gpu-device-census: FAIL: $sites Device::Gpu sites, want exactly one; take the device from parse_device or claim_gpu, which hold the Metal claim" >&2
    status=1
fi
if [ "$status" -eq 0 ]; then
    grep '^gpu-device: ' <<<"$hits"
    echo "check-gpu-device-census: ok (1 site, $nfiles file(s) scanned)"
fi
exit "$status"
