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
# Library mode (`--library`): library code takes its device from its caller and
# never picks the GPU itself, so there is no count to hold, only a position.
#   gpu-value     `Device::Gpu` used as a value: an argument, a `let`, a field,
#                 a return. A comparison is not a value: after `==` or `!=`,
#                 inside `matches!(`, a match-arm or `|` pattern, or the pattern
#                 of an `if let` / `while let`.
#   device-alias  as above.
# The whole file is scanned, `#[cfg(test)]` items in a non-test file included:
# the scan does not read attributes, so it fails closed on them. A comparison
# wrapped so that its operator or its `matches!(` sits on another line reads as
# a value, which also fails closed. The default roots are rmlx-audio,
# rmlx-server and rmlx-models; rmlx-kv-quant is not one, because its GPU-only
# encoders are valid only behind the up-front refusal of GPU-only codecs under
# `--device cpu`.
#
# What it cannot see: MLX's default device is the GPU, so a call that passes
# no device, or runs on a default-device stream, reaches Metal with no
# `Device::Gpu` anywhere; a GPU device returned by a helper in a crate outside
# the scan; and nothing outside the scan roots is counted. The text readers do
# not know raw strings (scripts/lib/awk_text.sh).
#
# Usage: check_gpu_device_census.sh [<rmlx-cli-src-dir>]
#        (default: crates/rmlx-cli/src)
#        check_gpu_device_census.sh --library [<src-dir>...]
#        (default: crates/rmlx-audio/src crates/rmlx-server/src
#         crates/rmlx-models/src)
# Exit 0 = exactly one site (library mode: no value site), no alias, no dropped
# claim. 1 = a rule failed. 2 = cannot scan (a root is not a directory, or it
# holds no in-scope file).

set -uo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/awk_text.sh
. "$REPO_ROOT/scripts/lib/awk_text.sh"

# in_scope <root>: the in-scope files under <root>, relative, one per line.
in_scope() {
    (
        cd "$1" || exit 2
        find . -path ./bin -prune -o -type f -name '*.rs' \
            ! -name '*_tests.rs' ! -name 'tests.rs' -print |
            sed 's|^\./||' | sort
    )
}

# check_root <root>: refuse a root that cannot be scanned.
check_root() {
    if [ ! -d "$1" ]; then
        echo "check-gpu-device-census: unavailable: scan root '$1' is not a directory" >&2
        exit 2
    fi
    if [ -z "$(in_scope "$1")" ]; then
        echo "check-gpu-device-census: unavailable: no in-scope .rs file under '$1'" >&2
        exit 2
    fi
}

IFS= read -r -d '' AWK_ALIAS <<'EOF'
function check_alias(code) {
    if (code ~ /Device[ \t]*::[ \t]*([*]|[{])/ || code ~ /(^|[^A-Za-z0-9_])Device[ \t]+as[ \t]/ ||
        code ~ /(^|[^A-Za-z0-9_])type[ \t]+[A-Za-z0-9_]+[ \t]*(<[^=]*>)?[ \t]*=[ \t]*(::[ \t]*)?([A-Za-z0-9_]+[ \t]*::[ \t]*)*Device[ \t]*;/)
        print "device-alias: " file ":" FNR ": " $0
}
EOF

if [ "${1:-}" = "--library" ]; then
    shift
    if [ "$#" -eq 0 ]; then
        set -- "$REPO_ROOT/crates/rmlx-audio/src" "$REPO_ROOT/crates/rmlx-server/src" \
            "$REPO_ROOT/crates/rmlx-models/src"
    fi
    for root in "$@"; do
        check_root "$root"
    done

    IFS= read -r -d '' AWK_LIBRARY <<'EOF'
{
    code = blank_strings(decomment($0))
    rest = code
    seen = ""
    while (match(rest, /Device[ \t]*::[ \t]*Gpu([^A-Za-z0-9_]|$)/)) {
        start = RSTART
        len = RLENGTH
        pre = seen substr(rest, 1, start - 1)
        post = substr(rest, start + len - 1)
        path = "((::)?[A-Za-z0-9_]+[ \t]*::[ \t]*)*$"
        compared = pre ~ ("(==|!=)[ \t]*" path) ||
            pre ~ ("(^|[^|])[|][ \t]*" path) ||
            inside_matches(pre) ||
            post ~ /^[ \t]*(=>|[|]([^|]|$))/ ||
            (pre ~ ("(if|while)[ \t]+let[ \t]+" path) && post ~ /^[ \t]*=([^=]|$)/)
        if (!compared)
            print "gpu-value: " file ":" FNR ": " $0
        seen = pre substr(rest, start, len)
        rest = substr(rest, start + len)
    }
    check_alias(code)
}
function inside_matches(s,   at, depth, i, ch) {
    at = 0
    while (match(substr(s, at + 1), /matches![ \t]*\(/))
        at += RSTART + RLENGTH - 1
    if (at == 0)
        return 0
    depth = 1
    for (i = at + 1; i <= length(s); i++) {
        ch = substr(s, i, 1)
        if (ch == "(") depth++
        else if (ch == ")" && --depth == 0) return 0
    }
    return 1
}
EOF

    status=0
    for root in "$@"; do
        label="${root#"$REPO_ROOT"/}"
        hits=""
        while IFS= read -r f; do
            out="$(awk -v file="$label/$f" "$AWK_TEXT_FNS $AWK_ALIAS $AWK_LIBRARY" "$root/$f")"
            [ -n "$out" ] && hits="${hits}${out}"$'\n'
        done <<<"$(in_scope "$root")"
        values="$(grep -c '^gpu-value: ' <<<"$hits")"
        aliases="$(grep -c '^device-alias: ' <<<"$hits")"
        [ -n "$hits" ] && printf '%s' "$hits"
        echo "check-gpu-device-census --library: $label: $values value site(s), $aliases alias(es)"
        if [ "$values" -gt 0 ] || [ "$aliases" -gt 0 ]; then
            status=1
        fi
    done
    if [ "$status" -ne 0 ]; then
        echo "check-gpu-device-census --library: FAIL: library code names the GPU device as a value; take the device from the caller" >&2
    else
        echo "check-gpu-device-census --library: ok ($# root(s) scanned)"
    fi
    exit "$status"
fi

SRC="${1:-$REPO_ROOT/crates/rmlx-cli/src}"
check_root "$SRC"
files="$(in_scope "$SRC")"

IFS= read -r -d '' AWK_RULES <<'EOF'
{
    code = blank_strings(decomment($0))
    rest = code
    while (match(rest, /Device[ \t]*::[ \t]*Gpu([^A-Za-z0-9_]|$)/)) {
        print "gpu-device: " file ":" FNR ": " $0
        rest = substr(rest, RSTART + RLENGTH)
    }
    check_alias(code)
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
    out="$(awk -v file="$f" "$AWK_TEXT_FNS $AWK_ALIAS $AWK_RULES" "$SRC/$f")"
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
