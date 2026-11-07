#!/usr/bin/env bash
# Recall test for the release tool's six modes: a scratch repo, stub external tools, no network.
#
# Recall test for: release/release.sh package sign sha bottle tap-sync notes default
# Not a mode of: release/release_cpu_selftest.sh — that one tests the target-CPU check alone; this one runs the steps around it
# Runs in make ci, not in hosted CI: it uses BSD sed -i '' and date and tar flags.
#
# The tool is copied into a scratch repo under ST_WORK, beside a planted Cargo.toml, CHANGELOG.md,
# formula, licenses and tap directory, and run there on a PATH of its own whose cargo, curl, cosign,
# brew, gh and git record their argv and return a planted result. Every case asserts the literal
# exit code, what was printed, and the files written or the argv the stubs received. The modes are
# named in one place each: the `tool_*` functions.
#
# Not reached: the build itself (the cargo stub writes a binary and cargo's fingerprint record), the
# download of the source tarball (the curl stub serves a planted body), the cosign signature and its
# verdict, the real brew bottle, and the clone and push of the tap.
#
# Usage:  bash scripts/release/release_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init release-selftest

REPO="$ST_WORK/repo"
REL="$REPO/scripts/release"
STUBS="$ST_WORK/stubs"
S="$STUBS/state"
TAP="$ST_WORK/tap"
NOJQ="$ST_WORK/nojq"
SRC_SHA=690fbeb8c101c0c02b4ef985b639c7475fb34a836572ab83a2339812d77482e9
FORMULA=packaging/homebrew/rmlx.rb
TARBALL_REL=dist/rmlx-v0.4.5-aarch64-apple-darwin.tar.gz
NOREPLY=4850452+Pushkinist@users.noreply.github.com
WHY=""

export HOME="$ST_WORK/home"
mkdir -p "$HOME/.config/rmlx" "$STUBS" "$NOJQ"
export PATH="$STUBS:/usr/bin:/bin"

# ---- the six modes, one function each ---------------------------------------------------------

place_tools() {
    local f
    for f in release.sh release_cpu.py; do
        cp "$REPO_ROOT/scripts/release/$f" "$REL/" || st_die "cannot copy $f"
    done
    mkdir -p "$REPO/scripts/lib"
    cp "$REPO_ROOT/scripts/lib/args.sh" "$REPO_ROOT/scripts/lib/identity.sh" "$REPO/scripts/lib/" || st_die "cannot copy the lib helpers"
}

tool_package() { st_bounded 120 bash "$REL/release.sh" package "$@"; }
tool_sign() { st_bounded 60 bash "$REL/release.sh" sign "$@"; }
tool_sha() { st_bounded 60 bash "$REL/release.sh" sha "$@"; }
tool_bottle() { st_bounded 60 bash "$REL/release.sh" bottle "$@"; }
tool_tap_sync() { st_bounded 60 bash "$REL/release.sh" tap-sync "$@"; }
tool_notes() { st_bounded 60 bash "$REL/release.sh" notes "$@"; }

# tool_default: the tool with the arguments given, the first of which is not one of the six modes.
tool_default() { st_bounded 60 bash "$REL/release.sh" "$@"; }

# tool_bottle_no_jq: the bottle mode on a PATH that holds brew and dirname but no jq.
tool_bottle_no_jq() { PATH="$NOJQ" /bin/bash "$REL/release.sh" bottle; }

# tool_without <tool> <cmd...>: run the command with that stub taken off the PATH.
tool_without() {
    local t="$1" rc=0
    shift
    mv "$STUBS/$t" "$ST_WORK/off_$t" || st_die "no stub $t"
    "$@" || rc=$?
    mv "$ST_WORK/off_$t" "$STUBS/$t"
    return "$rc"
}

# ---- stubs -------------------------------------------------------------------------------------

stub() {
    cat >"$STUBS/$1"
    chmod +x "$STUBS/$1"
}

RECORD='s=$(dirname "$0")/state
{ printf "[%s]" "$@"; printf "\n"; } >>"$s/NAME.argv"'

stub_with_record() { { printf '#!/bin/sh\n%s\n' "${RECORD//NAME/$1}"; cat; } | stub "$1"; }

stub_with_record cargo <<'EOF2'
printf '[%s][%s]\n' "${RUSTC_WRAPPER-unset}" "${RUSTC_WORKSPACE_WRAPPER-unset}" >>"$s/cargo.wrappers"
printf '%s' "$CARGO_ENCODED_RUSTFLAGS" | tr '\037' ' ' >"$s/cargo.flags"
mode=ok
[ -f "$s/cargo.mode" ] && mode=$(cat "$s/cargo.mode")
[ "$mode" = nobin ] && exit 0
unit=target/release/.fingerprint/rmlx-cli-0123456789abcdef
mkdir -p target/release/deps "$unit"
if [ "$mode" = home ]; then
    printf 'panic at %s/src/main.rs\n' "$HOME" >target/release/rmlx
else
    printf 'stub release binary\n' >target/release/rmlx
fi
chmod +x target/release/rmlx
cp target/release/rmlx target/release/deps/rmlx-0123456789abcdef
python3 - "$mode" "$unit/bin-rmlx.json" <<'PY'
import json, os, sys
if sys.argv[1] == "native":
    flags = ["-C", "target-cpu=native", "-C", "link-arg=-dead_strip"]
else:
    flags = os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
json.dump({"rustflags": flags}, open(sys.argv[2], "w"))
PY
EOF2

stub_with_record curl <<'EOF2'
rc=0
[ -f "$s/curl.rc" ] && rc=$(cat "$s/curl.rc")
[ "$rc" -eq 0 ] && cat "$s/curl.body"
exit "$rc"
EOF2

stub_with_record cosign <<'EOF2'
case "$1" in
sign-blob)
    while [ $# -gt 0 ]; do
        [ "$1" = --bundle ] && b=$2
        shift
    done
    printf 'bundle\n' >"$b"
    ;;
verify-blob)
    exit "$(cat "$s/cosign.verify_rc" 2>/dev/null || echo 0)"
    ;;
esac
EOF2

stub_with_record brew <<'EOF2'
case "$1" in
list) exit "$(cat "$s/brew.list_rc" 2>/dev/null || echo 0)" ;;
bottle)
    for f in "$s"/brew.files/*; do
        [ -e "$f" ] && cp "$f" .
    done
    ;;
esac
exit 0
EOF2

stub_with_record gh <<'EOF2'
[ "$1 $2" = "repo clone" ] && mkdir -p "$4"
exit 0
EOF2

stub_with_record git <<'EOF2'
pwd >>"$s/git.cwd"
[ "$1" = diff ] && exit "$(cat "$s/git.diff_rc" 2>/dev/null || echo 1)"
exit 0
EOF2

stub uname <<'EOF2'
s=$(dirname "$0")/state
if [ "$1" = -m ] && [ -f "$s/uname.m" ]; then cat "$s/uname.m"; else exec /usr/bin/uname "$@"; fi
EOF2

printf '#!/bin/sh\nexec /usr/bin/jq "$@"\n' | stub jq
printf '#!/bin/sh\nd=$(cd "$(dirname "$0")/.." && pwd)/clone\nmkdir -p "$d"\nprintf "%%s\\n" "$d"\n' | stub mktemp
ln -sf "$STUBS/brew" "$NOJQ/brew"
ln -sf /usr/bin/dirname "$NOJQ/dirname"

# ---- planting ----------------------------------------------------------------------------------

# plant_formula <path> <url-version> <sha>
plant_formula() {
    cat >"$1" <<EOF2
class Rmlx < Formula
  desc "planted"
  # Bump: url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.1.0.tar.gz"
  url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v$2.tar.gz"
  sha256 "$3"
  license "MIT"

  resource "other" do
    url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.3.0.tar.gz"
    sha256 "1111111111111111111111111111111111111111111111111111111111111111"
  end

  bottle do
    sha256 cellar: :any, arm64_tahoe: "2222222222222222222222222222222222222222222222222222222222222222"
  end
end
EOF2
}

# fresh [<version>]: a scratch repo holding the tool and everything it reads; stub state cleared.
fresh() {
    local ver="${1:-0.4.5}"
    rm -rf "$REPO" "$S" "$TAP"
    mkdir -p "$REL" "$REPO/.cargo" "$REPO/packaging/homebrew" "$S" "$TAP"
    place_tools
    cat >"$REPO/Cargo.toml" <<EOF2
[workspace]
members = ["crates/a"]

[workspace.package]
edition = "2021"
rust-version = "1.98.1"
version = "$ver"

[workspace.dependencies]
dep = { version = "1.0.0" }
EOF2
    cat >"$REPO/CHANGELOG.md" <<'EOF2'
# Changelog

## [Unreleased]

- nothing yet

## [0.4.5] - 2026-11-01

### Fixed

- fix five

- fix six

## [0.4.4] - 2026-10-01

### Added

- add four

## [0.4.3] - 2026-09-01
- last one
EOF2
    printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "link-arg=-dead_strip"]\n' >"$REPO/.cargo/config.toml"
    printf 'mit\n' >"$REPO/LICENSE-MIT"
    printf 'apache\n' >"$REPO/LICENSE-APACHE"
    printf 'readme\n' >"$REPO/README.md"
    printf 'public key\n' >"$REPO/cosign.pub"
    plant_formula "$REPO/$FORMULA" 0.4.4 0000000000000000000000000000000000000000000000000000000000000000
    printf 'arm64\n' >"$S/uname.m"
    printf 'planted source tarball\n' >"$S/curl.body"
    printf 'private key\n' >"$HOME/.config/rmlx/cosign.key"
    printf 'other key\n' >"$ST_WORK/other.key"
}

plant_tarball() {
    mkdir -p "$REPO/dist"
    printf 'tarball\n' >"$REPO/$1"
}

# plant_bottle <cellar>: what `brew bottle` leaves in the working directory.
plant_bottle() {
    mkdir -p "$S/brew.files"
    printf 'bottle bytes\n' >"$S/brew.files/rmlx--0.4.5.arm64_tahoe.bottle.tar.gz"
    cat >"$S/brew.files/rmlx--0.4.5.arm64_tahoe.bottle.json" <<EOF2
{"pushkinist/rmlx/rmlx":{"bottle":{"cellar":"$1","tags":{"arm64_tahoe":{"filename":"rmlx-0.4.5.arm64_tahoe.bottle.tar.gz","local_filename":"rmlx--0.4.5.arm64_tahoe.bottle.tar.gz","sha256":"abcd1234"}}}}}
EOF2
}

# ---- judging -----------------------------------------------------------------------------------

expect_eq() {
    if [ "$2" != "$3" ]; then WHY="${WHY:+$WHY$'\n'}$1: got '$2', want '$3'"; fi
}

# expect_argv <stub> <line>...: the stub's argv record, one bracketed line per call.
expect_argv() {
    local tool="$1" got want
    shift
    got=$(cat "$S/$tool.argv" 2>/dev/null || true)
    want=$(printf '%s\n' "$@")
    [ "$#" -gt 0 ] || want=""
    expect_eq "$tool argv" "$got" "$want"
}

expect_absent() {
    if [ -e "$1" ]; then WHY="${WHY:+$WHY$'\n'}$1 exists, want it absent"; fi
}

expect_present() {
    if [ ! -e "$1" ]; then WHY="${WHY:+$WHY$'\n'}$1 missing"; fi
}

# finish <name> <want-exit> [st_check options]: judge the last st_run and everything expected since.
finish() {
    local name="$1" want="$2" why
    shift 2
    why="$(st_check "$want" "$@")"
    if [ -n "$WHY" ]; then why="${why:+$why$'\n'}$WHY"; fi
    WHY=""
    st_verdict "$name" "$why"
}

cd "$ST_WORK"

# ---- notes -------------------------------------------------------------------------------------

fresh
st_run tool_notes 0.4.5
finish notes_explicit_version 0 --is $'\n### Fixed\n\n- fix five\n\n- fix six'

fresh
st_run tool_notes
finish notes_default_is_cargo_version 0 --is $'\n### Fixed\n\n- fix five\n\n- fix six'

fresh 0.4.4
st_run tool_notes
finish notes_default_follows_cargo_toml 0 --is $'\n### Added\n\n- add four'

fresh
st_run tool_notes 0.4.3
finish notes_last_section 0 --is '- last one'

fresh
st_run tool_notes 9.9.9
finish notes_missing_version_prints_nothing 1 --is 'error: CHANGELOG.md has no section for version 9.9.9'

fresh
{ printf '## [0x4y5] - 2026-11-02\n\n- decoy\n\n'; cat "$REPO/CHANGELOG.md"; } >"$ST_WORK/changelog"
mv "$ST_WORK/changelog" "$REPO/CHANGELOG.md"
st_run tool_notes 0.4.5
finish notes_version_is_matched_literally 0 --is $'\n### Fixed\n\n- fix five\n\n- fix six'

fresh
rm "$REPO/CHANGELOG.md"
st_run tool_notes 0.4.5
finish notes_no_changelog 2 --has "CHANGELOG.md"

fresh
printf '[workspace.package]\nname = "x"\n' >"$REPO/Cargo.toml"
st_run tool_notes
finish notes_no_version_anywhere 1 --has "error: no version"

# ---- sha ---------------------------------------------------------------------------------------

fresh
st_run tool_sha
expect_argv curl "[-fsSL][https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.4.5.tar.gz]"
plant_formula "$ST_WORK/formula_before" 0.4.4 0000000000000000000000000000000000000000000000000000000000000000
expect_eq "formula" "$(cat "$REPO/$FORMULA")" "$(cat "$ST_WORK/formula_before")"
finish sha_prints_the_digest 0 --has "sha256: $SRC_SHA" --has "==> fetching https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.4.5.tar.gz" --lacks "patched"

fresh 0.9.1
st_run tool_sha
expect_argv curl "[-fsSL][https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.9.1.tar.gz]"
finish sha_url_follows_cargo_toml 0 --has "fetching https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.9.1.tar.gz"

fresh
st_run tool_sha --write
plant_formula "$ST_WORK/formula_want" 0.4.5 "$SRC_SHA"
expect_eq "formula" "$(cat "$REPO/$FORMULA")" "$(cat "$ST_WORK/formula_want")"
finish sha_write_patches_url_and_sha_only 0 --has "sha256: $SRC_SHA" --has "==> patched $FORMULA (url -> v0.4.5, sha256)"

fresh
: >"$S/curl.body"
st_run tool_sha --write
plant_formula "$ST_WORK/formula_before" 0.4.4 0000000000000000000000000000000000000000000000000000000000000000
expect_eq "formula" "$(cat "$REPO/$FORMULA")" "$(cat "$ST_WORK/formula_before")"
finish sha_empty_download_is_refused 1 --has "error: the download was empty" --lacks "sha256:" --lacks "patched"

fresh
printf '22\n' >"$S/curl.rc"
st_run tool_sha --write
plant_formula "$ST_WORK/formula_before" 0.4.4 0000000000000000000000000000000000000000000000000000000000000000
expect_eq "formula" "$(cat "$REPO/$FORMULA")" "$(cat "$ST_WORK/formula_before")"
finish sha_download_fails 22 --lacks "sha256:" --lacks "patched"

fresh
rm "$REPO/$FORMULA"
st_run tool_sha --write
finish sha_write_without_formula 1 --has "sha256: $SRC_SHA" --has "$FORMULA"

fresh
printf 'name = "x"\n' >"$REPO/Cargo.toml"
st_run tool_sha
expect_argv curl
finish sha_no_version 1 --has "error: could not read version from Cargo.toml"

# ---- sign --------------------------------------------------------------------------------------

BUNDLE_REL="$TARBALL_REL.cosign.bundle"

fresh
plant_tarball "$TARBALL_REL"
st_run tool_sign
expect_argv cosign "[sign-blob][--yes][--key][$HOME/.config/rmlx/cosign.key][--bundle][$BUNDLE_REL][$TARBALL_REL]" \
    "[verify-blob][--key][cosign.pub][--bundle][$BUNDLE_REL][$TARBALL_REL]"
expect_present "$REPO/$BUNDLE_REL"
finish sign_default_key 0 --has "==> wrote $BUNDLE_REL (verified against cosign.pub)" --has "gh release upload v0.4.5 $BUNDLE_REL"

fresh
plant_tarball "$TARBALL_REL"
st_run tool_sign "$ST_WORK/other.key"
expect_argv cosign "[sign-blob][--yes][--key][$ST_WORK/other.key][--bundle][$BUNDLE_REL][$TARBALL_REL]" \
    "[verify-blob][--key][cosign.pub][--bundle][$BUNDLE_REL][$TARBALL_REL]"
finish sign_explicit_key 0 --has "==> signing $TARBALL_REL"

fresh 0.9.1
plant_tarball dist/rmlx-v0.9.1-aarch64-apple-darwin.tar.gz
st_run tool_sign
expect_present "$REPO/dist/rmlx-v0.9.1-aarch64-apple-darwin.tar.gz.cosign.bundle"
finish sign_tarball_follows_cargo_toml 0 --has "gh release upload v0.9.1 dist/rmlx-v0.9.1-aarch64-apple-darwin.tar.gz.cosign.bundle"

fresh
plant_tarball "$TARBALL_REL"
printf '1\n' >"$S/cosign.verify_rc"
st_run tool_sign
expect_absent "$REPO/$BUNDLE_REL"
finish sign_bundle_does_not_verify 1 --has "error: the bundle does not verify against cosign.pub" --lacks "==> wrote"

fresh
plant_tarball "$TARBALL_REL"
st_run tool_without cosign tool_sign
finish sign_no_cosign 1 --has "error: cosign not found"

fresh
plant_tarball "$TARBALL_REL"
rm "$HOME/.config/rmlx/cosign.key"
st_run tool_sign
expect_argv cosign
finish sign_no_default_key 1 --has "error: private key $HOME/.config/rmlx/cosign.key not found"

fresh
plant_tarball "$TARBALL_REL"
st_run tool_sign "$ST_WORK/missing.key"
finish sign_no_explicit_key 1 --has "error: private key $ST_WORK/missing.key not found"

fresh
plant_tarball "$TARBALL_REL"
rm "$REPO/cosign.pub"
st_run tool_sign
finish sign_no_public_key 1 --has "error: cosign.pub not found at the repository root"

fresh
printf 'name = "x"\n' >"$REPO/Cargo.toml"
st_run tool_sign
finish sign_no_version 1 --has "error: could not read version from Cargo.toml"

fresh
st_run tool_sign
expect_argv cosign
finish sign_no_tarball 1 --has "error: $TARBALL_REL not found"

# ---- package -----------------------------------------------------------------------------------

PKG=rmlx-v0.4.5-aarch64-apple-darwin

fresh
st_run tool_package
expect_argv cargo "[build][--release][-p][rmlx-cli]"
expect_eq "cargo flags" "$(cat "$S/cargo.flags")" "-Ctarget-cpu=apple-m1 -Clink-arg=-dead_strip --remap-path-prefix=$HOME=~"
expect_eq "cargo wrappers" "$(cat "$S/cargo.wrappers")" "[][]"
expect_eq "tar members" "$(tar -tzf "$REPO/$TARBALL_REL" | sort)" \
    "$(printf '%s\n' "$PKG/" "$PKG/LICENSE-APACHE" "$PKG/LICENSE-MIT" "$PKG/README.md" "$PKG/rmlx")"
expect_eq "sha256 check" "$(cd "$REPO/dist" && shasum -a 256 -c "$PKG.tar.gz.sha256" 2>&1)" "$PKG.tar.gz: OK"
expect_eq "sha256 lines" "$(grep -Ec "^[0-9a-f]{64}  $PKG.tar.gz\$" "$REPO/$TARBALL_REL.sha256")" 1
finish package_tarball_and_checksum 0 --has "==> done" --has "    $TARBALL_REL" --has "release-cpu: ok $PKG.tar.gz built with target-cpu=apple-m1"

fresh 0.9.1
st_run tool_package
expect_present "$REPO/dist/rmlx-v0.9.1-aarch64-apple-darwin.tar.gz"
expect_present "$REPO/dist/rmlx-v0.9.1-aarch64-apple-darwin.tar.gz.sha256"
expect_absent "$REPO/$TARBALL_REL"
finish package_name_follows_cargo_toml 0 --has "==> building rmlx v0.9.1"

fresh
printf 'x86_64\n' >"$S/uname.m"
st_run tool_package
expect_argv cargo
finish package_not_apple_silicon 1 --has "error: must build on Apple Silicon (arm64)"

fresh
printf 'name = "x"\n' >"$REPO/Cargo.toml"
st_run tool_package
expect_argv cargo
finish package_no_version 1 --has "error: could not read version from Cargo.toml"

fresh
printf '[build]\nrustflags = ["-C", "target-cpu=native", "-C", "target-feature=+neon"]\n' >"$REPO/.cargo/config.toml"
st_run tool_package
expect_argv cargo
finish package_config_sets_a_target_feature 1 --has "release-cpu: FAIL .cargo/config.toml sets"

fresh
printf 'nobin\n' >"$S/cargo.mode"
st_run tool_package
expect_absent "$REPO/dist"
finish package_no_binary 1 --has "error: ./target/release/rmlx not found"

fresh
printf 'home\n' >"$S/cargo.mode"
st_run tool_package
expect_absent "$REPO/dist"
finish package_binary_carries_home_directory 1 --has "contains the build machine's home directory; nothing is packaged"

fresh
printf 'native\n' >"$S/cargo.mode"
st_run tool_package
expect_eq "dist" "$(ls "$REPO/dist")" ""
finish package_built_for_another_cpu 1 --has "release-cpu: FAIL" --has "tarball and staging removed"

# ---- bottle ------------------------------------------------------------------------------------

ROOT_URL=https://github.com/Pushkinist/rMLX/releases/download/v0.4.5
BOTTLE=rmlx-0.4.5.arm64_tahoe.bottle.tar.gz
BOTTLE_JSON=rmlx--0.4.5.arm64_tahoe.bottle.json

fresh
plant_bottle any_skip_relocation
st_run tool_bottle
expect_argv brew "[list][rmlx]" "[bottle][--json][--root-url=$ROOT_URL][--no-rebuild][--force-core-tap][rmlx]"
expect_present "$REPO/dist/$BOTTLE"
expect_present "$REPO/dist/$BOTTLE_JSON"
expect_absent "$REPO/rmlx--0.4.5.arm64_tahoe.bottle.tar.gz"
expect_absent "$REPO/$BOTTLE_JSON"
finish bottle_files_and_block 0 --has "==> bottle:  dist/$BOTTLE" --has "==> json:    dist/$BOTTLE_JSON" \
    --has "    root_url \"$ROOT_URL\"" --has "    sha256 cellar: :any_skip_relocation, arm64_tahoe: \"abcd1234\"" \
    --has "  gh release upload v0.4.5 dist/$BOTTLE"

fresh
plant_bottle /opt/homebrew/Cellar
st_run tool_bottle
finish bottle_absolute_cellar_is_quoted 0 --has "sha256 cellar: \"/opt/homebrew/Cellar\", arm64_tahoe:"

fresh
plant_bottle any
st_run tool_bottle --root-url https://example.invalid/dl
expect_argv brew "[list][rmlx]" "[bottle][--json][--root-url=https://example.invalid/dl][--no-rebuild][--force-core-tap][rmlx]"
finish bottle_root_url_override 0 --has "    root_url \"https://example.invalid/dl\""

fresh 0.9.1
plant_bottle any
st_run tool_bottle
finish bottle_root_url_follows_cargo_toml 0 --has "root_url: https://github.com/Pushkinist/rMLX/releases/download/v0.9.1" \
    --has "==> building bottle for rmlx v0.9.1"

fresh
plant_bottle any
printf '{"pushkinist/rmlx/rmlx":{"bottle":{"cellar":"any","tags":{"arm64_tahoe":{"sha256":"abcd1234"}}}}}\n' >"$S/brew.files/$BOTTLE_JSON"
st_run tool_bottle
expect_absent "$REPO/dist/$BOTTLE_JSON"
expect_present "$REPO/$BOTTLE_JSON"
finish bottle_json_without_filename 1 --has "error: $BOTTLE_JSON names no filename for the bottle" --lacks "bottle do"

fresh
plant_bottle any
st_run tool_bottle --root-url
expect_argv brew
finish bottle_root_url_without_value 2 --has "ERROR: --root-url requires a value"

fresh
plant_bottle any
st_run tool_without brew tool_bottle
finish bottle_no_brew 1 --has "error: brew not found"

fresh
plant_bottle any
st_run tool_bottle_no_jq
finish bottle_no_jq 1 --has "error: jq not found"

fresh
plant_bottle any
printf 'x86_64\n' >"$S/uname.m"
st_run tool_bottle
expect_argv brew
finish bottle_not_apple_silicon 1 --has "error: must run on Apple Silicon (arm64)"

fresh
plant_bottle any
printf 'name = "x"\n' >"$REPO/Cargo.toml"
st_run tool_bottle
finish bottle_no_version 1 --has "error: could not read version from Cargo.toml"

fresh
plant_bottle any
printf '1\n' >"$S/brew.list_rc"
st_run tool_bottle
expect_argv brew "[list][rmlx]"
finish bottle_keg_not_installed 1 --has "error: rmlx is not installed in Homebrew."

fresh
plant_bottle any
rm "$S/brew.files/rmlx--0.4.5.arm64_tahoe.bottle.tar.gz"
st_run tool_bottle
finish bottle_no_tarball_produced 1 --has "error: brew bottle did not produce a .bottle.tar.gz file"

fresh
plant_bottle any
rm "$S/brew.files/$BOTTLE_JSON"
st_run tool_bottle
finish bottle_no_json_produced 1 --has "error: brew bottle did not produce a .bottle.json file"

# ---- tap-sync ----------------------------------------------------------------------------------

COMMIT="[-c][user.name=Pushkinist][-c][user.email=$NOREPLY][commit][-q][-m][rmlx 0.4.5]"

fresh
st_run tool_tap_sync "$TAP"
expect_argv git "[add][Formula/rmlx.rb]" "[diff][--cached][--quiet]" "$COMMIT" "[push][-q]"
expect_eq "git cwd" "$(sort -u "$S/git.cwd")" "$TAP"
expect_eq "tap formula" "$(cat "$TAP/Formula/rmlx.rb")" "$(cat "$REPO/$FORMULA")"
expect_argv gh
finish tap_sync_commits_and_pushes 0 --has "==> pushed Formula/rmlx.rb (rmlx 0.4.5) to Pushkinist/homebrew-rmlx"

fresh 0.9.1
st_run tool_tap_sync "$TAP"
expect_argv git "[add][Formula/rmlx.rb]" "[diff][--cached][--quiet]" \
    "[-c][user.name=Pushkinist][-c][user.email=$NOREPLY][commit][-q][-m][rmlx 0.9.1]" "[push][-q]"
finish tap_sync_commit_follows_cargo_toml 0 --has "(rmlx 0.9.1)"

fresh
printf '0\n' >"$S/git.diff_rc"
st_run tool_tap_sync "$TAP"
expect_argv git "[add][Formula/rmlx.rb]" "[diff][--cached][--quiet]"
expect_present "$TAP/Formula/rmlx.rb"
finish tap_sync_already_up_to_date 0 --has "==> tap already up to date" --lacks "pushed"

fresh
rm -rf "$ST_WORK/clone"
st_run tool_tap_sync
clone_dir="$ST_WORK/clone"
expect_argv gh "[repo][clone][Pushkinist/homebrew-rmlx][$clone_dir]"
expect_present "$clone_dir/Formula/rmlx.rb"
expect_eq "git cwd" "$(sort -u "$S/git.cwd")" "$clone_dir"
finish tap_sync_clones_when_no_dir_given 0 --has "==> cloning Pushkinist/homebrew-rmlx -> $clone_dir" --has "==> pushed Formula/rmlx.rb"

fresh
rm "$REPO/$FORMULA"
st_run tool_tap_sync "$TAP"
expect_argv git
expect_argv gh
finish tap_sync_no_formula 1 --has "error: $FORMULA missing"

# ---- usage -------------------------------------------------------------------------------------

USAGE="usage: release.sh {package|sign [KEY]|sha [--write]|bottle [--root-url URL]|tap-sync [DIR]|notes [VER]}"

fresh
st_run tool_default
finish usage_no_mode 2 --has "$USAGE"

fresh
st_run tool_default frobnicate
finish usage_unknown_mode 2 --has "$USAGE"

fresh
st_run tool_package extra
expect_argv cargo
finish usage_package_takes_no_argument 2 --has "$USAGE"

fresh
st_run tool_sign one.key two.key
expect_argv cosign
finish usage_sign_takes_one_argument 2 --has "$USAGE"

fresh
st_run tool_sha --read
expect_argv curl
finish usage_sha_unknown_option 2 --has "$USAGE"

fresh
st_run tool_sha --write extra
expect_argv curl
finish usage_sha_extra_argument 2 --has "$USAGE"

fresh
st_run tool_bottle --root
expect_argv brew
finish usage_bottle_unknown_option 2 --has "$USAGE"

fresh
st_run tool_bottle --root-url https://example.invalid/dl extra
expect_argv brew
finish usage_bottle_extra_argument 2 --has "$USAGE"

fresh
st_run tool_tap_sync "$TAP" extra
expect_argv git
finish usage_tap_sync_takes_one_argument 2 --has "$USAGE"

fresh
st_run tool_notes 0.4.5 0.4.4
finish usage_notes_takes_one_argument 2 --has "$USAGE"

st_done
