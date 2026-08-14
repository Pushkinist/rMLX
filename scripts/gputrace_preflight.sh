#!/usr/bin/env bash
# Check the host-side prerequisites for Apple's GPU tools to work with a Metal
# capture of this binary.
#
# Two different things gate a session, and confusing them is why a trace can be
# written successfully and still be useless:
#
#   - capture needs the Metal capture layer (MTL_CAPTURE_ENABLED at launch) and
#     a binary built with the metal-capture feature. scripts/gpu_capture.sh owns
#     both; they are build-side, not host-side, so they are not checked here.
#   - the GPU tools attaching to the process needs the host in developer mode,
#     full Xcode selected, and the process marked debuggable — i.e. signed with
#     com.apple.security.get-task-allow. Cargo's linker-signed ad-hoc signature
#     carries no entitlements at all, so a freshly built binary fails that one
#     by default.
#
# What this does NOT promise: per-dispatch timings. Only Xcode's GUI Profile
# replay writes .gpuprofiler_raw, it cannot be driven headlessly, and the
# counters behind it are unsupported on this GPU — see docs/PROFILING.md §5,
# which points at Metal System Trace for wall-clock GPU time.
#
# Every failure names its exact fix. Run it standalone before a session, or let
# scripts/gpu_capture.sh run it — it refuses there before the multi-GB write.
#
# Usage:
#   bash scripts/gputrace_preflight.sh
#   bash scripts/gputrace_preflight.sh --binary target/release-debug/rmlx
#
# Exit: 0 = every prerequisite met (warnings may still print)
#       1 = at least one prerequisite unmet (each printed with its fix)
#       2 = usage error

set -uo pipefail

BIN="target/release-debug/rmlx"

while [ $# -gt 0 ]; do
	case "$1" in
	--binary)
		if [ $# -lt 2 ]; then
			echo "--binary needs a path" >&2
			exit 2
		fi
		BIN="$2"
		shift 2
		;;
	-h | --help)
		echo "usage: $0 [--binary <path>]"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		echo "usage: $0 [--binary <path>]" >&2
		exit 2
		;;
	esac
done

cd "$(dirname "$0")/.." || exit 1

fail=0

pass() { echo "  ok    $1"; }
warn() { echo "  warn  $1"; }
bad() {
	echo "  FAIL  $1" >&2
	shift
	for line in "$@"; do echo "        $line" >&2; done
	fail=1
}

echo "gputrace preflight (binary: $BIN)"

# --- 1. full Xcode selected -------------------------------------------------
# Command Line Tools ship no GPU tools, so a trace cannot be opened at all.
dev_dir=$(xcode-select -p 2>/dev/null)
if [ -z "$dev_dir" ]; then
	bad "xcode-select has no developer directory selected." \
		"sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
elif [ "${dev_dir##*/}" = "CommandLineTools" ]; then
	bad "xcode-select points at Command Line Tools, not Xcode ($dev_dir)." \
		"sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
else
	pass "Xcode selected: $dev_dir"
fi

# --- 2. developer mode ------------------------------------------------------
# Without it the GPU tools cannot attach at all. Note the third branch: an
# unrecognised DevToolsSecurity output is reported as unreadable rather than
# quietly treated as "disabled" or "fine".
dts=$(DevToolsSecurity -status 2>&1)
case "$dts" in
*"currently enabled"*) pass "developer mode enabled" ;;
*"currently disabled"*)
	bad "developer mode is disabled — Apple's GPU tools cannot attach at all." \
		"sudo DevToolsSecurity -enable"
	;;
*)
	bad "could not read developer-mode status; DevToolsSecurity said: ${dts:-<no output>}" \
		"Check it by hand:  DevToolsSecurity -status" \
		"Enable it with:    sudo DevToolsSecurity -enable"
	;;
esac

# --- 3. the capture binary is debuggable ------------------------------------
# `codesign -d --entitlements -` prints the entitlement plist on stdout. A
# cargo-built binary is ad-hoc *linker-signed* and has none, which is the
# default state of a fresh `cargo build` — `make build-capture` re-signs it.
SIGN_HINT_A="make build-capture     # builds and re-signs with the entitlement"
SIGN_HINT_B="or: codesign --force --sign - --entitlements scripts/rmlx-capture.entitlements $BIN"
if [ ! -x "$BIN" ]; then
	bad "capture binary not found at $BIN." "$SIGN_HINT_A"
else
	ents=$(codesign -d --entitlements - --xml "$BIN" 2>/dev/null)
	cs_rc=$?
	# The key is dotted and plutil reads dots as key-path separators, so it has
	# to be escaped — unescaped, plutil reports "no value at that key path" for
	# a binary that does carry the entitlement, i.e. a silent false negative.
	gta_key='com\.apple\.security\.get-task-allow'
	if [ $cs_rc -ne 0 ]; then
		bad "codesign could not read $BIN (exit $cs_rc) — is it a signed Mach-O?" "$SIGN_HINT_A"
	elif [ -z "$ents" ]; then
		# A cargo binary is linker-signed: signed, but with an empty entitlement set.
		bad "$BIN carries no entitlements at all (cargo's linker-signed ad-hoc signature)." \
			"Apple's GPU tools may not attach to a process that is not marked" \
			"debuggable, so a capture from it is not usable in them. Re-sign it:" \
			"$SIGN_HINT_A" \
			"$SIGN_HINT_B"
	elif ! printf '%s' "$ents" | plutil -lint - >/dev/null 2>&1; then
		# Neither "entitled" nor "not entitled": codesign printed something this
		# check cannot read. Say so instead of guessing an answer.
		bad "codesign printed an entitlement blob that is not a readable plist." \
			"Inspect it by hand:  codesign -d --entitlements - $BIN"
	else
		gta=$(printf '%s' "$ents" | plutil -extract "$gta_key" raw - 2>/dev/null)
		if [ "$gta" = "true" ]; then
			pass "get-task-allow entitlement present"
		elif [ -n "$gta" ]; then
			bad "$BIN has com.apple.security.get-task-allow set to '$gta', not true." \
				"$SIGN_HINT_A"
		else
			bad "$BIN carries no com.apple.security.get-task-allow entitlement." \
				"Apple's GPU tools may not attach to a process that is not marked" \
				"debuggable, so a capture from it is not usable in them. Re-sign it:" \
				"$SIGN_HINT_A" \
				"$SIGN_HINT_B"
		fi
	fi
fi

# --- 4. Metal toolchain (advisory) ------------------------------------------
# Anything that recompiles the captured pipelines needs it, so its absence
# turns into a confusing Xcode-side failure. Advisory, not fatal: capture
# itself does not need it.
if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
	pass "Metal toolchain present: $(xcrun -sdk macosx metal --version 2>&1 | head -1)"
else
	warn "Metal toolchain not installed — anything recompiling captured shaders will fail."
	warn "Install it with:  xcodebuild -downloadComponent MetalToolchain"
fi

if [ "$fail" -ne 0 ]; then
	echo "preflight FAILED — fix the items above before capturing." >&2
	exit 1
fi
echo "preflight ok — Apple's GPU tools can attach to this binary on this host"
exit 0
