#!/usr/bin/env bash
# Verify the linked MLX stack is sane before benching, and — on hardware that
# has a Neural Accelerator — that it is actually nax-capable.
#
# Two separate concerns, deliberately not conflated:
#
#   1. Portable checks (every Apple Silicon host): the `opt` symlinks resolve,
#      Mach-O install names are relocated, and the built binary can launch.
#      A broken stack here means `rmlx` does not run at all. These read the
#      package manager's view, which is a pre-filter and not the truth — see
#      step 4.
#
#   2. NA-class hosts only (M5 and later): `mlx.metallib` must actually contain
#      `steel_gemm_fused_nax` GEMM kernels, and the linked pair must be the one
#      the pin names. Some homebrew-core `arm64_tahoe` bottles ship ZERO nax
#      kernels, costing ~2-3.8x on GEMM-bound prefill. Decode is largely
#      unaffected, so the failure is silent: benches still run and still look
#      plausible. See docs/FFI.md for which bottles and the measured cost.
#
# On M1-M4 both the nax check and the pinned-pair check are skipped, not failed
# — those bottles legitimately contain no nax kernels because the hardware has
# no Neural Accelerator, so the pinned pair buys nothing there.
#
# The pinned pair is read from crates/rmlx-mlx/mlx-pin.txt, never restated here.
#
# Run before any measurement. Exits non-zero and names the fix on failure.
# Background: .rmlx/mlx-homebrew-nax-regression.md

set -uo pipefail

PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"

# The pinned pair, read from its one declaration by the one parser. A copy
# here would drift the moment the pin moves, and the drift would be silent.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/mlx_pin.sh"
mlx_pin_load "$REPO_ROOT/crates/rmlx-mlx/mlx-pin.txt" || exit 1

fail() {
	echo "PREFLIGHT FAIL: $*" >&2
	return 1
}

hint_restore() {
	echo "" >&2
	echo "  Restore the nax-capable pair:  make mlx-restore-pin" >&2
}

# --- host class -------------------------------------------------------------
# The Neural Accelerator arrives with M5. Anything earlier legitimately has no
# nax kernels, so requiring them there would be a false failure.
brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
na_class=0
if [[ "$brand" =~ Apple\ M([0-9]+) ]]; then
	[ "${BASH_REMATCH[1]}" -ge 5 ] && na_class=1
fi

# --- 1. portable: symlinks resolve -----------------------------------------
for keg in mlx mlx-c; do
	link="$PREFIX/opt/$keg"
	if [ ! -e "$link" ]; then
		fail "$link does not resolve (keg unlinked or removed)"
		hint_restore
		exit 1
	fi
done

mlx_ver=$(basename "$(readlink "$PREFIX/opt/mlx")")
mlxc_ver=$(basename "$(readlink "$PREFIX/opt/mlx-c")")

# --- 2. portable: install names relocated ----------------------------------
# A hand-poured bottle keeps Homebrew's @@HOMEBREW_PREFIX@@ placeholders, which
# dyld cannot resolve.
for lib in "$PREFIX/opt/mlx/lib/libmlx.dylib" "$PREFIX/opt/mlx-c/lib/libmlxc.dylib"; do
	if [ ! -f "$lib" ]; then
		fail "$lib missing"
		hint_restore
		exit 1
	fi
	if otool -L "$lib" 2>/dev/null | grep -q '@@HOMEBREW_PREFIX@@'; then
		fail "$lib still carries @@HOMEBREW_PREFIX@@ placeholders (unrelocated pour)"
		hint_restore
		exit 1
	fi
done

# --- 3. NA-class only: nax kernels must be present --------------------------
metallib="$PREFIX/opt/mlx/lib/mlx.metallib"
nax="n/a"
if [ "$na_class" = "1" ]; then
	if [ ! -f "$metallib" ]; then
		fail "$metallib missing"
		hint_restore
		exit 1
	fi
	# `grep -c` prints 0 whether the file has no kernels or `strings` could not
	# read it at all, and the exit status that tells them apart is swallowed by
	# the pipe. Take the reader's status first: "could not look" must not be
	# reported as "ships none", which sends the operator to restore a bottle
	# over a broken toolchain.
	if ! symbols=$(strings "$metallib"); then
		fail "cannot read $metallib (is \`strings\` present?) — the nax kernel check" \
			"could not run, so this is not a finding about the bottle"
		exit 1
	fi
	nax=$(printf '%s\n' "$symbols" | grep -c steel_gemm_fused_nax)
	if [ "$nax" -lt 1 ]; then
		fail "$brand has a Neural Accelerator but mlx $mlx_ver ships 0 nax GEMM kernels" \
			"— GEMM-bound prefill would be ~2-3.8x slow (pinned: mlx $PIN_MLX + mlx-c $PIN_MLXC)"
		hint_restore
		exit 1
	fi
	# The pair is the validated unit even when the kernels are present: mlx and
	# mlx-c are ABI-coupled, and any prefill number measured across the pin
	# boundary is not comparable to one from the other side. On a host the pin
	# binds, that is a refusal to measure, not a note.
	if [ "$mlx_ver" != "$PIN_MLX" ] || [ "$mlxc_ver" != "$PIN_MLXC" ]; then
		fail "linked mlx $mlx_ver + mlx-c $mlxc_ver is not the pinned pair" \
			"(mlx $PIN_MLX + mlx-c $PIN_MLXC, crates/rmlx-mlx/mlx-pin.txt)"
		hint_restore
		exit 1
	fi
fi

# --- 4. the binary's own answer, which outranks everything above ------------
# Steps 1-3 read the package manager's symlinks. That is a cheap pre-filter,
# not the truth: `MLX_PREFIX` / `MLX_C_PREFIX` (crates/rmlx-mlx/build.rs) can
# link a build against an install nothing above inspects. Only the binary can
# say what dyld resolved for it, so when one exists it is the authority.
#
# `rmlx baseline` and `rmlx bench` refuse on their own if this is wrong; asking
# here just moves the failure before the model load.
bin="target/release-perf/rmlx"
if [ -x "$bin" ]; then
	if ! "$bin" --version >/dev/null 2>&1; then
		fail "$bin cannot launch against the linked MLX (dyld failure? ABI mismatch?)"
		hint_restore
		exit 1
	fi
	# Only the mlx_pin line: a red elsewhere in healthcheck (no registry, no
	# metrics DB) is not this gate's business.
	pin_line=$("$bin" healthcheck --human 2>/dev/null | grep '^mlx_pin:')
	if [ -z "$pin_line" ]; then
		fail "$bin reported no mlx_pin line — cannot confirm what the binary loaded"
		exit 1
	fi
	case "$pin_line" in
	*"GREEN"*) echo "binary agrees: $pin_line" ;;
	*)
		fail "the built binary did not load the pinned pair: $pin_line"
		hint_restore
		exit 1
		;;
	esac
else
	echo "note: no $bin yet — the symlink checks above are a pre-filter; the binary's" \
		"own verdict is what gates the measurement (rmlx baseline / rmlx bench)."
fi

if [ "$na_class" = "1" ]; then
	echo "preflight ok: $brand (NA-class), pinned mlx $mlx_ver + mlx-c $mlxc_ver, $nax nax GEMM kernel occurrences"
else
	echo "preflight ok: $brand (no Neural Accelerator), mlx $mlx_ver + mlx-c $mlxc_ver, nax check skipped"
fi
