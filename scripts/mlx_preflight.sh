#!/usr/bin/env bash
# Verify the linked MLX stack is sane before benching, and — on hardware that
# has a Neural Accelerator — that it is actually nax-capable.
#
# Two separate concerns, deliberately not conflated:
#
#   1. Portable checks (every Apple Silicon host): the `opt` symlinks resolve,
#      Mach-O install names are relocated, and the built binary can launch.
#      A broken stack here means `rmlx` does not run at all.
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

# The pinned pair, read from its one declaration. A copy here would drift the
# moment the pin moves, and the drift would be silent.
PIN_FILE="$(cd "$(dirname "$0")/.." && pwd)/crates/rmlx-mlx/mlx-pin.txt"
[ -f "$PIN_FILE" ] || {
	echo "FAIL: no MLX pin at $PIN_FILE" >&2
	exit 1
}
PIN_MLX=$(awk '$1 == "mlx" { print $2; n++ } END { exit n != 1 }' "$PIN_FILE") ||
	PIN_MLX=""
PIN_MLXC=$(awk '$1 == "mlx-c" { print $2; n++ } END { exit n != 1 }' "$PIN_FILE") ||
	PIN_MLXC=""
[ -n "$PIN_MLX" ] && [ -n "$PIN_MLXC" ] || {
	echo "FAIL: $PIN_FILE must declare exactly one 'mlx <version>' and one 'mlx-c <version>' line" >&2
	exit 1
}

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
	nax=$(strings "$metallib" | grep -c steel_gemm_fused_nax)
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

# --- 4. portable: the binary actually launches ------------------------------
bin="target/release-perf/rmlx"
if [ -x "$bin" ]; then
	if ! "$bin" --version >/dev/null 2>&1; then
		fail "$bin cannot launch against the linked MLX (dyld failure? ABI mismatch?)"
		hint_restore
		exit 1
	fi
fi

if [ "$na_class" = "1" ]; then
	echo "preflight ok: $brand (NA-class), pinned mlx $mlx_ver + mlx-c $mlxc_ver, $nax nax GEMM kernel occurrences"
else
	echo "preflight ok: $brand (no Neural Accelerator), mlx $mlx_ver + mlx-c $mlxc_ver, nax check skipped"
fi
