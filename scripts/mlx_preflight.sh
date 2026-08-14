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
#      `steel_gemm_fused_nax` GEMM kernels. The homebrew-core `arm64_tahoe`
#      bottle of mlx 0.32.0 ships ZERO of them where 0.31.2 ships 145, costing
#      ~2-3.8x on GEMM-bound prefill. Decode is largely unaffected, so the
#      failure is silent: benches still run and still look plausible.
#
# On M1-M4 the nax check is skipped, not failed — those bottles legitimately
# contain no nax kernels because the hardware has no Neural Accelerator. Nothing
# here pins a version on those hosts.
#
# Run before any measurement. Exits non-zero and names the fix on failure.
# Background: .rmlx/mlx-homebrew-nax-regression.md

set -uo pipefail

PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
# Known-good nax-capable pair for NA-class hosts (see the report).
GOOD_MLX="0.31.2"
GOOD_MLXC="0.6.0_2"

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
			"— GEMM-bound prefill would be ~2-3.8x slow (known good: mlx $GOOD_MLX + mlx-c $GOOD_MLXC)"
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
	echo "preflight ok: $brand (NA-class), mlx $mlx_ver + mlx-c $mlxc_ver, $nax nax GEMM kernel occurrences"
	if [ "$mlx_ver" != "$GOOD_MLX" ] || [ "$mlxc_ver" != "$GOOD_MLXC" ]; then
		echo "  note: not the recorded known-good pair (mlx $GOOD_MLX + mlx-c $GOOD_MLXC);" \
			"nax kernels are present, so this is informational only."
	fi
else
	echo "preflight ok: $brand (no Neural Accelerator), mlx $mlx_ver + mlx-c $mlxc_ver, nax check skipped"
fi
