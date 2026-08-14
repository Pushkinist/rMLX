#!/usr/bin/env bash
# Verify the linked MLX stack is the pinned, nax-capable pair before benching.
#
# The homebrew-core `arm64_tahoe` bottle of mlx 0.32.0 ships ZERO
# `steel_gemm_fused_nax` GEMM kernels where 0.31.2 ships 145 distinct
# (288 `grep -c` occurrences), costing ~3.8x on GEMM-bound prefill. Decode is
# largely unaffected, so the failure is silent: benches still run and still look
# plausible, they are just measuring a different engine. A `brew cleanup` also
# removes the pinned kegs outright (`brew pin` does not protect against it),
# leaving `rmlx` unable to launch at all.
#
# Run this before any measurement. Exits non-zero and says what to do on failure.
# See docs in .rmlx/mlx-homebrew-nax-regression.md.

set -uo pipefail

PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
WANT_MLX="0.31.2"
WANT_MLXC="0.6.0_2"
# `grep -c` occurrences of the nax GEMM symbol in a good arm64_tahoe metallib.
MIN_NAX=1

fail() {
	echo "PREFLIGHT FAIL: $*" >&2
	echo "" >&2
	echo "  Restore the pinned pair:  bash scripts/mlx_restore_pin.sh" >&2
	exit 1
}

# 1. The opt symlinks must resolve — this is what rmlx links against.
for keg in mlx mlx-c; do
	link="$PREFIX/opt/$keg"
	[ -e "$link" ] || fail "$link does not resolve (keg unlinked or removed)"
done

mlx_target=$(readlink "$PREFIX/opt/mlx")
mlxc_target=$(readlink "$PREFIX/opt/mlx-c")
mlx_ver=$(basename "$mlx_target")
mlxc_ver=$(basename "$mlxc_target")

# 2. Versions must be the ABI-coupled pair. mlx-c 0.6.0_3 needs symbols mlx
#    0.31.2 does not export, so a mismatched pair aborts at dlopen time.
[ "$mlx_ver" = "$WANT_MLX" ] ||
	fail "mlx is $mlx_ver, want $WANT_MLX (0.32.0 ships no nax kernels)"
[ "$mlxc_ver" = "$WANT_MLXC" ] ||
	fail "mlx-c is $mlxc_ver, want $WANT_MLXC (ABI-coupled to mlx $WANT_MLX)"

# 3. The metallib must actually contain nax GEMM kernels. A correct version
#    number is not proof: the bottle is what ships the kernels.
metallib="$PREFIX/opt/mlx/lib/mlx.metallib"
[ -f "$metallib" ] || fail "$metallib missing"
nax=$(strings "$metallib" | grep -c steel_gemm_fused_nax)
[ "$nax" -ge "$MIN_NAX" ] ||
	fail "mlx.metallib contains $nax nax GEMM kernels (want >= $MIN_NAX) — prefill would be ~3.8x slow"

# 4. Install names must be relocated. A hand-poured bottle keeps Homebrew's
#    @@HOMEBREW_PREFIX@@ placeholders, which dyld cannot resolve.
for lib in "$PREFIX/opt/mlx/lib/libmlx.dylib" "$PREFIX/opt/mlx-c/lib/libmlxc.dylib"; do
	[ -f "$lib" ] || fail "$lib missing"
	if otool -L "$lib" 2>/dev/null | grep -q '@@HOMEBREW_PREFIX@@'; then
		fail "$lib still carries @@HOMEBREW_PREFIX@@ placeholders (unrelocated pour)"
	fi
done

# 5. The built binary, if present, must actually load the stack.
bin="target/release-perf/rmlx"
if [ -x "$bin" ]; then
	"$bin" --version >/dev/null 2>&1 ||
		fail "$bin cannot launch against the linked MLX (dyld failure?)"
fi

echo "preflight ok: mlx $mlx_ver + mlx-c $mlxc_ver, $nax nax GEMM kernels"
