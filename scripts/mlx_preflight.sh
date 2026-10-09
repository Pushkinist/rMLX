#!/usr/bin/env bash
# Stop a measurement exactly where `rmlx baseline` and `rmlx bench` refuse:
# an mlx-c C API mismatch on every host, and a pair that is not the pinned,
# nax-capable one on M5 and later or on a host whose chip cannot be
# identified. On an identified M1-M4 host the pin does not bind, so the
# preflight passes there when the C API matches.
#
# A C API mismatch is fixed by a rebuild. The restore is named for it only
# when the binary's line names it: where the pinned pair is required (the pin
# binds, or the chip is not identified) and the loaded pair is not the pinned
# one.
#
# Two sources, never both:
#
#   1. A built binary (target/release-perf/rmlx). It must launch, and its own
#      `mlx_pin` healthcheck line decides: GREEN or INFO passes, anything else
#      stops. That line is `PinCheck::refusal` in crates/rmlx-mlx/src/pin.rs,
#      the verdict `rmlx baseline` and `rmlx bench` apply, read in the process
#      that has the pair loaded. `MLX_PREFIX` / `MLX_C_PREFIX`
#      (crates/rmlx-mlx/build.rs) can load a pair that the `opt` records do
#      not name, so with a binary the records are not read.
#
#   2. No binary yet: a pre-filter on the `opt` records, for the pair a build
#      would load. The records must resolve to relocated dylibs. Unless the
#      chip is an identified M1-M4, mlx.metallib must carry
#      `steel_gemm_fused_nax` kernels and the pair must be the pinned one. The
#      C API cannot be known without a binary.
#
# The pinned pair is read from crates/rmlx-mlx/mlx-pin.txt, never restated here.
#
# Run before any measurement. Exits non-zero and names the fix on failure.
# Background: docs/MLX_PAIR.md

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

# --- 1. the built binary's own verdict ---------------------------------------
bin="target/release-perf/rmlx"
if [ -e "$bin" ] && [ ! -x "$bin" ]; then
	fail "$bin is not executable, so its verdict cannot be read. Build it again" \
		"(make build-perf), then run this again"
	exit 1
fi
if [ -x "$bin" ]; then
	if ! "$bin" --version >/dev/null 2>&1; then
		fail "$bin cannot launch against the MLX it loads (dyld failure? ABI mismatch?)"
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
	"mlx_pin: GREEN "* | "mlx_pin: INFO "*)
		echo "preflight ok: the built binary reports $pin_line"
		exit 0
		;;
	esac
	fail "the built binary refuses to measure: $pin_line"
	# A restore does not change the binary, so a C API mismatch takes the
	# restore only when its line names it.
	case "$pin_line" in
	*"C API mismatch"*"make mlx-restore-pin"*) hint_restore ;;
	*"C API mismatch"*) ;;
	*) hint_restore ;;
	esac
	exit 1
fi

# --- 2. no binary: pre-filter on the opt records -----------------------------
echo "note: no $bin yet — the opt records are a pre-filter; the binary's own" \
	"verdict is what gates the measurement (rmlx baseline / rmlx bench)."

# The rule of rmlx_core::apple_gpu::parse_apple_generation: only an identified
# M1-M4 is exempt. M5 and later, and a brand it cannot parse, need the pin.
brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
[ -n "$brand" ] || brand="unknown"
exempt=0
if [[ "$brand" =~ ^[[:space:]]*(Apple|apple|APPLE)\ [Mm]0*[1-4]([^0-9]|$) ]]; then
	exempt=1
	host="no Neural Accelerator, the pin does not bind"
else
	host="the pin binds, or the chip is not identified"
fi

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

if [ "$exempt" = 1 ]; then
	echo "preflight ok: $brand ($host), mlx $mlx_ver + mlx-c $mlxc_ver, nax check skipped"
	exit 0
fi

metallib="$PREFIX/opt/mlx/lib/mlx.metallib"
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
	fail "$brand ($host): mlx $mlx_ver ships 0 nax GEMM kernels" \
		"— GEMM-bound prefill would be ~2-3.8x slow (pinned: mlx $PIN_MLX + mlx-c $PIN_MLXC)"
	hint_restore
	exit 1
fi
# The pair is the validated unit even when the kernels are present: mlx and
# mlx-c are ABI-coupled, and any prefill number measured across the pin
# boundary is not comparable to one from the other side.
if [ "$mlx_ver" != "$PIN_MLX" ] || [ "$mlxc_ver" != "$PIN_MLXC" ]; then
	fail "linked mlx $mlx_ver + mlx-c $mlxc_ver is not the pinned pair" \
		"(mlx $PIN_MLX + mlx-c $PIN_MLXC, crates/rmlx-mlx/mlx-pin.txt)"
	hint_restore
	exit 1
fi
echo "preflight ok: $brand ($host), pinned mlx $mlx_ver + mlx-c $mlxc_ver, $nax nax GEMM kernel occurrences"
