#!/usr/bin/env bash
# Recall test for mlx_preflight.sh: throwaway Homebrew prefixes and stubs, asserting output.
#
#
# Recall test for: mlx_preflight.sh
#
#     bash scripts/mlx_preflight_selftest.sh [<script under test>]
#
# Each case builds a throwaway Homebrew prefix whose `opt` records name a pair,
# a repo tree whose pin names mlx 9.9.9 + mlx-c 8.8.8, and stubs first on
# PATH: `sysctl` names the chip (or fails), `otool` prints nothing. A stub
# built binary at target/release-perf/rmlx prints the `mlx_pin` line of the
# case beside a red line of another check; a case with no built binary takes
# `none`. A case can change its tree before it runs. The case runs the script
# from the repo tree and asserts the exit code, lines that must occur in the
# output and lines that must not. No case reads the real Homebrew prefix.
#
# Portable to GNU and BSD userlands: the hosted CI runs it on Linux.

set -euo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="${1:-$HERE/mlx_preflight.sh}"
# shellcheck source=scripts/lib/selftest_support.sh
. "$HERE/lib/selftest_support.sh"
st_init mlx-preflight-selftest

MLX_V=9.9.9
MLXC_V=8.8.8

# new_case <name> <brand or `unknown`> <mlx> <mlx-c> <nax: yes|no> <mlx_pin line or `none`>
#   -> prints the case directory. `unknown` makes `sysctl` fail; `none` builds
#   no binary.
new_case() {
	local c="$ST_WORK/$1" brand=$2 mlx=$3 mlxc=$4 nax=$5 line=$6 kernel
	mkdir -p "$c/repo/scripts/lib" "$c/repo/crates/rmlx-mlx" "$c/repo/target/release-perf" \
		"$c/bin" "$c/prefix/opt" "$c/prefix/Cellar/mlx/$mlx/lib" "$c/prefix/Cellar/mlx-c/$mlxc/lib"
	cp "$SUT" "$c/repo/scripts/mlx_preflight.sh"
	cp "$HERE/lib/mlx_pin.sh" "$c/repo/scripts/lib/mlx_pin.sh"
	printf 'mlx    %s\nmlx-c  %s\n' "$MLX_V" "$MLXC_V" >"$c/repo/crates/rmlx-mlx/mlx-pin.txt"

	printf 'file' >"$c/prefix/Cellar/mlx/$mlx/lib/libmlx.dylib"
	printf 'file' >"$c/prefix/Cellar/mlx-c/$mlxc/lib/libmlxc.dylib"
	if [ "$nax" = yes ]; then kernel=steel_gemm_fused_nax_bfloat16; else kernel=steel_gemm_fused_bfloat16; fi
	printf 'header %s trailer' "$kernel" >"$c/prefix/Cellar/mlx/$mlx/lib/mlx.metallib"
	ln -s "$c/prefix/Cellar/mlx/$mlx" "$c/prefix/opt/mlx"
	ln -s "$c/prefix/Cellar/mlx-c/$mlxc" "$c/prefix/opt/mlx-c"

	if [ "$brand" = unknown ]; then
		printf '#!/usr/bin/env bash\nexit 1\n' >"$c/bin/sysctl"
	else
		printf '#!/usr/bin/env bash\necho "%s"\n' "$brand" >"$c/bin/sysctl"
	fi
	printf '#!/usr/bin/env bash\nexit 0\n' >"$c/bin/otool"
	chmod +x "$c/bin/sysctl" "$c/bin/otool"
	if [ "$line" != none ]; then
		printf '%s\n' "registry: RED — no registry" "$line" >"$c/pin_line"
		cat >"$c/repo/target/release-perf/rmlx" <<STUB
#!/usr/bin/env bash
case "\$1" in
--version) echo "rmlx 0.0.0" ;;
healthcheck) cat "$c/pin_line" ;;
esac
STUB
		chmod +x "$c/repo/target/release-perf/rmlx"
	fi
	printf '%s\n' "$c"
}

# preflight_run <case>: the script under test, run from the case's repo tree.
preflight_run() {
	(cd "$1/repo" && PATH="$1/bin:$PATH" HOMEBREW_PREFIX="$1/prefix" bash scripts/mlx_preflight.sh)
}

CAPI="mlx_pin: RED — mlx-c C API mismatch: this binary was compiled against the mlx-c 0.6 C API, but the loaded libmlxc.dylib has the mlx-c 0.7 C API. Rebuild rMLX against the loaded mlx-c"

c=$(new_case m2-info "Apple M2 Pro" 9.9.8 8.8.7 no \
	"mlx_pin: INFO — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9 (the pin does not bind here)")
st_case "an M1-M4 Mac whose binary reports info passes" 0 \
	--has "preflight ok: the built binary reports mlx_pin: INFO" -- preflight_run "$c"

c=$(new_case m2-c-api "Apple M2 Pro" 9.9.8 8.8.7 no "$CAPI")
st_case "a C API mismatch on an M1-M4 Mac stops and does not name the restore" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: $CAPI" \
	--lacks "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m5-c-api "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "$CAPI")
st_case "a C API mismatch on an M5 Mac stops and does not name the restore" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: $CAPI" \
	--lacks "make mlx-restore-pin" -- preflight_run "$c"

# The binary was built against the pinned pair, and `opt` moved to another
# pair: the binary's line names the restore, and so does the preflight.
c=$(new_case m5-c-api-off-pin "Apple M5 Max" 9.9.8 8.8.7 no \
	"$CAPI. On this host a rebuild against the loaded mlx-c is refused for the pair: the loaded pair does not pass the pin either (dyld resolved mlx 9.9.8). Run \`make mlx-restore-pin\` first, then rebuild if this binary was not built against the pinned pair.")
st_case "a C API mismatch whose line names the restore names it" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: $CAPI" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m5-pair "Apple M5 Max" "$MLX_V" "$MLXC_V" yes \
	"mlx_pin: RED — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9")
st_case "an M5 Mac whose binary loaded another pair stops and names the restore" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED" \
	--has "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m5-green "Apple M5 Max" "$MLX_V" "$MLXC_V" yes \
	"mlx_pin: GREEN — loaded mlx 9.9.9 + mlx-c 8.8.8")
st_case "an M5 Mac whose binary loaded the pinned pair passes" 0 \
	--has "preflight ok: the built binary reports mlx_pin: GREEN" -- preflight_run "$c"

# The binary loads the pinned pair through MLX_PREFIX; the opt records name a
# pair without NAX kernels. The binary decides, so the records are not read.
c=$(new_case m5-green-opt-elsewhere "Apple M5 Max" 9.9.8 8.8.7 no \
	"mlx_pin: GREEN — loaded mlx 9.9.9 + mlx-c 8.8.8")
st_case "with a binary the opt records are not read" 0 \
	--has "preflight ok: the built binary reports mlx_pin: GREEN" \
	--lacks "PREFLIGHT FAIL" -- preflight_run "$c"

c=$(new_case unknown-host unknown 9.9.8 8.8.7 no \
	"mlx_pin: RED — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9 (the chip could not be identified)")
st_case "a host the binary cannot identify, on another pair, stops" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED" -- preflight_run "$c"

# Only the two passing statuses pass: a status the script does not know stops.
c=$(new_case unknown-status "Apple M2 Pro" 9.9.8 8.8.7 no "mlx_pin: AMBER — a status no rmlx prints")
st_case "a status other than green or info stops" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: AMBER" -- preflight_run "$c"

# The status is the word after `mlx_pin:`, not a word anywhere in the line.
c=$(new_case red-naming-green "Apple M2 Pro" 9.9.8 8.8.7 no \
	"mlx_pin: RED — dyld resolved mlx 9.9.8 in /opt/EVERGREEN/lib, see INFO above")
st_case "a red line that names GREEN and INFO stops" 1 \
	--has "PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED" -- preflight_run "$c"

c=$(new_case no-launch "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "mlx_pin: GREEN — loaded mlx 9.9.9")
printf '#!/usr/bin/env bash\nexit 1\n' >"$c/repo/target/release-perf/rmlx"
st_case "a binary that cannot launch stops" 1 \
	--has "PREFLIGHT FAIL: target/release-perf/rmlx cannot launch" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

# The opt records would pass: a binary that cannot run must not fall through
# to them.
c=$(new_case not-executable "Apple M2 Pro" 9.9.8 8.8.7 no "mlx_pin: GREEN — loaded mlx 9.9.9")
chmod -x "$c/repo/target/release-perf/rmlx"
st_case "a binary that is not executable stops" 1 \
	--has "PREFLIGHT FAIL: target/release-perf/rmlx is not executable" \
	--lacks "no target/release-perf/rmlx yet" -- preflight_run "$c"

c=$(new_case no-pin-line "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "")
st_case "a binary that prints no mlx_pin line stops" 1 \
	--has "PREFLIGHT FAIL: target/release-perf/rmlx reported no mlx_pin line" -- preflight_run "$c"

# No binary yet: the opt records stand in for the pair a build would load.
c=$(new_case m2-no-binary "Apple M2 Pro" 9.9.8 8.8.7 no none)
st_case "no binary, an M1-M4 Mac on another pair passes" 0 \
	--has "note: no target/release-perf/rmlx yet" \
	--has "preflight ok: Apple M2 Pro (no Neural Accelerator, the pin does not bind), mlx 9.9.8 + mlx-c 8.8.7, nax check skipped" -- preflight_run "$c"

c=$(new_case m5-no-binary "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
st_case "no binary, an M5 Mac on the pinned pair passes" 0 \
	--has "preflight ok: Apple M5 Max (the pin binds, or the chip is not identified), pinned mlx 9.9.9 + mlx-c 8.8.8, 1 nax GEMM kernel occurrences" -- preflight_run "$c"

c=$(new_case m5-no-binary-no-nax "Apple M5 Max" "$MLX_V" "$MLXC_V" no none)
st_case "no binary, an M5 Mac without NAX kernels stops" 1 \
	--has "ships 0 nax GEMM kernels" \
	--has "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m5-no-binary-pair "Apple M5 Max" 9.9.8 8.8.7 yes none)
st_case "no binary, an M5 Mac on another pair stops" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair" \
	--has "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case unknown-no-binary unknown 9.9.8 8.8.7 yes none)
st_case "no binary, a chip sysctl cannot name is held to the pin" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair" -- preflight_run "$c"

# rmlx_core::apple_gpu identifies a chip only from a leading "Apple M<n>".
c=$(new_case virtual-no-binary "Virtual Apple M2" 9.9.8 8.8.7 yes none)
st_case "no binary, a brand that does not start with Apple M<n> is held to the pin" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair" -- preflight_run "$c"

# The spellings rmlx_core::apple_gpu::parse_apple_generation also reads as
# M1-M4: lower and upper case, leading space, a zero before the number.
for brand in "apple m2" "APPLE M3" "  Apple M3 Max" "Apple M04"; do
	c=$(new_case "exempt-$(printf '%s' "$brand" | tr -c 'A-Za-z0-9' _)" "$brand" 9.9.8 8.8.7 no none)
	st_case "no binary, \"$brand\" is an M1-M4 Mac" 0 \
		--has "(no Neural Accelerator, the pin does not bind), mlx 9.9.8 + mlx-c 8.8.7, nax check skipped" -- preflight_run "$c"
done

# parse_apple_generation takes only these three spellings of the vendor.
c=$(new_case mixed-case-no-binary "aPPLE M2" 9.9.8 8.8.7 yes none)
st_case "no binary, a vendor spelling the Rust parser refuses is held to the pin" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair" -- preflight_run "$c"

c=$(new_case m5-no-binary-mlx-only "Apple M5 Max" 9.9.8 "$MLXC_V" yes none)
st_case "no binary, an M5 Mac on the pinned mlx-c and another mlx stops" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.8 is not the pinned pair" \
	--has "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m5-no-binary-mlxc-only "Apple M5 Max" "$MLX_V" 8.8.7 yes none)
st_case "no binary, an M5 Mac on the pinned mlx and another mlx-c stops" 1 \
	--has "linked mlx 9.9.9 + mlx-c 8.8.7 is not the pinned pair" \
	--has "make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case m41-no-binary "Apple M41" 9.9.8 8.8.7 yes none)
st_case "no binary, M41 is not M4" 1 \
	--has "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair" -- preflight_run "$c"

c=$(new_case no-opt "Apple M2 Pro" 9.9.8 8.8.7 no none)
rm "$c/prefix/opt/mlx-c"
st_case "no binary, an opt record that does not resolve stops" 1 \
	--has "PREFLIGHT FAIL: $c/prefix/opt/mlx-c does not resolve" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case no-opt-mlx "Apple M2 Pro" 9.9.8 8.8.7 no none)
rm "$c/prefix/opt/mlx"
st_case "no binary, an mlx opt record that does not resolve stops" 1 \
	--has "PREFLIGHT FAIL: $c/prefix/opt/mlx does not resolve" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case no-libmlx "Apple M2 Pro" 9.9.8 8.8.7 no none)
rm "$c/prefix/Cellar/mlx/9.9.8/lib/libmlx.dylib"
st_case "no binary, a missing libmlx.dylib stops" 1 \
	--has "PREFLIGHT FAIL: $c/prefix/opt/mlx/lib/libmlx.dylib missing" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case no-libmlxc "Apple M2 Pro" 9.9.8 8.8.7 no none)
rm "$c/prefix/Cellar/mlx-c/8.8.7/lib/libmlxc.dylib"
st_case "no binary, a missing libmlxc.dylib stops" 1 \
	--has "PREFLIGHT FAIL: $c/prefix/opt/mlx-c/lib/libmlxc.dylib missing" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case unrelocated "Apple M2 Pro" 9.9.8 8.8.7 no none)
printf '#!/usr/bin/env bash\necho "@@HOMEBREW_PREFIX@@/opt/mlx/lib/libmlx.dylib"\n' >"$c/bin/otool"
st_case "no binary, an unrelocated dylib stops" 1 \
	--has "carries @@HOMEBREW_PREFIX@@ placeholders" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case unrelocated-mlxc "Apple M2 Pro" 9.9.8 8.8.7 no none)
printf '#!/usr/bin/env bash\ncase "$2" in *libmlxc.dylib) echo "@@HOMEBREW_PREFIX@@/opt/mlx-c/lib/libmlxc.dylib" ;; esac\n' >"$c/bin/otool"
st_case "no binary, an unrelocated libmlxc.dylib stops" 1 \
	--has "$c/prefix/opt/mlx-c/lib/libmlxc.dylib still carries @@HOMEBREW_PREFIX@@ placeholders" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case no-metallib "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
rm "$c/prefix/Cellar/mlx/$MLX_V/lib/mlx.metallib"
st_case "no binary, an M5 Mac without mlx.metallib stops" 1 \
	--has "mlx.metallib missing" \
	--has "Restore the nax-capable pair:  make mlx-restore-pin" -- preflight_run "$c"

c=$(new_case no-strings "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
printf '#!/usr/bin/env bash\nexit 1\n' >"$c/bin/strings"
chmod +x "$c/bin/strings"
st_case "no binary, a metallib strings cannot read names the tool, not the restore" 1 \
	--has "cannot read $c/prefix/opt/mlx/lib/mlx.metallib (is \`strings\` present?)" \
	--lacks "make mlx-restore-pin" -- preflight_run "$c"

st_done
