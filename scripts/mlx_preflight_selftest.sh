#!/usr/bin/env bash
# scripts/mlx_preflight_selftest.sh — recall test for mlx_preflight.sh.
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

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="${1:-$HERE/mlx_preflight.sh}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mlx-preflight-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

MLX_V=9.9.9
MLXC_V=8.8.8
pass=0
fail=0

# new_case <name> <brand or `unknown`> <mlx> <mlx-c> <nax: yes|no> <mlx_pin line or `none`>
#   -> prints the case directory. `unknown` makes `sysctl` fail; `none` builds
#   no binary.
new_case() {
	local c="$WORK/$1" brand=$2 mlx=$3 mlxc=$4 nax=$5 line=$6 kernel
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

# expect <name> <case> <exit> <needles> [<lines that must not occur>]
#   <needles>  one or more lines; each must occur in the output
expect() {
	local name=$1 c=$2 want=$3 needles=$4 banned=${5:-} out rc ok=1 why="" needle
	out=$(cd "$c/repo" && PATH="$c/bin:$PATH" HOMEBREW_PREFIX="$c/prefix" \
		bash scripts/mlx_preflight.sh 2>&1)
	rc=$?
	[ "$rc" -eq "$want" ] || { ok=0; why="exit $rc, wanted $want"; }
	while IFS= read -r needle; do
		grep -qF -- "$needle" <<<"$out" || { ok=0; why="$why; no line with: $needle"; }
	done <<<"$needles"
	if [ -n "$banned" ]; then
		while IFS= read -r needle; do
			! grep -qF -- "$needle" <<<"$out" || { ok=0; why="$why; a line with: $needle"; }
		done <<<"$banned"
	fi
	if [ "$ok" = 1 ]; then
		pass=$((pass + 1))
		echo "  PASS  $name"
	else
		fail=$((fail + 1))
		echo "  FAIL  $name ($why)"
		while IFS= read -r line; do echo "        $line"; done <<<"$out"
	fi
}

CAPI="mlx_pin: RED — mlx-c C API mismatch: this binary was compiled against the mlx-c 0.6 C API, but the loaded libmlxc.dylib has the mlx-c 0.7 C API. Rebuild rMLX against the loaded mlx-c"

c=$(new_case m2-info "Apple M2 Pro" 9.9.8 8.8.7 no \
	"mlx_pin: INFO — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9 (the pin does not bind here)")
expect "an M1-M4 Mac whose binary reports info passes" "$c" 0 \
	"preflight ok: the built binary reports mlx_pin: INFO"

c=$(new_case m2-c-api "Apple M2 Pro" 9.9.8 8.8.7 no "$CAPI")
expect "a C API mismatch on an M1-M4 Mac stops and does not name the restore" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: $CAPI" "make mlx-restore-pin"

c=$(new_case m5-c-api "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "$CAPI")
expect "a C API mismatch on an M5 Mac stops and does not name the restore" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: $CAPI" "make mlx-restore-pin"

# The binary was built against the pinned pair, and `opt` moved to another
# pair: the binary's line names the restore, and so does the preflight.
c=$(new_case m5-c-api-off-pin "Apple M5 Max" 9.9.8 8.8.7 no \
	"$CAPI. On this host a rebuild against the loaded mlx-c is refused for the pair: the loaded pair does not pass the pin either (dyld resolved mlx 9.9.8). Run \`make mlx-restore-pin\` first, then rebuild if this binary was not built against the pinned pair.")
expect "a C API mismatch whose line names the restore names it" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: $CAPI
Restore the nax-capable pair:  make mlx-restore-pin"

c=$(new_case m5-pair "Apple M5 Max" "$MLX_V" "$MLXC_V" yes \
	"mlx_pin: RED — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9")
expect "an M5 Mac whose binary loaded another pair stops and names the restore" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED
make mlx-restore-pin"

c=$(new_case m5-green "Apple M5 Max" "$MLX_V" "$MLXC_V" yes \
	"mlx_pin: GREEN — loaded mlx 9.9.9 + mlx-c 8.8.8")
expect "an M5 Mac whose binary loaded the pinned pair passes" "$c" 0 \
	"preflight ok: the built binary reports mlx_pin: GREEN"

# The binary loads the pinned pair through MLX_PREFIX; the opt records name a
# pair without NAX kernels. The binary decides, so the records are not read.
c=$(new_case m5-green-opt-elsewhere "Apple M5 Max" 9.9.8 8.8.7 no \
	"mlx_pin: GREEN — loaded mlx 9.9.9 + mlx-c 8.8.8")
expect "with a binary the opt records are not read" "$c" 0 \
	"preflight ok: the built binary reports mlx_pin: GREEN" "PREFLIGHT FAIL"

c=$(new_case unknown-host unknown 9.9.8 8.8.7 no \
	"mlx_pin: RED — dyld resolved mlx 9.9.8, but crates/rmlx-mlx/mlx-pin.txt pins 9.9.9 (the chip could not be identified)")
expect "a host the binary cannot identify, on another pair, stops" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED"

# Only the two passing statuses pass: a status the script does not know stops.
c=$(new_case unknown-status "Apple M2 Pro" 9.9.8 8.8.7 no "mlx_pin: AMBER — a status no rmlx prints")
expect "a status other than green or info stops" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: AMBER"

# The status is the word after `mlx_pin:`, not a word anywhere in the line.
c=$(new_case red-naming-green "Apple M2 Pro" 9.9.8 8.8.7 no \
	"mlx_pin: RED — dyld resolved mlx 9.9.8 in /opt/EVERGREEN/lib, see INFO above")
expect "a red line that names GREEN and INFO stops" "$c" 1 \
	"PREFLIGHT FAIL: the built binary refuses to measure: mlx_pin: RED"

c=$(new_case no-launch "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "mlx_pin: GREEN — loaded mlx 9.9.9")
printf '#!/usr/bin/env bash\nexit 1\n' >"$c/repo/target/release-perf/rmlx"
expect "a binary that cannot launch stops" "$c" 1 \
	"PREFLIGHT FAIL: target/release-perf/rmlx cannot launch"

# The opt records would pass: a binary that cannot run must not fall through
# to them.
c=$(new_case not-executable "Apple M2 Pro" 9.9.8 8.8.7 no "mlx_pin: GREEN — loaded mlx 9.9.9")
chmod -x "$c/repo/target/release-perf/rmlx"
expect "a binary that is not executable stops" "$c" 1 \
	"PREFLIGHT FAIL: target/release-perf/rmlx is not executable" "no target/release-perf/rmlx yet"

c=$(new_case no-pin-line "Apple M5 Max" "$MLX_V" "$MLXC_V" yes "")
expect "a binary that prints no mlx_pin line stops" "$c" 1 \
	"PREFLIGHT FAIL: target/release-perf/rmlx reported no mlx_pin line"

# No binary yet: the opt records stand in for the pair a build would load.
c=$(new_case m2-no-binary "Apple M2 Pro" 9.9.8 8.8.7 no none)
expect "no binary, an M1-M4 Mac on another pair passes" "$c" 0 \
	"preflight ok: Apple M2 Pro (no Neural Accelerator, the pin does not bind), mlx 9.9.8 + mlx-c 8.8.7, nax check skipped"

c=$(new_case m5-no-binary "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
expect "no binary, an M5 Mac on the pinned pair passes" "$c" 0 \
	"preflight ok: Apple M5 Max (the pin binds, or the chip is not identified), pinned mlx 9.9.9 + mlx-c 8.8.8"

c=$(new_case m5-no-binary-no-nax "Apple M5 Max" "$MLX_V" "$MLXC_V" no none)
expect "no binary, an M5 Mac without NAX kernels stops" "$c" 1 \
	"ships 0 nax GEMM kernels
make mlx-restore-pin"

c=$(new_case m5-no-binary-pair "Apple M5 Max" 9.9.8 8.8.7 yes none)
expect "no binary, an M5 Mac on another pair stops" "$c" 1 \
	"linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair
make mlx-restore-pin"

c=$(new_case unknown-no-binary unknown 9.9.8 8.8.7 yes none)
expect "no binary, a chip sysctl cannot name is held to the pin" "$c" 1 \
	"linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair"

# rmlx_core::apple_gpu identifies a chip only from a leading "Apple M<n>".
c=$(new_case virtual-no-binary "Virtual Apple M2" 9.9.8 8.8.7 yes none)
expect "no binary, a brand that does not start with Apple M<n> is held to the pin" "$c" 1 \
	"linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair"

# The spellings rmlx_core::apple_gpu::parse_apple_generation also reads as
# M1-M4: lower and upper case, leading space, a zero before the number.
for brand in "apple m2" "APPLE M3" "  Apple M3 Max" "Apple M04"; do
	c=$(new_case "exempt-$(printf '%s' "$brand" | tr -c 'A-Za-z0-9' _)" "$brand" 9.9.8 8.8.7 no none)
	expect "no binary, \"$brand\" is an M1-M4 Mac" "$c" 0 \
		"(no Neural Accelerator, the pin does not bind), mlx 9.9.8 + mlx-c 8.8.7, nax check skipped"
done

# parse_apple_generation takes only these three spellings of the vendor.
c=$(new_case mixed-case-no-binary "aPPLE M2" 9.9.8 8.8.7 yes none)
expect "no binary, a vendor spelling the Rust parser refuses is held to the pin" "$c" 1 \
	"linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair"

c=$(new_case m41-no-binary "Apple M41" 9.9.8 8.8.7 yes none)
expect "no binary, M41 is not M4" "$c" 1 "linked mlx 9.9.8 + mlx-c 8.8.7 is not the pinned pair"

c=$(new_case no-opt "Apple M2 Pro" 9.9.8 8.8.7 no none)
rm "$c/prefix/opt/mlx-c"
expect "no binary, an opt record that does not resolve stops" "$c" 1 \
	"PREFLIGHT FAIL: $c/prefix/opt/mlx-c does not resolve"

c=$(new_case unrelocated "Apple M2 Pro" 9.9.8 8.8.7 no none)
printf '#!/usr/bin/env bash\necho "@@HOMEBREW_PREFIX@@/opt/mlx/lib/libmlx.dylib"\n' >"$c/bin/otool"
expect "no binary, an unrelocated dylib stops" "$c" 1 "carries @@HOMEBREW_PREFIX@@ placeholders"

c=$(new_case no-metallib "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
rm "$c/prefix/Cellar/mlx/$MLX_V/lib/mlx.metallib"
expect "no binary, an M5 Mac without mlx.metallib stops" "$c" 1 "mlx.metallib missing"

c=$(new_case no-strings "Apple M5 Max" "$MLX_V" "$MLXC_V" yes none)
printf '#!/usr/bin/env bash\nexit 1\n' >"$c/bin/strings"
chmod +x "$c/bin/strings"
expect "no binary, a metallib strings cannot read names the tool, not the restore" "$c" 1 \
	"cannot read $c/prefix/opt/mlx/lib/mlx.metallib (is \`strings\` present?)" "make mlx-restore-pin"

echo "mlx_preflight_selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
