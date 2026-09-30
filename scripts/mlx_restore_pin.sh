#!/usr/bin/env bash
# Restore the MLX pair named by crates/rmlx-mlx/mlx-pin.txt, then link and pin
# exactly those two kegs.
#
# The pinned pair is a source build with the NAX kernels (docs/MLX_PAIR.md).
# No Homebrew bottle for macOS 26 carries them, so this script pours no
# bottle. It takes each keg from, in this order:
#
#   1. the Cellar, when the keg is still there;
#   2. the durable copy in $RMLX_BOTTLE_STORE/source-built (default
#      ~/.rmlx/bottles/source-built): a tar of the Cellar keg directory, listed
#      in the SHA256SUMS file there.
#
# When neither has a keg, it stops and names the source build
# (docs/MLX_PAIR.md, "Building the pinned pair").
#
# Linking and pinning use Homebrew's own Keg and FormulaPin on the exact keg.
# `brew link <f>` and `brew pin <f>` act on the newest keg in the Cellar, not on
# the one the pin names.
#
# Verify afterwards with scripts/mlx_preflight.sh.

set -uo pipefail

PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
CELLAR="$PREFIX/Cellar"
STORE="${RMLX_BOTTLE_STORE:-$HOME/.rmlx/bottles}/source-built"

# The pinned pair, read from its one declaration by the one parser. The
# versions pass the allowlist in scripts/lib/mlx_pin.sh before they reach a
# path or the Ruby text below.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/mlx_pin.sh"
mlx_pin_load "$REPO_ROOT/crates/rmlx-mlx/mlx-pin.txt" || exit 1

die() {
	echo "restore FAIL: $*" >&2
	exit 1
}

# The dylib that a complete keg of the formula holds.
keg_lib() {
	case $1 in
	mlx) echo libmlx.dylib ;;
	mlx-c) echo libmlxc.dylib ;;
	esac
}

# from_store <formula> <version>: extract that keg from the durable copy.
# A tar is found by what it holds, not by its file name. Returns 1 when no
# listed tar holds the keg.
from_store() {
	local f=$1 v=$2 sums="$STORE/SHA256SUMS" sha file top got
	[ -f "$sums" ] || return 1
	while read -r sha file; do
		[ -f "$STORE/$file" ] || continue
		top=$(tar -tzf "$STORE/$file" 2>/dev/null | head -1)
		[ "$top" = "$f/$v/" ] || continue
		got=$(shasum -a 256 "$STORE/$file" | awk '{print $1}')
		[ "$got" = "$sha" ] || die "$STORE/$file has sha256 $got, but SHA256SUMS lists $sha"
		mkdir -p "$CELLAR/$f"
		tar -xzf "$STORE/$file" -C "$CELLAR" "$f/$v" || die "cannot extract $f $v from $file"
		echo "[store] $f $v from $file"
		return 0
	done <"$sums"
	return 1
}

for f in mlx mlx-c; do
	if [ "$f" = mlx ]; then v=$PIN_MLX; else v=$PIN_MLXC; fi
	keg="$CELLAR/$f/$v"
	lib=$(keg_lib "$f")
	if [ -f "$keg/lib/$lib" ]; then
		echo "[cellar] $f $v"
	elif [ -e "$keg" ]; then
		die "$keg exists but has no lib/$lib. Remove that directory, then run this again"
	else
		from_store "$f" "$v" ||
			die "no $f $v in $CELLAR and no copy of it in $STORE. Build the pair from source:" \
				"docs/MLX_PAIR.md, \"Building the pinned pair\""
		[ -f "$keg/lib/$lib" ] || die "the copy of $f $v has no lib/$lib"
	fi
done

# The kernels are the reason for the pin. A keg of the right version without
# them is a bottle build, not the source build the pin names.
metallib="$CELLAR/mlx/$PIN_MLX/lib/mlx.metallib"
symbols=$(strings "$metallib") || die "cannot read $metallib, so the NAX check could not run"
nax=$(printf '%s\n' "$symbols" | grep -c steel_gemm_fused_nax)
[ "$nax" -ge 1 ] ||
	die "mlx $PIN_MLX in $CELLAR has no NAX GEMM kernels. It is a bottle build, not the" \
		"source build the pin names: docs/MLX_PAIR.md, \"Building the pinned pair\""
echo "[ok] mlx $PIN_MLX has $nax NAX GEMM kernel occurrences"

echo "[link] link and pin mlx $PIN_MLX + mlx-c $PIN_MLXC"
brew ruby -e "
[['mlx', '$PIN_MLX'], ['mlx-c', '$PIN_MLXC']].each do |name, version|
  formula = Formula[name]
  pin = FormulaPin.new(formula)
  pin.unpin
  formula.installed_kegs.each(&:unlink)
  keg = Keg.new(HOMEBREW_CELLAR/name/version)
  keg.lock { keg.link }
  pin.pin_at(keg.version)
end" || die "brew could not link and pin the pair"

for f in mlx mlx-c; do
	if [ "$f" = mlx ]; then v=$PIN_MLX; else v=$PIN_MLXC; fi
	want=$(cd "$CELLAR/$f/$v" && pwd -P)
	for record in "opt/$f" "var/homebrew/linked/$f" "var/homebrew/pinned/$f"; do
		got=$(cd "$PREFIX/$record" 2>/dev/null && pwd -P)
		[ "$got" = "$want" ] || die "$PREFIX/$record resolves to '${got:-nothing}', not $want"
	done
done
echo "[done] opt, linked and pinned records name mlx $PIN_MLX + mlx-c $PIN_MLXC"
echo "       rebuild rmlx against them:  cargo clean -p rmlx-mlx && make build-perf"
exec "$(dirname "$0")/mlx_preflight.sh"
