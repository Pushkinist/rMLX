#!/usr/bin/env bash
# Restore the MLX pair named by crates/rmlx-mlx/mlx-pin.txt, then link and pin
# exactly those two kegs.
#
# The pinned pair is a source build with the NAX kernels (docs/MLX_PAIR.md).
# No Homebrew bottle for macOS 26 carries them, so this script pours no
# bottle. It takes each keg from, in this order:
#
#   1. the Cellar, when a complete keg is there;
#   2. the durable copy in $RMLX_BOTTLE_STORE/source-built (default
#      ~/.rmlx/bottles/source-built): a tar of the Cellar keg directory, listed
#      in the SHA256SUMS file there.
#
# A keg is usable when it holds every file rMLX builds against and loads and,
# for mlx, the NAX GEMM kernels. When neither source has a usable keg, the
# script stops and names the source build (docs/MLX_PAIR.md, "Building the
# pinned pair").
#
# A copy is extracted into a staging directory and moved into the Cellar only
# when it is usable, so a refused copy leaves no keg for a later run or for
# Homebrew to take as installed.
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
RECORDS="opt var/homebrew/linked var/homebrew/pinned"

# The pinned pair, read from its one declaration by the one parser. The
# versions pass the allowlist in scripts/lib/mlx_pin.sh before they reach a
# path or the Ruby text below.
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/lib/mlx_pin.sh"
mlx_pin_load "$REPO_ROOT/crates/rmlx-mlx/mlx-pin.txt" || exit 1

# Staging is dot-named in the Cellar root: Homebrew counts every directory in a
# formula directory as a keg but skips dot-named formulas, and a move inside one
# file system is one rename.
STAGE=""
trap '[ -z "$STAGE" ] || rm -rf "$STAGE"' EXIT
# A run stopped by SIGKILL leaves its staging directory, with a full keg in it.
rm -rf "${CELLAR:?}"/.rmlx-restore.*

die() {
	echo "restore FAIL: $*" >&2
	exit 1
}

pin_of() {
	if [ "$1" = mlx ]; then echo "$PIN_MLX"; else echo "$PIN_MLXC"; fi
}

# The files of a keg that rMLX builds against and loads.
keg_files() {
	case $1 in
	mlx) echo lib/libmlx.dylib lib/libjaccl.dylib lib/mlx.metallib include/mlx/version.h ;;
	mlx-c) echo lib/libmlxc.dylib include/mlx/c/fast.h ;;
	esac
}

# keg_problem <directory> <formula>: print why the directory is not a usable
# keg of the formula. Exit 1 when it is usable.
keg_problem() {
	local file symbols
	for file in $(keg_files "$2"); do
		if [ ! -s "$1/$file" ]; then
			echo "has no $file, or the file is empty"
			return 0
		fi
	done
	[ "$2" = mlx ] || return 1
	# The kernels are the reason for the pin. The reader's status comes first:
	# `grep -c` prints 0 also when `strings` could not run.
	if ! symbols=$(strings "$1/lib/mlx.metallib"); then
		echo "has a lib/mlx.metallib that strings cannot read, so the NAX check could not run"
		return 0
	fi
	if [ "$(printf '%s\n' "$symbols" | grep -c steel_gemm_fused_nax)" -lt 1 ]; then
		echo "has no NAX GEMM kernels. It is a bottle build, not the source build the pin names"
		return 0
	fi
	return 1
}

# from_store <formula> <version>: move that keg from the durable copy into the
# Cellar. A tar is found by what it holds, not by its file name. Returns 1 when
# no listed tar holds the keg.
from_store() {
	local f=$1 v=$2 sums="$STORE/SHA256SUMS" sha file top got problem
	[ -f "$sums" ] || return 1
	while read -r sha file; do
		[ -f "$STORE/$file" ] || continue
		top=$(tar -tzf "$STORE/$file" 2>/dev/null | head -1)
		[ "$top" = "$f/$v/" ] || continue
		got=$(shasum -a 256 "$STORE/$file" | awk '{print $1}')
		[ "$got" = "$sha" ] || die "$STORE/$file has sha256 $got, but SHA256SUMS lists $sha"
		STAGE=$(mktemp -d "$CELLAR/.rmlx-restore.XXXXXX") || die "cannot make a staging directory in $CELLAR"
		tar -xzf "$STORE/$file" -C "$STAGE" "$f/$v" ||
			die "cannot extract $f $v from $file. Nothing was written to $CELLAR/$f." \
				"Remove $STORE/$file and its line in $sums, then run this again"
		if problem=$(keg_problem "$STAGE/$f/$v" "$f"); then
			die "the copy of $f $v in $file $problem. Nothing was written to $CELLAR/$f." \
				"Remove $STORE/$file and its line in $sums, then run this again"
		fi
		mkdir -p "$CELLAR/$f" && mv "$STAGE/$f/$v" "$CELLAR/$f/$v" ||
			die "cannot move $f $v into $CELLAR/$f"
		rm -rf "$STAGE"
		STAGE=""
		echo "[store] $f $v from $file"
		return 0
	done <"$sums"
	return 1
}

for f in mlx mlx-c; do
	v=$(pin_of "$f")
	keg="$CELLAR/$f/$v"
	if [ -e "$keg" ]; then
		if problem=$(keg_problem "$keg" "$f"); then
			die "$keg $problem. Remove that directory, then run this again"
		fi
		echo "[cellar] $f $v"
	else
		from_store "$f" "$v" ||
			die "no $f $v in $CELLAR and no copy of it in $STORE. Build the pair from source:" \
				"docs/MLX_PAIR.md, \"Building the pinned pair\""
	fi
done

# record_keg <record> <formula>: the directory a prefix record resolves to.
record_keg() {
	(cd "$PREFIX/$1/$2" 2>/dev/null && pwd -P)
}

# The link step can stop with one formula relinked and the other unlinked or
# unpinned, so a failure names what each record resolves to now.
link_failed() {
	local f record
	{
		echo "restore FAIL: $1"
		echo "  The records of the pair are now:"
		for f in mlx mlx-c; do
			for record in $RECORDS; do
				echo "    $PREFIX/$record/$f -> $(record_keg "$record" "$f" || echo nothing)"
			done
		done
		echo "  Fix the cause above, then run make mlx-restore-pin again."
	} >&2
	exit 1
}

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
end" || link_failed "brew could not link and pin the pair"

for f in mlx mlx-c; do
	want=$(cd "$CELLAR/$f/$(pin_of "$f")" && pwd -P)
	for record in $RECORDS; do
		got=$(record_keg "$record" "$f")
		[ "$got" = "$want" ] || link_failed "$PREFIX/$record/$f resolves to '${got:-nothing}', not $want"
	done
done
echo "[done] opt, linked and pinned records name mlx $PIN_MLX + mlx-c $PIN_MLXC"
echo "       rebuild rmlx against them:  cargo clean -p rmlx-mlx && make build-perf"
exec "$(dirname "$0")/mlx_preflight.sh"
