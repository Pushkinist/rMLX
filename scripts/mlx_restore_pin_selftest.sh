#!/usr/bin/env bash
# scripts/mlx_restore_pin_selftest.sh — recall test for mlx_restore_pin.sh.
#
#     bash scripts/mlx_restore_pin_selftest.sh [<script under test>]
#
# Each case builds a throwaway Homebrew prefix, a durable store and a repo tree
# whose pin names mlx 9.9.9 + mlx-c 8.8.8, runs the script with a stub `brew`
# first on PATH, and asserts the exit code, a line of the output, and the
# records the run left. The stub acts as Homebrew does: `brew link` and
# `brew pin` take the newest keg, and a `brew ruby` Keg link takes the keg it
# names. No case reads or changes the real Homebrew prefix.

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUT="${1:-$HERE/mlx_restore_pin.sh}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/mlx-restore-pin-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

MLX_V=9.9.9
MLXC_V=8.8.8
NEWER_V=9.9.10
pass=0
fail=0

# The stub `brew`. It logs every call and changes only the prefix it is given.
mkdir -p "$WORK/bin"
cat >"$WORK/bin/brew" <<'STUB'
#!/usr/bin/env bash
set -uo pipefail
P="$HOMEBREW_PREFIX"
echo "brew $*" >>"$P/brew.log"
[ -f "$P/brew.fail" ] && exit 1
record() { # <dir> <formula> <version>
	mkdir -p "$P/$1"
	ln -sfn "$P/Cellar/$2/$3" "$P/$1/$2"
}
newest() { ls "$P/Cellar/$1" | sort -V | tail -1; }
case "${1:-}" in
ruby)
	pairs=$(printf '%s' "${3:-}" | grep -oE "\['[a-z-]+', '[^']+'\]" | tr -d "[]',")
	while read -r name version; do
		[ -n "$name" ] || continue
		rm -f "$P/var/homebrew/pinned/$name" "$P/var/homebrew/linked/$name"
		record opt "$name" "$version"
		record var/homebrew/linked "$name" "$version"
		record var/homebrew/pinned "$name" "$version"
	done <<<"$pairs"
	;;
link)
	shift
	for name in "$@"; do
		record opt "$name" "$(newest "$name")"
		record var/homebrew/linked "$name" "$(newest "$name")"
	done
	;;
pin)
	shift
	for name in "$@"; do record var/homebrew/pinned "$name" "$(newest "$name")"; done
	;;
unpin | unlink)
	shift
	for name in "$@"; do rm -f "$P/var/homebrew/pinned/$name"; done
	;;
*) exit 0 ;;
esac
STUB
chmod +x "$WORK/bin/brew"

# keg <root> <formula> <version> <nax: yes|no>: a keg directory as brew leaves it.
keg() {
	local d="$1/$2/$3"
	mkdir -p "$d/lib"
	if [ "$2" = mlx ]; then
		printf 'dylib' >"$d/lib/libmlx.dylib"
		if [ "$4" = yes ]; then
			printf 'header steel_gemm_fused_nax_bfloat16 trailer' >"$d/lib/mlx.metallib"
		else
			printf 'header steel_gemm_fused_bfloat16 trailer' >"$d/lib/mlx.metallib"
		fi
	else
		printf 'dylib' >"$d/lib/libmlxc.dylib"
	fi
}

# new_case <name> -> prints the case directory, with repo, prefix and store.
new_case() {
	local c="$WORK/$1"
	mkdir -p "$c/repo/scripts/lib" "$c/repo/crates/rmlx-mlx" "$c/prefix/Cellar" \
		"$c/store/source-built"
	cp "$SUT" "$c/repo/scripts/mlx_restore_pin.sh"
	cp "$HERE/lib/mlx_pin.sh" "$c/repo/scripts/lib/mlx_pin.sh"
	printf '#!/usr/bin/env bash\necho "[preflight stub]"\n' >"$c/repo/scripts/mlx_preflight.sh"
	chmod +x "$c/repo/scripts/mlx_preflight.sh"
	printf 'mlx    %s\nmlx-c  %s\n' "$MLX_V" "$MLXC_V" >"$c/repo/crates/rmlx-mlx/mlx-pin.txt"
	printf '%s\n' "$c"
}

# store_tar <case> <file> <formula> <version> <nax>: a tar of a keg, listed in
# SHA256SUMS with its true sha256.
store_tar() {
	local c=$1 file=$2 src="$1/tarsrc"
	keg "$src" "$3" "$4" "$5"
	tar -czf "$c/store/source-built/$file" -C "$src" "$3/$4"
	(cd "$c/store/source-built" && shasum -a 256 "$file") >>"$c/store/source-built/SHA256SUMS"
	rm -rf "$src"
}

# resolves <case> <record> -> the real directory a prefix record names.
resolves() { (cd "$1/prefix/$2" 2>/dev/null && pwd -P); }

# expect <name> <case> <exit> <needle> [<record> <keg dir under Cellar>]...
expect() {
	local name=$1 c=$2 want=$3 needle=$4 out rc ok=1 why=""
	shift 4
	out=$(PATH="$WORK/bin:$PATH" HOMEBREW_PREFIX="$c/prefix" RMLX_BOTTLE_STORE="$c/store" \
		bash "$c/repo/scripts/mlx_restore_pin.sh" 2>&1)
	rc=$?
	[ "$rc" -eq "$want" ] || { ok=0; why="exit $rc, wanted $want"; }
	grep -qF -- "$needle" <<<"$out" || { ok=0; why="$why; no line with: $needle"; }
	while [ "$#" -ge 2 ]; do
		local got expected
		got=$(resolves "$c" "$1")
		expected=$(cd "$c/prefix/Cellar/$2" 2>/dev/null && pwd -P)
		[ -n "$expected" ] && [ "$got" = "$expected" ] ||
			{ ok=0; why="$why; $1 -> '${got:-nothing}', wanted Cellar/$2"; }
		shift 2
	done
	if [ "$ok" = 1 ]; then
		pass=$((pass + 1))
		echo "  PASS  $name"
	else
		fail=$((fail + 1))
		echo "  FAIL  $name ($why)"
		sed 's/^/        /' <<<"$out"
	fi
}

LINKED=(opt/mlx "mlx/$MLX_V" opt/mlx-c "mlx-c/$MLXC_V"
	var/homebrew/linked/mlx "mlx/$MLX_V" var/homebrew/pinned/mlx "mlx/$MLX_V"
	var/homebrew/linked/mlx-c "mlx-c/$MLXC_V" var/homebrew/pinned/mlx-c "mlx-c/$MLXC_V")

# A drifted Mac: the pinned kegs are in the Cellar, a newer mlx keg is linked
# and pinned. The restore must take the exact keg, not the newest one.
c=$(new_case cellar)
keg "$c/prefix/Cellar" mlx "$MLX_V" yes
keg "$c/prefix/Cellar" mlx "$NEWER_V" yes
keg "$c/prefix/Cellar" mlx-c "$MLXC_V" yes
mkdir -p "$c/prefix/opt" "$c/prefix/var/homebrew/pinned"
ln -sfn "$c/prefix/Cellar/mlx/$NEWER_V" "$c/prefix/opt/mlx"
ln -sfn "$c/prefix/Cellar/mlx/$NEWER_V" "$c/prefix/var/homebrew/pinned/mlx"
expect "kegs in the Cellar: link and pin the exact pinned kegs" "$c" 0 "[preflight stub]" "${LINKED[@]}"

c=$(new_case store)
store_tar "$c" mlx-copy.tar.gz mlx "$MLX_V" yes
store_tar "$c" mlx-c-copy.tar.gz mlx-c "$MLXC_V" yes
expect "no keg, a durable copy: extract, link and pin it" "$c" 0 "[store] mlx $MLX_V" "${LINKED[@]}"

c=$(new_case store-by-content)
store_tar "$c" "mlx-$MLX_V.tar.gz" mlx "$NEWER_V" yes
store_tar "$c" other-name.tar.gz mlx "$MLX_V" yes
store_tar "$c" mlx-c-copy.tar.gz mlx-c "$MLXC_V" yes
expect "a copy is found by the keg it holds, not by its file name" "$c" 0 \
	"[store] mlx $MLX_V from other-name.tar.gz" "${LINKED[@]}"

c=$(new_case store-sha)
store_tar "$c" mlx-copy.tar.gz mlx "$MLX_V" yes
store_tar "$c" mlx-c-copy.tar.gz mlx-c "$MLXC_V" yes
sed -i '' 's/^./0/' "$c/store/source-built/SHA256SUMS"
expect "a copy whose sha256 disagrees with SHA256SUMS is refused" "$c" 1 "but SHA256SUMS lists"

c=$(new_case nothing)
expect "no keg and no copy: stop and name the source build" "$c" 1 "Building the pinned pair"

c=$(new_case no-nax)
keg "$c/prefix/Cellar" mlx "$MLX_V" no
keg "$c/prefix/Cellar" mlx-c "$MLXC_V" yes
expect "a keg without NAX kernels is refused" "$c" 1 "has no NAX GEMM kernels"

c=$(new_case store-no-nax)
store_tar "$c" mlx-copy.tar.gz mlx "$MLX_V" no
store_tar "$c" mlx-c-copy.tar.gz mlx-c "$MLXC_V" yes
expect "a durable copy without NAX kernels is refused" "$c" 1 "has no NAX GEMM kernels"

c=$(new_case incomplete)
mkdir -p "$c/prefix/Cellar/mlx/$MLX_V"
keg "$c/prefix/Cellar" mlx-c "$MLXC_V" yes
expect "an incomplete keg directory is not overwritten" "$c" 1 "has no lib/libmlx.dylib"

c=$(new_case brew-fails)
keg "$c/prefix/Cellar" mlx "$MLX_V" yes
keg "$c/prefix/Cellar" mlx-c "$MLXC_V" yes
touch "$c/prefix/brew.fail"
expect "a failed brew link is a failure" "$c" 1 "brew could not link and pin the pair"

echo "mlx_restore_pin_selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
