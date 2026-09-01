#!/usr/bin/env bash
# Restore the nax-capable MLX pair named by crates/rmlx-mlx/mlx-pin.txt.
#
# Needed because `brew pin` does NOT protect a keg from `brew cleanup`: the
# pinned versions can disappear entirely, leaving only the nax-less 0.32.0
# bottle (unlinked), at which point `rmlx` cannot launch at all.
#
# Sources the bottles from ~/.rmlx/bottles first (durable local copy), falling
# back to ghcr. Pours alongside whatever else is in the Cellar, repoints the
# `opt` symlinks, relocates install names, and re-signs.
#
# Verify afterwards with scripts/mlx_preflight.sh.
# Background: .rmlx/mlx-homebrew-nax-regression.md

set -uo pipefail

PREFIX="${HOMEBREW_PREFIX:-/opt/homebrew}"
CELLAR="$PREFIX/Cellar"
STORE="${RMLX_BOTTLE_STORE:-$HOME/.rmlx/bottles}"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

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
MLX_VER="$PIN_MLX"
MLXC_VER="$PIN_MLXC"
# sha256 from the homebrew-core *bottle-update* commits (cbfd9632d44 / b2763d78a34),
# for the pair the pin currently names. Moving the pin without moving these makes
# the extracted version directory disagree with the requested one, which the
# check below turns into a hard failure rather than a wrong restore.
# NOTE: the version-bump commit still carries the PREVIOUS release's hashes —
# taking the sha from there silently yields mlx 0.31.1.
MLX_SHA="def8a7ae1e6a6506eed4dea45bf52b55be0f52f8364f8a928da6e65b1204a371"
MLXC_SHA="b30c755158db3f9d9090b69a1a74f3e05ae8e7a3969695d9f0974f1bc1d3df1b"

die() {
	echo "restore FAIL: $*" >&2
	exit 1
}

obtain() { # name version sha -> stages <name>.tar.gz
	local name=$1 ver=$2 sha=$3
	local out="$STAGE/$name.tar.gz"
	local local_copy="$STORE/$name--$ver.arm64_tahoe.bottle.tar.gz"

	if [ -f "$local_copy" ]; then
		echo "[local] $name $ver from $STORE"
		cp "$local_copy" "$out"
	else
		echo "[fetch] $name $ver from ghcr"
		curl -sSL -H "Authorization: Bearer QQ==" \
			"https://ghcr.io/v2/homebrew/core/$name/blobs/sha256:$sha" -o "$out" ||
			die "download failed for $name"
	fi

	local got
	got=$(shasum -a 256 "$out" | awk '{print $1}')
	[ "$got" = "$sha" ] || die "$name sha mismatch: got $got want $sha"
	echo "[ok] $name sha verified"
}

obtain mlx "$MLX_VER" "$MLX_SHA"
obtain mlx-c "$MLXC_VER" "$MLXC_SHA"

tar -xzf "$STAGE/mlx.tar.gz" -C "$STAGE" || die "extract mlx"
tar -xzf "$STAGE/mlx-c.tar.gz" -C "$STAGE" || die "extract mlx-c"

# The version directory inside the tarball is the ground truth — a correct
# checksum does not prove it is the version you asked for.
[ -d "$STAGE/mlx/$MLX_VER" ] || die "bottle is not mlx $MLX_VER (got: $(ls "$STAGE/mlx"))"
[ -d "$STAGE/mlx-c/$MLXC_VER" ] || die "bottle is not mlx-c $MLXC_VER (got: $(ls "$STAGE/mlx-c"))"

staged_nax=$(strings "$STAGE/mlx/$MLX_VER/lib/mlx.metallib" | grep -c steel_gemm_fused_nax)
[ "$staged_nax" -ge 1 ] || die "staged mlx $MLX_VER has $staged_nax nax kernels — wrong bottle"
echo "[ok] staged mlx has $staged_nax nax GEMM kernel occurrences"

echo "[install] pouring into Cellar (other versions left intact)"
rm -rf "${CELLAR:?}/mlx/$MLX_VER" "${CELLAR:?}/mlx-c/$MLXC_VER"
mkdir -p "$CELLAR/mlx" "$CELLAR/mlx-c"
cp -R "$STAGE/mlx/$MLX_VER" "$CELLAR/mlx/$MLX_VER" || die "install mlx"
cp -R "$STAGE/mlx-c/$MLXC_VER" "$CELLAR/mlx-c/$MLXC_VER" || die "install mlx-c"

echo "[link] repointing opt symlinks"
ln -sfn "$CELLAR/mlx/$MLX_VER" "$PREFIX/opt/mlx"
ln -sfn "$CELLAR/mlx-c/$MLXC_VER" "$PREFIX/opt/mlx-c"

# Homebrew rewrites @@HOMEBREW_PREFIX@@ at install time; a hand-poured bottle
# keeps the placeholders and dyld cannot resolve them.
echo "[relocate] rewriting install names"
for lib in "$CELLAR/mlx/$MLX_VER/lib/libmlx.dylib" \
	"$CELLAR/mlx/$MLX_VER/lib/libjaccl.dylib" \
	"$CELLAR/mlx-c/$MLXC_VER/lib/libmlxc.dylib"; do
	[ -f "$lib" ] || continue
	old_id=$(otool -D "$lib" | tail -1)
	install_name_tool -id "${old_id//@@HOMEBREW_PREFIX@@/$PREFIX}" "$lib" 2>/dev/null
	otool -L "$lib" | tail -n +2 | awk '{print $1}' | grep '@@HOMEBREW_PREFIX@@' |
		while read -r dep; do
			install_name_tool -change "$dep" "${dep//@@HOMEBREW_PREFIX@@/$PREFIX}" "$lib" 2>/dev/null
		done
	# install_name_tool invalidates the signature on arm64.
	codesign --force --sign - --preserve-metadata=entitlements,flags "$lib" 2>/dev/null
done

for script in "$CELLAR/mlx/$MLX_VER"/bin/*; do
	[ -f "$script" ] && sed -i '' "s|@@HOMEBREW_PREFIX@@|$PREFIX|g" "$script" 2>/dev/null
done

echo "[cache] keeping a durable copy in $STORE"
mkdir -p "$STORE"
cp -n "$STAGE/mlx.tar.gz" "$STORE/mlx--$MLX_VER.arm64_tahoe.bottle.tar.gz" 2>/dev/null
cp -n "$STAGE/mlx-c.tar.gz" "$STORE/mlx-c--$MLXC_VER.arm64_tahoe.bottle.tar.gz" 2>/dev/null

echo "[done] rebuild rmlx so it links the restored pair:  touch crates/rmlx-mlx/build.rs && make build-perf"
exec "$(dirname "$0")/mlx_preflight.sh"
