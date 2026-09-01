#!/usr/bin/env bash
# Read the validated MLX / mlx-c pair from its one declaration.
#
#     REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"   # adjust depth per script
#     source "$REPO_ROOT/scripts/lib/mlx_pin.sh"
#     # -> PIN_MLX, PIN_MLXC, both non-empty; exits 1 with a reason otherwise.
#
# The grammar here is the same one `parse_pin` (crates/rmlx-mlx/src/pin.rs)
# applies, and `the_shell_pin_parser_agrees_with_the_rust_one` holds the two
# together. A file the Rust gate calls unusable must not be one these scripts
# quietly proceed on.
#
# The version shape is an allowlist, not a sanity check: these values are
# interpolated into `rm -rf`, `cp -R` and `ln -sfn` targets under the Cellar,
# so a value like `..` would delete or repoint far more than a keg.

# One keg version. Anchored, and the first character must be alphanumeric so
# `.`, `..` and anything that reads as an option are rejected outright.
MLX_PIN_VERSION_RE='^[0-9A-Za-z][0-9A-Za-z._+-]*$'

# Parse a pin file. Prints `<mlx>\n<mlx-c>` and exits 0, or exits 1 silently.
#
# Rejects the whole file — never a partial read — on: a line that is not a
# comment and not exactly `<formula> <version>`, an unknown formula, a repeated
# formula, or a version outside the allowlist. The pair is the unit that was
# validated, so half of it is not a usable answer.
mlx_pin_parse() {
	awk -v re="$MLX_PIN_VERSION_RE" '
		NF == 0                        { next }
		substr($1, 1, 1) == "#"        { next }
		NF != 2                        { bad = 1; exit }
		$1 != "mlx" && $1 != "mlx-c"   { bad = 1; exit }
		$2 !~ re                       { bad = 1; exit }
		$1 == "mlx"                    { if (m++) { bad = 1; exit } mlx = $2; next }
		                               { if (c++) { bad = 1; exit } mlxc = $2 }
		END {
			if (bad || m != 1 || c != 1) { exit 1 }
			print mlx
			print mlxc
		}
	' "$1"
}

mlx_pin_load() { # <pin-file> -> sets PIN_MLX / PIN_MLXC
	local file=$1 parsed
	if [ ! -f "$file" ]; then
		echo "FAIL: no MLX pin at $file" >&2
		return 1
	fi
	if ! parsed=$(mlx_pin_parse "$file"); then
		echo "FAIL: $file must declare exactly one 'mlx <version>' and one" \
			"'mlx-c <version>' line, comments aside, each version matching" \
			"$MLX_PIN_VERSION_RE" >&2
		return 1
	fi
	PIN_MLX=$(printf '%s\n' "$parsed" | sed -n 1p)
	PIN_MLXC=$(printf '%s\n' "$parsed" | sed -n 2p)
	# Belt and braces: these are about to be interpolated into paths that get
	# removed and overwritten, and a parser change upstream must not be able to
	# reach that without also passing here.
	if ! [[ "$PIN_MLX" =~ $MLX_PIN_VERSION_RE ]] || ! [[ "$PIN_MLXC" =~ $MLX_PIN_VERSION_RE ]]; then
		echo "FAIL: $file yielded a version outside $MLX_PIN_VERSION_RE" >&2
		return 1
	fi
	return 0
}
