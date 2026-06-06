#!/usr/bin/env bash
# rMLX installer — builds from source and links the system MLX / mlx-c libraries.
#
# Quick install:
#   curl -fsSL https://raw.githubusercontent.com/Pushkinist/rMLX/main/install.sh | bash
#
# Safer (inspect before running — recommended for any curl | bash):
#   curl -fsSL https://raw.githubusercontent.com/Pushkinist/rMLX/main/install.sh -o install.sh
#   less install.sh
#   bash install.sh
#
# Env knobs:
#   RMLX_REF  git ref to build (default: main)
set -euo pipefail

REPO="Pushkinist/rMLX"
REF="${RMLX_REF:-main}"

info() { printf '==> %s\n' "$1"; }
err()  { printf 'error: %s\n' "$1" >&2; exit 1; }

# 1. Platform — Apple Silicon macOS only (Metal-only backend).
[[ "$(uname -s)" == "Darwin" ]] || err "rMLX is macOS only."
[[ "$(uname -m)" == "arm64"  ]] || err "rMLX is Apple Silicon (arm64) only."

# 2. Homebrew + MLX. rMLX links libmlxc.dylib at runtime; MLX must be present.
command -v brew >/dev/null 2>&1 || err "Homebrew is required — see https://brew.sh"
if ! brew list mlx-c >/dev/null 2>&1; then
  info "Installing MLX via Homebrew (provides mlx-c)..."
  brew install mlx-c
fi
MLX_C_PREFIX="$(brew --prefix mlx-c)"
export MLX_C_PREFIX
[[ -f "$MLX_C_PREFIX/lib/libmlxc.dylib" ]] \
  || err "libmlxc.dylib not found under $MLX_C_PREFIX/lib — set MLX_C_PREFIX to your MLX install."

# 3. Rust toolchain.
if ! command -v cargo >/dev/null 2>&1; then
  info "Installing Rust via rustup..."
  curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi

# 4. Build + install from source. The repo's .cargo/config.toml targets the
#    building machine's own chip (target-cpu=native), so this is optimal for
#    whatever Mac runs it. Compiling the workspace takes a few minutes.
info "Building rMLX ($REF) from source — this compiles the full workspace..."
# No --root: cargo installs to $CARGO_HOME/bin (~/.cargo/bin), which rustup
# already puts on PATH, so `rmlx` is usable immediately with no PATH edit.
cargo install \
  --git "https://github.com/${REPO}" \
  --branch "$REF" \
  --bin rmlx \
  rmlx-cli

CARGO_BIN="${CARGO_HOME:-$HOME/.cargo}/bin"
info "Installed: ${CARGO_BIN}/rmlx"
case ":$PATH:" in
  *":${CARGO_BIN}:"*) ;;
  *) info "Add to your PATH (rustup normally does this):  export PATH=\"${CARGO_BIN}:\$PATH\"" ;;
esac
if command -v rmlx >/dev/null 2>&1; then rmlx --version; else "${CARGO_BIN}/rmlx" --version; fi || true
