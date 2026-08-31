# typed: strict
# frozen_string_literal: true

# Homebrew formula for rMLX.
#
# This file is the source-of-truth; the published tap lives in a separate repo
# so users can `brew tap`. To set up the tap:
#
#   1. Create repo github.com/Pushkinist/homebrew-rmlx
#   2. Copy this file to that repo as Formula/rmlx.rb
#   3. After tagging v0.1.0, fill in the sha256 below:
#        curl -fsSL https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.1.0.tar.gz | shasum -a 256
#
# Then end users install with:
#
#   brew tap Pushkinist/rmlx
#   brew install rmlx
#
# Builds from source (depends_on "rust" => :build) and links the Homebrew MLX
# (depends_on "mlx-c"), so the mlx-c rpath always matches the user's MLX install.
#
# The `mlx-c` dependency is deliberately unversioned, and must stay that way.
# rMLX pins one validated mlx / mlx-c pair for development
# (crates/rmlx-mlx/mlx-pin.txt), but that pin exists for a bottle regression
# that only costs anything on M5-and-later hardware — the generation that has a
# GPU Neural Accelerator. On M1-M4 the same MLX is entirely correct, and
# requiring an older release there would force a downgrade for a benefit that
# hardware cannot use. The pin is a this-machine, this-generation workaround,
# not a product requirement.
#
# What ships to users instead is a runtime check: rmlx probes the mlx.metallib
# of the library it actually loaded and warns on startup only when the host has
# a Neural Accelerator and the kernels are missing. That stays true after a
# `brew upgrade mlx` moves the symlink underneath an already-installed rmlx,
# which no version constraint here could. See crates/rmlx-mlx/src/nax.rs and
# docs/FFI.md. There is no `caveats` block for the same reason: it would print
# for every user on every Mac, and the runtime warning reaches exactly the
# hosts the finding applies to.
class Rmlx < Formula
  desc "Rust-native, single-binary MLX inference + conversion backend for Apple Silicon"
  homepage "https://github.com/Pushkinist/rMLX"
  url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.4.0.tar.gz"
  sha256 "55d13c77b62d20d901c5fddec8c319216ba4e9c7770c470aaa732623fb3dea03"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/Pushkinist/rMLX.git", branch: "main"

  bottle do
    root_url "https://github.com/Pushkinist/rMLX/releases/download/v0.3.0"
    sha256 cellar: :any, arm64_tahoe: "4bd5ad4f87cdac86646e2da43015b1fd9b376ada0516ed6bca79d47dd7ac3aa7"
  end

  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "mlx-c"

  def install
    # build.rs needs BOTH prefixes; mlx-c pulls mlx transitively.
    ENV["MLX_C_PREFIX"] = Formula["mlx-c"].opt_prefix
    ENV["MLX_PREFIX"] = Formula["mlx"].opt_prefix
    system "cargo", "install", *std_cargo_args(path: "crates/rmlx-cli")
  end

  test do
    assert_match "rmlx", shell_output("#{bin}/rmlx --version")
  end
end
