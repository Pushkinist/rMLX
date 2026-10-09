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
# rMLX follows Homebrew's MLX: it uses the mlx / mlx-c that Homebrew installs,
# on every Mac. The pair in crates/rmlx-mlx/mlx-pin.txt is a development and
# measurement pin, not a product requirement. It exists because MLX builds its
# Neural-Accelerator kernels only for a macOS 26.2 deployment target, so the
# Homebrew bottle for macOS 26 has none. Only M5 and later hardware, the
# generation that has a GPU Neural Accelerator, can use those kernels. On
# M1-M4 the Homebrew MLX is entirely correct.
#
# What ships to users instead is a runtime check: rmlx probes the mlx.metallib
# of the library it actually loaded and warns on startup only when the host has
# a Neural Accelerator and the kernels are missing. That stays true after a
# `brew upgrade mlx` moves the symlink underneath an already-installed rmlx,
# which no version constraint here could. A second check reads the C API of
# the loaded mlx-c and names `brew reinstall rmlx` when it is not the one this
# build compiled against. See crates/rmlx-mlx/src/nax.rs and docs/MLX_PAIR.md.
# There is no `caveats` block for the same reason: it would print for every
# user on every Mac, and the runtime warning reaches exactly the hosts the
# finding applies to.
class Rmlx < Formula
  desc "Rust-native, single-binary MLX inference + conversion backend for Apple Silicon"
  homepage "https://github.com/Pushkinist/rMLX"
  url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.4.2.tar.gz"
  sha256 "4ffd415cb488767aac9b4b033f4522c72ed844dea54e1a82c6a6f79d6ed01377"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/Pushkinist/rMLX.git", branch: "main"

  depends_on "rust" => :build
  depends_on arch: :arm64
  depends_on :macos
  depends_on "mlx-c"

  def install
    # build.rs needs BOTH prefixes; mlx-c pulls mlx transitively.
    ENV["MLX_C_PREFIX"] = Formula["mlx-c"].opt_prefix
    ENV["MLX_PREFIX"] = Formula["mlx"].opt_prefix
    # A bottle runs on Macs other than the one that built it, so it takes the
    # release tarball's flags (apple-m1) instead of config.toml's target-cpu=native.
    if build.bottle?
      ENV["CARGO_ENCODED_RUSTFLAGS"] =
        Utils.safe_popen_read("python3", "scripts/release/release_cpu.py", "rustflags")
    end
    system "cargo", "install", *std_cargo_args(path: "crates/rmlx-cli")
  end

  test do
    assert_match "rmlx", shell_output("#{bin}/rmlx --version")
  end
end
