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
class Rmlx < Formula
  desc "Rust-native, single-binary MLX inference + conversion backend for Apple Silicon"
  homepage "https://github.com/Pushkinist/rMLX"
  url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.2.5.tar.gz"
  sha256 "0b37ed5856efe7274ce0c6af27309c89740f732dc492cb797242be67558ea3a8"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/Pushkinist/rMLX.git", branch: "main"

  bottle do
    root_url "https://github.com/Pushkinist/rMLX/releases/download/v0.2.5"
    sha256 cellar: :any, arm64_tahoe: "5dfc6d98c581a9dcebe7d84f91d489d59dc1404603347447c83622d1a8704650"
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
