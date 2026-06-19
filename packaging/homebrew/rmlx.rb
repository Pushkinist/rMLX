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
  url "https://github.com/Pushkinist/rMLX/archive/refs/tags/v0.2.4.tar.gz"
  sha256 "37577deaf4d00e1a263386ba6ad46929dc71f3b96227d7e57c605fcea9e3b603"
  license any_of: ["MIT", "Apache-2.0"]
  head "https://github.com/Pushkinist/rMLX.git", branch: "main"

  bottle do
    root_url "https://github.com/Pushkinist/rMLX/releases/download/v0.2.4"
    sha256 cellar: :any, arm64_tahoe: "a017a84871b83643d3132685158937a8245b9654d80245e521476b0b213f4abf"
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
