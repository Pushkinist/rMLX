# The MLX pair

Which MLX and mlx-c rMLX runs on, how to get that pair on each Mac, and what
protects a binary from the wrong one. The pin gate itself is in
[`docs/FFI.md`](FFI.md#pinned-mlx--mlx-c-pair).

## What to do on each Mac

The pinned pair is mlx 0.32.3 + mlx-c 0.7.0, built with the Neural-Accelerator
(NAX) kernels. MLX builds those kernels only for a deployment target of macOS
26.2 or later. Only M5 and later chips can use them.

| Mac | What to do |
|---|---|
| M1–M4, any macOS | Nothing. `brew install mlx-c` gives a pair rMLX compiles against; the pin does not bind. |
| M5, macOS 27 or later | The Homebrew bottle for macOS 27 carries the kernels: `brew upgrade mlx mlx-c`, or `brew reinstall mlx` at the same version. |
| M5, macOS 26.2 to 26.x | Build the pair from source ([below](#building-the-pinned-pair)). The macOS 26 bottle has no NAX kernels, and a plain `brew install --build-from-source mlx` has none either: the formula sets the deployment target to `26`. |
| M5, below macOS 26.2 | No fix. Update macOS. |

The runtime warning in `crates/rmlx-mlx/src/nax.rs` names the same fix for the
host it runs on.

## Building the pinned pair

This edits the two formulas in a local `homebrew/core` tap and builds them.
rMLX ships no formula of its own.

1. Install the Metal toolchain (Xcode 16.3 and later ship it separately):

   ```sh
   xcodebuild -downloadComponent MetalToolchain
   ```

2. Edit `$(brew --repo homebrew/core)/Formula/m/mlx.rb`:
   - `url` → `https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.3.tar.gz`,
     `sha256` → `4129039ddcb36cb860982b616c975a5b9b8e6d8fa97efa86335b4427c29c4b4b`;
   - delete the `stable do` wrapper, its nanobind backport `patch :DATA` and the
     `__END__` patch (mlx 0.32.3 carries that fix), and the `bottle do` block;
   - in `install`, set `ENV["MACOSX_DEPLOYMENT_TARGET"] = "26.2"`. This line
     turns the NAX kernels on.

3. Edit `Formula/m/mlx-c.rb`:
   - `url` → `https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.7.0.tar.gz`,
     `sha256` → `ee726bb38e191bb3c516a6bae47dc8abad9e5f273873385839019ce46bfceab5`;
   - delete `revision`, the `bottle do` block and the four backport `patch do`
     blocks (v0.7.0 contains them);
   - replace the `__END__` patch with this one. mlx 0.32.3 adds `global_scale`
     in front of `sorted_indices` in `gather_qmm`, and mlx-c 0.7.0 still passes
     the old argument list. `std::nullopt` is the upstream default; the C API of
     `mlx_gather_qmm` does not change.

     ```diff
     --- a/mlx/c/ops.cpp
     +++ b/mlx/c/ops.cpp
     @@ -1708,6 +1708,7 @@
                  (bits.has_value ? std::make_optional<int>(bits.value)
                                  : std::nullopt),
                  std::string(mode),
     +            std::nullopt,
                  sorted_indices,
                  mlx_stream_get_(s)));
        } catch (std::exception& e) {
     ```

4. Build, from a directory outside `~/Documents` (the sandboxed relocation step
   fails there), mlx first:

   ```sh
   export HOMEBREW_NO_INSTALL_FROM_API=1 HOMEBREW_NO_AUTO_UPDATE=1
   brew unpin mlx mlx-c
   brew install --build-from-source mlx      # log: "Building for macOS 26.2"
   brew install --build-from-source mlx-c
   ```

   mlx-c must compile against the new mlx: before the second install, check
   that `readlink "$(brew --prefix)/opt/mlx"` names `0.32.3`.

5. Link and pin exactly these kegs ([why](#brew-link-and-brew-pin-take-the-newest-keg)),
   then check the kernels:

   ```sh
   make mlx-restore-pin
   strings "$(brew --prefix mlx)/lib/mlx.metallib" | grep -c steel_gemm_fused_nax   # non-zero
   ```

6. Keep a durable copy, which `make mlx-restore-pin` reads when a keg is gone
   from the Cellar:

   ```sh
   store=~/.rmlx/bottles/source-built; mkdir -p "$store"
   (cd "$(brew --cellar)" && tar czf "$store/mlx-0.32.3.tar.gz" mlx/0.32.3 &&
     tar czf "$store/mlx-c-0.7.0.tar.gz" mlx-c/0.7.0)
   (cd "$store" && shasum -a 256 *.tar.gz > SHA256SUMS)
   ```

The kegs link `libjaccl.dylib` through `@rpath`, so it must stay beside
`libmlx.dylib`. `libmlx` of this build has `minos 26.2`; a link from a lower
target prints an `ld` warning, not an error.

## `brew link` and `brew pin` take the newest keg

In Homebrew 7, `brew pin <f>` pins the newest keg in the Cellar
(`formula_pin.rb`, `def pin`), and `brew link <f>` links the newest keg
(`cmd/link.rb` resolves the name to the latest keg). Neither acts on the keg
that is linked now, and neither can go back to an older keg. So they give the
pinned pair only while it is the newest pair in the Cellar.

`make mlx-restore-pin` does not depend on that. It calls Homebrew's own
`Keg#link` and `FormulaPin#pin_at` on the exact kegs `mlx-pin.txt` names, then
checks that the `opt`, linked and pinned records all resolve to them.

## Restoring the pair: `make mlx-restore-pin`

`scripts/mlx_restore_pin.sh` takes each pinned keg from:

1. the Cellar, when the keg is still there;
2. the durable store (`~/.rmlx/bottles/source-built`, or
   `$RMLX_BOTTLE_STORE/source-built`): a tar of the keg directory, found by
   the keg it holds and checked against `SHA256SUMS`.

It stops when neither has the keg, when a keg directory is incomplete, and
when the mlx keg has no NAX kernels. It pours no Homebrew bottle. After it,
run `cargo clean -p rmlx-mlx` and build again. `make mlx-restore-pin-selftest`
is its recall test.

## Two mlx-c C APIs

rMLX compiles against the mlx-c 0.6 and the 0.7 C API. Of the functions it
calls, one differs: 0.7 adds `bool force_fused` in front of the stream of
`mlx_fast_scaled_dot_product_attention`. `build.rs` classifies the generated
bindings (9 parameters and no `mlx_compile_cache_new`, or 10 parameters and
that function; anything else stops the build) and sets `cfg(mlxc_c_api_0_7)`
for the call. rMLX passes `force_fused = false`, which keeps MLX's own
routing.

The symbol name is the same in both, so dyld links a binary built against
one C API to a `libmlxc.dylib` of the other with no error. This happens when
Homebrew moves `mlx-c` under an installed `rmlx`, and to a release tarball on
a Mac with the other `mlx-c`. The call then reads its stream from the wrong
argument slot: measured, a 0.7 binary on the 0.6 library fails with
`expected a non-empty mlx_stream`, and a 0.6 binary on the 0.7 library reads
`force_fused = true`.

`src/c_api.rs` finds the C API of the loaded library from whether it exports
`mlx_compile_cache_new`, which the same upstream mlx-c commit adds. The
verdict is taken once per process:

- the SDPA wrapper returns `Err` naming both C APIs and does not call mlx-c;
- the one-shot MLX init logs the mismatch as an error;
- the pin gate reads `CApiMismatch`, so `rmlx healthcheck`, the preflight,
  `rmlx baseline` and `rmlx bench` refuse before a model loads.

The fix is a rebuild against the loaded mlx-c (`cargo clean -p rmlx-mlx`,
then build).

## Moving the pin

1. Build the new pair ([above](#building-the-pinned-pair)) and check the NAX
   kernel count.
2. Bump both lines of `crates/rmlx-mlx/mlx-pin.txt` to the keg directory
   names (`brew list --versions mlx mlx-c`).
3. `cargo clean -p rmlx-mlx` and build. `cargo build -p rmlx-mlx -vv` shows
   whether `build.rs` ran. A new mlx-c C API stops the build in `build.rs`
   until [Two mlx-c C APIs](#two-mlx-c-c-apis) knows it.
4. Re-derive the evaluation-lock reach set (`scripts/check_eval_lock.sh`
   header), re-run `layout_flag_classifies_views_once_they_are_evaluated` and
   the JIT language probe (`docs/FFI.md`), and derive the GPU census again
   (`docs/GPU_TESTS.md`).
5. Re-run a prefill cell and compare its prefill rate with the old pair's.
