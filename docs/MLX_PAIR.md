# The MLX pair

Which MLX and mlx-c rMLX runs on, how to get that pair on each Mac, and what
protects a binary from the wrong one. How the build links the pair is in
[`docs/FFI.md`](FFI.md#pinned-mlx--mlx-c-pair).

## What to do on each Mac

The pinned pair is mlx 0.32.3 + mlx-c 0.7.0, built with the Neural-Accelerator
(NAX) kernels. MLX builds those kernels only for a deployment target of macOS
26.2 or later. Only M5 and later chips can use them.

| Mac | What to do |
|---|---|
| M1–M4, any macOS | Nothing for the kernels: `brew install mlx-c` gives a pair rMLX compiles against, and the pin does not bind. On that pair `rmlx healthcheck` reports `mlx_pin` info, and `rmlx baseline`, `rmlx bench` and `make mlx-preflight` run. They stop on an mlx-c C API mismatch, as on every Mac ([Two mlx-c C APIs](#two-mlx-c-c-apis)). |
| M5, macOS 27 or later | The Homebrew bottle for macOS 27 carries the kernels, so `brew upgrade mlx mlx-c` (or `brew reinstall mlx` at the same version) gives `rmlx serve` the NAX kernels. That bottle is not the pinned pair (today it is mlx 0.32.1 + mlx-c 0.6.0_4), so `rmlx baseline` and `rmlx bench` refuse it and `rmlx healthcheck` reports `mlx_pin` red. To measure, build the pinned pair from source ([below](#building-the-pinned-pair)). |
| M5, macOS 26.2 to 26.x | Build the pair from source ([below](#building-the-pinned-pair)). The macOS 26 bottle has no NAX kernels, and a plain `brew install --build-from-source mlx` has none either: the formula sets the deployment target to `26`. |
| M5, below macOS 26.2 | No fix. Update macOS. Until then `rmlx baseline` and `rmlx bench` refuse. |

The runtime warning in `crates/rmlx-mlx/src/nax.rs` names the kernel fix for
the host it runs on. To measure on M5 you also need the pinned pair (see the
macOS 27 row).

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
     tar czf "$store/mlx-c-0.7.0.tar.gz" mlx-c/0.7.0) &&
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

1. the Cellar, when the keg is there and usable;
2. the durable store (`~/.rmlx/bottles/source-built`, or
   `$RMLX_BOTTLE_STORE/source-built`): a tar of the keg directory, found by
   the keg it holds and checked against `SHA256SUMS`.

A usable keg holds the files rMLX builds against and loads, none of them
empty: `libmlx.dylib`, `libjaccl.dylib`, `mlx.metallib` and
`include/mlx/version.h` for mlx, `libmlxc.dylib` and `include/mlx/c/fast.h`
for mlx-c. The mlx keg must also carry the NAX kernels. The script extracts a
copy into a staging directory in the Cellar, checks it there, and moves it
into the Cellar only when it is usable. So a refused copy leaves no keg that a
later run, or Homebrew, can take as installed. One restore runs at a time: a
second run stops on the lock `Cellar/.rmlx-restore-lock`. A run stopped by
SIGKILL or a crash leaves that lock, and the refusal tells you how to remove
it. Staging that such a run left is removed at the start.

It stops when neither source has a usable keg. A refusal names the keg
directory in the Cellar, or the tar in the store and its line in
`SHA256SUMS`, for removal only when that keg or tar is bad: a file is missing
or empty, there are no NAX kernels, the tar does not read to its end, or its
sha256 is not the listed one. When `strings` cannot run, or the disk cannot
take the keg, the refusal names that cause and keeps the copy. It pours no
Homebrew bottle. When the link step fails, or
leaves a record on another keg, it prints what the `opt`, linked and pinned
records of both formulas resolve to, and tells you to run it again. After it, run
`cargo clean -p rmlx-mlx` and build again: a move to an older keg does not
re-run `build.rs`, so the crate would keep bindings from the wrong headers.
`make mlx-restore-pin-selftest` (in `make ci` and the hosted CI) is its
recall test.

## The pin gate

`crates/rmlx-mlx/src/pin.rs` reads the two dylibs dyld resolved for this
process, canonicalises them to their kegs, compares both versions with
`mlx-pin.txt`, scans the `mlx.metallib` beside `libmlx.dylib` for
`steel_gemm_fused_nax`, and reads the mlx-c C API verdict.
`linked_mlx_matches_the_pinned_pair` (`src/pin_tests.rs`) fails unless all of
that agrees.

The metallib scan is the load-bearing half: bottle contents vary by build
runner, so a version match does not prove the kernels are there. The version
check covers the ABI coupling.

The check is not in a build script. Cargo re-runs a build script only when a
`rerun-if-changed` path is *newer*, statting through symlinks. Repointing
`opt/mlx` at an older keg moves the mtime backwards, so cargo would replay a
stale verdict.

Every failure is its own verdict, so an inconclusive probe never reads as a
pass:

| Verdict | Means |
|---|---|
| `Match` | both kegs are the pinned pair, the metallib carries the kernels, and the C API matches |
| `NotLoaded` | dyld listed no such image; reported first |
| `CApiMismatch` | the loaded `libmlxc` has another C API than the compiled one; attention cannot run. It carries the verdict on the pair, which decides whether the restore is part of the fix |
| `KernelsMissing` | the metallib was read and has none |
| `VersionMismatch` | a keg version disagrees with the pin |
| `NotAKeg` | the resolved library is not in a keg, so it has no version |
| `KernelsUnverified` | the metallib could not be read |
| `PinUnparsable` | `mlx-pin.txt` declares no pair |

The pin grammar is parsed twice, because the preflight and the restore script
run before any binary exists: `parse_pin` (`src/pin.rs`) and
`scripts/lib/mlx_pin.sh`. `the_shell_pin_parser_agrees_with_the_rust_one`
holds them together. Versions must look like a keg directory name, because the
restore script interpolates them into Cellar paths and Ruby text.

**Scoped to Neural-Accelerator hosts** (Apple GPU family 10 and later, from
`rmlx_core::apple_gpu`). Earlier chips have no Neural Accelerator, so the
kernels buy nothing there. A host whose chip cannot be identified is held to
the pin, so that it cannot pass without a check, and
`the_gate_can_tell_which_host_it_is_on` fails there. The C API verdict is not
scoped: a mismatch fails on every host ([Two mlx-c C APIs](#two-mlx-c-c-apis)).

**It runs where numbers are made.** `PinCheck::refusal` is the one verdict,
and it names its cause: a C API mismatch, or a pair that is not the pinned
one. `rmlx baseline` and `rmlx bench` refuse when the loaded pair is not the
pinned one on a host the pin binds or a host that cannot be identified, and
on every host when the C API differs. `rmlx healthcheck` reports the same
verdict as an `mlx_pin` line: red where they refuse, info for another pair on
an identified M1–M4 host, green for the pinned pair.
`scripts/mlx_preflight.sh` (`make mlx-preflight`, run by `make canary`,
`canary-ab` and `bench-codec-cell`) stops on the same verdict.
With a built binary it reads only that binary's `mlx_pin` line: green or info
passes, anything else stops. Only the process taking the measurement knows
what dyld resolved, and `MLX_PREFIX` can load a pair that the `opt` symlinks
do not name. With no binary it reads the `opt` symlinks as a pre-filter for
the pair a build would load, and holds every host but an identified M1–M4 to
the pin, as the binary does. It cannot see the C API without a binary.
`make mlx-preflight-selftest` (in `make ci` and the hosted CI) is its recall
test.

### Run identity: `events.mlx_nax`

`rmlx_mlx::nax_capability()` returns `present`, `absent` or `unknown` from
the same runtime scan, once per process. `rmlx-cli`'s `main` forwards it to
`rmlx_metrics::identity::set_mlx_nax`, so every `events` row records whether
that run had the kernels. `unknown` means the metallib could not be
inspected. See `docs/METRICS_SCHEMA.md` §3.6.

mlx's version comes from `include/mlx/version.h`. mlx-c ships no version
header; its identity is the keg directory name, the only place the revision
suffix appears.

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
`mlx_compile_cache_new`, which the same upstream mlx-c commit adds. That tells
the 0.6 C API from the 0.7 one and nothing else: a later mlx-c that keeps the
function and changes another signature reads as 0.7. The verdict is taken
once per process, and a mismatch fails on every Mac, M1–M4 included:

- the SDPA wrapper returns `Err` naming both C APIs and does not call mlx-c,
  so any command fails at its first attention call;
- the one-shot MLX init logs the mismatch as an error;
- `rmlx healthcheck` reports `mlx_pin` red, `rmlx baseline` and `rmlx bench`
  refuse before a model loads, and `scripts/mlx_preflight.sh` stops.

The fix is a rebuild against the loaded mlx-c (`cargo clean -p rmlx-mlx`, then
build; `brew reinstall rmlx` for a Homebrew install), or the mlx-c the binary
was built against, which is the only fix for a release tarball. The refusals
name that fix. A restore does not change the binary, so they name
`make mlx-restore-pin` only where a measurement also needs the pinned pair: on
a host that requires it, when the loaded pair is not the pinned one. There a
rebuild against the loaded mlx-c gives a binary that is refused again, for
the pair. Restore first, then rebuild if the binary was not built against the
pinned pair.

On a matched pair the compiled and the loaded C API agree, so no test there
can see a probe that always answers the compiled one. A cross-pair run can:
build against one pair, then run with `DYLD_LIBRARY_PATH` set to the `lib`
directories of both kegs of the other pair (set it on the command itself;
macOS drops `DYLD_*` at a protected binary such as `/usr/bin/env`).
Measured on an M5 Max, macOS 26.6, with a binary built against mlx 0.32.1 +
mlx-c 0.6.0_4 run on the pinned mlx 0.32.3 + mlx-c 0.7.0, and the reverse:

| Check | 0.6 binary on the pinned pair | 0.7 binary on 0.32.1 + 0.6.0_4 |
|---|---|---|
| `rmlx healthcheck --human` | `mlx_pin: RED — mlx-c C API mismatch: this binary was compiled against the mlx-c 0.6 C API, but the loaded libmlxc.dylib has the mlx-c 0.7 C API. …`, naming the rebuild | the same with the C APIs swapped, then `The loaded pair is not the pinned one either: … run make mlx-restore-pin, then rebuild if …` |
| `rmlx baseline --model <missing path>` | exit 1, `Error: rmlx baseline refuses to run: mlx-c C API mismatch: …` | exit 1, the same with the restore sentence |
| `scripts/mlx_preflight.sh`, the binary as `target/release-perf/rmlx` | exit 1 on that `mlx_pin` line, no `make mlx-restore-pin` | exit 1, and `Restore the nax-capable pair:  make mlx-restore-pin` |
| `the_loaded_c_api_is_the_compiled_one` | fails | fails |
| `scaled_dot_product_attention_on_cpu_follows_the_c_api_verdict` | passes through its mismatch arm | passes through its mismatch arm |

On a matched pair the same `rmlx baseline` fails on the model path.

With `loaded()` changed to return the compiled C API, every test passes on a
matched pair. Across pairs the CPU SDPA test then fails: the 0.6 binary on the
0.7 library gets `force_fused=True but no fused kernel is available`. With
`pin_check` changed to report a matching C API,
`the_pin_check_carries_the_c_api_verdict` passes on a matched pair and fails
across pairs.

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
