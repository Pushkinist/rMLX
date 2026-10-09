# The MLX pair

Which MLX and mlx-c rMLX runs on, how to get that pair on each Mac, and what
protects a binary from the wrong one. How the build links the pair is in
[`docs/FFI.md`](FFI.md#pinned-mlx--mlx-c-pair).

## What to do on each Mac

rMLX uses the `mlx` and `mlx-c` that Homebrew installs, on every Mac. Its
formula depends on `mlx-c` with no version, and rMLX ships no MLX formula of
its own. `brew info mlx mlx-c` shows what Homebrew ships.

`crates/rmlx-mlx/mlx-pin.txt` names the pinned pair: the pair rMLX is
developed and measured on, built with the Neural-Accelerator (NAX) kernels.
The pin does not limit `rmlx serve`. MLX builds the NAX kernels only for a
deployment target of macOS 26.2 or later, and the Homebrew formula builds each
bottle for the macOS of that bottle. Only M5 and later chips can use the
kernels. Without them on such a chip, prefill is slower; decode and the output
do not change.

| Mac | What to do |
|---|---|
| M1–M4, any macOS | Nothing for the kernels: `brew install mlx-c` gives a pair rMLX compiles against, and the pin does not bind. On that pair `rmlx healthcheck` reports `mlx_pin` info, and `rmlx baseline`, `rmlx bench` and `make mlx-preflight` run. They stop on an mlx-c C API mismatch, as on every Mac ([Two mlx-c C APIs](#two-mlx-c-c-apis)). |
| M5 and later, macOS 27 or later | The Homebrew bottle for macOS 27 carries the kernels, so `brew upgrade mlx mlx-c` (or `brew reinstall mlx` at the same version) gives `rmlx serve` the NAX kernels. While Homebrew ships the versions that `mlx-pin.txt` names, that bottle also passes the pin gate. When Homebrew ships another version, `rmlx baseline` and `rmlx bench` refuse it and `rmlx healthcheck` reports `mlx_pin` red. To measure then, build the pinned pair from source ([below](#building-the-pinned-pair)). |
| M5 and later, macOS 26.2 to 26.x | `rmlx serve` runs on the Homebrew bottle, with slower prefill and the startup warning. For the kernels, build the pair from source ([below](#building-the-pinned-pair)). The macOS 26 bottle has no NAX kernels, and a plain `brew install --build-from-source mlx` has none either: the formula sets the deployment target to `26`. `rmlx baseline` and `rmlx bench` refuse the bottle. |
| M5 and later, below macOS 26.2 | No fix. Update macOS. Until then `rmlx serve` runs with slower prefill and the warning, and `rmlx baseline` and `rmlx bench` refuse. |

The runtime warning in `crates/rmlx-mlx/src/nax.rs` fires only on an M5 or
later chip whose loaded MLX has no NAX kernels, and names the kernel fix for
the macOS it runs on.

## Building the pinned pair

This edits the two formulas in a local `homebrew/core` tap and builds them.
The versions and checksums below are those of the pin. The steps were written
against the formulas Homebrew ships for those versions; `brew info mlx mlx-c`
shows whether Homebrew is still there.

1. Install the Metal toolchain (Xcode 16.3 and later ship it separately):

   ```sh
   xcodebuild -downloadComponent MetalToolchain
   ```

2. Edit `$(brew --repo homebrew/core)/Formula/m/mlx.rb`:
   - `url` must be `https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.3.tar.gz`,
     and `sha256` `4129039ddcb36cb860982b616c975a5b9b8e6d8fa97efa86335b4427c29c4b4b`;
   - delete the `bottle do` block;
   - delete the `stable do` wrapper around `url` and `sha256`, with its
     `patch :DATA` and the `__END__` patch. That patch is a compile fix for
     macOS 27, and the pinned kegs were built without it;
   - in `install`, set `ENV["MACOSX_DEPLOYMENT_TARGET"] = "26.2"`. This line
     turns the NAX kernels on.

3. Edit `Formula/m/mlx-c.rb`:
   - `url` must be `https://github.com/ml-explore/mlx-c/archive/refs/tags/v0.7.0.tar.gz`,
     and `sha256` `ee726bb38e191bb3c516a6bae47dc8abad9e5f273873385839019ce46bfceab5`;
   - delete the `bottle do` block;
   - keep the `gather_qmm` patch (`patch do`, mlx-c commit `cfc471f`). mlx
     0.32.3 adds `global_scale` in front of `sorted_indices` in `gather_qmm`,
     and mlx-c 0.7.0 passes the old argument list. The patch passes
     `std::nullopt`, the upstream default, in that slot; the C API of
     `mlx_gather_qmm` does not change. The pinned kegs were built with that
     one line as a local patch, without the version guard the commit adds.

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

   `make mlx-restore-pin` needs a checkout of rMLX. Without one,
   `brew pin mlx mlx-c` pins these kegs while they are the newest in the
   Cellar.

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
| `rmlx healthcheck --human` | `mlx_pin: RED — mlx-c C API mismatch: this binary was compiled against the mlx-c 0.6 C API, but the loaded libmlxc.dylib has the mlx-c 0.7 C API. …`, naming the rebuild | the same with the C APIs swapped, then `On this host a rebuild against the loaded mlx-c is refused for the pair: the loaded pair does not pass the pin either (…). Run make mlx-restore-pin first, then rebuild if …` |
| `rmlx baseline --model <missing path>` | exit 1, `Error: rmlx baseline refuses to run: mlx-c C API mismatch: …` | exit 1, the same with the restore sentence |
| `scripts/mlx_preflight.sh`, the binary as `target/release-perf/rmlx` | exit 1 on that `mlx_pin` line, no `make mlx-restore-pin` | exit 1, and `Restore the nax-capable pair:  make mlx-restore-pin` |
| `the_loaded_c_api_is_the_compiled_one` | fails | fails |
| `scaled_dot_product_attention_on_cpu_follows_the_c_api_verdict` | passes through its mismatch arm | passes through its mismatch arm |

On the pinned pair the same `rmlx baseline` fails on the model path.

With `loaded()` changed to return the compiled C API, every test passes on a
matched pair. Across pairs the CPU SDPA test then fails: the 0.6 binary on the
0.7 library gets `force_fused=True but no fused kernel is available`. With
`pin_check` changed to report a matching C API,
`the_pin_check_carries_the_c_api_verdict` passes on a matched pair and fails
across pairs.

## The attention row rule

MLX 0.32.3 runs an attention call at `head_dim` 256, with an array mask and at
least 1024 query rows, on its head-dim-split kernel. The kernel is compiled per
call for "query rows a multiple of 64" and "key rows a multiple of 32". With
both false it returns `+inf` rows under Metal device-memory shader validation.
The cause is the instrumented compile of that kernel; no run without the
instrument has shown the fault, and mlx 0.31.2 and 0.32.1 do not show it.

`sdpa_under` in `crates/rmlx-mlx/src/fast_ops.rs` keeps every call away from
it: it pads the query rows of such a call to the next multiple of 64
(`FFI.md`, "`scaled_dot_product_attention`"). The rule is on the query rows
alone: a call with aligned key rows was measured clean, and a clean cell is
only a bound on a rate. It has no `cfg` and runs on both mlx-c C API arms,
because the C API a binary is built against is not the MLX it runs on. Query
rows are independent in attention, so each real row is the row the caller
asked for. `crates/rmlx-models/tests/prefill_attention_configuration.rs`,
`crates/rmlx-mlx/tests/attention_check_allocations.rs` and the padded-against-
unpadded cell in `crates/rmlx-mlx/src/fast_ops_tests.rs` hold the rule.

The cost of the copies (`FFI.md`, same section) was measured with
`rmlx baseline --kv-quant none` on Qwen3.6-35B-A3B-8bit against a build
without the rule, in alternating runs: 12 for each build at 4,000 and at
32,518 prompt tokens, 8 at 130,790. The last prefill chunk is padded in each
of the 10 full-attention layers, so a prefill has 10 padded nodes. The limits
were declared before the runs. The figures are the build with the rule against
the build without it.

| Prompt tokens | Prefill rate | Verdict | `metal_gen_alloc_mb` | Verdict |
|---:|---:|---|---:|---|
| 4,000 | -0.39 % | PASS | -12.3 MB (-0.67 %) | PASS |
| 32,518 | +0.82 % | PASS | 0.0 MB | PASS |
| 130,790 | -1.59 % | INCONCLUSIVE | +268.8 MB (+5.43 %) | FAIL |

- Prefill rate: PASS is a median within 1 % and no run under 95 % of the
  median of the other build; FAIL is a median under 97 %. The 130,790 cell has
  one run at 92.3 %. Two sets of runs of one build at 32,518 tokens differ by
  0.64 % in the median, with one run at 94.5 %, and 8 runs for each build do
  not separate the 130,790 cell from that.
- `metal_gen_alloc_mb` (`CLI.md`): PASS is within 1 % and FAIL is more than
  3 %. Each build gave one value in all its runs. At 130,790 tokens the padded
  mask copy, 1792 x 130790 x bf16 = 469 MB in each padded layer, sets the peak
  of the generation. At 32,518 tokens the copy is 121 MB and stays below a
  peak that another step sets. `metal_peak_mb`, which includes the weights,
  and the peak RSS are PASS in the three cells.
- Decode rate, 200 tokens after 4,000 prompt tokens: PASS on Qwen3.6
  (+0.65 %), gemma-4-e4b (-0.24 %) and Ternary-Bonsai-8B (+0.11 %).

Not measured: the HTTP server path, speculative prefill, and
`metal_gen_alloc_mb` with a settled start count.

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
5. Measure the cells of `MEASURED_CELLS`
   (`crates/rmlx-models/tests/prefill_attention_configuration.rs`) in pure MLX
   on the new pair, under device-memory shader validation. The probe is not
   in the tree, and with the rule in place no public call builds the unpadded
   node: only `attention_node` in `crates/rmlx-mlx/src/fast_ops.rs` does. If
   no cell returns a non-finite value, delete the query-row rule in
   `sdpa_under` ([above](#the-attention-row-rule)) and its tests. If the
   faulty shapes changed, change the rule and the table together.
6. Re-run a prefill cell and compare its prefill rate with the old pair's.
