# Changelog

All notable changes to rMLX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The bench scripts record the decode rate the engine measured, not one they
  derive themselves.** `scripts/spec_bench.sh` read the speculative round loop's
  `done` line and divided `emitted` by `elapsed_ms`. That elapsed covers the
  prompt prefill, so the quotient is not decode throughput and reads low —
  233.6 tok/s against the 276.3 the round loop measured, on a code prompt, on
  this host. The engine has reported the prefill-excluded rate on that same line
  as `decode_tps` since the round loops were corrected; nothing was reading it.
  Its no-drafter arm divided the completion tokens by the whole curl request,
  which is `overall_tps` recorded under `decode_tps_warm` and measured 9.6% low
  on the prompt those rows used. `scripts/perf-iter/bench_decode_tps.sh`, which
  `make perf-iter` runs across three models, had the same whole-request form.

  All of them now report the first-emitted-token to last-emitted-token window
  `docs/SPECULATIVE.md` has always claimed for these tables. Two readers own it:
  `scripts/lib/spec_round_log.py` for a round loop's own logged rate, and
  `scripts/lib/server_decode_tps.py` for the rate the server derives from a
  request's inter-token gaps and publishes at `GET /metrics/cache`, which is the
  same quantity for an arm that has no round loop. Each reading is cross-checked
  against the same window timed client-side, and a disagreement past a stated
  band stops the run instead of choosing between them.

  The readers refuse rather than guess: a `decode_tps` that is a bare number
  instead of `Some(x)` / `None` means an older binary wrote the log and the only
  rate in it is the contaminated one; a `None` is the engine saying there was no
  interval to measure; a log holding fewer round-loop records than requests
  served no longer silently averages whatever is left; and a request the server
  cannot attribute an inter-token sample to is not read off the previous one.

  Six `scripts/bench/` campaign drivers that wrote the same wrong column
  (`t1`/`t2`/`t3_final_bench.sh`, `fullctx_regression_bench.sh`,
  `gemma_matrix_bench.sh`, `final_matrix_bench.sh`) are deleted rather than
  fixed: none had a caller, and a driver nobody invokes that writes an
  uncorrectable row when someone does is not reproducibility.

  `scripts/spec_bench_selftest.sh` drives the whole script against a stub server
  and is a hard-fail step of `make ci`. The rows written before this are named
  by a predicate in `docs/METRICS_DB.md`, since `observations` is append-only
  and the values sit inside the metric's plausible-value bound where no gate can
  reach them.

- **A speculative-decode arm no longer takes the champion cell of a plain-decode
  one.** The `bests` cell key named what was measured but nothing about how the
  tokens were produced, so a drafter's rate and a plain rate for the same model,
  quant and prompt ranked against each other and the larger published as that
  model's decode throughput — 276 tok/s standing in for the 142 a request
  without a drafter gets. Migration 005 adds `observations.decode_config` and
  the view partitions on it; `NULL` is ordinary decode, which is what every
  earlier row carries, so no existing plain-decode cell moves.

- **`spec_bench.sh` records the codec and prompt length the run actually had.**
  It wrote `kv_quant = "k8v8"` and `prompt_tokens = 14` as constants while
  starting its server with no `--kv-quant` at all — the engine resolved `none`
  and said so — and while being run against three different prompt files. The
  codec now comes from the run's `cache-type resolved` event and the length from
  the response's `usage.prompt_tokens`; a run that reports neither is refused
  rather than filed under a guess. That event now names the codec through
  `Display` rather than `Debug`, so what it prints is the name the flag accepts
  and the DB records instead of `None` or `Mixed { .. }`.

- **`perf-iter/bench_decode_tps.sh` keeps its state under `<RMLX_HOME>`.** It
  created `metrics/buffer/` relative to the working directory, so running it
  from anywhere but the repo root left a stray tree there, and it stamped a
  `git_sha` with a `-dirty` suffix — not a commit, and nothing can look the row's
  code up by it. Identity comes from `lib/identity.sh` like every other bench
  script, and a dirty tree now records no `git_sha` at all.

## [0.4.1] - 2026-09-02

A patch release carrying one decode-path fix, one correction to what the
operator surfaces claim, and the gates that keep both honest. The HTTP surface
is unchanged and no default moves.

The decode fix is narrow and its boundary is worth stating up front: it removes
a per-layer, per-step copy of the bf16 V prefix that only the iso / rotor /
planar flash-decode dispatchers reached, and only at `kv_h > 1`. The default is
still `DEFAULT_KV_QUANT = KvQuant::None`, which dispatches none of those
kernels, so a run that does not name one of those codecs sees no change — the
`kv_h == 1` control cell measures exactly zero. Alongside it, a silent
wrong-output check that had degraded to a `debug_assert` — compiled out of
`release-perf` and executed by no gate — is a real error again in every profile.

The correction is to what rMLX told operators about its own KV codecs. Four
`iso*` codecs were filed under "measures LARGER than bf16" while holding 0.758x
and 0.879x of it, and `--kv-bits` recommended `--kv-quant none` as the smallest
cache on shared-KV architectures where `iso3_sym` is three quarters of that.
Rather than retype the numbers, the help now points at a table a program
computes.

### Fixed

- **The flash-decode dispatchers stride over the bf16 V mirror instead of
  materialising its prefix.** `update_decode_fp16_v_only` returned a `..offset`
  cut of a head-major `[b, kv_h, max_seq, head_dim]` allocation, cut on axis 2.
  That view is row-contiguous only when `b * kv_h == 1`, so the iso / rotor /
  planar dispatchers' flatten of it materialised the whole valid V prefix — once
  per layer per decode step, with no `contiguous()` call anywhere to make the
  copy visible. The site straddles two files, which is why two earlier
  investigations each got it half right: the slice is in `kvcache/update.rs`,
  the reshape in the dispatcher.

  All three now take the mirror whole and carry its sequence extent as a `dims`
  slot the kernel indexes V's sequence axis with; `kv_seq` still bounds the tile
  loop, the seq-major K index and the mask index. One helper, `flatten_v_mirror`,
  owns the shape contract for all three, and `VMirror` pairs the allocation with
  its valid prefix so the two cannot drift apart among five same-typed `i32`
  parameters.

  Measured on Ternary-Bonsai-8B at 4k / 16 tokens, `--emit-token-ids`, binaries
  snapshotted and verified distinct by sha256 **and** by a symbol present in
  only one — `metal_gen_alloc_mb`:

  | model | `kv_h` | codec | before | after | token ids |
  |---|---|---|---|---|---|
  | Ternary-Bonsai-8B | 8 | `k_iso3` | 2 176.2 | **1 886.5** | identical |
  | Ternary-Bonsai-8B | 8 | `k_rotor3` | 2 307.1 | **2 003.8** | identical |
  | gemma-4-e2b | 1 | `k_iso3` | 1 077.6 | 1 077.6 | identical |

  −289.7 MB against a shape-derived prediction of
  `36 x 8 x 3786 x 128 x 2 B = 279.2 MB`. The `kv_h == 1` row is the negative
  control: unchanged in bytes and in output, because that slice was already
  contiguous.

  **No throughput claim is made.** At the gate shape the ABBA ranges overlap at
  n=6 per arm and TPS drifts 11% across runs on this host, so the oracle is
  resident growth per decoded token — where every `max_seq`-sized buffer is
  allocated before the window and cancels, leaving only prefix-scaled terms.
  Over 1 024 decode steps at `kv_h=8`: 1.86 MB with the mirror passed whole,
  6.05 MB with the cut passed instead, against a 3.15 MB budget. Its first
  design was vacuous and only mutation-checking caught it — a peak-memory
  bracket reads green against a deliberately reintroduced copy, because Metal
  releases a dispatch's buffers on the *next* dispatch. Each codec's clean floor
  is measured rather than shared (886 per mille of one V prefix for `iso3`,
  1 144 for `rotor3`), so K-side drift gets its own named failure instead of
  eating the V-side margin. (#469, closes #467)

- **The V mirror's valid length is checked in every profile.** Striding over the
  mirror removed an always-on check without replacing it: the old flatten
  targeted `b * kv_h * kv_seq * head_dim` elements, and MLX's `reshape` rejects
  an element-count mismatch, so a mirror shorter than the attended length could
  not reach the kernel anywhere. Flattening the allocation matches whatever was
  passed, leaving only a `debug_assert` — compiled out of `release-perf`, and
  behind no gate, since `make ci` runs `dev` with no `--ignored` and `test-perf`
  executes no `Device::Gpu` test. A step whose store ran ahead of its mirror
  would have attended the mirror's tail and returned plausible-but-wrong output
  with no error. `flatten_v_mirror` now rejects `valid != kv_seq` as an
  `Error::Quant`, and its regression test is `Device::Cpu` and deliberately not
  `#[ignore]`d, so it runs under `make test`, `make ci` and `make test-perf`.
  (#469)

- **The operator surfaces stop quoting hand-written residency ratios.** The
  `--kv-quant` help filed `iso3_sym`, `iso4_sym`, `k_iso3` and `k_iso4` under
  "Runs its codec — and measures LARGER than bf16" at 1.00x–1.05x. All four are
  under it: 12.125 and 14.0625 bits per value at `head_dim` 128, 0.758x and
  0.879x of bf16 on a global layer, on both topologies. `--kv-bits` closed by
  recommending `--kv-quant none` as the smallest cache on a shared-KV
  architecture, where `iso3_sym` is three quarters of it. The rotor ratios were
  roughly twenty times their true distance from 1.0.

  The stale numbers were the symptom: all six wrong figures were written when
  they were correct and carried unchanged through the two commits that moved the
  stores underneath them, because no rule in `check_kv_codec_disposition.sh`
  reads a number. So the ratios are gone rather than corrected —
  `rmlx info --list-cache-types` now prints, per codec and for both topologies,
  what `estimated_resident_bytes_per_layer` says a global layer holds, and the
  help points at it. (#461)

- **The MLX pin gate keys on what dyld resolved, and runs where measurements are
  taken.** The pin was checked only where it cannot fail: `build.rs` warned on
  keg drift and on a metallib without `steel_gemm_fused_nax`, but cargo re-runs
  a build script only for a *newer* mtime and stats through symlinks, so
  repointing `opt/mlx` at an older keg moves the observed mtime backwards and
  cargo replays its cached output. The drift costs ~3.8x on GPU matmul while
  output stays correct and decode stays flat, so only prefill numbers go quietly
  wrong. The gate now reads the two dylibs dyld resolved for the running
  process, compares both keg versions to `mlx-pin.txt`, and scans the resolved
  metallib for the kernel family — the load-bearing half, because bottle
  contents vary by build runner and a version match is not evidence. `rmlx
  baseline` and `rmlx bench` call it before anything else, which is
  dyld-truthful by construction: the preflight's `readlink` oracle can read
  green while the process loads something else entirely. (#462)

- **The pin parsers are fenced.** Pin versions reached `rm -rf`, `cp -R` and
  `ln -sfn` targets under the Cellar after nothing but a non-empty check; a pin
  line of `mlx ..` yielded `rm -rf "$CELLAR/mlx/.."`. Both parsers now allowlist
  the shape of a keg directory name, and the shell one moved into
  `scripts/lib/mlx_pin.sh` so there is a single grammar, held to the Rust one by
  a differential test with controls in both directions. (#462)

### Tested

- **mxfp8 measured against MLX affine 8-bit at equal rate.** Both spend 8.250
  bits per value, measured from the arrays MLX returns (mxfp8 at group 32 ties
  affine at group 128 exactly), which is the only rate at which the comparison
  means anything. At that rate mxfp8 is 4.39x–10.24x worse on relative Frobenius
  error across 42 cells spanning three fixtures, two head dims, five outlier
  magnitudes and three seeds. A best-case E4M3 arm is carried alongside, because
  MLX's encoder rounds its E8M0 shared exponent to nearest and clips the group
  maximum in 46.7% of groups when that rounds down; under a non-clipping scale
  the format still never reaches the declared 0.85x material threshold. The
  fixture's power is measured rather than assumed — the outlier fixture
  separates affine-g32 from affine-g128 at 0.53–0.58x where a uniform LCG
  fixture reads 1.06x and sees nothing. (#464)

- **A rotation-sensitive fidelity gate for the turbo family.** Neither committed
  turbo fidelity surface could price the transform the family lacks: the
  rate-distortion fixture is i.i.d. Gaussian, where the Lloyd-Max codebook's
  assumption already holds and an identity rotation passes, and the
  outlier-fixture comparison against iso / rotor / planar confounds the missing
  rotation with four-times-coarser scale granularity. The new gate holds codec,
  width and group size fixed and moves only a full-`head_dim` Walsh-Hadamard.
  Measured on the outlier fixture, in bits of SQNR the rotation buys: +2.077 /
  +2.004 / +1.574 / +0.961 at 1 / 2 / 3 / 4 bits, against exactly 0.000 with the
  transform removed. It accounts for 87.5% of `turbo3`'s reconstruction-error
  deficit against `iso3` and 79.4% of `turbo4`'s against `iso4`, and is worth
  about 0.01 bits on i.i.d. Gaussian — the shape the literature reports for the
  Value cache.

  The criterion was declared before measuring, and the result is recorded in the
  bucket it landed in rather than the one it was hoped for: two of four
  pre-declared conditions are unmet, so the verdict is **FAIL**, stated as such
  and reconciled condition by condition. The shortfall at the widest width is
  pinned rather than tuned away. (#465)

### Changed

- `rmlx baseline` and `rmlx bench` refuse to run when the pin binds and the
  loaded MLX pair is wrong, rather than producing numbers from an unknown
  library. `mlx_preflight.sh` likewise refuses an unpinned pair on the hosts it
  binds; both it and `mlx_restore_pin.sh` read `mlx-pin.txt` instead of carrying
  their own copies of the versions.
- `perf_ceiling.py`'s KV byte model is held to the engine's by
  `make check-kv-byte-model-parity`, which sweeps `ALL_KV_QUANTS` across both
  topologies and two shapes. The second copy had drifted on three axes at once —
  up to 3.46x on `mixed_k8g64_v4g64`. The engine is the oracle; the Python half
  chooses nothing about what is covered, so a new codec reaches the gate without
  anyone adding it to a list. (#461)
- The KV-codec disposition gate's help scope is derived from the clap attributes
  across the whole CLI crate rather than from a hand-written list, so a help
  constant in any module is in scope. It found `CACHE_TYPE_K_LONG_HELP` and
  `CACHE_TYPE_V_LONG_HELP` — operator-facing KV help under no rule at all — and
  `KV_PRESET_LONG_HELP`, which named five inert codecs outside any INERT marker.
  A new rule rejects any ratio-shaped figure in those constants outright: a
  corrected number is the same defect. (#461)
- `make check-doc-source-citations` gates that every `crates/...` path cited in
  `docs/` resolves. Nine such paths named files that had moved crate or become
  `mod.rs`. (#471, #472)

### Removed

- `RMLX_MLX_NAX`. It was a `cargo:rustc-env` set by the build script, not an
  operator knob; `events.mlx_nax` now comes from the runtime metallib scan, so a
  bench row cannot claim a capability the run did not have. (#462)

### Documentation

- **"Quantized KV always loses at decode" is retired.** It was a statement about
  8-bit K that had been reading as a statement about codecs. The same harness,
  the same model and the same ABBA design run with K at 4 bits
  (`mixed_k4g64_v4g64`) change the sign: on the dense 8B target the 4-bit-K arm
  beats `none` with the arms' per-slot ranges disjoint, at a 32k prompt and again
  at a 63k one — where the 8-bit-K arm at identical shapes returns INCONCLUSIVE.
  The earlier null is sound and was not a measurement error; one of the new cells
  independently reproduces the old Bonsai row in a different session. It was
  measured at the wrong K width, and no 4-bit-K cell existed anywhere in this
  tree until then.

  Read the consequence narrowly. The win is on the `Mixed` path through MLX's
  `quantized_matmul`, not on a fused flash-decode kernel over a packed store, so
  it does not transfer to the iso / rotor / planar ring. It is not a
  long-context effect — it holds at 32k as well as 63k. No default changes on
  this result; acting on it is a preset question, not a patch-release one.
  `docs/PERF_BASELINE.md`'s codec-cell section is scoped to 8-bit K accordingly.
  (#471)

- **The decode ceiling.** An ablation ladder that deletes 100% of a fused codec's
  decode math still leaves it far short of `none`, so the kernel *shell* — not
  codec arithmetic — is the majority of the gap. That puts every decode-math
  proposal under one ceiling below parity and redirects the next measurement to a
  shell variant. Three of the ladder's four arms are throwaway instrumented
  binaries reproducible from no committed ref, so no ratio from it is written
  down: the rung ordering is the durable result. (#471)

- The residency cells withdrawn in 0.4.0 now point at the `runs.db` rows that
  measured them, with the query, in both pages that carried them.
  `docs/PERF_BASELINE.md` was still carrying pre-elision ratios in four table
  cells and five sentences while `docs/KV_QUANT.md` had withdrawn them — one
  number, two pages, two answers. The attribution is corrected: the mirror
  elision moves part of the distance and the later in-family boundary floor moves
  the rest. (#471)

- The turbo family's docs stop claiming a Walsh-Hadamard transform the encoder
  does not have — `turboquant.rs` contains no Hadamard code, and the `_wht_`
  layout tags are SSD geometry identifiers. Turbo is primarily a V codec, and the
  rotation is worth ~a bit or two on K-shaped data against ~a hundredth of a bit
  on V-shaped, so its value and its cost are inverted. (#471)

- Architecture facts corrected against the checkpoints and the loader: e2b's
  depth is not e4b's and its alternation period differs, so `layer_types` is the
  only authority; Gemma4 splits KV heads by layer class on every size, not just
  where the override field ships; Bonsai-8B's context caps below the 131k column
  its `kv_frac` row projects. The GQA screen does not reproduce on 8:1, and the
  dilution arithmetic was backwards — sliding layers short-circuit to bf16, so
  only full-attention layers carry a K codec and the treatment there is total.
  (#471)

- `docs/FFI.md`: NAX's M floor of 16 is MLX's one shipped tile, not an
  instruction-set limit, and it is not why decode cannot reach NAX — decode is
  bandwidth-bound and capped at `1/heads_per_kv` by grid geometry. The
  drift-recovery command needs `brew unpin` first; pinning while a newer keg is
  linked is what produced the split link it was meant to prevent. (#462)

- The gate table in `CLAUDE.md` names the three checks added with the KV
  campaign. (#472)

## [0.4.0] - 2026-08-31

This release is the KV-cache subsystem measured against its own claims, and the
silent-corruption class that was hiding behind them. Seventeen of the
twenty-eight KV codecs never built a packed store and decoded byte-identically
to bf16; the ten that did held *more* resident bytes than plain bf16, not fewer;
and two `auto` resolvers disagreed about which one a request got. `--kv-quant
auto` is now unquantised bf16 on every architecture and every context length,
the `--kv-quant` help and `docs/KV_QUANT.md` state each codec's runtime
disposition and a CI gate fails the build when they drift from it, and — after
the sideband and mirror work — the first codecs in this tree's history hold
fewer bytes than bf16, on the architectures whose topology pays for them.

Running alongside that is a set of silent-wrong-output fixes. `quantized_matmul`
read past its split-K partition on every nvfp4 checkpoint, corrupting prefill; a
sparse-V kernel corrupted every `mixed_*` / `rot_k_*` decode above 8 192 tokens;
the GDN speculative rollback replayed through a scratch KV stack at six of seven
call sites; a `Mixed` truncate reported positions it did not hold; and the
speculative serve path kept a second full copy of the verifier. Every published
speculative accept rate and speedup is re-derived on the fixed tree, and several
moved in both directions.

The HTTP surface keeps its shape. The behaviour changes to read before upgrading
are in **Changed**: `auto` no longer quantises, `--paged-kv` requires an explicit
codec, Gemma4 with `mixed_*` / `rot_k_*` and an SSD-tier hit is refused rather
than silently wrong, `--turbo-flash auto` holds OFF, and a `projects.toml`
`draft_model` naming the served model fails at load.

### Performance

- **Prefill attention masks are built on device, not scalar-filled on the
  host.** `build_chunked_prefill_mask` and `build_swa_prefill_mask` allocated
  three full-size buffers per call — an `f32` `Vec`, its upload, and the bf16
  cast — for a mask that is O(`seq` × `kv_len`). A 68 898-token gemma-4-e2b
  prefill spent 69.5% of its main-thread samples there and drove free memory to
  zero, which made every repeated in-process generation slower than the last
  (gen3/gen1 prefill 1.24–1.33). The builders now compose the mask from MLX
  position vectors (`arange` → broadcast compare → `where`), so it is produced
  where it is consumed and never crosses the host boundary. Shared by every
  architecture that chunk-prefills, and bit-identical: temp=0 token digests are
  unchanged on all three test-target families at every context measured.

  Measured `rmlx bench --kv-quant none --warmup 0 --runs 3`, free memory
  settled before each cell, before/after pairs run back-to-back at matched
  host load. `prefill_ms` per generation, gen-1 first:

  | cell | before | after | gen-1 Δ |
  |---|---|---|---|
  | gemma-4-e2b @4 096 | 254.9 / 213.8 / 219.4 ms | 226.1 / 184.9 / 185.5 ms | −11.3% |
  | gemma-4-e2b @68 898 | 13 673–14 024 ms | 4 762–5 277 ms | −63% |
  | Qwen3.6-35B-A3B @4 096 | 1 461.2 / 1 090.2 / 1 111.9 ms | 1 155.2 / 1 089.9 / 1 091.3 ms | −20.9% |
  | Qwen3.6-35B-A3B @34k | 12 828.4 ms | 12 267.2 ms | −4.4% |
  | Ternary-Bonsai-8B @4 096 | 1 347.8 / 1 358.3 ms | 1 364.7 ms | +0.9% |
  | Ternary-Bonsai-8B @68 898 | 60 032.2 ms | 59 296.7 ms | −1.2% |

  The small-shape case is the one a device-built mask could lose — it trades a
  host upload for a handful of MLX dispatches — so the 4 096-token cells are
  measured, not argued: both improve. Decode TPS, `kv_cache_bytes` and token
  digests are unchanged on every cell.

  The gemma-4-e2b @68 898 cell is where the repeated-generation drift lived. It
  no longer crosses the host's free-memory line, so the drift is gone rather
  than reduced:

  | | before | after |
  |---|---|---|
  | gen3/gen1 prefill | 1.24–1.33 | 0.97–1.00 |
  | free memory low-water | 0.07–0.10 GiB | 12.4–23.9 GiB |
  | compressor growth | +14.0–15.2 GiB | +0.00 GiB |
  | decompressions | 6.2–6.3 M | 0.002–0.003 M |
  | `build_attn_mask` share of main-thread samples | 69.5% | 0.07% |

  Ternary-Bonsai-8B's prefill time is unchanged (±3%, it is GPU-bound behind
  MLX's fused `head_dim=128` kernel) but at 68 898 tokens its free memory now
  bottoms out at 18.6 GiB instead of 0.1 GiB.

  Sharing one prefill mask across a forward call's layers — which
  `qwen3_5_moe` and `qwen3_vl_moe` do and continue to do — was measured on both
  architectures that use it and is **not** uniformly good or bad: on gemma-4-e2b
  @68 898 sharing costs 2× the prefill time, on Qwen3.6 @34k it wins by 1.6%
  (inside run-to-run spread). No hoist was added or removed here. `mask.rs`
  records both numbers and does not claim a mechanism for the gemma-4 one.

- **KV kernel dispatchers stop blocking the host per layer — up to 3.06×
  decode.** All four iso/rotor flash-decode dispatchers called `Array::eval()`
  on their kernel inputs immediately before dispatch. That is a synchronous
  graph evaluation, and it ran once per attention layer per decode step — 26
  host waits per token on Ternary-Bonsai-8B, with the forward pass advancing one
  layer at a time and nothing queued ahead. It bought nothing: the row
  contiguity the kernels need already comes from `MetalKernel::new`, which
  passes `ensure_row_contiguous`. Paired A/B on one binary pair, `rmlx bench`
  n=3 per cell, token digest / KV bytes / TTFT identical in every cell. Decode
  TPS:

  | cell | before | after |
  |---|---:|---:|
  | Bonsai-8B `iso3_sym` @4k | 19.09 | 55.15 |
  | Bonsai-8B `iso3_sym` @16k | 11.00 | 19.01 |
  | Bonsai-8B `k_iso3` @16k | 14.90 | 24.37 |
  | Bonsai-8B `rotor3_sym` @16k | 10.13 | 16.03 |
  | gemma-4-e2b `iso3_sym` @4k | 65.57 | 100.20 |

  Absolute decode figures recorded on these codecs before this change are not
  comparable with figures taken after it (#292, #334).

- **The rotor / iso flash-decode shell was rewritten, and the rotor fused decode
  stopped downloading its tail to the host each step.** The per-token QK dot
  used a `log2(head_dim)`-round threadgroup barrier tree with 127 of 128 lanes
  idle; it now folds through simdgroup reductions (~8 → 2–3 barriers per token),
  and the rotor's ~64-FMA inverse Clifford sandwich is decoded once per group by
  the block leader rather than once per lane. Separately, the rotor K-only fused
  path ran `rotor_gpu_outputs_to_cpu` inside every decode step, building a
  per-layer per-token CPU block the flash kernel never read — a host copy and a
  GPU sync per layer per step, in the path whose whole purpose is removing host
  work from decode. The GPU ring is now the source of truth for the decode tail
  and the CPU blocks are rebuilt on demand at the two consumer boundaries,
  `dequant()` and the SSD-spill / prompt-cache clone. Long-context kernel
  microbenchmark 1.5–1.8× at 32k on both codecs; 4k decode TPS improves on every
  cell (Bonsai `k_iso3` 17.9 → 22.5, `k_rotor3` 21.3 → 22.5; gemma-4-e2b
  `k_iso3` 60.7 → 72.2, `k_rotor3` 62.3 → 70.4). Keyed off codec and shape
  (`head_dim`, `kv_heads`, `bits`), never an architecture (#231, #234, #278,
  #280).

### Fixed

- **`quantized_matmul` silently corrupted prefill on every nvfp4 checkpoint.**
  MLX's `qmm_splitk` aligns each split-K partition to `group_size`, but the
  `qmm_t_splitk` kernels step K by a fixed 32-wide tile and do not bound that
  loop. At nvfp4's group of 16 a partition can be 16 or 80 — not a whole number
  of tiles — so the kernel reads past its partition into the next group's codes
  and scales. Upstream aligns to `max(group_size, 32)` on main; neither 0.31.2
  nor 0.32.0 carries that fix.

  It is live by dispatch, not by inference: a 16-token chat prompt on
  `gemma-4-E4B-it-qat-nvfp4` issues 40 `qmm_t_splitk` calls at N=512, K=2560,
  partition 80 — `k_proj` and `v_proj` in 20 of 42 layers — at 43–63% relative
  error. Decode is unaffected: M=1 routes to `qmv`, which is correct, and that is
  why it hid. The predictor is `K/split_k % 32 != 0` rather than the partition
  being narrower than the tile — at K=2560 the partition is 80, wider than the
  tile and still not tile-whole — and over a 216-cell sweep that predicate has no
  false positives and no false negatives.

  `quantized_matmul` now mirrors MLX's partition choice and, where the resulting
  partition is not tile-whole, pads the batch with zero rows onto one that is and
  slices them off. Both gating tests are scalar and run before any array access,
  so a wide-group model pays one modulo and one branch. On a 600-prompt
  ground-truth-scored greedy battery, accuracy on corrupting shapes goes
  31.5% → 49.5% (McNemar 98 vs 26, p=5e-11) while tile-whole shapes are unchanged
  at 12.0% and 200/200 byte-identical; gemma-4-e4b-mxfp8 and Ternary-Bonsai-8B
  are each 600/600 byte-identical across the change, and mxfp4 / mxfp8 / affine
  show only bf16 noise across the sweep. **Anyone who evaluated an nvfp4 model on
  a previous release got degraded results.** A test fails the build once the
  linked MLX is past the last version carrying the defect, so the guard is
  deleted on a bump rather than degrading into a prefill tax on a kernel that no
  longer needs it.

- **A speculative round loop's `decode_tps` counted the prompt prefill.** The
  MTP, DFlash and EAGLE-3 round loops started their timer before the verifier
  prefill and then divided the emitted token count by the whole elapsed window,
  so the field named `decode_tps` was not a decode rate. On a short prompt that
  cost 1–3%; on a 4k prompt the loop reported 13.8 tok/s where the streamed
  tokens arrived at 21.3 — an understatement of 55%, and it grows with the
  prompt. Every published speculative throughput figure was scraped from that
  line, and each was being compared against `rmlx baseline`'s `decode_tps`,
  which excludes prefill — so the speculative arm wore a penalty the arm it was
  measured against did not. Every round loop now reports the window between the
  first and last emitted token, `(marks - 1) / (last - first)`, which is the
  same window `rmlx baseline` reports. `elapsed_ms` still covers the whole call.
  Emitted tokens, accept rates and throughput are unchanged; only the reported
  number moves.

  Three details of the new field. It is carried as an `Option` and rendered
  `Some(x)` / `None`, because a one-token generation has no interval to measure
  and `0.0` in that slot prints, averages and wins a champion cell exactly like
  a real throughput of zero — the same reason `rmlx baseline` carries its phase
  timings as `Option`. The window counts the tokens it saw rather than trusting
  a total passed in, so a loop that emits without going through the shared
  `emit_step` cannot report a rate faster than it ran. And **all five**
  speculative round loops now carry the field on that one basis:
  `spec_generate_greedy_cached`, `spec_generate_stochastic_cached` and
  `mtp_assistant_generate_greedy` previously logged `elapsed_ms` and no rate at
  all, which left the Gemma4 assistant pair — a published row in
  `docs/SPECULATIVE.md` — with no serve-log decode rate to read.

- **The speculative serve path loaded the verifier twice.** The MTP / EAGLE-3 /
  DFlash branches each called `load_speculative(verifier_dir, verifier_dir, ..)`
  to fill a draft slot their round loops never read, and `load_speculative` has
  no reuse path — so `load_model` ran twice on one directory and MLX took an
  owning copy each time. Resident memory measured 52.5 GB for
  `Qwen3.8-27B-mxfp8` and 9.7 GB for `gemma-4-e2b-it-mxfp8`, roughly double what
  a single copy needs, leaving no headroom for a second model or long-context
  prefill scratch on a 128 GB machine. The dispatcher's draft slot is now
  optional and the sidecar branches load the verifier once, so the second copy
  is impossible rather than merely unused: 52.5 GB → 26.7 GB and 9.7 GB →
  5.2 GB, with temp=0 output byte-identical either side. Qwen3.6-35B-A3B holds
  one 35 GB copy across all three drafter kinds.

  `--draft-model` naming the same snapshot as `--model` is now refused at load
  rather than silently doubling the weights — a draft that is the verifier costs
  exactly as much to run as the verifier it is meant to outrun. The affected
  configuration is a **profile-supplied `draft_model`**: clap's
  `requires = "draft_kind"` only binds CLI-supplied flags, so a `profiles.toml`
  / `projects.toml` that sets `draft_model` to the `--model` path reached the
  two-model path and now fails at startup with a message naming the fix. Point
  `draft_model` at a smaller model, or drop the key.

- **`KvStorage::truncate_to` reset the `Mixed` store instead of truncating it,
  so a cache reported `n` positions and held zero.** `KvCache::truncate_to` sets
  `offset = n` immediately afterwards, and the reset was never needed — the
  store is a capacity buffer whose `offset` *is* the fill marker, so rolling the
  marker back is the truncation. This is the whole of the gemma4-assistant MTP
  crash under `--kv-quant mixed_k8g64_v4g64`: the first partial-accept round
  scored `seq` keys against an `n+seq`-wide mask. **The crash was the lucky
  case.** A single-token decode needs no mask, so an ordinary decode after a
  prompt-cache trim silently attended the current token alone and produced
  wrong output with no error. `MixedKvState::truncate_to` was likewise silent in
  the over-long direction, and the `debug_assert` that would have caught it is
  compiled out of `release-perf`; `Mixed` cannot clamp — `offset` *is* its
  coverage — so it keeps its fill and says so.

- **The GDN speculative rollback replayed through a scratch KV stack, corrupting
  the verifier on every partial-accept round.** The recurrent state has no
  sequence axis, so it is restored from a snapshot and replayed over the kept
  prefix — but that replay runs the whole layer stack, and on a GDN hybrid the
  full-attention layers sit between the GDN layers and feed them. Replaying
  through fresh caches made those FA layers attend a `kept`-token prefix at
  positions `0..kept`, so every downstream GDN layer advanced on a wrong hidden.
  The rollback now rolls the *real* caches back to the pre-round offset and
  replays into them.

  **It was live at six of seven call sites.** Two round loops had been fixed and
  four left; `eagle3_generate_greedy` carried the pre-fix block verbatim, and
  the shared helper `restore_and_replay_lin` — whose own doc argued *for* the
  scratch stack — sat behind the classic two-model loop's four call sites.
  `speculative::rollback_round_caches` now owns the whole rollback (the FA
  `truncate_to` loop, the GDN snapshot restore, and the replay) and picks its
  arm from the arch: seven callers, one implementation.

  Measured greedy at temp=0 against a plain greedy arm on the same verifier,
  scratch-stack replay vs real-cache replay — tokens shared with the
  no-drafter reference:

  | path | before | after |
  |---|---|---|
  | MTP sidecar (Qwen3.8-27B) | 4/31 | 31/31 |
  | EAGLE-3 (Qwen3.6-35B-A3B) | 13/96 | 93/96 |
  | two-model (Qwen3.8-27B + ornith-1.0-9b) | 17/96 | 96/96 |
  | Gemma4-e4b + e2b (full-attention arm) | 96/96 | 96/96 |

  The two-model loop could not have reached the rollback at all: it read the
  pre-round offset from `caches[0]`, and on a GDN hybrid layer 0 is a recurrent
  layer whose `KvCache::offset` never leaves 0, so the truncation target went
  negative every round.

- **Token selection resolved ties differently on the host than on the device,
  and `top_p` / `top_k` resolved them differently on every call.** MLX `argmax`
  resolves a tie to the lowest token id. The host greedy path
  (`argmax_with_penalties`, taken whenever a repetition/presence/frequency
  penalty or a `logit_bias` is set at `temperature == 0`) used
  `Iterator::max_by`, which returns the *last* maximum — and which also let a
  `NaN` reset the running best, since `partial_cmp` is `None` against one. So
  adding a penalty to an otherwise greedy request could change the token on a
  tied row, and an all-`-inf` (fully constraint-masked) row returned the last id
  instead of id 0. `filter_top_k`, `filter_top_p` and `compute_top_logprobs`
  each left their tied order to a sort's pivot choice or to a selection's swaps.
  All four now use one rule — equal values rank by lowest token id — so
  `top_k = 1` is the argmax on a tied row, the `top_p` nucleus is the lowest
  tied ids rather than a non-contiguous scatter (a 64-wide row with one 0.4 and
  63 identical tail values at `top_p 0.5` kept `{0, 29, 54..63}`), and logprob
  rank 0 agrees with the device argmax. `top_p` matters most: it ships set in
  several `generation_config.json` snapshots, so it is on by default on the
  served path.

  Ties are not exotic. On a realistic 262144-wide BF16-derived softmax row,
  259416 of the 262143 adjacent pairs are exactly equal — 8 mantissa bits give
  0.125 spacing at logit magnitude 16.

  Neither filter uses a comparator any more, which also removes a pre-existing
  crash: **a `NaN` probability could abort the decode step.** Folding an
  unordered pair to `Equal` makes a `NaN` compare equal to everything, which is
  intransitive, and `sort_unstable_by` panics with "user-provided comparison
  function does not correctly implement a total order" — reproduced on the
  shipped comparator. Both filters now order integers under the standard `Ord`,
  so no comparator exists to be intransitive.

  They do it differently because they have different jobs, and this is also a
  **speedup on both**. `filter_top_k` partitions (`select_nth_unstable`) over
  packed `u64` keys — it needs a set, not an order, which is what mlx-lm's
  `argpartition` says too. `filter_top_p` needs the full ascending order for its
  cumulative sum, so it sorts the *values alone* and applies the id rule once,
  to the single tied group the cut lands in; packing the id into that sort key
  would make every key distinct and destroy the equal-element partition a
  tie-dense row hands the sort, which measured *slower* than no tie rule at all.
  At a 262144-token vocabulary, best-of-9 across three runs:

  | Filter | fixture | before | after |
  |---|---|---:|---:|
  | `top_k` (k=64) | tie-dense | 2.02–2.13 ms | 0.30–0.33 ms |
  | `top_k` (k=64) | all-distinct | 3.67–4.31 ms | 0.32–0.41 ms |
  | `top_p` (0.95) | tie-dense | 2.02–2.06 ms | 1.31–1.34 ms |
  | `top_p` (0.95) | all-distinct | 3.73–4.32 ms | 2.17–2.68 ms |

  The rank keys use the IEEE total-order flip rather than the raw bit pattern,
  so they order every `f32`. The raw pattern is monotone only over non-negative
  values; `probs` are non-negative today, but the failure mode if that stopped
  holding is a silently wrong token — `-0.0` outranks every positive, so
  `top_k = 1` on `[-0.0, 0.5, 0.25]` would zero the real maximum and keep
  `-0.0` — and `release-perf` disables debug assertions, so an assert would not
  catch it.

  The pre-existing tests could not reach any of this: every row they used had a
  unique maximum.
- **The sampler emitted a constant token, silently, on a `NaN` logits row.**
  `softmax_scaled` propagated the `NaN` into `probs`; `renormalise` then no-oped
  (`total > 0.0` is false on `NaN`), `sample_inverse_cdf`'s `total <= 0.0` guard
  was also false, its `cum > target` never fired, and it returned `last_nonzero`
  — the same id on every step, **independent of the RNG**, with the request
  reporting success. Measured on a 16-wide row with one `NaN`: a constant token
  for every seed tried, against a varied healthy control; with `top_k = 1` the
  `NaN` took the only surviving slot and the stream collapsed to id 0.

  `softmax_scaled` now returns `Err` when the exponentials do not sum to a
  finite value, which happens exactly when a logit is `NaN` or `+inf`. This is
  the decode-step half of the rule the prefill guard already enforces, and it
  has to live here because that guard is a *prefill* guard: on every
  test-target architecture it runs once before the loop and never again, so
  nothing downstream reported a `NaN` arriving at decode step 300. The check is
  free — `sum` is already computed. Greedy deliberately does **not** refuse such
  a row: it mirrors the device reduction, which skips `NaN` and returns the
  largest real logit, and erroring there would re-create the host/device split
  the tie contract exists to close.
- **An all-`false` constraint mask would produce a stuck stream instead of an
  error.** No token satisfies the grammar, so every token the selection could
  return violates it — and because the engine state that produced the empty mask
  persists, the same arbitrary token would be emitted for the rest of the
  generation while the request reported success. All three mask-accepting entry
  points (`apply_mask_argmax`, `argmax_with_penalties`, `sampling_distribution`)
  now return `Err`. Logging instead would not have worked: the check sits on the
  per-token decode path, so a `warn!` fires once per emitted token and, at a few
  hundred bytes a line, evicts the whole log directory under `RMLX_LOG_CAP_MB`
  within hours — deleting the evidence it exists to provide. This guard is
  **unit-tested only and has no demonstrated production trigger**: the
  all-`false` state was not reachable through the HTTP surface (`{"enum": []}`
  is rejected at schema parse with HTTP 400 before a mask exists, and byte-level
  BPE means exotic `const` / single-`enum` values never starve the mask). The
  constraint engine does engage, so the path is live; the empty-mask state is a
  future constraint-engine defect being pre-empted, not an observed one.
- **Mixed / RotK decode produced wrong output above 8 192 context tokens.** The
  V side of `mixed_quantized_sdpa` diverted to a separate MSL kernel
  (`sparse_v_weighted_sum`) once the cache held 8 192 tokens or more. That
  kernel applied *symmetric* dequant (`code − 2^(bits−1)`) to *affine* data, so
  every V element came back offset by `−2^(bits−1) · scale`: measured against
  `mx.dequantize`, `scale·raw + bias` agrees to 2.4e-7 while
  `scale·(raw − 2^(bits−1)) + bias` is off by 2.96. Its dispatch was one thread
  per output element at a threadgroup of 1, each thread walking the whole
  context serially, which cost 17× the `quantized_matmul` it replaced. The
  kernel is deleted and the V side now always goes through `quantized_matmul`.
  Affected every `--kv-quant mixed_*` / `rot_k_*` cell on every architecture
  past 8 192 tokens, including the arch default on `Qwen3ForCausalLM`; below
  that threshold this change alone moves nothing and temp=0 token digests are
  byte-identical (the truncation entry below does move some short-context
  digests, so the shipped build is not digest-identical at short context).
  At 16k the fix takes Ternary-Bonsai-8B from 75.2 to 10.0 ms per decode step
  (7.3× → 0.97× of `none`) and gemma-4-e2b from 18.4 to 8.2 ms (2.2× → 1.00×).
  A decode-path gate now checks `mixed_quantized_sdpa` against an oracle built
  from `mx.dequantize` plus stock SDPA, at context lengths either side of 8 192.
  The kernel's own tests could not have caught this: one reimplemented the
  kernel's dequant formula as its "reference CPU", and the other used codes
  equal to the midpoint, where the offset is exactly zero.
- **Attention probabilities below 1e-6 are no longer truncated to zero before
  the V matmul.** The truncation existed to feed the sparse-V kernel above, on
  the theory that a zeroed row costs nothing downstream. `quantized_matmul` is
  opaque and reads every V row regardless, so it bought no bandwidth while
  dropping attention mass it never renormalised. Against the untruncated oracle
  it cost 28–73× the relative L2 error (6.5e-5–1.7e-4 with it, 1.1e-6–2.4e-6
  without) across GQA, single-KV-head and MHA shapes. It is also a small decode
  speedup (two fewer ops per layer per step): Ternary-Bonsai-8B at 16k goes
  10.011 → 9.756 ms per step.

  Unlike the kernel removal above, this changes the V-matmul input at *every*
  context length, not only past 8 192. Whether that moves the sampled ids is
  shape-dependent: measured changed on Ternary-Bonsai-8B at 3 833 and 15 692
  context tokens and gemma-4-e2b at 4 180, and measured unchanged on
  Ternary-Bonsai-8B at 7 802 and gemma-4-e2b at 17 211, and unchanged on the
  32-token shape pinned by `bonsai_8b_mixed_k8g64_v4g64.golden.txt`.
- **The fused-QK dispatch table listed eight codecs it could never serve, and
  a strict-mode test asserted four of them dispatch.** The head-major fused-QK
  shadow is seeded by re-encoding the bf16 K mirror, so a codec only reaches
  that path when it keeps one (`KvQuant::feeds_bf16_k_at_decode`).
  `Iso{3,4}Sym`, `IsoKOnly{3,4}`, `Rotor{3,4}Sym` and `RotorKOnly{3,4}` keep
  none by design — each decodes through its own flash-decode-over-quant kernel
  straight off the packed ring — so listing them was listing entries no shape
  on no architecture could reach. The tables in `rmlx-kv-quant` and
  `rmlx-models` are pruned to the reachable set (q8, `TurboSym3`, `TurboSym4`,
  `RotorK{3,4}Asym`), and a unit test pins the entry ⇒ bf16-K-mirror
  implication so a ninth cannot be added silently.
  `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs` becomes a routing
  contract: for each rotor codec it asserts which kernel family fired **and**
  that the other two did not, which the previous test never checked in either
  direction.
- **The rotor fused-QK kernel is reachable and does dispatch.** Proven on two
  architectures at both supported head widths, counting per-dispatch `trace!`
  events in the run's `.jsonl` under `--log verbose`:
  - Ternary-Bonsai-8B (`Qwen3ForCausalLM`, `kv_h=8`, `head_dim=128`),
    `rmlx --log verbose --metrics off --fused-qk on bench --prompt-tokens 4096
    --max-ctx 8192 --ctk rotor_k_3 --ctv q4_g64 --max-tokens 32 --runs 2
    --warmup 1` → codec resolves to `rotor_k_3_asym_v4_g64`, 2418
    `rotor_fused_qk_sdpa: dispatch`.
  - ornith-1.0-9b (`kv_h=4`, `head_dim=256`), same flags with `--ctk rotor_k_4
    --ctv q4_g64 --max-tokens 8` → codec resolves to `rotor_k_4_asym_v4_g64`,
    126 dispatches.

  Both need an explicit affine `--ctv`: `--ctk rotor_k_*` alone takes the
  arch-default V and `combo_to_kv_quant` then yields `RotorKOnly*`, which
  routes to `rotor_flash_decode` instead. Only an accepted affine V produces
  the asym codec this kernel serves.

  It is the *only* GPU decode kernel for the two rotor-asym codecs, which have
  no flash-decode arm.
- **`planar_flash_decode` was documented as byte-identical to the split
  chain; it is not — but only some cells show it.** Both arms decode the same
  packed K; the flash kernel folds the softmax into a per-tile online
  log-sum-exp reduction while the split chain materialises the score row and
  calls `softmax_precise`. Different summation orders, different low mantissa
  bits. Measured with production dtypes throughout — bf16 Q as the model
  streams it, and the closing `astype(queries.dtype())` both arms apply — over
  three contexts on two head shapes:

  | `kv_h` × `hpkv` | `head_dim` | `kv_seq` | f32 accumulator differs | max abs err | **bf16 output differs** |
  |---|---:|---:|---:|---:|---:|
  | 8 × 4 | 128 | 64 | 3569/4096 | 8.94e-8 | **0/4096** |
  | 8 × 4 | 128 | 512 | 3643/4096 | 2.98e-8 | **0/4096** |
  | 8 × 4 | 128 | 4096 | 3863/4096 | 2.05e-8 | **3/4096** |
  | 1 × 8 | 256 | 64 | 2048/2048 | 1.13e-4 | **273/2048** |
  | 1 × 8 | 256 | 512 | 2048/2048 | 3.55e-5 | **280/2048** |
  | 1 × 8 | 256 | 4096 | 2048/2048 | 1.46e-5 | **298/2048** |

  Every cell differs in the f32 accumulator, but the divergence only survives
  the bf16 output cast in 4 of 6 — the two `head_dim=128` short-context cells
  are bit-identical to a caller. That is the same shape as the TurboFlash
  retraction: a single-cell check at `head_dim=128, kv_seq<=512` would have
  "confirmed" byte-identity outright. The claim is now stated as measured, with
  the sweep, in `docs/KV_QUANT.md`, `docs/CLI.md` and the
  `--planar-flash-decode` rustdoc, and a GPU test pins both halves (some cell
  differs at bf16; every cell differs at f32, the null control).

  The serve-path A/B cannot settle it either way: with the warm-TTFT bf16-K
  seed live, **neither** `--planar-flash-decode on` nor `off` dispatches the
  kernel (measured 0 and 0, 2418 warm-TTFT bypasses, identical digests) — an
  A/B whose arms both skip the kernel confirms any equivalence put to it.
- **The golden-token decode gates had no configuration in which they ran.** All
  five (`bonsai`, `gemma4`, `qwen3`, `bitnet`, `medgemma`) resolved their
  snapshot from a single `RMLX_KV_TEST_MODEL`, so at most one of them could be
  armed per invocation and none was armed by any shared gate: `make ci` passes
  no `--ignored` and never runs them, and `make gpu-test` / `make ci-perf` —
  which do run them, via the cross-file classifier reach into
  `common::run_golden_test` — set no such variable, so every golden returned at
  its first line and libtest reported `ok`. A committed fixture, a test that
  reads it, and no configuration in which it runs is the shape of a gate that
  cannot fail.

  Each golden now names its own snapshot (architecture + slug) and resolves it
  from exactly two variables: `RMLX_KV_TEST_MODEL`, then the slug under
  `RMLX_O_MODELS_ROOT`. The second arms them by default — every `make` target
  exports that root when it resolves, so a machine holding the snapshots runs
  every golden whose model is on disk, and an operator sets nothing.

  The override applies **only to the golden whose architecture it serves**;
  pointed elsewhere, resolution falls through to the slug instead of standing
  the golden down. `RMLX_KV_TEST_MODEL` is not a golden-only variable —
  `gemma4_kv_cache_equivalence.rs`, `cli_flags_e2e.rs` and `projects_toml_e2e.rs`
  all require it, typically at a Gemma4 path — so a plain override-wins rule
  would have left four of the five goldens silently disarmed for any developer
  with it exported, which is the original defect surviving for exactly the
  developer who most needs these gates. Ranking the slug first instead would
  break the other direction: `RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=<path>`
  would record the fixture from the slug snapshot and ignore the named one.
  `make model-check-full MODEL=…` therefore covers at least the named model,
  and more on a machine with a populated root.

  The per-architecture `RMLX_TEST_MODEL_*` family is deliberately **not** a
  third source. Those variables mean "a snapshot of this family" for the smoke,
  template and NIAH suites, and `docs/TESTING.md` tells operators to export the
  three primary ones persistently for a whole `cargo test --workspace`. A golden
  is a byte-exact fixture over ONE checkpoint's weights, so consulting them would
  let a shell export retarget it to a same-family substitute — a QAT rebuild, a
  re-quantized sibling — producing a token mismatch indistinguishable from a
  decode regression, past an architecture check the substitute passes. Nothing
  is lost: each golden is its own test binary, so
  `RMLX_KV_TEST_MODEL=<path> cargo test --test <arch>_golden_tokens` retargets
  one deliberately and per-invocation, and a snapshot living outside the root
  can be symlinked in under its slug, which every other slug-addressed consumer
  benefits from too.

  Absence and misconfiguration are no longer the same outcome. Nothing
  configured, an existing models root that does not hold the slug, or a
  half-written snapshot under it still **skips** — a developer without the
  weights cannot run the gate, and an interrupted download is an absence rather
  than a wrong pointer. `RMLX_KV_TEST_MODEL` naming a path that is not a
  runnable snapshot, `RMLX_O_MODELS_ROOT` set to something that is not an
  existing directory, and a slug resolving to a snapshot of the wrong
  architecture all now **fail**: skipping on a stale or typo'd export is how a
  wrong pointer reports success without asserting anything, and a mistyped root
  disarms all five gates at once.

  "Runnable" means every file the harness opens by name: `config.json`,
  `tokenizer.json`, and one of `model.safetensors.index.json` /
  `model.safetensors` — the same disjunction `rmlx_loader::load_shard_index`
  tries, mirrored so the check cannot drift from the loader it stands in front
  of. Checking only the JSONs missed the *modal* half-written snapshot, because
  a download writes the small files first and the multi-GB shards last; that
  shape resolved as runnable and panicked inside `load_shard_index`, inverting
  the intended asymmetry in which a fully missing directory is a benign skip and
  a half-present one was fatal.

  Recording is stricter than checking. With `RMLX_REGEN_GOLDENS` set, an
  override pointed at another architecture is a hard failure rather than a
  fall-through: writing a committed fixture from a snapshot the operator did not
  name, while discarding the one they did, gives that golden untraceable
  provenance, and regenerating the whole set under one override would give each
  fixture a different origin silently. On the read path the fall-through is now
  announced (`NOTE <test>: … using <path> instead`) instead of being dropped.
  An override whose `config.json` is present but unparseable fails rather than
  falling through — only a legible, *different* architecture is a statement
  about another golden, and the slug branch already treated the same empty
  string as fatal.

  `crates/rmlx-models/tests/common/snapshot_tests.rs` pins the whole table
  weights-free (26 cases): both probes, every arm of the choice including the
  arch fall-through and its regen-time refusal, the empty-variable spelling of
  "unset", each partial-snapshot shape, both weight entrypoints, and the
  decision-to-return mapping with a `#[should_panic]` case over the `Fail` edge.
  One case builds a directory from the harness's own required-file constants and
  asserts every path the harness opens *by name* — transcribed from the call
  sites in `rmlx-loader` and `run_golden_test`, not from the constants — is
  present, so an under-specified constant fails there instead of being ratified.
  Verified by mutation: ranking the slug first, deleting the fall-through,
  accepting a config-only directory, dropping the weight requirement, emptying
  the weight-entrypoint list, demoting a bad root to absence, removing the
  regen-time refusal, dropping the stood-down note, letting an unreadable
  override config fall through, and making the return mapping swallow `Fail`
  each turn only the cases that claim them red.

  Four Makefile defects fed this and are fixed with it. `model-check-full`
  guarded `MODEL` with `test -n`, which could never fire because `MODEL` has an
  unconditional default; on a machine lacking that snapshot the target
  fabricated a path and forwarded it as `RMLX_KV_TEST_MODEL`, which the harness
  correctly reads as an operator naming a snapshot. It now guards the path. It
  also ran four of the five goldens — `medgemma_golden_tokens` was missing from
  the list — and passed `--ignored` without `--test-threads=1`, so with a Bonsai
  `MODEL` it drove four `#[ignore]` GPU tests across one Metal context from
  parallel libtest threads: the abort the `#[ignore]` rule exists to prevent, in
  the target most likely to be pointed at Bonsai.

  And `RMLX_O_MODELS_ROOT` was exported unconditionally, including a repo-local
  `models/` fallback that need not exist, handing every child a root that was
  never there. It is now exported when an operator **named** one — through
  `.env`, a shell export or the command line — and, for the invented fallback
  only, when that directory exists. The distinction matters because `.env` is
  `-include`d: its values are make variables, not environment ones, so they
  reach a child only through this `export`. Gating the export on the path
  existing would have suppressed it exactly when the path was wrong, and the
  child would have reported "no snapshot configured" and skipped green at the
  one operator who did configure something.

  **Overwriting a fixture is itself gated.** A regenerated golden with no
  recorded reason is indistinguishable from a hidden regression, so
  `RMLX_REGEN_GOLDENS=1` no longer writes unconditionally. When the ids differ
  from the committed fixture the harness re-decodes once at `top_logprobs_k = 2`
  and measures the top-2 gap at the first differing index
  (`first_divergence`), writing only when that gap is at or below
  `REGEN_MAX_TIE_MARGIN` (0.10) — otherwise it panics `REFUSED`, naming the
  index, both ids and the margin. The written file's reason line carries the
  margin, so the fixture records why it moved. A token-count change is refused
  at any margin, and an unmeasurable margin — missing step, absent logprobs, or
  a probe run whose ids differ from the first run anywhere in the prefix up to
  that index — is refused too.

  The 0.10 floor is derived, not chosen: the top-2 logprob gap equals the top-2
  logit gap (the log-sum-exp normaliser cancels), those logits are bf16 after
  the load-time cast, and one bf16 ULP is ~0.0625 for |logit| in [8, 16) and
  ~0.125 in [16, 32). So it admits an exact tie at every magnitude and a
  one-ULP gap only in the lower octave. Tighten rather than widen if a case ever
  lands between.

  `bonsai_8b_mixed_k8g64_v4g64.golden.txt` is **not** regenerated here. It has
  not been touched since the 0.1.0 squash and is stale at index 18 as a
  consequence of the bf16 uniformity cast; regenerating it is a separate change
  so the arming and the fixture stay independently revertable.

- **A shared-KV architecture forced every codec onto the legacy bf16 route.**
  The cross-layer-KV producer path was missing the two fused-over-quant-store
  arms `update_and_sdpa` has, so any model declaring shared-KV layers fell
  through to `update()`'s O(`seq`) CPU dequant with the GPU idle, whatever the
  codec supported — the same rotor kernel that lifts a non-sharing model 4–7×
  logged zero dispatches on one that shares KV. A consumer never needed bf16
  *tensors*; it needed access to the producer's K/V, which the quant store
  already provides. The codec now says which it can offer: `SharedKv::Bf16`
  reuses the tensors the producer's own SDPA materialised (rotating rings,
  `Mixed` / `RotK`, TurboFlash / fused-QK and the legacy fallback all keep their
  behaviour bit for bit), and `SharedKv::Store` re-enters the same fused kernel.
  Keyed off sharing topology and codec, never an arch name (#228, #232).

- **Generation died the moment it crossed the window the prompt happened to
  provision.** `max_seq` is provisioned lazily and only the prefill path grew
  it, so the cap froze at what the prompt needed even when `--max-ctx` allowed
  the sequence. The surviving headroom was incidental — `next_pow2(prompt) -
  prompt` — which is why it looked healthy: a prompt well under the bound
  generates for thousands of tokens first, while one that saturates it dies on
  the first generated token. Measured on Ternary-Bonsai-8B at `--max-ctx 65536`,
  tokens generated before decode died were exactly `headroom + 2` at every
  power-of-two boundary (4096 → 2, 8190 → 4, 16380 → 6, 32760 → 10). Each store
  failed differently — the packed rotor ring raised a shape error, the paged
  code buffers clamped and sliced to zero length, and the bf16 mirrors
  `slice_update` out of bounds, a silent no-op that drops the token while
  `offset` marches on — so `ensure_decode_capacity` sits at the shared decode
  seam, with the `--max-ctx` ceiling and the hard cap kept as loud, typed
  backstops (#233, #238). The K8V4 head-major TurboFlash path bypasses that seam
  and needed the same call plus a re-size of its latched head-major buffers: it
  died with `reshape: Cannot reshape array of size 0` at the next power-of-two
  boundary, and because a deterministic crash reproduces on replay, the retry
  envelope was replacing the real error with a synthetic "replay prefix
  divergence" message. The engine error is now preserved across replay attempts
  and reaches the client (#261, #271).

- **A speculative partial accept left four different KV stores holding the wrong
  prefix.** Four defects at one seam, all reachable from an ordinary partial
  accept or a prompt-cache trim:

  - `truncate_to` compared a block store's `n_tokens` — a row count,
    `b · kv_h · seq` — against a *sequence* target, dropping valid blocks at
    `kv_h > 1` and staying invisible at `b · kv_h == 1` (#284).
  - It kept only the blocks that fit whole, but a verifier writes its
    `K + 1`-token chunk as one block, so every partial accept cuts inside it.
    `blocks` then covered fewer rows than `shape[2]` and the next `dequant` /
    `try_deep_clone` aborted the request. `truncate_plan` now splits the
    trailing block, and each block type cuts every per-row buffer — codes,
    per-group scales, per-group quaternions, per-token norms, and the rotor QJL
    sideband (#378, #391).
  - Twelve arms of `KvStorage::truncate_to` did nothing but set `shape[2] = n`,
    while those stores accumulate their CPU payload independently of it. The
    payload kept over-covering the target, the next append stacked on top, and
    the dequant read back a prefix of the stale buffer: attention over the
    tokens the verifier *rejected*, with the accepted correction missing. Every
    arm now delegates to a store-level `truncate_to` (#382, #393).
  - Every block-accumulating store ended `dequant` by reordering the
    *concatenation* of its blocks as one `[B, S_total, kv_h, D]` run. Each block
    is only `[B, S_block, kv_h, D]`, so above one batch element every block
    restarts at batch 0 and the blocks interleave — measured on `QuantRotorV3`
    at `b = 2`, exactly the batch-1 half of a two-block store disagreed with a
    one-block control, while the `b = 1` control matched bit for bit (#383,
    #392, #400).

- **Short prompts aborted at MSL compile on the `iso*_sym` / `rotor*_sym`
  quant-V kernels.** MLX binds the small per-token norms buffer as `constant`
  where the kernel header wants `device const float*`, which at `kv_h == 1`
  (Gemma4 global layers) killed the dispatch. A shared
  `pad_norms_to_device_floor` zero-pads norms to 16 elements to force the
  `device` binding, so decode stays on-GPU at every `kv_seq >= 1` rather than
  falling back to the host (#220, #279, #286).

- **`--rotor-qjl` defaults to off, and a store's QJL decision is sticky.** The
  rotor update path gated its GPU encode on the process-global flag while the
  SDPA fast path read the store's own decision, so a later toggle reinterpreted
  bytes already written; the update path now reads the store, matching SDPA
  (#230). And QJL forces the rotor fused decode off — there is no MSL kernel for
  it — putting both axes on the CPU at stock defaults. Measured at temp=0 on
  Ternary-Bonsai-8B and gemma-4-e2b across a 4k–32k sweep, QJL-on cost 16–71×
  decode and up to ~3.7× TTFT while buying zero measured accuracy:
  byte-identical short-context output and identical 6k-context needle retrieval,
  on and off. It is the opt-in fidelity knob now, and the loud CPU-path warning
  fires only when it is chosen (#245, #268).

- **A KV kernel returned f32 to a caller holding bf16, re-instantiating the
  decode graph at f32.** `turbo_flash_sdpa` declared f32 kernel outputs —
  correct for online-softmax accumulation — and returned that f32 up. MLX then
  did what it is designed to do: the attention output promoted the residual add,
  and the promotion propagated through the next layer's RMSNorm, its weight
  GEMV, its elementwise ops and the sampler, at no point erroring or warning.
  Proven by kernel identity rather than by timing — two GPU captures one
  `astype` apart show nine f32 kernels leave the `--turbo-flash on` arm's list,
  each replaced by its bf16 twin, and nothing new appears.
  `rot_k_fwht_quantize_gpu` had the same defect by a second mechanism, returning
  f32 *scales and biases* where `mx.quantize` returns them at K's dtype, which
  promoted the graph just as effectively and made the fused and non-fused arms
  of one codec run at different widths. The six flash-decode dispatchers now
  restore the query dtype themselves (#413, #421).

- **The Qwen-MoE KV guard read the declared architecture, not the one the loader
  built.** Both Qwen3.5 arch strings load through one loader into one
  `Architecture`, and that loader decides dense-versus-sparse from a per-layer
  tensor witness, so `architectures[0]` and the model that gets built can
  disagree. The guard rejects rotor / iso / low-bit-K codecs because they were
  measured to destroy perplexity on Qwen sparse MoE — and a checkpoint declaring
  the dense name while shipping MoE tensors ran every one of them to completion:
  no error, a correct-looking run, wrong output. Verified against a real
  256-expert snapshot whose only modification was `architectures[0]`;
  `rotor3_sym` completed at 11.4 TPS. `arch_class()` now reports the resolved
  class and the invariant table is re-run against it after load, which also
  closes a wider hole — the server's resolver never called the guard at all, so
  an explicit `--kv-quant` and a per-request `kv_quant` override reached the
  model unvalidated on every architecture (#352, #379).

- **K8V8 `exit_prefill` could throw "There is no Stream(cpu, 0) in current
  thread"** on an axum worker, once per layer, yielding zero tokens and HTTP
  503. Since MLX 0.31 the default CPU and GPU streams are thread-local, and the
  generate entry points registered the worker's GPU stream but not its CPU one.
  `ensure_cpu_default_stream` is now called at every worker-thread eval entry
  point — generation, embeddings, image, audio, transcribe, speculative (#206,
  #210).

- **Choosing a KV codec to fit a longer context measured *larger* than
  `--kv-quant none`.** `exit_prefill` bulk-encoded the codec store for every
  quant, including the whole family whose decode reads only the bf16 mirror. For
  those codecs the store was written once and then held, unread, for the entire
  decode window — a second full copy of the context per layer, on top of a
  mirror that is already exactly bf16-sized, and monotonically worse as context
  grew. The bulk encode is gated on an exhaustive `decode_reads_packed_store()`
  classifier now; the store is still built for the families whose decode reads
  it and for any cache with no mirror to read (SSD hydrate, `Device::Cpu`, a
  cache that never bracketed a prefill). Measured on Ternary-Bonsai-8B and
  gemma-4-e2b at 4k/8k/32k/64k, every `k8v8` / `k8v4` / `planar` cell now equals
  its `none` cell exactly — ratio 1.0000 at all 32 cells, against 1.33 / 1.60
  flat on Bonsai and 1.43 → 1.92 rising on gemma before. Process memory follows
  where the store was the allocator peak: Bonsai 32k `planar` 9 052 → 5 671 MB,
  64k 17 347 → 10 568 MB. Token ids are byte-identical at temp=0 (#404, #420).

- **`--kv-quant none` allocated a packed q8 store that nothing read.**
  `kv_quant_for_layer` promoted the first two and last eight layers to `K8V8`
  under every base codec, `None` included — on top of the bf16 buffers those
  layers already hold, and a `K8V8` layer early-returns into the bf16 mirror
  anyway. The store could not change an output bit and was pure residency:
  +14.3% on Ternary-Bonsai-8B and +16.0% on gemma-4-26b at 32k, 2.6 GiB at 128k.
  The promotion exists to recover quantization loss, so it is applied only where
  there is loss to recover — a codec property (both sides already at model
  dtype), not an arch or head-count branch, and the K-only families stay
  promoted because their K is 3- or 4-bit. Token ids byte-identical in every
  pair (#411, #419).

- **Rotor K-only appends kept the prefill prefix resident twice.** `k_rotor3` /
  `k_rotor4` never called `drop_blocks_when_ring_live`, so once the GPU ring
  went live the CPU blocks stayed for the whole request — which is why they read
  ~1.5× bf16 while `rotor3_sym` / `rotor4_sym` read ~1.24×. Measured A/B on one
  binary pair, 3 runs per cell: Ternary-Bonsai-8B 990.0 → 717.2 MB at 4k
  (−27.6%) and 8 227.6 → 5 943.7 MB at 32k (−27.8%); gemma-4-e2b 53.9 →
  37.0 MB at 4k (−31.4%) and 395.6 → 254.2 MB at 32k (−35.8%). Every other codec
  measured byte-identical, temp=0 output hashes match across 16 cells, and
  decode TPS is unchanged. The reduced figure is still above bf16, and that is
  the format rather than a bug (#310, #315).

- **`kv_bytes` could not see the GPU ring, and three accountings of the same
  memory had drifted apart.** `KvCache::approx_bytes` — the source of the
  `kv_bytes` trace event — was a per-codec bits-per-element formula that never
  read the store: on Ternary-Bonsai-8B it reported byte-identical KV for
  `k_iso3`, `k_rotor3` and `k8v4`, three storage layouts and one number. Each
  store also had its own `byte_size`, which did count the ring, but nothing in
  production called it. The store is the single source now
  (`KvCache::resident_bytes` → `KvStorage::resident_bytes` →
  `<Store>::byte_size`), built from `Array` shape × dtype and `Vec` length ×
  element size, so a buffer that grows or changes dtype reports the truth with
  nothing to update, and adding one anywhere in the chain fails to compile until
  it is classified as payload or metadata (#246, #258). The metric is sampled at
  one lifecycle point too — post-decode, behind a witness minted by a completed
  decode loop. Five architectures recorded it at the prefill snapshot, before the
  decode-time ring was allocated, and one recorded either figure depending on
  whether the prompt cache hit: on ring-backed codecs one process read 9 228 480
  bytes pre-decode against 35 047 808 post-decode (#259, #273).

- **A decode step that errored was reported as a clean stop.** The loop logged a
  warning, broke, and handed back the tokens produced so far; the server read
  the last token, found it was not EOS, and reported `finish_reason="length"` —
  byte-identical to hitting the token cap, so a caller could not tell "generated
  2 tokens then crashed" from "hit `max_tokens`". That defeats every automated
  gate in the repo, because they all read exactly those two signals: measured
  with decode dying at step 140 of 300, `rmlx baseline` exited 0 and printed
  `decode_tps=138.312`, a plausible number that would have been recorded as a
  legitimate measurement. The forward error now propagates out of the shared
  decode loop and the three per-arch copies. `finish_reason="error"` is
  deliberately not introduced — it is not in the OpenAI enum, and Anthropic's
  `map_stop_reason` catch-all resolves unknown reasons to `end_turn`, so it
  would re-create the same bug on another surface — and each surface instead
  reports the failure the way its own protocol already does: HTTP 503 with the
  standard envelope when blocking, an error event in place of the terminal chunk
  on OpenAI streaming, and the native `error` event on Anthropic streaming
  (#235, #244).

- **A failed prefill completed as a successful run with an empty step list.**
  The shared `chunked_prefill` returned `Ok(None)` on a chunk failure,
  destroying the cause at the boundary, and all five arch callers answered that
  `None` identically. This shipped fabricated zeros as data: `rmlx baseline`
  over the max-ctx ceiling exited 0 and printed
  `ttft_ms=0 decode_tps=0.000 prefill_tps=0.0` as a measurement, which the perf
  canary read, exited green on, appended to its CSV and recorded into `runs.db`.
  The signature is `Result<Array>` now and the cause reaches the caller verbatim,
  which also fixes classification — a `KvCeilingExceeded` is Fatal, where the
  swallowed path degraded to `Error::Other` and was replayed as Migratable
  (#243, #253). The speculative `prefill_chunked` gets the same mandatory
  `exit_prefill` sweep on failure; it previously `?`-returned mid-prefill,
  stranding every cache with `in_prefill = true` so a retained `Vec<KvCache>`
  decoded on un-finalized caches (#251, #263).

- **A `NaN` prefill returned one junk token and reported success.** Every
  architecture detected `NaN` in its prefill logit row and then discarded the
  finding — `if nan_count > 0 { return Ok(steps) }`, logging nothing at any
  level. Greedy selection over an all-`NaN` row returns index 0 whatever the
  model computed, so the run emitted one fixed token, skipped the post-decode
  KV-byte store, and returned success: `rmlx baseline` printed a summary and
  under `--record` wrote a permanent row to the append-only store, and the
  server returned HTTP 200 carrying that token. All six prefill sites go through
  a shared `reject_nan_prefill` now, which logs `nan_count`, `max_abs_logit` and
  `prompt_len` and propagates, raised before the poisoned token is handed to
  `step_fn`. It is classified Migratable: the fault is intermittent, a temp=0
  replay has a real chance of completing, and raising before any delivery is what
  makes that replay safe. `qwen3_5_moe` and `qwen3_vl_moe` hard-coded
  `nan_count: 0` and never computed it, so the guard would have been unreachable
  on Qwen3.6 (#346, #380).

- **`json_schema` could return HTTP 200 with the schema unenforced.**
  `SchemaGrammar::step` treated whitespace as an unconditional no-op wherever it
  was legal, and withholding EOS until the value is complete turns any
  accept-without-progress byte into a cycle a greedy decoder never leaves: the
  mask kept offering whitespace, kept refusing EOS, and the request ran to
  `max_tokens` carrying nothing but indentation. Both JSON engines now cap a run
  of insignificant whitespace at 16 bytes, reset by any content or structural
  byte, so no document becomes unreachable — only indentation deeper than the
  cap is clipped. Whitespace was also being swallowed *inside object key
  strings*, which re-opened the cycle there and made a schema property name
  containing a space unmatchable; raw C0 control bytes are now rejected inside
  any string, key or value, in both engines, matching RFC 8259. The initial
  reasoning channel is no longer inferred from the architecture (#388, #399).

- **`max_tokens` reached `Vec::with_capacity` with no ceiling at all**, in
  `ArchGenerator`, in the speculative path and in every per-arch step vector,
  because the default `--max-tokens-cap` was `u32::MAX`. A request asking for
  4 294 967 295 picked a 275 GB pre-allocation. `enforce_max_tokens_cap` takes
  `cap.min(MAX_COMPLETION_TOKENS)` now, so the operator flag can only lower the
  ceiling, and both routes reject with HTTP 400 rather than clamping. The
  ceiling is 1 Mi tokens: a completion cannot outgrow the context holding it,
  and that is the ceiling the input side and the Anthropic `ctx_max` report
  already use.

- **The retry envelope re-issues the original prompt.** `build_request` appended
  the already-delivered tokens to the prompt *and* the replay loop skipped
  `delivered.len()` engine-output tokens, so on a partial-delivery replay those
  tokens were consumed as prompt and skipped on output. The skip then compared
  mismatched positions and reported a spurious prefix divergence, turning a
  recoverable mid-stream migratable error into a hard failure. The engine
  re-generates the delivered prefix deterministically at temp=0 and the replay
  loop skips it; genuine divergence detection is preserved (#272, #276).

- **`KvCeilingExceeded` no longer calls decode "prefill", and two catch-alls
  stop laundering unknowns into success.** Since decode grows the window too,
  both ceiling errors fire on the decode path, and their message still said
  "prefill" — which the response payload carries, so a user whose prompt fit
  comfortably was told their "prefill request" exceeded the ceiling when
  generation crossed it. The phase word is dropped. Separately, Anthropic's
  `map_stop_reason` mapped an unrecognised finish reason to the successful
  `end_turn`, and retry classification had a `_ => Fatal` arm; classification now
  delegates to an exhaustive match in the crate that defines the error type, so
  a new variant fails the build until it is explicitly classified (#239, #240,
  #264).

- **A single over-cap prompt-cache snapshot stalled the next warm request.** The
  RAM-cap eviction loop only evicted *other* slots and never refused the
  incoming entry, so an empty cache silently accepted a snapshot many times
  larger than the cap. The next identical request deep-cloned it and
  copy-on-wrote the whole KV on the first decode append — a second full-size
  residency, which at long context pushed total residency past physical RAM and
  stalled decode with one multi-hundred-second pause while steady-state ITL
  stayed healthy. Admission is refused when the incoming entry alone exceeds the
  cap, so the repeat request re-prefills exactly like the cold one and peak
  residency is bounded to one live copy; an SSD hydrate of an over-cap block is
  treated as a miss. Keyed off `kv_bytes` against the cap, never an arch or
  codec name (#212, #225).

- **`--prompt-cache-slots 0` rebuilt the cache instead of disabling it.**
  `PromptCache::new` clamped capacity to `max(1)` on the way in while
  `ArchPromptCache::ensure` compared the unclamped argument, so the rebuild arm
  fired on every call — once per generation on all eight architectures. A
  "zero-slot" run therefore discarded a freshly built cache each time,
  re-installing the SSD sinks with it, and reset the hit/miss counters, which
  silently zeroes any measurement taken as `after - before` around a generation.
  Capacity is stored as asked now and `push` refuses admission at 0. In the same
  area, the KV-byte counter was a per-arch-type static: two models of the same
  architecture resident at once wrote the same location, so model B's store could
  advance the sequence model A's recording bracket was watching and be returned
  as A's — into the append-only `events` table, where a wrong row is permanent.
  It lives on each arch's model struct now (#319, #321, #348).

- **The SSD KV tier hydrated nothing, and enforced its budget only at attach.**
  The prompt-cache key gained a model term and the hydrate probe did not; chained
  digests are seeded, so one missing term made block 0 differ and every candidate
  prefix miss — not an error and not a wrong answer, just `ssd_hits=0` forever
  and a full re-prefill on every repeat. `cache_seed` now lives in
  `rmlx_kv_ssd::hashing`, below both consumers, with the model's own signature
  threaded through `attach_at_load` rather than re-derived from a name string,
  so there is no formula left to retype (#350). Separately, `evict_lru_until` was
  reachable only from the once-per-load attach path, so a `serve` that stayed up
  ran past `--kv-ssd-cache-gb` for its whole lifetime and `rmlx_ssd_bytes_used`
  froze at the figure measured when the model loaded: on a 4-request session at
  a ceiling that holds one block, gemma-4-e2b held 2.2× over and
  Ternary-Bonsai-8B 3.0× over, with zero evictions in both. The spill drain
  thread — the only writer that grows the tier, and off the inference path — now
  runs the evict-to-budget pass after every block it records (#30, #342, #344).
  And `last_used` was unix seconds against an `ORDER BY last_used ASC` with no
  tiebreak, so under any realistic request rate many blocks shared a second and
  LRU degraded toward random replacement exactly when the tier matters most;
  stamps are wall-clock microseconds clamped above the highest the process has
  already issued, which is a strict total order across threads and survives a
  restart (#343, #349).

- **A checkpoint whose affine weight-quant bit width has no dequant kernel in
  this build's mlx-c "loaded" successfully**, then failed per token at first
  prefill, spamming a buried Metal kernel-load error 48× per request before
  returning a generic 503. `arch::load_model` pre-flights the resolved affine
  bits — the global default and every `tensor_overrides` entry — against
  `rmlx_quant::affine::SUPPORTED_BITS` before any tensor I/O, so the model fails
  once with one actionable error naming the offending tensor instead of
  advertising itself as loaded (#208, #209).

- **`rmlx baseline` silently truncated an over-length prompt on the GPU.** The
  65 536-token cap has an O(N²) CPU-forward rationale that does not apply to
  `--device gpu`, and truncation was logged at WARN only, so an over-length run
  recorded a shorter measurement that looked like a valid full-length one.
  `--device gpu` hard-errors now unless the caller opts in with an explicit
  `--max-prompt-tokens` or the new `--allow-truncate`; `--device cpu` keeps the
  historical behaviour, where the rationale is real (#213, #223). It also read a
  chat-JSON prompt fixture with a plain string read and tokenized the envelope,
  keys and syntax along with the content, so `--prompt-tokens N` measured N raw
  *file* tokens — inflating counts past both the model's context ceiling and the
  prompt cap on long-context fixtures. It renders the messages through the
  model's own `chat_template.jinja` first now, the same render-then-tokenize path
  the HTTP chat-completions route uses, and hard-errors on a chat-shaped fixture
  it cannot parse rather than falling back to the envelope (#291, #312).

- **`kv_quant` and `model_namespace` are recorded labels, not validated enums.**
  The metrics-side allow-list was a stale hand-maintained mirror of the codec
  grammar, missing roughly 18 real tokens, so every observation and event row for
  the rotation / sym / planar / turbo families was silently rejected at ingest.
  `canonicalize_kv_quant` no longer rejects anything: it lowercases, trims,
  normalizes a tiny alias set, and records everything else verbatim — including a
  codec name this binary has never heard of. The grammar-mirroring helpers are
  deleted with it, because the mirror could only drift again (#214, #224).

- **An implausible row could win a `bests` cell and publish.** `observations`
  held 20 `prefill_tps` rows storing `(prompt_tokens - 242) * 1000` under
  `unit='tps'` — up to 998× any real rate — plus 95 rows across four metrics
  whose value is exactly `0.0`, and four of them had published into
  `BENCHMARK_CHAMPIONS.md`. The view ranks by magnitude and nothing on the way in
  bounded the number; the registry carried `(unit, direction)` only, and
  `doctor`'s unit check compares the label, never the value. The registry gains a
  per-metric `Bounds` — a ceiling, plus whether `0.0` is itself a measurement, as
  a rate is zero only when nothing was produced — enforced at ingest, *generated*
  into the `bests` view so the two cannot drift, and reported by `doctor`.
  `rmlx baseline` no longer fabricates `0.0` for an unmeasured phase:
  `PhaseTiming` is `Option` per phase, prints `n/a`, writes an empty CSV field
  and serializes to `null` (#401, #418).

- **The missing-nax-kernel warning fired on every build regardless of
  hardware**, including runners that never had a Neural Accelerator and
  legitimately ship none of the kernels — while asserting, in the same breath as
  reporting a confirmed absence, that the pinned metallib ships them. The
  warn-or-stay-silent decision is a pure function over (NA-class host, kernels
  present) now, and the message no longer asserts what an uninspected bottle
  contains (#262).

### Added

- **The first KV codecs in this tree's history that hold fewer resident bytes
  than bf16.** Two independent changes, and neither is universal — read the
  architecture column before picking one.

  *The iso ring cleared the floor.* `QuantKGpuRing`'s scale and norm planes were
  `f32`; they are stored at `KV_SIDEBAND_DTYPE` now. That takes 4.125 bits per
  value off iso and 5.5 off rotor: **iso 16.25 → 12.125** bits/value at
  `head_dim = 128` (0.758× bf16, the first member under 16.0) and **rotor
  21.75 → 16.25**, which is still above the floor and always will be — rotor
  spends one whole `u32` code word per 3 head-dim slots, 10.67 bits per value
  before any sideband, and no sideband change can fix a code cadence. Planar is
  unchanged at 22.00. The affine sideband is **32 bits per group** (bf16 scale +
  bf16 bias), measured, not the 64 the byte model previously assumed.

  *The `Mixed` mirror is built only where something reads it.* `mixed_*` and
  `rot_k_*` held a full bf16 K/V mirror beside their packed store on every
  architecture, which is why they measured **1.29× `none`**. The mirror exists
  for a cross-layer-KV consumer, so it is now built only where a layer reads
  another layer's K/V; combined with an 8-bit in-family boundary floor,
  `mixed_k8g64_v4g64` on Ternary-Bonsai-8B went **9.29 → 7.29** bits per value
  at 4k and **9.16 → 7.08** at 32k, 0.784× / 0.774× `none`, at no resolvable
  decode cost (ABBA-paired decode-TPS ratio 0.9957, n=3, sd 0.0028, inside a
  same-code-path control band of 1.0061 ± 0.0152 — INCONCLUSIVE on throughput,
  greedy token ids byte-identical on three architectures).

  Measured end-to-end with `rmlx serve` at a 928-token prompt, `kv_cache_bytes`
  from the server's own N16 event, temp=0, ×`none` in parentheses:

  | model | `mixed_k8g64_v4g64` | `rot_k_v8g64` | `iso3_sym` | `k_iso3` |
  |---|---|---|---|---|
  | Ternary-Bonsai-8B | 78 679 040 (**0.519**) | 97 145 856 (0.641) | 145 858 560 (0.962) | 148 496 384 (0.980) |
  | Ternary-Bonsai-27B | 187 011 072 (0.843) | 199 778 304 (0.901) | 217 759 744 (0.982) | 219 766 784 (0.991) |
  | Qwen3.8-27B | 188 172 288 (0.838) | 201 240 576 (0.896) | 218 103 808 (0.971) | 221 315 072 (0.986) |
  | Qwen3.6-35B-A3B | 74 952 192 (0.876) | 80 023 040 (0.935) | arch-refused | arch-refused |
  | gemma-4-e2b | 15 495 168 (1.232) | 19 556 352 (1.555) | 11 022 336 (**0.876**) | 11 784 192 (0.937) |
  | gemma-4-12B | 357 219 840 (1.021) | 365 336 064 (1.044) | 342 896 640 (0.980) | 348 395 520 (0.996) |

  The two rows that matter for expectation-setting: `mixed_*` / `rot_k_*` are a
  win **only** on an architecture whose layers do not share K/V — on shared-KV
  Gemma4 they are *larger* — and the iso family pays for its bytes in decode:
  **0.64–0.69×** `none`'s TPS on Bonsai-8B, 0.77–0.86× on gemma-4-e2b, and
  0.85–0.97× on the 27B/12B models. Rotor is worse on every one (0.58–0.96×).
  No codec in the tree is both smaller and faster than bf16.

- **Qwen3.8-27B serves, including with its MTP sidecar.** The
  `Qwen3.8-27B-MTP-mxfp8` sidecar ships a plain SwiGLU `layers.0.mlp` and no
  expert keys; `num_experts` was read as a required key (rejecting it at config
  parse) and `MlpBlock::Moe` was built unconditionally (rejecting it at tensor
  load). Both key off facts now — the counts read against the same
  `num_experts == 0` "dense, no experts" sentinel `Qwen3_5MoeConfig` already
  uses, and `MtpLayer` probes `mlp.switch_mlp.gate_proj.weight` exactly as
  `build_mlp` does per layer. Read the MTP throughput note under **Changed**
  before enabling it.

- **Per-dispatch `trace!` on the two PlanarQuant kernels**
  (`planar_flash_decode_sdpa: dispatch`, `planar_fused_qk: dispatch`), matching
  every sibling KV kernel. Their in-process dispatch counters have no caller
  outside tests, so these events are the only way a shipped binary can answer
  "did this kernel run".
- **`fused_qk: skipped` trace with a `reason` field.** Every fall-through in
  `try_fused_qk_dispatch` now names the gate that rejected, and the `head_dim`
  gate carries the observed value. This is what identified the Gemma4 result
  below in one run instead of by reading the dispatcher.

- **Fused flash-decode kernels for the iso and rotor KV families.** `k_iso3` /
  `k_iso4` and `k_rotor3` / `k_rotor4` decoded by CPU-dequantizing the whole K
  prefix on every step — O(`seq`) host work per token with the GPU idle, which is
  what pinned them at single-digit TPS. Three MSL kernels close it:
  `rotor_flash_decode` (QK over the packed rotor store, online softmax, bf16-V
  SV, in two dispatches per step; the Cl(3,0) rotor decode runs inside the
  attention inner loop, so no bf16 or f32 K is materialised and nothing restages
  through the host — #217, #229), `iso_flash_decode` (one left Hamilton product
  per lane, no threadgroup staging and no barrier, sharing the codec-agnostic
  pass-2 LSE merge with the rotor and planar kernels — #218, #247), and
  `rotor_flash_decode_symv`, which reads V straight from its own packed ring so
  `rotor3_sym` / `rotor4_sym` need no bf16 mirror on either axis (#219, #281).
  Each carries a GPU-versus-CPU-dequant numerical oracle across `head_dim`
  64/128/256/512, GQA and additive masks.

- **`rmlx bench`** — a repeated-run decode instrument. It serves one (model, KV
  codec, context, generation length) cell `--warmup` + `--runs` times in-process
  and reports TTFT, ITL p50/p99, decode TPS, prefill TPS and filled-prefix KV
  bytes as a median with the observed min/max and range%. It writes nothing;
  `baseline --record` remains the path to the append-only store. It also refuses
  to produce a number it cannot stand behind: `Architecture::kv_cache_bytes()`
  returned a bare `u64` in which `0` was both the never-written initialiser and a
  legal reading, and in which a generation that returned before its KV-byte store
  left the *previous* generation's figure readable, indistinguishable from a
  fresh one. The accessor returns `KvBytesSample { bytes, seq }` now, and an
  unadvanced sequence and a reported zero are two differently-worded hard errors
  (#303, #322).

- **GPU profiling that works headlessly.** `rmlx baseline --gpu-capture` (with
  `--gpu-capture-skip` / `--gpu-capture-steps`) captures a bounded decode window
  as a replayable Metal trace. The window is selected in the shared decode loop,
  so it is model- and codec-agnostic with no per-arch wiring, and the whole path
  sits behind the `metal-capture` cargo feature — an ordinary build has no flag,
  no branch and no undefined reference to `mlx_metal_start_capture`, so it cannot
  capture accidentally. It conflicts with `--record`, because capture drops
  decode to single-digit TPS and that number must never reach `runs.db` (#307,
  #324). `make build-capture` signs the binary with
  `com.apple.security.get-task-allow`, since Cargo's linker-signed ad-hoc
  signature carries no entitlements and a freshly built binary is not attachable
  by Apple's GPU tools, and a preflight checks the toolchain before a run writes
  several GB (#325, #329). GPU time itself comes from `xctrace`, and
  `make gpu-test` now pins the whole `MTL_SHADER_VALIDATION_*` environment rather
  than inheriting it and asserts the validation banner appeared — an
  out-of-bounds device store from a Metal kernel is otherwise dropped silently,
  the command buffer completes, the process exits 0, and the assertions over the
  frozen buffer still pass (#328, #330, #347).

- **MLX identity is recorded, and checked at run time.** The build stamps whether
  the resolved MLX metallib carries the `steel_gemm_fused_nax` GEMM family, and
  migration `004` records `mlx_nax` on every `events` row as a free-form label
  (#275). That stamp describes the machine that *built* the binary, which is the
  wrong answer for anything shipped: the bottle and the release tarball both link
  `libmlx.dylib` through the moving `opt` symlink, so they run against the
  installing user's MLX. A runtime probe now walks dyld's image list for the
  library actually loaded, scans its colocated metallib, and warns only when the
  host has a GPU Neural Accelerator and the kernels are confirmed absent — on
  M5-class hardware that absence is a silent 2.2–3.7× prefill and TTFT loss with
  correct output, which reads as a model-code defect rather than a toolchain one.
  Host-class gating runs first and short-circuits before any file access, because
  M1–M4 legitimately ship zero of these kernels and warning there would bury the
  one host where the absence costs something (#305, #339). A version skew between
  the MLX compiled against and the MLX loaded warns for the same reason (#216,
  #249).

### Documentation

- **Lifting ε is answered negative, and the residual it was blamed on was the
  wrong residual.** Two standing proposals — re-index the flash-decode P1 grid
  by KV head, and lift the `mixed_*` packed-store path to ≥ 400 GB/s — are
  recorded as answered-negative in `docs/KV_QUANT.md` § "Lifting ε does not pay"
  and `docs/PERF_BASELINE.md`. The grid mechanism is true and re-verified at
  source (`turbo_flash_p1.metal:17,20,27`; `:29-32` in each iso/rotor P1), but
  its consequence is not: removing the *entire* query-head class moves the ON
  arm only 0.231× → 0.311× of the generic path, because the kernel is
  issue-bound (Integer and Conditional 50.45%) rather than memory-bound (LLC
  10.66%), the redesign adds ~54–126 f32 per lane against 22.24% occupancy with
  no spill headroom, the one store dense enough to clear a lifted ceiling
  (`tsym3`) is byte- and token-identical to `none`, and the best real codec on
  that kernel (`iso4_sym` @32k) decodes at 0.170× of `none` while holding more
  bytes than bf16. The ≥ 400 GB/s and ≤ 4× pass criteria both presume a bound
  the counters contradict. `docs/KV_QUANT.md` had attributed the residual to
  "the f32 `partial_o` P1→P2 round trip plus the thread-0-serial softmax
  between threadgroup barriers" — `turbo_flash_p1` has zero
  `threadgroup_barrier`, no thread-0 section and no `partial_o` at all, and the
  P2 that does have them is 3.66% of GPU time; the iso/rotor P1s do carry both
  and have never been profiled. `docs/models/bonsai/27B/rMLX.md` restated the
  same attribution as "chiefly because". Both corrected, and the one unmeasured
  cell (`iso_flash_decode_symv_p1`) is recorded with a pre-registered decision
  rule.

- **Metal System Trace granularity is per encoder, and the driver-coalescing
  claim was unsupported.** `scripts/mst_capture.sh`, `docs/PROFILING.md` and the
  XML unescaper in `rmlx-mlx` all said the driver merges consecutive compute
  encoders into one GPU kick so a row can cover several. Re-derived from the two
  bundles under `<RMLX_HOME>/traces/mst`: 14 140 rmlx `metal-gpu-intervals` rows
  carry 13 996 distinct `encoder-id`s, and the same run's
  `metal-application-command-buffer-submissions` sums 13 997 encoders over
  14 512 command buffers, of which 13 592 hold exactly one. No row is a
  coalesced kick, and no `gpu-channel-name` in either export is anything but
  `Compute` / `Fragment` / `Vertex` — the `&` the unescaper exists for comes
  from the compositor's IOSurface labels. Also corrected: "no pipeline or
  function names in the export" is true of `metal-gpu-intervals` only. The
  bundle names 52 rmlx pipelines in `metal-shader-profiler-shader-list`; what is
  missing is a join key, because the stock template records `Shader Timeline:
  Disabled` and `metal-shader-profiler-intervals` exports zero rows —
  configuration, not a device ceiling. Counters genuinely are dead headlessly:
  the bundle holds exactly one, `RT Unit Active`.

- **The gemma4 SWA comment claimed quantized codecs take a full-size path.**
  They do not, and never did on this tree: `KvCache::with_quant_max_seq_window`
  selects the rotating ring on `window > 0` alone, `update` / `enter_prefill` /
  `exit_prefill` all return before any codec dispatch when it is set, and
  `KvStorage` is allocated lazily, so a windowed layer under `k8v8` holds
  exactly what it holds under `none`. There was no "pending follow-up" branch
  behind the comment. Corrected at the gemma4 site and at the five places that
  restated it — `gemma3/generate.rs`, `speculative/mod.rs` (which additionally
  claimed the window is *ignored* for quantized modes, and named a per-arch
  default table that no longer exists), the `rotating` module doc, the
  `KvCache::rotating` field doc, and `docs/KV_CACHE.md`'s windowed-ring scope
  note.

- **`CLAUDE.md` filed ParoQuant as a rotation-based KV family and both
  ParoQuant and IsoQuant as "rotation-KV references".** ParoQuant is a
  weight-only INT4 method — the token `kv` does not occur in its upstream repo
  and its calibration path drops `use_cache` — and rMLX has no `KvQuant::Paro*`
  variant, no ParoQuant `KvStorage` and no `--kv-quant` name for it. IsoQuant
  upstream is five files, two stage-1 CUDA kernels, no cache and no decode
  path, so rMLX's `iso*` codecs have no upstream KV counterpart to port. The
  capability line now names the four families that are KV codecs
  (TurboQuant, IsoQuant, PlanarQuant, RotorQuant) and says where ParoQuant
  actually lives; the reference entries state each repo's real scope. The same
  parenthetical in `README.md` carried the same error twice and is corrected
  with it. No code changes — `docs/WEIGHT_QUANTS.md` §7 already filed ParoQuant
  correctly.

- **`docs/PERF_BASELINE.md` H2's "active bytes/step" was nameplate arithmetic.**
  The four figures (`~2 / ~4 / ~3.5 / ~3.5 GB`) were active-parameter counts
  times the weight-quant bit width. They dropped the quantization sidebands,
  the tied `lm_head` (all four models set `tie_word_embeddings`), the per-arch
  auxiliaries — and KV traffic entirely, while being divided into a decode rate
  measured at a 4 096-token prompt. Replaced with a tensor census from
  `scripts/perf_ceiling.py` (`config.json` + safetensors headers), with the
  invocation recorded so the row is re-checkable without a device. The KV term
  is a **second producer** — the script transcribes
  `decode_reads_packed_store`, `feeds_bf16_{k,v}_at_decode` and
  `kv_quant_for_layer` into Python by hand and nothing gates the two copies
  against each other, which the doc now states rather than calling the term
  "the engine's own accounting". The correction is not a constant bias: three
  ceilings fall and Qwen3.6-35B's rises 9%, because 30 of its 40 layers are GDN
  and hold no attention projections. The band tightens from 1.84x–2.66x to
  1.69x–2.15x, dissolving the reading that Bonsai is a factor-of-1.4 outlier
  with arch-specific overhead worth hunting — most of its excess was the
  missing KV term, 13.9% of its stream at that shape. `measured decode_tps` is
  untouched; only the denominator moved. Carried through every dependent claim
  in the file — the decode-only re-baseline table, the H2 addendum's per-step
  overhead comparison against llama.cpp (still INCONCLUSIVE, ranges still
  overlap), H9b, H10 and the net narrative — and through
  `docs/KV_CACHE.md`'s restatement of the band, which also repeated an
  unmeasured literature envelope that `PERF_BASELINE.md` had already retracted.
  H9b and H10 additionally ranked Qwen3.6 "the best of the four models" by
  ratio-vs-ceiling; that is a comparison across models of different size, which
  the same document forbids, so both now read the scale-free quantity
  (per-step overhead, 5.30 ms against a 4.65–6.93 ms range) and reach the same
  conclusion.

- **The iso / rotor stored bit rate is now stated symbolically and at the point
  of selection.** `docs/KV_QUANT.md` said the 16.25 bits/value result is
  "head_dim independent" and then gave two different values for two head dims.
  The rate is `16 + 32/head_dim` for iso and `(64·⌈D/3⌉ + 32)/D` for rotor,
  floors 16.0 (approached from above, never reached) and 21.33 — so it is the
  *sign*, not the rate, that holds at every finite head dim, and both are
  strictly above bf16's 16.0. Both formulas are derivations from
  `QuantKGpuRing::alloc`, marked as such. The "Memory and bit-rate summary"
  table omitted the ring families entirely while listing seven codecs below
  bf16; the four ring rows are added, flagged as the only rows whose decode
  reads the store they describe. `rmlx info --list-cache-types` — where these
  tags are actually chosen — now says the bit width in an `iso_*`/`rotor_*`
  name is its codebook rather than its stored rate.

- **Three stale claims found beside the above and corrected.** `--rotor-qjl`
  has defaulted to `off` since the rotor Metal path landed, but four places in
  `docs/KV_QUANT.md` still called `on` the default — including the decode-cost
  caveat, which therefore described the CPU path as what an operator gets.
  `docs/KV_QUANT.md` also said the V-only `iso3`/`iso4` codecs "measure ≈2.1×
  `none`", which its own codec-disposition section measures as byte-identical
  (they build no store); the 48.25 bits/value figure is the rate they would
  cost once a kernel reads one, and now says so. And `KvQuant::K8VTurbo3`'s doc
  comment still described itself as the auto default for Gemma4 small, which
  the retired per-arch table used to make true.

- **Metal System Trace does instrument `rmlx` on this host.**
  `docs/PROFILING.md` claimed the headless path "exports zero rows for `rmlx`".
  Reproduced twice at xctrace 16.0 / Xcode 26.6 on M5 Max — 6 931 rmlx rows
  (gemma-4-e2b `none` @4096, `target/release/rmlx`) and 14 140 (Bonsai-8B
  `k8v4` @8192, `target/release-perf/rmlx`), `Compute` channel,
  `start-latency` populated, no `sudo` and no entitlement. The recordings that
  produced the claim held 24 rows for the *whole machine* over 25 s, against
  36 441 here over 20 s, so what failed was the recording. The false sentence
  is removed and the real boundary stated: MST carries no kernel names and no
  counters, which is what the Xcode GUI replay is for.

- **The GPU suite is not clean, and `docs/TESTING.md` said it was.** The Metal
  shader-validation aggregate reports 160 invalid accesses in MLX's own
  `affine_qmm_t_splitk`, so the first reader to hit the aggregate had nothing to
  compare against. They are out-of-bounds device *loads* in
  `QuantizedBlockLoader::load_safe`, which bounds its row index against the
  compile-time `BK` instead of the runtime `num_outs`, so the guard cannot fire
  and a transposed quantized matmul over an unaligned N dequantizes the tile's
  out-of-range rows straight from device memory. Reads only: the store side is
  clipped by the sole reachable store branch for that instantiation, and the
  block MMA performs no reduction across n. The values are shown bitwise not to
  reach the output over 66 controlled cells spanning bits, group size, codec,
  dtype and both kernel families, with the primary control striding a view over a
  NaN-padded quantized triple so the same instantiation runs with the
  out-of-range rows provably poisoned; end to end, `eval ppl` on two
  architectures is identical across validation off, zerofill and allow. Two rules
  go with it, both got wrong once: the unaligned unit is 32 on the non-batched
  path and 64 on the batched and gather paths, and no diagnostic never licenses
  no out-of-bounds read. The suite's own totals are corrected with it — 352 GPU
  tests passed in a full run, not 3 532; the runner does not print that total
  when a validation hit makes it exit early, so it has to be summed from cargo's
  per-crate result lines.

- **Two Bonsai-27B benchmark corrections.** The `k8v4` crater is real: a clean,
  boundary-safe re-measurement reproduces 50.4 / 15.5 / 5.0 TPS at 4k/32k/128k
  against a same-machine `k8v8` control of 45.1 / 37.3 / 21.4, so the tq4-V
  dequant cost stands. But the same re-measurement surfaced that every recorded
  256-token `k8v4` cell was a truncated crashing run — 242–250 of 256 tokens, at
  the power-of-two decode boundary, masked to the streaming client — so the row
  is marked as a crashing cell and "avoid 4-bit V (slow)" becomes "broken and
  slow" (#241, #260). The reported 2.6–3.4× prefill deficit against the sibling
  backends does *not* survive: the campaign ran on a Homebrew MLX bottle that
  silently shipped zero NAX GEMM kernels on this host, a prefill-only ~3.8×
  matmul loss. Re-measured with the pin verified active, cold TTFT is
  4.8 / 10.7 / 24.4 / 53.4 / 136.6 / 408.8 s from 4k to 128k against the
  published 14.8 / 32.0 / 67.9 / 147.5 / 335.0 / 815.4 — a 3.08× deficit becomes
  1.99×, and roughly 1.0–1.3× against the NAX-correct mlx-lm champion, parity at
  4k–8k. The GDN-recurrence explanation was numerology; the recurrence is ~2% of
  prefill. Decode TPS and KV bytes are NAX-independent and unaffected (#248,
  #274).

### Changed

- **Every speculative accept rate and speedup in the docs is re-derived, and
  the MTP sidecar is a net win on both GDN hybrids.** Three faults had to be
  cleared before any of these numbers meant anything: the GDN rollback replaying
  through a scratch KV stack, the sidecar serve path holding a second copy of
  the verifier, and a round-loop `decode_tps` field that divided emitted tokens
  by prefill-plus-decode (all three in **Fixed**). Re-measured on this branch at
  temperature 0, `--kv-quant none`, n=6 pooled over two passes in palindromic
  order, decode measured first-emitted-token to last:

  | Verifier | Drafter | Block | Accept | Decode vs no drafter |
  |---|---|---|---|---|
  | Qwen3.8-27B-mxfp8 | MTP sidecar | 2 | 0.67–0.88 | **1.09–1.36×** |
  | Qwen3.8-27B-mxfp8 | MTP sidecar | 3 | 0.53–0.73 | 0.92–1.23× |
  | Qwen3.6-35B-A3B-8bit | MTP-5bit | 3 | 0.65–0.90 | **1.02–1.34×** |
  | Qwen3.6-35B-A3B-8bit | DFlash | 16 | 0.49–0.61 | 0.78–0.97× |
  | Qwen3.6-35B-A3B-8bit | Eagle3 | 5 | 0.26–0.36 | 0.61–0.74× |
  | gemma-4-e4b-it-mxfp8 | E4B assistant | 6 | 0.24–0.73 | 0.79–1.90× |

  Each range spans the prompt classes `prose`, `code` and a 4k-context prompt;
  accept rate is a property of the (verifier, drafter, prompt) triple and a
  single-prompt figure predicts nothing. **`--draft-block-size` above 3 is a
  no-op on both shipped Qwen3.5-family MTP sidecars** — neither config carries a
  `block_size` key, so both take the loader default of 3 — and block 2 measured
  faster than block 3 on every prompt class for Qwen3.8-27B. Every cell is now a
  row in `runs.db`; before this change the database held no speculative
  observation on any GDN hybrid at all, which is why the drift went unnoticed.

  Two previously published figures do not survive and are removed rather than
  footnoted: **Qwen3.8-27B MTP at `block_size 3` is not 0.86× baseline** (0.92×
  on a 4k prompt, 0.98× on prose, 1.23× on code — that reading was taken with
  the doubled verifier load live), and **Qwen3.6-35B-A3B MTP is not +4.2%**
  (+2% to +34%, with DFlash at −3% to −22% rather than −37% and Eagle3 at −26%
  to −39%). The pre-rollback-fix `0.755 / 23.9 TPS / 1.28×` reading for
  Qwen3.8-27B stays retracted — a corrupted verifier agrees with its drafter
  more often than a correct one does — but the correction to it was itself
  measured on the doubled load and is superseded by the table above. The blanket
  form of that retraction does not survive either: DFlash (0.488–0.608) and
  EAGLE-3 (0.263–0.362) reproduce close to their recorded accept rates, so the
  claim that *every* accept rate taken on a GDN hybrid before this branch is
  inflated is withdrawn. The EAGLE-3 `fcs` norms raise accept by 1.04–1.52×, not
  the "more than doubled" previously claimed.

- **Gemma4 with `mixed_*` / `rot_k_*` and an SSD-tier hit now fails loudly where
  it previously produced silently-wrong output.** `none_bf16_payloads` never
  persists a bf16 mirror for `KvStorage::Mixed`, so a hydrated cache on a
  cross-layer-KV architecture has no mirror to rebuild and no way to serve the
  layers that read another layer's K/V. That combination used to hydrate anyway
  and decode off a mirror that was not there. It is refused now, on the artifact
  rather than on the declaration. This is a **behaviour regression for anyone
  running that exact combination** — the remedy is `--kv-quant none` (or any
  mirror-fed codec) on Gemma4 with the SSD tier enabled, and the better end
  state (persist the mirror, or refuse to hydrate and re-prefill) is filed
  separately.

- **`--kv-quant` help states each codec's real disposition, and is gated against
  it.** The help text used to name codecs without saying that 17 of the 28 build
  no packed store and decode byte-identically to bf16. It now carries INERT
  banners derived from `ALL_KV_QUANTS` + `decode_reads_packed_store` /
  `feeds_bf16_{k,v}_at_decode`, and `make check-kv-codec-disposition` (in
  `make ci`) fails the build when the help or `docs/KV_QUANT.md` disagrees with
  the runtime disposition. `auto` and `DEFAULT_KV_QUANT` resolve to unquantised
  bf16 on every architecture and every context length, and the help says so.

  Verified end-to-end, not just in the type: all **17** inert codecs served
  Ternary-Bonsai-8B at a 928-token prompt with `kv_cache_bytes` **151 584 768**
  — byte-identical to `none` — and greedy token digest `4f26f49e2b3529f6`,
  byte-identical to `none`. 18/18 cells, both axes.

- **`xctrace`'s "no rows" refusal splits into the two states it was
  conflating.** A table with no rows at all (the recording captured nothing)
  and a table holding other processes' rows and none of this one's (the
  recording ran; this process was not in it) have opposite remedies, and both
  were reported as `parsed but contains no rows` — which `scripts/mst_capture.sh`
  then annotated with "the run itself failed", sending the reader to re-run a
  workload that is fine. `XctraceError::NoRowsForProcess` is the second case
  and carries the row count and the process census the recording *did* see, on
  both the plain and the `--skip-ms` entry branch. A third state,
  `SkipRemovedEveryRow`, covers the case those two cannot describe honestly: the
  process IS in the recording and `--skip-ms` cut past its work.
  `SkipExceedsSpan` does not subsume it — that guard fires on
  `origin >= max(start + duration)`, so one long submission straddling the
  origin keeps it silent while every row's `start` is below the floor, and the
  refusal would otherwise claim the process was absent while printing its own
  row count in the census. This is the diagnostic that would have identified the
  four zero-row recordings above as failed recordings.

- **`--kv-preset auto` resolves to `DEFAULT_KV_QUANT`, the same constant
  `--kv-quant auto` resolves to.** It previously ran its own resolver — a
  decision tree over `sysctl hw.memsize` and a `config.json` parameter estimate
  that returned a "compressing" preset when the model plus its bf16 KV would not
  fit. Two `auto` surfaces resolving independently are two defaults that can
  disagree, and these did: at identical flags
  (`--max-ctx 131072 --prompt-tokens 4096`) `--kv-preset auto` resolved
  `TurboSym4` on Ternary-Bonsai-8B and `K8V8` on gemma-4-e2b while
  `--kv-quant auto` resolved `None` on both.

  Worse, the answer bought nothing. Every preset that tree could return holds
  resident KV **byte-identical** to `fp16` — measured below — so it warned that
  the model might not fit and then picked a codec with no memory effect. Its own
  KV estimate was, by its docstring, 10–30× off, so it could not be repurposed
  into a diagnostic either. The tree and its two hardware queries
  (`rmlx_core::unified_memory`, `rmlx_loader::model_size`, whose only caller it
  was) are removed. Every named preset still resolves to its own codec.

- **No `--kv-preset` row is described as a memory setting any more, because none
  is one.** `q8`, `speed`, `quality`, `planar`, `planar3` and `k_only_planar`
  each resolve to a codec whose decode reads the bf16 mirror, so `exit_prefill`
  never builds its packed store. `--help`, `docs/CLI.md` and `docs/KV_QUANT.md`
  now say so.

- **A KV codec that changes nothing says so at resolve time.** `validate_resolved`
  emits a `warn!` when the resolved codec keeps no packed store and is not
  `none`: 17 of the 28 codecs the enum spells are in that class, and selecting
  one previously produced a confident "resolved KV cache quant" log line and no
  hint that resident KV and every generated token were identical to bf16.
  Warn-and-proceed, like the existing CPU-hot-path classification beside it.

  Both honesty warns are now emitted **once per `(arch, codec)` per process**.
  They classify a resolved configuration, not a request, and `validate_resolved`
  runs per request on the normal and speculative paths — so an operator serving
  under one of them was getting the same paragraph for the process lifetime.

  The warning says the codec is not known to cost anything either. That is
  deliberate: the per-layer dispatch cost an earlier draft charged it with is
  INCONCLUSIVE at all five recorded ABBA cells, so the class is *equivalent* to
  bf16, not beaten by it. `docs/KV_QUANT.md` § "Codec disposition" carries the
  axis-by-axis reading and the consequence for the dominated-vs-unused split:
  the codecs that are genuinely dominated by the baseline are the ten that read
  their own store and measure 1.003×–1.541× larger, not the seventeen that tie
  it.

- **`--kv-preset auto` works in `--registry` mode.** It was rejected there with
  exit 78 and "auto-selection needs config.json to estimate model size" — a
  reason that stopped existing when the selector did, since the resolver now
  reads a constant and opens nothing. `--kv-quant auto`, the same constant under
  another flag, was accepted on the same command line.

  Measured with `scripts/bench/codec_inertness_probe.sh` — one `rmlx baseline`
  per codec at temperature 0, 27 codec spellings × 2 architectures × 2 contexts,
  108 runs, all exit 0. gemma-4-e2b is `kv_h == 1` with shared-KV and
  sliding-window layers; Ternary-Bonsai-8B is `kv_h == 8` dense.

  | | e2b 4k | e2b 32k | Bonsai 4k | Bonsai 32k |
  |---|---:|---:|---:|---:|
  | `none` resident KV (B) | 32 194 560 | 217 976 832 | 570 507 264 | 4 667 277 312 |
  | of 27 driven spellings, those byte- and id-identical to `none` | 17 | 17 | 17 | 17 |
  | spellings larger than `none` | 10 | 10 | 10 | 10 |
  | spellings **smaller** than `none` | **0** | **0** | **0** | **0** |

  The two 17s in this entry are different sets of the same size and it is a
  coincidence: 17 of the 28 enum variants are in the inert class (`none` is
  not one of them), while 17 of the 27 driven spellings measure identical to
  `none` (16 inert ones plus `none` itself — the 28th variant,
  `rotor_k_3_asym_*`, was left to the family-parameter test).

  No codec is removed. The full disposition — which codecs are dominated, which
  are merely losing today, and why the rotation families stay despite both — is
  `docs/KV_QUANT.md` § "Codec disposition", pinned by a
  `DISPOSITIONS` table that every variant must appear in exactly once.

- **`--kv-quant auto` resolves to unquantised bf16, on every architecture and
  every prompt length.** Previously two resolvers disagreed with each other: a
  per-arch table returned `K8V8` / `K8V4` / `Planar` / `Mixed{k8g64,v4g64}`
  from arch class, `hidden_size`, the MoE flag, the PARO flag and
  `quantization.bits`, and a separate per-prompt-length server policy then
  re-picked `K8V4` / `None` / `K8V8` / `Planar` per request, overriding it.
  Both are removed.

  The second one's reach was narrower than it looks, and worth stating so the
  removal is not oversold: on `rmlx serve --model` it never fired, because the
  CLI resolves `auto` before the server starts and hands the generator a
  concrete codec, which the server reads as operator-supplied. It was live in
  `--registry` (multi-model) mode, where no codec is pre-resolved — measured
  there on one gemma-4-e2b: the same model served `K8V4`, `K8V4`, `None` and
  `K8V8` across four requests of 110 / 3 010 / 9 010 / 30 010 prompt tokens. A single constant,
  `rmlx_models::kv_cache::DEFAULT_KV_QUANT`, is now read by the CLI, the server
  load path, the image branch, the arch dispatcher and all six speculative
  drafter stacks. `KvCacheBuilder::for_arch_default`,
  `KvCacheBuilder::resolve_default`, `ResolverSignals`, `kv_quant_for_ctx` and
  `Architecture::preferred_auto_kv` are gone with it.

  **What changes for you.** If you passed an explicit `--kv-quant`,
  `--cache-type-k/-v`, `--kv-bits` or `--kv-preset`, nothing changes — explicit
  always wins, and every codec remains selectable by name. If you passed no
  flag, output is **byte-identical at temp=0** on every architecture whose old
  default was a bf16-mirror codec (`K8V8`, `K8V4`, `Planar`), because those
  codecs already decoded off the bf16 mirror and, since the packed store was
  elided, already held exactly bf16's resident bytes — verified byte-identical
  `kv_cache_bytes` and identical token ids on gemma-4-e2b at 4k/8k/32k. It is
  **not** byte-identical on `Qwen3ForCausalLM` at `weight_bits == 2` (Bonsai
  ternary), whose old default `Mixed{k8g64,v4g64}` genuinely quantises: that one
  gets smaller and lossless instead of larger and lossy. Pass
  `--kv-quant mixed_k8g64_v4g64` to reproduce the old bits.

  This is not a claim that quantised KV cannot pay — it is a claim that these
  implementations do not, today, on this hardware. `DEFAULT_KV_QUANT` is the
  single place a future answer changes.

  **Two flag surfaces change with it.** `--paged-kv` now requires an explicit
  `--kv-quant`: it pages a codec's packed store, `auto` is bf16, and bf16 has no
  store, so `rmlx serve --model M --paged-kv` exits 1 with a message naming a
  codec instead of inheriting a quantised per-arch default. It is deliberately
  not auto-promoted — picking a codec because a storage-layout flag was passed
  would be a second codec resolver keyed on something other than `--kv-quant`.
  And a single-sided `--cache-type-k` / `--cache-type-v` now fills the side you
  left `auto` with that codec's canonical `q8_g128` partner rather than with the
  engine default: naming one side quantised is an opt-in to quantisation, and
  decomposing the other side from a bf16 default would have made every
  single-sided invocation a startup refusal (`--ctv tq4`, `--ctk q8_g128`, …).

  The `perf_canary` anchors in `docs/PERF_BASELINE.md` are re-taken at the new
  default, because the Bonsai one silently changed meaning: the canary passes no
  `--kv-quant`, so its Bonsai row measured `Mixed{k8g64,v4g64}` before and bf16
  now. The 2026-05-21 rows are kept and marked as belonging to the retired
  defaults; the new rows are anchors, not a measured gain over them.


- **The five `OnceLock` kernel gates are one threaded `DispatchPolicy` value.**
  `fused_qk_enabled`, `sparse_attn_enabled`, `turbo_flash_enabled`,
  `turbo_flash_lock_enabled`, `planar_flash_decode_enabled`,
  `rot_k_fused_enabled` and the two `_MIN` thresholds each latched an
  environment read on first call, so the first dispatch froze the kernel path
  for the process lifetime and two arms could only be compared across two
  processes — two model loads and two thermal states. They are replaced by
  `rmlx_core::DispatchPolicy`, a `Copy` value resolved once from the existing
  clap surface, captured by each `KvCache` at construction and read at the
  dispatch sites. Two caches built under different policies now run side by
  side in one process. Behaviour is unchanged: every flag keeps its precedence
  (`on` → on, `off` → hard override, `auto` → the `RMLX_*` variable), the CLI
  no longer mutates its own environment to communicate with the gates, and
  temp=0 token digests are identical before and after on gemma-4-e2b and
  Ternary-Bonsai-8B, in both the default and the `--turbo-flash on` arm.

- **`--rot-k-fused {on|off|auto}`** — `RMLX_ROT_K_FUSED` had no flag and so did
  not appear in `--help`. `auto` (the default) still reads the variable, so an
  existing opt-in is unaffected.

- **The SSD hydrate path carries the caller's `DispatchPolicy`.**
  `KvCache::from_storage`, `block_io::read_caches{,_timed}`,
  `SsdHydrator::lookup{,_seeded,_with_recorder}`, `SsdHydrate::hydrate` and
  `PromptCache::hydrate_from_ssd` all take it. A hydrated cache replaces a live
  one, so reconstructing it under the process default rather than the policy in
  hand would put that one path back on process-global behaviour — invisible
  while every cache shares the default, wrong the moment two do not. Same
  per-request contract the trait already states for `seed` and `kv_quant`.

- **Documented that no Gemma4 model can reach a fused-QK kernel.** Gemma4
  quantises only its full-attention layers, which use `global_head_dim = 512`,
  and the fused-QK shims are hard-gated on `head_dim ∈ {128, 256}`. A Gemma4
  run with a fused-QK codec logs `fused_qk: skipped` with
  `reason = "head_dim not in {128, 256}"` and `head_dim = 512`. The rotor and
  iso flash-decode kernels accept up to 512 and do fire there.

- **`--turbo-flash=auto` now resolves OFF on every host (HOLD).** It previously
  resolved ON for every recognised Apple family. On the one storage the kernel
  serves (K8V4, `kv_seq > 4096`) it decodes 2.0–4.25× *slower* than the generic
  path and holds ~722 MB more resident KV. `rmlx bench` n=3 on a quiet host,
  zero settle-gate refusals, with the loss scaling with `kv_seq` rather than
  sitting at a fixed penalty: Bonsai-8B k8v4 1.93× @~1.7k (threshold forced to
  zero), 2.74× @8k, 3.48× @16k, 4.25× @32k (63.25 → 14.89 TPS); Bonsai-27B
  k8v4 1.98× @16k. Dispatch proven by counter — 1638 ON vs 0 OFF. The kernel is
  also **not** bit-exact (SDPA cosine ≈0.997, the V turbo-4 codec floor), so at
  temp=0 it perturbs greedy argmax ties: two of those four production-threshold
  cells return a different token digest. gemma-4-e2b is a null control rather
  than a second architecture — its `kv_cache_bytes` is bit-identical in both
  arms, so the kernel never dispatches there at all. This retires the
  per-family default-ON policy; the validations behind it were crash/fidelity
  clearances (32k NIAH on Apple ≤9, the Apple10 `head_dim = 256` hazard
  re-drive) and are unaffected — lifting the HOLD needs a decode measurement.
  `--turbo-flash on` remains the opt-in, and `auto` still honours a pre-set
  `RMLX_TURBO_FLASH=1` — now with a `warn!` naming the cost, since in that case
  the flag reads OFF while the kernel runs. Consequence: `k8v4` on Bonsai-8B
  decodes 88.8 TPS @16k and 61.8 @32k out of the box, where the documented
  "crater from 8k up" had it at 39.3 @8k falling to 6.7 @64k. That crater was
  the kernel, not the codec.

### Removed

- **`iso_fused_qk` MSL kernel retired.** Its only possible callers were four
  codecs that keep no bf16 K mirror, so it could not dispatch from any
  production path; every iso codec decodes through `iso_flash_decode` /
  `iso_flash_decode_symv` instead. The dispatcher, both `.metal` bodies, both
  probe headers, the manifest rows, the tests and the doc references are all
  gone rather than left compiling and CI-checked for nothing.
- **`rmlx_models::kv_cache::attention_dispatch::FUSED_QK_TABLE`,
  `lookup_fused_qk` and `FusedQkEntry` removed**, with their tests. They were a
  public mirror of the codec layer's codec → kernel map with **zero** non-test
  callers: production dispatch has always gone through
  `rmlx_kv_quant`'s own `lookup_fused_qk_kernel`, because the codec layer
  cannot depend on `rmlx-models` per the workspace dep-graph rule. A second
  copy that nothing reads can only drift from the one that runs — which is
  what had happened. Same "nothing runs it" criterion as the iso kernel above.
  The module keeps its sparse-attention dispatch, which does have callers.
- **`rot_k_tq4v` retired.** Its decode appended to the packed store and then
  rebuilt a full bf16 K *and* a full bf16 V of the whole prefix on every step
  before running ordinary SDPA. `mx.quantized_matmul` cannot consume a Lloyd-Max
  codebook, so the affine-V pairing's fused route was never available to it, and
  the one kernel in tree that reads a tq4 V at decode is `auto`-OFF and slower
  than the generic path. Re-measured against its affine sibling `rot_k_v4g64` at
  the same shape on Ternary-Bonsai-8B and gemma-4-e2b at 4k/8k/32k, it is slower
  at all six cells (per-slot ABBA ranges disjoint by 1.53× at the Bonsai 8k
  cell), never reproduces the sibling's token ids on the shared-KV architecture,
  and holds more resident KV. The name is rejected at parse and at
  `--ctk rot_k --ctv tq4`, each naming its successor. Its memory headline
  reproduces but its attribution does not: the +27–45% over `--kv-quant none` is
  a whole-`Mixed` / `RotK`-family property, of which tq4-V against affine-4-V is
  0.5–1.0% (#408, #409, #422).
- **`tcq_v2_msl` removed** — a GPU Viterbi kernel with zero production dispatch
  path. `K8VTurbo2Tcq` forces `Device::Cpu` on its hot V-side update and
  `tcq_quantize_v2_gpu` had no caller outside its own test file, so repairing its
  kernel-load failure would only have re-hidden the same rot (#265).

### Dependencies

- `spin` 0.9.8 → 0.9.9, off the yanked release (#257).
- Two `cargo-minor-patch` group bumps, 9 crates and 6 crates (#287, #302).

## [0.3.0] - 2026-07-13

Metrics run-identity is now trustworthy. `observations.backend_version` was
wrong on 11 of 12 rMLX emitters — hard-coded `'0.0.1'` literals, absent values
that silently became NULL, and raw git SHAs stuffed into a semver field. The
root cause was structural: the §8.5 record had **12 construction sites and no
single integration point**, so identity was merely the first field group to rot.
This release replaces all of them with one builder that cannot be bypassed, one
validator on every ingest path, and a rule the binary now follows without
exception: **it stamps only what it can honestly know, and refuses to invent the
rest.**

The serving surface — HTTP API, `serve`, `chat` — is unchanged. The breaking
changes are confined to the metrics/bench subsystem.

### Changed

- **BREAKING — §8.5 ingest now validates run identity.** A record with
  `backend: "rmlx"` must carry a semver-shaped `backend_version`; a missing or
  malformed value is rejected on *every* ingest path (`metrics record --file`,
  `--replay-pending`, and the in-process recorder) instead of failing open to a
  NULL row. Other backends keep the field free-form and optional — llama.cpp has
  no semver and legitimately emits `build_commit`. See `docs/METRICS_DB.md`
  §8.5.1.
- **BREAKING — the binary performs no git operations, at all.** Not at runtime,
  not in `build.rs`. It previously resolved `git_sha` by shelling out to `git` in
  the **process working directory**, so an installed `rmlx serve` launched from a
  user's project stamped *that project's* HEAD — plus its `-dirty` state — into
  every metrics row it produced. Baking the SHA in at compile time was tried and
  rejected: Cargo does not re-run `build.rs` on source edits, so a work-in-progress
  binary filed rows as if they came from the pristine commit.

  `git_sha` is therefore **caller-supplied provenance**, exactly like
  `hardware_tag`: bench scripts stamp it (they run `git -C <repo> rev-parse` in
  their own checkout, where the question is cheap and honest), or a caller passes
  the new `rmlx baseline --git-sha` / `rmlx eval ppl --git-sha`. Absent → `NULL`,
  never guessed. Live-telemetry rows from the server carry `NULL`, which is
  correct — nothing bisects them.
- **BREAKING — `run_id` is now `YYYYMMDD-HHMMSS-<version>`**, not
  `-<short-git-sha>`. Affects `logs/<run-id>.jsonl` filenames and `events.run_id`.
- `build_profile` now reliably distinguishes `release` / `release-perf` /
  `release-debug`. `cfg!(debug_assertions)` reported all three as `"release"`,
  so cross-profile perf comparisons were silently comparing unlike builds.
- `RunRecord` and `RunIdentity` can no longer be constructed or mutated outside
  `rmlx-metrics`. A hand-rolled record, a forged identity, or a post-hoc field
  write is now a compile error. Adding a new metric requires zero identity code.

### Added

- **`--metrics {off|events|full}`** (global, default `full`), mirroring the
  existing `--log` flag. `off` is a producer-side no-op — no database opened, no
  drainer thread spawned, no `runs.db` created.
- **`rmlx metrics identity --json`** — the measured binary reports its own
  identity block, so shell emitters never guess or hard-code it.
- **`--git-sha <SHA>`** on `rmlx baseline` and `rmlx eval ppl`, for callers that
  want commit attribution on a recorded run.
- Migration `003` adds `backend_version` and `build_profile` to the `events`
  table, stamped from the same identity source as `observations`.

### Fixed

- **One-time pending-buffer quarantine.** `rmlx metrics record --replay-pending`
  now rejects pre-contract `rmlx` buffer files (written before the
  `backend_version` requirement existed) rather than ingesting them as another
  NULL-version row. On the first run after upgrading, any such files move to
  `metrics/buffer/failed/` and the command exits **2**. This is expected,
  one-time behavior — not a regression. No file is deleted. See
  `docs/METRICS_DB.md` §8.5.1.
- **RUSTSEC-2026-0204** — `crossbeam-epoch` bumped 0.9.18 → 0.9.20 (transitive,
  via `criterion` → `rayon`). `make deny` and `make audit` are green again (#198,
  #202).
- Clippy lints introduced by Rust 1.97.0, which had turned `main` latently red:
  every PR failed `build + clippy` regardless of content (#200).

### Removed

- The compile-time git SHA, the `RMLX_SOURCE_ROOT` stamp, and the runtime
  working-tree `-dirty` probe — together roughly 300 lines, including the whole
  of `build.rs`'s git handling (201 → 50 lines, it now only resolves the Cargo
  profile). They were the source of a recurring wrong-but-plausible identity bug
  that reappeared one layer down after each fix. Do not reintroduce them: the
  binary cannot honestly answer "what commit am I?", so it no longer tries.
- `events.git_sha` — a column no caller could ever fill. `events` is written only
  by the binary, which has no SHA to give, and nothing read the column.

### Dependencies

- `rustc-hash` 2.1.2 → 2.1.3, `uuid` 1.23.4 → 1.23.5, `time` 0.3.51 → 0.3.53
  (#201).

## [0.2.8] - 2026-06-30

Qwen3.5-family model-loading correctness and a CI-gateable smoke probe. The
weight-quant loaders no longer corrupt mxfp8/mxfp4 scales, dense Qwen3.5 mxfp8
checkpoints now load via fact-driven dispatch (no longer hardwired to the PARO
path), and `rmlx info --probe-smoke` returns distinct exit codes so a load
failure can no longer masquerade as success. No breaking changes.

### Added

- **Dense Qwen3.5 mxfp8 loader + fact-driven dispatch.** Both
  `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`
  now route by checkpoint facts, not the arch string: `is_paroquant()` selects
  the PARO vs the standard loader (the two share an arch string and differ only
  by `quantization_config.quant_method`), a shared `resolve_prefix` probes shard
  headers for the tensor prefix, and the MLP block is chosen per layer by tensor
  presence (dense SwiGLU vs sparse MoE). A defensive guard hard-errors if a PARO
  checkpoint ships MoE expert tensors. Dense Qwen3.5 mxfp8 snapshots now serve
  end-to-end. (#191, closes #189)

### Fixed

- **mxfp8/mxfp4 uint8 E8M0 scales corrupted at load → MoE prefill crash.** The
  Qwen3.5-MoE and Qwen3 loaders blanket-cast every quantized `.scales` tensor to
  bf16, which is correct for affine (float) scales but corrupts mxfp's uint8 E8M0
  scales, crashing the first prefill with `dequantize: Scale type must be uint8`.
  A new per-tensor `bf16_scales` gate casts only float scales and passes uint8
  scales through verbatim. (#190, closes #188)

### Changed

- **`rmlx info --probe-smoke` now returns distinct exit codes for CI gating.**
  Previously every non-`Broken*` outcome — including a supported-arch load
  failure and an inconclusive zero-token run — exited 0, so a loader regression
  read as a pass. Exit codes are now `0` ok, `1` broken, `3` load-fail, `4`
  inconclusive, `5` unsupported (`2` is reserved for clap arg-parse errors).
  `healthcheck` marks load-fail / inconclusive / broken as Red and unsupported
  as a non-fatal skip. (#193, closes #192)
- Bumped `anyhow` 1.0.102 → 1.0.103 (fixes a Stacked-Borrows UB in
  `Error::downcast_mut`) and `uuid` 1.23.3 → 1.23.4. (#187)

## [0.2.7] - 2026-06-28

Constrained-decode hot-path and Gemma4-unified vision tuning. The `json_schema`
and `json_object` per-token allow-mask probes no longer deep-clone their grammar
across the ~152K-token vocab on every decode step, and a whitespace stall in
schema-constrained decode is fixed. Gemma4-unified gains a per-request image-token
budget. No breaking changes.

### Added

- **Per-request + CLI image-token budget for Gemma4-unified vision.** A
  `image_max_tokens` request field (and matching CLI flag) caps soft image
  tokens per request; default 280, ceiling 1120. Lets callers trade vision
  fidelity for prefill cost on the unified any-to-any path. (#181, closes #180)

### Fixed

- **Schema-constrained decode whitespace loop.** Under `response_format:
  json_schema`, enum / scalar leaves accepted insignificant whitespace in
  states where it must be rejected (inside a literal, inside a string, at the
  root scalar start), letting temp=0 decode loop on `\n`. The allow-mask now
  matches whitespace per-leaf-state and rejects raw control chars (`0x00..=0x1f`)
  inside strings. (#183)

### Performance

- **`json_schema` constrained decode no longer deep-clones the schema per
  vocab token.** The allow-mask probe reset a scratch `SchemaGrammar` ~152K
  times per decode step, deep-copying the immutable parsed schema each reset.
  The schema is now held behind `Arc` (`Object.props`, `Union`, `Array.items`),
  so entering a container/property/union branch is a refcount bump and the
  per-token reset reuses buffers in place. Per-step cost on the production path
  drops ~8–25× (was 20–40× heavier than the `json_object` engine; now
  comparable). Tool / function-calling agents pay this directly. (#184,
  closes #182)
- **`json_object` constrained decode allow-mask reset is scratch-reused.** The
  `JsonGrammar` reset became a state copy + `Vec` clear/extend (the stack frame
  is `Copy`) instead of a fresh clone per vocab token — ~2× on the per-step
  probe. The two engines now share one `fill_allow_mask` kernel over a
  `ProbeGrammar` trait. (#183)

## [0.2.6] - 2026-06-24

f32-KV-leak class hardening. The `--kv-quant none` KV cache no longer widens to
f32 on the Qwen3 path, and the leak is now structurally closed for every
architecture. Headline: Qwen3-dense (Bonsai-8B-2bit) `none` decode is ~+32…+87 %
across 4k–64k and now beats the mlx-lm reference at every context, with KV
residency halved. No breaking changes.

### Fixed

- **Qwen3 dense (`Qwen3ForCausalLM`) `--kv-quant none` KV stored f32, not bf16.**
  Bonsai ships RMSNorm weights and quant scales/biases as fp16; bf16 activations
  × fp16 params promoted the residual — and the K/V projection outputs — to f32,
  so the cache stored f32 (4 B/element). Casting all Qwen3 float params to bf16
  at load (`bf16_param`) keeps the stream and the cache bf16. On Bonsai-8B-2bit:
  none-KV halved (≈0.53× the f32 MB), decode +32 / +47 / +68 / +82 / +87 % at
  4k / 8k / 16k / 32k / 64k, and prefill ~0.55×. (#168)
- **Qwen3.6 MoE (`Qwen3_5MoeForConditionalGeneration`) hardened to bf16-param
  parity**, including the GatedDeltaNet norm + conv1d weights; audited clean for
  the same f32-KV leak. (#171)

### Added

- **Model-agnostic bf16 floor at the KV-cache store boundary.** The
  `--kv-quant none` cache casts K/V to bf16 at the single store choke point, so
  no architecture can store f32 there regardless of upstream dtype — a durable
  backstop for the per-arch fixes. Bytes-per-element invariant test wired into
  `make model-check`. (#169)
- **CI gate `make check-no-scalar-f32-leak`** flags unguarded `scalar_f32(` in
  arch-layer code (the f32-leak idiom). Surfaced and fixed 13 latent leaks across
  gemma3, gemma4 vision/audio, jina, bitnet, and dflash. (#170)

### Dependencies

- safetensors 0.7→0.8, rusqlite 0.32→0.40, miniz_oxide 0.8→0.9, plus the
  cargo-minor-patch group; CI `actions/checkout` 6→7 and `Swatinem/rust-cache`.
  (#162–167)

### Security

- memmap2 0.9.10 → 0.9.11, clearing **RUSTSEC-2026-0186** (unsound out-of-bounds
  `offset`/`len` in `advise_range` / `flush_range`). rMLX maps safetensors
  read-only and does not call the affected functions, so it was not reachable —
  bumped to keep the advisory gate clean.

### Docs

- Bonsai-8B (2-bit) full rMLX KV-quant matrix + sibling-backend champions. (#177)

## [0.2.5] - 2026-06-20

Prefill / time-to-first-token fix for the MoE families, plus a baseline
correction. Headline: Qwen 3.6 prefill is ~4× faster at short context and now at
mlx-lm parity. No breaking changes.

### Performance

- **Qwen 3.6 (Qwen3.5-MoE) prefill is ~4× faster at short context.** The
  GatedDeltaNet recurrence flipped from the `gated_delta_step_gpu` Metal kernel
  to a lazy ops-graph at `T≥256`, which pinned the prefill chunk at 64 — a 4k
  prompt ran ~64 forward passes where mlx-lm runs ~2. Making the GDN always use
  the kernel (a byte-for-byte port of mlx-lm's `gated_delta_kernel`; chaining
  across chunks is f32-state-exact) unblocked raising the prefill chunk to 2048
  (mlx-lm's `prefill_step_size`). Warm-TTFT on `Qwen3.6-35B-A3B-8bit` (kv-none):
  4k 4240→1065 ms (4.0×), 8k 9008→2136 ms (4.2×), 16k 19489→4712 ms (4.1×);
  decode unchanged, no Metal watchdog through 64k. `gated_delta_prefill_ops` is
  retained as the test-only kernel-equivalence oracle. (#155)
- **Gemma 4 prefill chunk raised 512 → 1024.** A real-model sweep found 1024 the
  shared TTFT sweet spot: e4b 4k +6% / 8k +4.5%, 26b-a4b +17%; decode flat, no
  watchdog. `chunk=2048` regresses the e4b dense path (a sliding-window /
  exec-unit cliff above 1024 = 2×window), so the shared `gemma4` default
  stays 1024. (#155)

### Documentation

- **Prefill/TTFT is at mlx-lm parity, not "40–50× slower".** The earlier
  "~40–50× slower than mlx-lm / 4k TTFT 144 ms / 28000 tok/s" framing was a
  non-physical baseline (the cited prompt-throughput exceeds the M5-Max
  bandwidth ceiling). A direct mlx-lm 0.31.3 run on the same `Qwen3.6-35B-A3B-8bit`
  snapshot + prompts measures 2711–3606 prompt tok/s vs rMLX's ~3050 — mlx-lm is
  only ~1.1–1.2× faster. README, `docs/models/qwen3.6/rMLX.md`, and
  `docs/models/qwen3.6/SIBLINGS.md` retract the claim. (#155)
- **Gemma 4 e4b QAT complex-image vision is a checkpoint limitation, not a bug.**
  Investigated degenerate / hallucinated output from the `e4b-it-qat-mxfp4` and
  `-qat-nvfp4` snapshots on high-detail screenshots (#153). The e4b QAT
  snapshots share a byte-identical SigLIP `vision_tower` and clipped-linear
  bounds with `e4b-it-mxfp8`; the unquantized `qat-bf16` checkpoint degrades on
  dense images identically to the fp4 variants, and the `mlx_vlm` Python
  reference reproduces the same failure on the same snapshots. So this is an
  intrinsic quality limit of the e4b QAT checkpoint on complex images, not an
  fp4-dequant defect — rMLX output is reference-faithful. No code change;
  `docs/MODELS.md` now documents the behavior and recommends `e4b-it-mxfp8` for
  complex-image OCR. (#153)

## [0.2.4] - 2026-06-19

Vision, KV, and embedding-lookup bug-fix batch for Qwen3-VL and Gemma 4, plus a
`/metrics/cache` recording/docs fix and a Homebrew bottle build+publish flow.
Highlights: Qwen3-VL large images now work end to end (KV sized from `--max-ctx`;
the O(seq²) embedding lookup that tripped the Metal GPU watchdog is gone), and
Gemma 4 image grounding is fixed by placing image tokens inside the user turn. No
breaking changes.

### Added

- **Homebrew bottle build+publish flow.** `scripts/release/build_bottle.sh` +
  `make bottle` drive `brew bottle` against an installed keg, rename the local
  bottle to the GitHub-Release asset name, and emit the ready-to-paste
  `bottle do` block; documented as a release-time step in `docs/RELEASING.md`.
  The committed formula stays source-build until a real bottle is uploaded, so
  existing tap installs are unaffected. (#143, #139)

### Fixed

- **`/metrics/cache` TTFT empty for non-streaming completions.** Both
  non-streaming paths (`generate_blocking`, OpenAI + Anthropic) measured TTFT
  but never pushed it into the in-memory `ttft_store` ring — only the streaming
  path did, so `ttft` stayed `[]` for non-streaming traffic. The ring is now
  written on both paths. `docs/SERVER.md` is realigned to the endpoint's actual
  shape (`models[]`, `itl`, `tokens_in/out`), dropping the never-emitted
  `prompt_cache` / `last_itl` keys. (#142, #141)
- **Gemma 4 image grounding (degenerate / image-independent output).** The
  per-image token block was spliced after BOS but *before* the user-turn opener,
  leaving the image outside the user message; the model then ignored it. Image
  blocks are now spliced inside the (final) user turn via a shared
  `splice_image_block`, matching the HF/mlx-vlm placeholder substitution. Fixes
  the reported e4b QAT-fp4 degeneration (the soft tokens were correct all along)
  and a latent flakiness that affected all Gemma 4 image requests; Qwen3-VL is
  unified onto the same path. (#144, #140)
- **Qwen3-VL ignored `--max-ctx`; large images failed with a `slice_update`
  broadcast.** The image and text generate paths built KV with the bare 4096
  default and never bracketed prefill, so any prompt over 4096 tokens (a large
  image tiles to thousands of soft tokens) overran the fixed buffer. Both paths
  now size the KV ring from the effective `--max-ctx` and chunk the prefill;
  an over-cap prompt returns a clean `context_overflow` instead of the broadcast
  panic. (#145, #138)
- **Qwen3-VL large images hit the Metal GPU watchdog.** The quantized embedding
  lookup used an O(seq²) `eye(seq) @ w` identity-matmul on CPU (plus a GPU↔CPU
  round-trip); embedding the whole augmented prompt for a large image produced a
  single command buffer that overran the ~10 s watchdog. Replaced with on-device
  `take + dequantize` (O(seq)); added query-tiled ViT attention as a faithful
  defense for very large single images. (#147, #146)
- **Qwen3.6 (`qwen3_5_moe`) embedding lookup** carried the same O(seq²)
  `eye(seq) @ w`-on-CPU trick (plus an `unsafe` block); ported to the same
  on-device `take + dequantize`. Numerically faithful, removes a per-step CPU
  round-trip. (#149, #148)

### Performance

- Qwen3-VL: large images (e.g. 2560×2560 → 6400 soft tokens) now complete
  end-to-end instead of aborting the process at the Metal GPU watchdog. (#145, #147)

### Tested

- New CI-gated tests: image-token placement (in-turn, last-turn, multi-image,
  after-BOS fallback), ViT attention tiling equals a single SDPA, and
  `qwen3_5_moe` embed_lookup numeric equivalence across both dtype arms (the
  prior coverage was `#[ignore]` + env-gated). Real-model proofs across Qwen3-VL
  (KV + large-image), Gemma 4 e4b QAT-fp4 vision, and Qwen3.6 (decode-TPS
  same-session A/B: no regression).

## [0.2.3] - 2026-06-18

Multi-model registry hardening. Two `--registry` serving bugs fixed: the
multimodal encoder-output cache no longer leaks vision/audio features across
models, and eager model preload now respects `--max-loaded-models`. No breaking
changes.

### Fixed

- **Multimodal encoder-output cache cross-model leak.** In `--registry`
  multi-model mode the vision/audio encoder cache was keyed on the
  post-preprocess content hash only, so a cached image encoding produced for one
  model (projected to its `hidden_size`) was returned to a different model for
  the same image — a vision-feature shape mismatch (HTTP 503) when the hidden
  sizes differed. The cache key now folds in a stable per-model signature, so
  entries are never shared across models; same-model repeats still hit. (#132)
- **Registry eager-preload ignored `--max-loaded-models`.** `rmlx serve
  --registry` preloaded every model at startup even with a smaller resident cap,
  paying the full load cost for models that were immediately evicted (a
  ~5-minute boot for a 13-model registry). Preload is now bounded to at most
  `--max-loaded-models` entries (the alphabetically-first ids, since the
  registry is id-sorted); the rest load on demand. (#133)

### Changed

- `README.md` refreshed to 0.2.3 with an accurate "What works" summary, and
  `docs/CLI.md` documents that the multimodal cache key now includes model
  identity (no cross-model sharing) and that registry preload is bounded to the
  resident cap.

## [0.2.2] - 2026-06-18

Multimodal release. Whisper transcription works end to end (decode correctness
+ long-form) behind a new model-agnostic `rmlx transcribe` CLI; the dense
Gemma 4 12B `gemma4_unified` any-to-any architecture is now supported for image
and audio input; the standard Gemma 4 family gains native audio input through
the serve path; and the unified vision color-fidelity bug is fixed. Plus
release-signing and CI-hardening housekeeping. No breaking changes.

### Added

- **`rmlx transcribe <audio> --model <snapshot> [--format vtt|srt|json|txt]`** —
  model-agnostic audio transcription CLI, arch-dispatched on `config.json`
  (Whisper today, a clean seam for future ASR). Decodes any container to 16 kHz
  mono internally (enabled `symphonia` isomp4+aac, so `.m4a` works). The HTTP
  endpoint and the CLI share one long-form engine. (#119)
- **Gemma 4 12B unified (`gemma4_unified`) image + audio input.** The dense
  any-to-any 12B has no SigLIP/Conformer tower — vision and audio are
  early-fusion via soft tokens projected straight into the shared 48-layer LM.
  Faithful encoder-free ports of `Gemma4UnifiedVisionEmbedder` (host patchify +
  3×3 merge → `patch_ln1` → quantized `patch_dense` → factorized 2D pos-emb →
  `embed_vision`) and `Gemma4UnifiedAudioFeatureExtractor` (raw 16 kHz waveform
  → fixed 640-sample frames → `embed_audio`). Dispatched off `is_unified_arch`;
  the standard e4b/26b/31b SigLIP path is unchanged. (#120)
- **Gemma 4 native audio input through the serve path.** The Conformer
  `audio_tower` + `embed_audio` projector + USM feature extractor now load at
  startup alongside the vision tower, and `input_audio` parts are decoded → mel
  → `AudioEncoder` → soft tokens scattered at `<|audio|>`, mirroring the vision
  flow. Submitting audio to a model without an audio tower (or combining image +
  audio) returns a clear 503 — no silent drop. (#122)

### Fixed

- **Whisper transcription was empty / garbage.** large-v3 has 100 language
  slots, shifting every special token +1 vs the v1/v2 layout the constants
  assumed — so `TOK_TRANSCRIBE` pointed at `<|translate|>` and the
  timestamp-begin hard-stop fired on `<|notimestamps|>`. Corrected the
  special-token layout and added the missing in-loop logit filters
  (`SuppressBlank`, `SuppressTokens` derived generally from the tokenizer, and a
  faithful `ApplyTimestampRules`). Long-form decode bounds are derived from
  `n_text_ctx` at runtime so the positional table can't overflow. Full 48-min
  real recording at temp 0 → normalized WER ≈ 0.079, deterministic. (#119)
- **Gemma 4 12B unified vision color corruption.** The encoder-free path read
  image soft tokens *causally*, but `gemma4_unified` conditions each image's
  soft tokens with **bidirectional** attention (the SigLIP path hides this by
  pre-integrating the image in its ViT). A per-prefill bidirectional overlay,
  keyed off the `<start_of_image>`/`<end_of_image>` markers and merged
  element-wise into each layer's causal/SWA mask, fixes color naming and layout;
  gated on `has_image` so text prefill is untouched. (LayerNorm eps also
  corrected to the PyTorch `nn.LayerNorm` default 1e-5.) A 100%-uniform
  achromatic fill still reads as one level — an inherent property of the
  encoder-free projection (`patch_ln1` normalizes the absolute level away),
  documented in `docs/MODELS.md`. (#127)
- **`--probe-smoke` false `BrokenPunctLoop` on instruction-tuned snapshots.**
  The probe fed a bare (no-chat-template) instruction; chat models degenerate on
  such out-of-distribution input (the mlx-lm reference reproduces it
  identically) — a probe artifact, not a 4-bit dequant bug. The smoke seed is
  now rendered through the snapshot's `chat_template.jinja` when present, falling
  back to the bare seed for base models; each entry point keeps its own canonical
  BOS resolver (no hardcoded id). (#121)

### Security

- Pin CI actions (`actions/checkout`, `dtolnay/rust-toolchain`,
  `Swatinem/rust-cache`) to commit SHAs, add keyless **cosign** release signing
  (`make release-sign`), and drop a stale RustSec advisory ignore. (#116)

### Changed

- `scripts/release/source_sha256.sh --write` now also bumps the formula `url`
  version, not just the sha256 (previously left the formula pointing at the old
  tag's tarball). (#118)
- `docs/RELEASING.md` documents the formula url-bump and Dependabot
  migration-push gotchas. (#115)

## [0.2.1] - 2026-06-17

Correctness + maintenance release. Closes a systemic KV-cache head-scramble
class that affected **every** flat quantized KV codec, hardens the SSD KV tier
and prompt cache, makes the single-MLX Metal claim self-heal after a crashed
holder, and unifies the per-architecture model code onto shared seams (decode
loop, loader, `Architecture` dispatch). Plus a round of dependency bumps. No
breaking changes.

### Fixed

- **Systemic KV head-scramble class.** Every flat quantized KV codec wrote its
  buffer sequence-major but reshaped it head-major on dequant — agreeing only
  when `batch × kv_heads == 1`, and scrambling per-head K/V on any multi-append
  (decode after a multi-token prefill, or after an SSD hydrate) when
  `kv_heads > 1` (grouped-query attention). Fixed family-wide with a canonical
  sequence-major layout (transpose on append + on dequant) and an explicit
  `Array::contiguous` before each custom MSL kernel, which reads its input by
  raw linear index and so cannot honor a lazy transpose. Covers `QuantK`
  (#103), `QuantV` / TurboSym-K / paged-K handoff (#108), the Iso/Rotor
  rotation codecs (#109), and PlanarQuant K/V plus its packed-K decode kernels
  (#110).
- **SSD KV tier.** Spill + restore now carry the bf16 K/V payload for
  `KvQuant::None` layers (#88); SSD-hydrated entries are excluded from the
  exact-hit fast path so a hydrate cannot be mistaken for an exact prompt-cache
  hit (#87); a Gemma 4 entry hydrated with an empty SWA layer degrades to a
  full re-prefill instead of decoding from a hole (#90).
- **Prompt cache unified across architectures.** A single model-agnostic
  `consume` engine replaces the per-arch hydrate/reuse glue and is retrofitted
  onto five architectures, so the SSD-hydrate / prefix-reuse correctness fixes
  above hold identically on every model (#98).
- **GPU default stream on every inference entry.** The image, speculative,
  audio, and embeddings blocking-thread entries now establish the thread-local
  GPU stream the text path already had, fixing intermittent
  `no Stream(gpu, N)` failures off the text path (#104). The adaptive
  prefill-chunk fallback resolves the loaded architecture instead of assuming
  Gemma 4 (#68).
- **Metal claim self-heals.** A stale claim left by a crashed holder is
  auto-reclaimed once the holder PID is proven dead (re-probed under the file
  lock); `SIGTERM`/`SIGINT` now shut the server down gracefully and release the
  claim (#112).
- **`Array::to_bytes` evaluates before reading the data pointer**, closing a
  lazy-eval race in the only reader of the raw MLX array buffer (#101).
- `MetalKernel::new` frees its input vector when output-name conversion fails
  (#60); the Planar3 V codec uses one packing path on CPU and GPU (#102) and
  warms its MSL kernels at precompile (#59); the resident-bytes estimator
  models Iso/Rotor sidebands exactly (#58); `chunked_prefill` exits prefill on
  every cache after a failure (#57); f16 negative subnormals no longer decode
  to `-0.0` (#56); the tensor-view loader distinguishes not-found from I/O /
  parse failures (#4b5ea54 → see history).
- **Gemma 4 loading:** unquantized bf16 and affine-int4 (QAT) snapshots load;
  affine biases pass through the MoE expert `gather_qmm`; the perplexity scorer
  prepends BOS to every sliding window.

### Changed

- **Shared decode loop.** Qwen 3, Qwen 3.5-MoE, Gemma 4, and Gemma 3 now run on
  one decode loop (per-arch copies removed); `ProbeStep` / `SmokeVerdict` live
  in the shared loop.
- **Shared loader seam.** All architecture loaders (Gemma 4 / 3, Qwen 3 /
  3.5-MoE / 3-VL-MoE, Laguna) adopt `load_util::Weights` — an index-first,
  header-truth tensor fetch; AWQ byte-math moved to `rmlx-quant`; a single
  `read_raw_config` helper replaces six per-loader clones.
- **`Architecture` dispatch.** Auto-KV default, KV-byte reporting, and
  prompt-cache stats now dispatch through the `Architecture` trait rather than
  arch-specific branches.
- Shared fused-QK setup scaffold (q8 / turbo-K3 / turbo-K4 / iso dispatchers
  ported onto it); arch modules construct arrays via
  `Array::from_{i32,f32}_slice` per `docs/FFI.md`; `refuses_qwen_moe` renamed to
  `k_below_8bit` (it is a codec property, not an arch rule).

### Dependencies

- `tokenizers` 0.20 → 0.23 (encode/decode add-special / skip-special semantics
  preserved; verified on Gemma 4, Qwen 3.6, and Bonsai tokenizers) (#97).
- `toml` 0.8 → 1.1 (#94), `tikv-jemallocator` 0.6 → 0.7 (#96),
  `criterion` 0.5 → 0.8 (dev / benches) (#95), `uuid` 1.23.2 → 1.23.3 and
  `time` 0.3.47 → 0.3.49 (#93).

### Tested

- Full KV-codec regression re-sweep after the head-scramble fixes: every codec
  class (QuantK/V, Iso/Rotor, Planar including its live fused-QK kernel) is
  within ±5 % of its recorded best decode cell on Bonsai, Gemma 4-e4b, and
  Qwen 3.6 — no decode regression. GPU round-trip tests assert each layout flip
  reconstructs true head-major K/V at quant noise (with pre-fix scramble
  controls).
- Tokenizer correctness re-proven on three tokenizer families (SentencePiece +
  BPE) at temp 0.

## [0.2.0] - 2026-06-10

Gemma 4 decode is now competitive with mlx-lm across the whole family, Gemma 4
speculative decoding (MTP) works end to end, the KV ring grows lazily with
per-request KV / context hot-swap, KV-cache metrics report live sizes, and the
env-var surface is cleaned up — **breaking** for shell configs that set removed
vars directly (see Removed).

### Added

- **Per-request KV-quant + `--max-ctx` hot-swap** on a resident model — switch
  the KV codec or context ceiling per request without reloading the model. (#26)
- **Per-layer KV net-benefit estimator** — warns when a KV codec costs more
  resident bytes than it saves on a given layer mix (general across arches). (#34)
- Five env-var-only knobs promoted to proper `--flag` / `env=` pairs (the flag
  always takes precedence): `--log-cap-mb`, `--yarn-factor`,
  `--yarn-original-max`, `--session-cache-max-sessions`, `--prompts-dir`.

### Fixed

- **Gemma 4 speculative (MTP) functional end to end.** Dispatch routes
  `--draft-kind mtp` by draft arch family and rejects a plain-`gemma4` draft
  cleanly (#23); the assistant SWA mask uses array mode instead of the rejected
  additive mode (#24); a verify-step SWA mask off-by-one in both the producer
  and consumer branches is fixed (#32); and the loader supports both assistant
  LM-head variants — sparse centroid-routed (e2b/e4b) and plain tied-head
  (26b/31b) (#49). All four Gemma 4 sizes load and run coherent under MTP.
- **Gemma 4 decode kept bf16 end to end.** `gelu_tanh` f32 constants plus the
  embed / per-layer scales no longer promote the dense activation stream to f32
  (#44), and the MoE router's strong-f32 root-size scalar no longer leaks f32
  into the routing weights and the downstream KV (#51). Net: e2b/e4b beat mlx-lm
  decode, 26b-a4b MoE closed from −10…−28 % to −4…+1 %, and global `--kv-quant
  none` KV is halved (bf16) on every model.
- **`--max-ctx` is a virtual ceiling** — the KV ring grows on demand, so a high
  ceiling no longer penalizes small-prompt decode. (#25)
- **Rotation / K-only KV codecs** precompile their MSL kernels at load and are
  truthfully classified Metal vs CPU (no silent host-CPU fallback). (#36)
- **Qwen3.6-MoE SSD-hydrated prefix skips prefill** via a hydrated-tail path — a
  cache hit no longer re-runs the full prefill. (#9)
- **Live KV-cache metrics** — `kv_cache_bytes` reports the real resident size
  (was always 0) and counts the filled prefix, not the `--max-ctx` ceiling.
  (#33, #39)

### Performance

- **MoE prefill ~4× faster** on gemma4-26b and Qwen3.5-MoE via sorted-index
  expert gather (contiguous per-expert access in `gather_qmm`) — 26b 128k cold
  TTFT ~403 s → ~117 s. (#46)

### Tested

- Falsified the 6× SWA-KV claim: windowed SWA KV is window-bounded, not
  full-context (#35, #40).
- Full Gemma 4 and Qwen 3.6 KV × context bench matrices (per-model decode /
  TTFT / KV-size across all codecs) recorded under `docs/models/`.

### Changed

- **Env-var surface cleanup** (`chore/env-var-cleanup`). Five previously
  env-var-only knobs are now proper `--flag` / `env=` pairs so the flag always
  takes precedence: `--log-cap-mb` (`RMLX_LOG_CAP_MB`), `--yarn-factor`
  (`RMLX_YARN_FACTOR`), `--yarn-original-max` (`RMLX_YARN_ORIGINAL_MAX`),
  `--session-cache-max-sessions` (`RMLX_SESSION_CACHE_MAX_SESSIONS`),
  `--prompts-dir` (`RMLX_PROMPTS_DIR`).
- `docs/CLI.md` env-var section restructured: split into **User / operational**
  and **Internal / advanced** subsections, with flag / default / description
  columns for every entry.
- `docs/TESTING.md`: added `RMLX_KV_TEST_MODEL`, `RMLX_DRAFT_TEST_MODEL`,
  `RMLX_VL_TEST_MODEL`, `RMLX_TEST_MODEL` to the specialised test-model table;
  added a **Test behaviour toggles** table covering `RMLX_SKIP_GPU`,
  `RMLX_REGEN_GOLDENS`, `RMLX_E2E_*`, `RMLX_REGISTRY_TEST`,
  `RMLX_NIAH_KV_QUANT`, and the `*_STRICT` flags.
- `.env.example` expanded to document all user-facing env vars: runtime data
  vars (`RMLX_HOME`, `RMLX_METRICS_DB`), all five newly-promoted flag-envs,
  audio path vars, `RMLX_MM_CACHE_BYTES`, `RMLX_SESSION_CACHE_MAX_SESSIONS`,
  draft compat keys, and prefill chunk tuning.
- Dependency bumps: `safetensors` 0.4 → 0.7, `symphonia` 0.5 → 0.6.

### Removed

The following env vars no longer have live readers in the Rust codebase.
**This is a breaking change** for any shell config that set them directly —
use the replacement flag instead.

| Removed variable | Replacement |
|---|---|
| `RMLX_KEEP_ALIVE` | `--idle-timeout-secs` |
| `RMLX_PROMPT_CACHE_MAX_BYTES` | `--prompt-cache-ram-gb` |
| `RMLX_PAGED_KV` | `--paged-kv` |
| `RMLX_KV_PAGE_SIZE` | `--paged-kv-page-tokens` |

The following debug / internal vars were dropped with no user-facing
replacement (they had no stable semantics across releases):

- `RMLX_SPEC_K` — undocumented experimental speculative-lookahead override.
  Its only value was the default; lookahead `K` is now fixed at 4. The
  independent `--draft-block-size` flag still controls the draft round size.
- `RMLX_MTP_DUMP`, `RMLX_DFLASH_DEBUG` — folded into `tracing` events; use
  `--log debug` or `RUST_LOG=rmlx=debug` instead.
- `RMLX_GIT_SHA` — was read for the metrics drainer's `git_sha` annotation but
  nothing ever set it (always `None`); the annotation now reuses the same
  `git rev-parse` helper the run ID uses, so it is populated in a git checkout.
- `RMLX_METAL_AVAILABLE`, `RMLX_METAL_CAPTURE` — doc-only, never implemented.
- `RMLX_METRICS_LOCK` — doc-only, never implemented (WAL handles concurrency).
- `RMLX_GPU_RESIDENT_ISO`, `RMLX_SPARSE_V_KERNEL`, `RMLX_SPARSE_V_THRESHOLD` —
  deep perf/kernel toggles, now hardcoded to their proven-best defaults
  (`off`, `on`, `1e-6`); the override env was removed (no perf change).
  *Correction:* "proven-best" and "no perf change" were wrong for the two
  sparse-V toggles. Pinning `RMLX_SPARSE_V_KERNEL` on left a kernel that
  produced wrong output and cost 17× past 8 192 context tokens, with no way to
  turn it off; the validation behind "proven-best" was taken at shapes below
  that threshold, where the kernel never runs. Both toggles and the kernel are
  gone as of the Unreleased section above.
- `RMLX_OMODELS_DIR` — bench-script alias renamed to the canonical
  `RMLX_O_MODELS_ROOT`.

## [0.1.1] - 2026-06-06

Bug-fix + dependency-maintenance release.

### Added

- `rmlx baseline --max-prompt-tokens <N>` — the prompt-truncation cap (previously
  a hardcoded 65536) is now configurable, enabling ≥128k-context baselines
  (validated `>= 1`). (#11)

### Fixed

- Eagle3 speculative decode crashed mid-generation on Qwen3-MoE
  (`slice_update` zero-length KV dim). The drafter KV cache is now sized to the
  verifier context limit instead of a hardcoded 4096. (#8)
- SSD KV-tier spill failed with `no Stream(gpu, N) in current thread` and skipped
  persisting blocks. KV/lin caches are now materialized on the inference thread
  before the prompt-cache store, so the drain thread's eval is a no-op. Applies
  to qwen3.5-moe, qwen3, and gemma4. (#10)

### Changed

- Dependency bumps: `bindgen` 0.72 (FFI codegen — golden-token-verified
  behaviorally identical), `sha2` 0.11, `actions/checkout` 6, and a minor/patch
  group (`serde_json`, `tokio`, `minijinja`, `chrono`, `uuid`).

## [0.1.0] - 2026-06-06

First release. Native, single-binary [MLX](https://github.com/ml-explore/mlx)
inference + conversion backend for Apple Silicon — no Python at runtime.

### Added

- Text generation — OpenAI `/v1/chat/completions` + `/v1/completions` and an
  Anthropic-compatible surface (temperature, top-k/p, penalties, thinking
  budget, constrained / schema-guided decoding).
- Image input — vision towers (Gemma 4 SigLIP, Qwen3-VL-MoE deepstack) via
  `image_url` content parts.
- Audio input — transcription / translation for audio-capable models.
- Multimodal embeddings — `/v1/embeddings`, including text + image (jina-v4).
- Tool / function calling — OpenAI `tool_calls` + Anthropic `tool_use`,
  multi-turn, multiple emit formats (Qwen XML, Hermes-JSON, Gemma).
- Quantization — affine 2–8 bit, mxfp4 / mxfp8, nvfp4, ParoQuant weights; KV
  quant incl. fp8, TurboQuant, RotorQuant, PlanarQuant, IsoQuant, paged-KV,
  mixed / asymmetric K/V, and an SSD KV tier — including rotation-based KV
  families no other MLX server ships.
- Speculative decoding — MTP, DFlash, and Eagle3 drafters.
- Prompt caching — automatic prefix caching with block hashing.
- Conversion — `rmlx convert` re-quantizes / repacks MLX → MLX.

### Tested

- Golden-token decode gates (temp=0) for Gemma 4
  (`Gemma4ForConditionalGeneration`), Qwen 3.6
  (`Qwen3_5MoeForConditionalGeneration`), Bonsai (`Qwen3ForCausalLM`), and
  BitNet (`BitNetForCausalLM`).
- Multimodal embeddings (`jina-embeddings-v4`).
- Speculative drafters validated against their verifiers: Qwen 3.6 MTP sidecar
  and the Gemma 4 assistant drafter.

[Unreleased]: https://github.com/Pushkinist/rMLX/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.4.1
[0.4.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.4.0
[0.3.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.3.0
[0.2.8]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.8
[0.2.7]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.7
[0.2.6]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.6
[0.2.5]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.5
[0.2.4]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.4
[0.2.3]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.3
[0.2.2]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.2
[0.2.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.1
[0.2.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.2.0
[0.1.1]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.1
[0.1.0]: https://github.com/Pushkinist/rMLX/releases/tag/v0.1.0
