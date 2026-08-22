# Changelog

All notable changes to rMLX are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

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

### Added

- **Per-dispatch `trace!` on the two PlanarQuant kernels**
  (`planar_flash_decode_sdpa: dispatch`, `planar_fused_qk: dispatch`), matching
  every sibling KV kernel. Their in-process dispatch counters have no caller
  outside tests, so these events are the only way a shipped binary can answer
  "did this kernel run".
- **`fused_qk: skipped` trace with a `reason` field.** Every fall-through in
  `try_fused_qk_dispatch` now names the gate that rejected, and the `head_dim`
  gate carries the observed value. This is what identified the Gemma4 result
  below in one run instead of by reading the dispatcher.

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

### Changed

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

[Unreleased]: https://github.com/Pushkinist/rMLX/compare/v0.3.0...HEAD
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
