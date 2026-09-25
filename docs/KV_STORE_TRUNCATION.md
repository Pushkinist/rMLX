# KV store truncation

This doc tells how `KvCache::truncate_to` rolls each KV store back to a
sequence position.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
API, CLI flags, the auto default, bit rates, codec disposition);
[`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) (which codec each layer gets);
[`KV_CODECS.md`](KV_CODECS.md) (storage per `KvStorage` variant, TurboQuant
calibration); [`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the iso and
rotor codecs); [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK, fused
flash-decode, sparse attention);
[`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md) (measured codec fidelity).

---

## Store truncation (`truncate_to`)

`KvCache::truncate_to(n)` rolls a cache back to its first `n` positions. Each
store cuts its own payload; the limits are below.

The rotor and iso `truncate_to` keeps the ring. It lowers `shape[2]` to `n`, and
the next append writes the ring from `n`. So a speculative rollback keeps a
ring-only tail up to `n`.

**Row vs. sequence units.** A rotor or iso block's `n_tokens` counts rows
(`b * kv_h * seq_of_block`). `truncate_to(n)` takes a sequence position. So
the planner compares cumulative rows with `n * b * kv_h`.

**A cut inside a block splits it.** A block spans one append. A speculative
partial accept cuts inside the verifier's `K + 1`-token chunk. The planner
splits the trailing block and cuts every per-row buffer to the kept rows:
codes, per-group scales, per-group quaternions, per-token norms and the QJL
sideband.

**The split is `b == 1` only.** A block's rows run `[B, S_block, kv_h, D]`.
So at `b > 1` a row prefix is not a sequence prefix. There the planner drops
the block, and the reconciliation guard reports the gap.
`quant_rotor_v3_truncate_at_b_gt_1_stays_loud` pins this. A block whose row
count is not a whole number of sequence positions stops the walk too.

**Multi-block decode at `b > 1`.** Each CPU store with a block list reorders
every block at its own sequence offset
(`seq_layout::transpose_chunked_seq_heads`). The iso GPU readers
(`QuantIsoK::dequant_gpu`, `QuantIsoV::dequant_gpu`) put each token row at its
head-major position through `iso_kernel_inputs_head_major`. The
`*_two_block_decode_matches_one_block_at_b_gt_1` tests, one per store, pin
this over `(b, kv_h) ∈ {1,2} × {1,2}`.

**Readers that refuse `b != 1`.** Two readers refuse `b != 1` with `S > 1`
instead of reordering:

* The flat GPU buffers of the turbo, planar and q8 stores (`QuantV`,
  `QuantKTurbo3/4`, `QuantPlanarK/V`, `QuantK`). Their prefix records no
  chunk boundary.
* `QuantK`'s CPU `codes` / `scales`, one flat pair with no per-append
  boundary. It refuses even a single-append `b > 1` store.

The eight rotor and iso K and V stores share one planner, `truncate_plan` in
`crates/rmlx-kv-quant/src/storage/mod.rs`, with one `BlockRows` implementation
per block type. Tests: `storage/truncate_plan_tests.rs`, and one store-level
round trip per block type in `quant_rotor_k_tests.rs`, `quant_rotor_v_tests.rs`
and `quant_iso_v_tests.rs`.

**Every store cuts its own payload.** The same planner drives the turbo and
planar blocks (`TurboBlocks`, `PlanarBlocks`). `QuantV`, `QuantKTurbo3/4`,
`QuantPlanarK` and `QuantPlanarV` implement `truncate_to`. `QuantK` cuts its
flat `codes` / `scales` pair to the first `n` positions. Every arm of
`KvStorage::truncate_to` and `KvStorage::reset` delegates to the store's own
`truncate_to` or `reset`. So no arm lowers `shape[2]` and leaves the payload in
place.

**Every CPU dequant checks its length.** Each path checks that its blocks decode
to `prod(shape)` elements. On a mismatch in either direction it returns an
error: `"CPU blocks decode to N elems but shape [...] implies M — refusing to
zero-pad / truncate"`. The stores refuse some cuts, and this check reports each
one. They refuse a cut at `b > 1` and a block that is not a whole number of
positions. `QuantK` also refuses a target inside a 128-element q8 group.

`TurboBlocks` and `PlanarBlocks` carry no `n_tokens`. Their row count is the
product of the first three axes of `original_shape` (`storage::block_rows`).
The CPU append and the SSD hydrate write those axes in different orders, and
the product is the same for both. A split writes `[1, 1, rows, width]`.
`cpu_block_truncate_tests::quant_v_truncate_reads_rows_from_the_shape_product`
pins it.

**Only the turbo, planar and q8 stores clamp the target.** They clamp it to
their own `shape[2]` (`storage::clamp_truncate_target`). A target past
`shape[2]` is reachable. A store-backed codec that also keeps a bf16 mirror
advances `KvCache::offset` on paths that the store does not follow. A `shape[2]`
raised to meet the target would claim tokens that no payload holds.

The rotor and iso stores do not clamp: they set `shape[2]` to the target. A
ring-only tail lies below `shape[2]`, and the ring readback returns `Err` on an
over-long target. So for `n > shape[2]` the mixed arms leave the two axes of one
codec at different lengths. These are `IsoV3`, `IsoV4`, `RotorV3` and `RotorV4`,
where the q8 K (`QuantK`) clamps. They are also `RotorKAsym3/4`, where the
TurboQuant V (`QuantV`) clamps. The guard on the unclamped side reports it at
spill.

**The `Mixed` arm truncates to its fill marker.** `MixedKvState` is a capacity
buffer that grows in `STEP` increments, with `offset` as its fill marker.
`truncate_to(n)` sets the marker to `n`, and the next append writes over the
rows from `n`. The bf16 mirror of a shared-KV producer follows the same offset.
For `n > offset` the state keeps its fill and emits an `error!` that names
both numbers. `offset` is the coverage, so there is nothing to clamp down to.
`kvcache/shared_source_tests.rs::mixed_truncate_to_keeps_the_prefix_it_was_told_to_keep`
compares the cut cache with a cache prefilled to the kept length.

Tests: `storage/cpu_block_truncate_tests.rs` covers the partial-accept round
trip per store and the `b > 1` and q8-group refusals. It also covers the zero,
exact and past-the-end targets, `KvStorage::reset`, and the
`KvStorage::truncate_to` dispatch. Each oracle is a reference store built from
the kept tokens only. `crates/rmlx-kv-ssd/src/hydrate_truncate_tests.rs`
truncates a hydrated cache inside a block, appends a correction and checks the
decoded V.

In production, `KvCache::truncate_to` is called by the prompt-cache prefix
trim (`PromptCacheEntry::truncate_kv_to`) and by the speculative
partial-accept rollback. `truncate_plan` has no arch or head-count branch.
`kv_cache_truncate_iso3_kv_h_gt_1_path` (`rmlx-models/src/kv_cache/tests.rs`)
drives the full `KvCache` dispatch at `kv_h = 4`.
