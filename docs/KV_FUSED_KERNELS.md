# Fused KV decode kernels

This doc gives the fused-QK kernels, the fused flash-decode kernels over the
rotor, iso and PlanarK stores, the fused-QK head-major K storage, the dispatch
axis and sparse attention.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
CLI flags, the auto default, bit rates, byte accounting, hot-swap, codec
disposition, public API); [`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) (which
codec each layer gets); [`KV_CODECS.md`](KV_CODECS.md) (storage per
`KvStorage` variant, TurboQuant calibration);
[`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the iso and rotor codecs);
[`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md) (`truncate_to` per store);
[`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md) (measured codec fidelity).

---

## Fused-QK kernels

A fused-QK kernel reads a packed K (codes, scales, rotation indices)
directly. It writes pre-softmax scores `[B, n_q_heads, 1, S_kv]`, so no
dequantized K is written to memory. After the softmax, V takes the split path:
a matmul with the bf16 V. Without a fused kernel, a Class 2 codec decodes off
the bf16 K mirror, with no dequant.

The q8, TurboSym and rotor-asym fused-QK kernels read a head-major K shadow
that is re-encoded from the bf16 mirror (§"Fused-QK head-major K storage").

### PlanarK fused-QK scope

`KvStorage::PlanarK` is the only storage whose own packed store a fused-QK
kernel reads.

- `crates/rmlx-kv-quant/src/planar_fused_qk_msl.rs`: the kernel and its
  dispatcher. The kernel reads the PlanarQuant `(codes, scales, rot32)`
  triple. It does the per-pair centroid lookup and the inverse Givens rotation
  in registers. The grid is `(S_kv, B * n_q_heads, 1)`, and each threadgroup
  is `head_dim` threads that reduce one score. `planar_fused_qk_msl_tests.rs`
  checks it against `planar_dequantize_v4_gpu` followed by a reference matmul.
- `crates/rmlx-kv-quant/src/planar_fused_qk.rs`: the process-wide toggle.
- `crates/rmlx-kv-quant/src/storage/quant_planar_k.rs::gpu_packed_view`: the
  GPU codes, scales and `rot32` for the `S` tokens held, not dequantized.
- `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused`:
  appends K to the packed store and V to the bf16 V buffer, then runs
  `planar_k_flash_over_store`. That function runs `planar_flash_decode` when
  the policy allows it and `head_dim` is a power of two. Otherwise it runs the
  fused-QK kernel, the additive mask, a precise softmax and a GQA-broadcast
  matmul with V.

The dispatcher takes this path only when **all** of these are true:

- the storage is `KvStorage::PlanarK`;
- `--planar-fused-qk` is `on`;
- the device is the GPU;
- the step is a decode step (`q_seq == 1`);
- no bf16 K seed is live (`decode_fp16_k` is `None`).

A cache that went through a prefill bracket holds a bf16 K seed, so it takes
the bf16 path. This is the warm-TTFT contract, and it is why `planar_k` is a
Class 2 codec. A cache with no seed takes the fused path. The dispatcher
also falls back to the bf16 path when the GPU packed view is not there.
`planar_k` needs `head_dim % 32 == 0` and is rejected on Qwen MoE
(`QwenMoePlanarKRejected`).

### Storage applicability

| Variant | K codec | Fused-QK? |
|---|---|---|
| `KvStorage::PlanarK` | PlanarQuant 4-bit | **Yes**, under the conditions above. |
| `KvStorage::Planar` (`planar`, `planar3`) | q8_0 | No. PlanarQuant is on the V axis; K is not Planar-packed. |

### CLI toggle

`--planar-fused-qk on|off` (default `on`). The CLI installs the value once at
start-up into a process-wide `OnceLock`. There is no environment variable.
`off` sends every PlanarK decode step through the dequant + SDPA path.

---

## `rotor_flash_decode` — fused MSL flash-decode over rotor-quant K

This is the decode kernel for `k_rotor3` and `k_rotor4`
(`KvStorage::RotorKOnly3` / `RotorKOnly4`). It computes QK over the packed
rotor K ring, an online softmax, and SV over the bf16 V mirror. It uses two
Metal dispatches per decode step. The Cl(3,0) K decode runs inside the attention
loop. So no bf16 or f32 K is built, and no K data goes through the host.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_msl.rs` — Rust dispatcher,
  header builder, dispatch counters.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_p1.metal` — pass-1 body,
  one body for both bit widths.
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — pass-2
  log-sum-exp merge. It is codec-agnostic. The rotor, iso and planar
  flash-decode kernels use it.
* `crates/rmlx-kv-quant/src/storage/quant_k_gpu_ring.rs` — `QuantKGpuRing`,
  the GPU-resident packed ring. Per token and KV head it holds the row's dense
  code plane, one bf16 scale per group and one bf16 L2 norm. It grows in pages
  and can be seeded from CPU blocks. The caller gives it `n_groups`, so the
  rotor and iso stores share it.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_rotor_k_fused` —
  dispatch site.

### Bit width is a header parameter

The header carries `bits ∈ {3, 4}` as `RF_BITS` / `RF_MASK`, with the matching
Lloyd-Max codebook. So one `.metal` body serves both widths. The dispatcher
takes the width from the storage variant. Any other width is an `Err`.

### Reusable K-decode half

The header emits the rotor decode as two MSL functions. `rf_decode_k_group`
decodes one Cl(3,0) group, and `rf_decode_k_lane` is a per-lane wrapper over
it. In each pass-1 body, lane 0 of a group decodes the group and writes its
lanes to threadgroup memory. So the decode runs once per group, not once per
lane. A body is a statement sequence inside a generated kernel signature, and
it cannot define a function. So a shared function must live in the header.

### Gate

There is no flag and no environment variable. The kernel runs when all of
these conditions are true:

- The cache is not a sliding-window ring. `update_and_sdpa` sends a rotating
  cache to `KvCache::update` and bf16 SDPA first.
- The device is GPU.
- The storage is `RotorKOnly3` or `RotorKOnly4`.
- The store carries no QJL sideband.
- `q_seq == 1`.
- `b == 1`.
- `head_dim` is a power of two and at most `ROTOR_FLASH_HEAD_DIM_MAX` (512).

On any other miss the non-fused fallback (`KvCache::update`, then SDPA)
dequantizes the K prefix on the CPU and runs bf16 SDPA.

**QJL.** `--rotor-qjl on` adds a 1-bit QJL residual to K. Its decode is a
per-token product with a dense `[head_dim, head_dim]` matrix, which the flash
loop does not reproduce. So a store with QJL keeps the CPU dequant path. The
gate reads the store's own decision (`use_qjl()`), not the global toggle. The
store fixes QJL at its first append. Before that append, the gate reads the
global toggle. The default is `--rotor-qjl off`.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `RotorKOnly3` / `RotorKOnly4`, QJL off, `b == 1` | **YES** | GPU ring + `rotor_flash_decode_sdpa`. |
| `RotorKOnly{3,4}`, QJL on | NO | The kernel does not reproduce the QJL residual. |
| `RotorKOnly{3,4}`, `b > 1` | NO | The ring stride does not interleave batch. |
| `RotorSym{3,4}` | NO | Decodes through `rotor_flash_decode_symv`, which reads V from its own ring. |
| `RotorKAsym{3,4}` | NO | V is a TurboQuant store (`QuantV`), not rotor. Its only GPU decode kernel is `rotor_fused_qk`, with `--fused-qk on` (§"Fused-QK head-major K storage"). |

### The caller of the encode chooses the ring feed (`RingFeed`)

The caller of the rotor or iso GPU encode decides whether the ring is fed. It
passes a `RingFeed`:

- **`Maintain`** — feed the ring and push a CPU block. The non-fused rotor
  K-only `update_*` entries use it (`LEGACY_ROTOR_K_ONLY_FEED`), because they
  dequantize the whole prefix on the same step.
- **`MaintainRingOnly`** — feed the ring and push no CPU block. This is a
  ring-only tail. The fused decode entries use it. `shape[2]` still advances.
- **`Skip`** — drop the ring. The non-fused rotor symmetric and asymmetric
  entries (`LEGACY_ROTOR_SYM_FEED`) and the non-fused iso entries use it.

A ring-only feed at `b > 1` takes the block path instead
(`is_ring_only_append`).

After a fused decode step the entry drops the CPU blocks
(`drop_blocks_when_ring_live_*`). The ring is then the only resident copy.

**Invariant: the CPU `blocks` cover `shape[2]`, or the GPU ring holds the
rest.** A store is in one of two states:

- *The blocks cover `shape[2]`*, after `Maintain`, a CPU `append` or an SSD
  hydrate. A live ring mirrors them.
- *Ring-only tail*, after a fused decode step. `synced_rotor_k_blocks`,
  `synced_rotor_v_blocks` and `synced_iso_v_blocks` rebuild the blocks from
  the ring at each consumer: `dequant`, the SSD spill and the prompt-cache
  clone (`try_deep_clone`).

A block push onto a store with a ring-only tail first moves that tail into
the blocks (`materialize_rotor_k_ring_tail`, `reconcile_ring`). So the blocks
stay a contiguous prefix. The fused entry requires `q_seq == 1`, so a
multi-token append on a warm cache takes the block path. A speculative verify
chunk is such an append.

A store whose blocks fall short of `shape[2]` with no ring to supply the rest
is an error, never a zero-padded read. `synced_rotor_k_blocks` refuses it at
the codec. `ensure_rotor_k_blocks_cover_shape` refuses it at SSD
serialization. Both return `Err` and are not `debug_assert`s, so they hold
under `release-perf`.

### Arch reachability

The gate keys off codec and shape, never an arch name. Any arch that calls
`KvCache::update_and_sdpa` or `update_and_sdpa_shared_source` reaches the
kernel. On a shared-KV arch the producer layer runs the kernel and reports
`SharedKv::Store`. Each consumer layer runs the same kernel over that store
through `KvCache::sdpa_shared`. Sliding-window layers stay bf16 (see Gate).
On a Qwen MoE arch, `validate_resolved` rejects every rotor and iso codec that
quantizes K (`QwenMoeRotorKRejected`, `QwenMoeIsoKRejected`), so no such cache
is built.

---

## `rotor_flash_decode_symv` — fused flash-decode over rotor-quant K **and** V

This is the decode kernel for `rotor3_sym` and `rotor4_sym`
(`KvStorage::RotorSym3` / `RotorSym4`). It reads both axes from their packed
rotor rings and keeps no bf16 K or V mirror. `feeds_bf16_k_at_decode` and
`feeds_bf16_v_at_decode` are both false for these variants. So `exit_prefill`
builds neither mirror, and the resident-byte estimate reads the same two
predicates.

### Reuse of the K-decode half

The kernel uses the header of `rotor_flash_decode`
(`build_rotor_flash_header`). The body calls `rf_decode_k_group` twice per
token, once on the K ring and once on the V ring. It stages each axis in its
own threadgroup array (`k_shared`, `v_shared`). The two rings share one
encoder (`rotor_gpu_encode_arrays`) and one bit width, so one `RF_BITS` serves
both.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_symv_msl.rs` — dispatcher and
  counters.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_symv_p1.metal` — pass-1
  body for both widths. Pass 2 is the shared merge.
* `crates/rmlx-kv-quant/src/storage/quant_rotor_v.rs` — the V store, with its
  own `QuantKGpuRing`. The fused entry feeds both rings with
  `RingFeed::MaintainRingOnly`.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_rotor_sym_fused`
  — dispatch site for the plain path and the shared-KV producer and consumer
  paths.

### Gate

The gate is the `rotor_flash_decode` gate, with `RotorSym3` or `RotorSym4` as
the storage. A store with QJL keeps the CPU dequant path on both axes.

No flash-decode dispatcher calls `Array::eval()` on its inputs. Such a call
blocks the host on the GPU once per layer per decode step.
`make check-no-kernel-input-eval` fails the build on one.

### The `norms` device floor

This kernel and `iso_flash_decode_symv` bind a per-token `norms` array. MLX's
custom-kernel builder can bind a small input in the `constant` address space.
The shared decode helpers declare `norms` in the `device` address space. On
an MLX build that binds it as `constant`, that mismatch fails the MSL compile.
So both dispatchers call `flash_decode_common::pad_norms_to_device_floor`. It
zero-pads `norms` to `NORMS_DEVICE_MIN` (16) elements when
`b * kv_h * kv_seq` is below it. The kernel loop stops at the real `kv_seq`,
so no pad element is read.

`iso_sym_short_kv_seq_kv_h1_stays_on_gpu`,
`iso_sym_transition_across_ring_norms_floor` and
`rotor_sym_transition_across_ring_norms_floor`
(`crates/rmlx-kv-ssd/src/block_io_tests.rs`) decode at `kv_h == 1` across the
floor. They compare the result with a `KvStorage::None` cache fed the same
tokens.

---

## `iso_flash_decode` — fused MSL flash-decode over iso-quant K

This is the decode kernel for `k_iso3` and `k_iso4`
(`KvStorage::IsoKOnly3` / `IsoKOnly4`). It has the same two-pass shell as
`rotor_flash_decode` and uses the shared merge. Only the K decode is
different.

### Files

* `crates/rmlx-kv-quant/src/iso_flash_decode_msl.rs` — Rust dispatcher,
  header builder, dispatch counters.
* `crates/rmlx-kv-quant/src/metal/iso_flash_decode_p1.metal` — pass-1 body for
  both bit widths.
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — the shared
  pass-2 merge.
* `crates/rmlx-kv-quant/src/storage/quant_iso_k.rs` — the iso K store
  (`QuantIsoK<BITS>`), with a `QuantKGpuRing`.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_iso_k_fused` —
  dispatch site. `try_dispatch_shared_store` and `sdpa_shared` serve
  shared-KV models.

### Decode: one left Hamilton product, not a sandwich

The iso codec encodes `r = q * v_unit` with one fixed golden-ratio unit
quaternion, `FIXED_QUAT`. So the decode is `q̄ * r`, one left Hamilton
product. The rotor codec's `R̃ * mv * R` sandwich is a different algebra. Do
not carry it across.

`if_decode_k_lane` decodes one lane. It reads the four codes of the lane's
group from the row's dense code plane and computes the product in registers.
It uses no threadgroup memory and no barrier.

### Fixed quaternion is baked into the header

The iso encoder uses `FIXED_QUAT` for every group. The header bakes `q̄` in,
and the ring carries no quaternion table. A store with per-group quaternions
would decode wrongly through this kernel. `assert_fixed_quat_blocks` checks a
quaternion table against `FIXED_QUAT`, but no dispatch path calls it.

### Bit width is a header parameter

The header carries `bits ∈ {3, 4}` as `IF_BITS` / `IF_MASK`, with the matching
Lloyd-Max codebook. So one `.metal` body serves both widths. Any other width
is an `Err`.

### Reusable K-decode half

`if_decode_k_lane` is a header function, so another kernel body can call it.
`iso_flash_decode_symv` calls it on the K ring and on the V ring.

### Gate

There is no flag and no environment variable. The kernel runs when all of
these conditions are true:

- The cache is not a sliding-window ring. A rotating cache takes
  `KvCache::update` and bf16 SDPA first.
- The device is GPU.
- The storage is `IsoKOnly3` or `IsoKOnly4`.
- `q_seq == 1`.
- `b == 1`.
- `head_dim` is a power of two, a multiple of 4, and at most
  `ISO_FLASH_HEAD_DIM_MAX` (512).

On any other miss the non-fused fallback (`KvCache::update`, then SDPA)
dequantizes the K prefix and runs bf16 SDPA. On a GPU device it dequantizes on
the GPU (`dequant_gpu`). On a CPU device it dequantizes on the CPU.

The iso codecs have no QJL residual. So `k_iso3` and `k_iso4` reach the kernel
at the default flags.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `IsoKOnly3` / `IsoKOnly4`, `b == 1` | **YES** | GPU ring + `iso_flash_decode_sdpa`. |
| `IsoKOnly{3,4}`, `b > 1` | NO | The ring stride does not interleave batch. |
| `IsoSym{3,4}` | NO (this kernel) | Decodes through `iso_flash_decode_symv` (`iso_flash_decode_symv_msl.rs`, `metal/iso_flash_decode_symv_p1.metal`), which reads V from its own ring. Its gate is this gate, with `IsoSym3` or `IsoSym4` as the storage. |

`KvStorage::resident_bytes` counts the CPU blocks and the ring of each iso and
rotor store (`byte_size`).

---

## `planar_flash_decode` — fused MSL flash-decode for PlanarK

This is a decode kernel for `planar_k` (`KvStorage::PlanarK`). Pass 1 computes
QK, an online softmax and SV for each 64-token tile. Pass 2 is the shared
merge. K is read from the `QuantPlanarK` packed buffers, and V from the bf16
mirror.

### Files

* `crates/rmlx-kv-quant/src/planar_flash_decode_msl.rs` — dispatcher, header
  builder, dispatch counter.
* `crates/rmlx-kv-quant/src/metal/planar_flash_decode_p1.metal` — pass-1 body.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::planar_k_flash_over_store` —
  dispatch site. It selects this kernel or the split fused-QK chain.
* `crates/rmlx-cli/src/commands/serve.rs::resolve_planar_flash_decode` —
  resolves `--planar-flash-decode`.

### Gate

`--planar-flash-decode on|off|auto` sets `DispatchPolicy::planar_flash_decode`.
The default, `auto`, resolves OFF unless `RMLX_PLANAR_FLASH_DECODE=1` is set.
`off` overrides that variable. The kernel runs only when all of these
conditions are true:

- The cache is not a sliding-window ring.
- The storage is `PlanarK`.
- `--planar-fused-qk` is on, which is the default.
- The device is GPU.
- `q_seq == 1`.
- The cache holds no bf16 K seed (`decode_fp16_k` is `None`).
- The packed K is on the GPU.
- `DispatchPolicy::planar_flash_decode` is set.
- `head_dim` is a power of two.

When only the last two conditions fail, the step runs the split fused-QK
chain (§"Fused-QK kernels").

### Arch reachability

The gate keys off codec and shape. The shared-KV path reaches the same arm.
On a Qwen MoE arch, `validate_resolved` rejects `planar_k`
(`QwenMoePlanarKRejected`).

The `niah_pflash_*` cells in `crates/rmlx-models/tests/niah_long_context.rs`
force `planar_k`, count dispatches and require the needle. With the policy on
and a live bf16 K seed, they assert zero dispatches.

### Numerical relationship to the split chain — measured, not bit-exact

The kernel is **not** byte-identical to the split chain. Both arms decode the
same packed K. The flash kernel folds the softmax into a per-tile online
log-sum-exp. The split chain builds the whole score row and calls
`softmax_precise`. The summation orders differ, so the f32 accumulators differ
in the low mantissa bits.

Whether a difference survives the closing cast to the query dtype depends on
how near each value is to a bf16 rounding boundary. That is a property of the
data. So some cells are clean and some are not, and one cell proves nothing.

`planar_flash_decode_is_not_bit_exact_vs_split_chain`
(`crates/rmlx-kv-quant/src/planar_flash_decode_msl_tests.rs`) runs both arms
over one packed store, with bf16 Q, bf16 V and the dispatcher's output cast.
It sweeps six cells: `(kv_h, heads_per_kv, head_dim)` of `(8, 4, 128)` and
`(1, 8, 256)`, each at `kv_seq` 64, 512 and 4096. It prints the per-cell
difference counts and asserts no count.

The test asserts three things. At least one cell differs at bf16, which is the
claim. At least one cell differs at f32, which shows that both arms ran. Every
cell's f32 error is below 1e-3, which separates rounding from a defect. The
kernel computes the same attention, but it is not lossless.

### The bf16 K seed bypasses the PlanarK kernels

`exit_prefill` builds a bf16 K seed (`decode_fp16_k`) for `planar_k`. While
the seed is live, `update_and_sdpa` sends a PlanarK decode step to bf16 SDPA.
It skips both the fused-QK chain and `planar_flash_decode`, and it logs
`warm_ttft_bypass` at `debug!` (target `rmlx_kv_quant::warm_ttft`). So an A/B
of `--planar-flash-decode on|off` on a normal generate flow runs the same
branch twice. Count the per-dispatch `trace!` events before you use such an
A/B. The kernels run on a cache with no seed: an SSD hydrate, or a cache that
never went through a prefill bracket.

---

## Fused-QK head-major K storage

The fused-QK MSL kernels compute pre-softmax `QK` from a head-major packed K
shadow, with no K dequant. These codecs reach them from the production
decode path: q8 (`K8V4` / `K8V8`), `TurboSym3`, `TurboSym4`, and the two
rotor-asym codecs (`RotorK3Asym` / `RotorK4Asym`).

### Which codecs can reach this path, and why the rest cannot

The shadow is built by re-encoding the bf16 K mirror (`decode_fp16_k`) that
`exit_prefill` builds. `exit_prefill` builds that mirror only for a codec whose
`KvQuant::feeds_bf16_k_at_decode()` is true. Eight codecs always return
false: `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4`, `Rotor3Sym`,
`Rotor4Sym`, `RotorKOnly3` and `RotorKOnly4`. Each of them decodes through
its own flash-decode kernel over the packed ring. `Mixed` and `RotK` return
`shares_kv`, and `update_and_sdpa` sends them to the mixed path first.

So those eight never reach the fused-QK path, at any `head_dim`, batch size or
architecture. `fused_qk_table_matches_the_bf16_k_mirror_contract` pins the
kernel table to the codecs that keep the mirror.

Decode routing for the rotation-KV families, at `b = 1`:

| Codec | Decode kernel | Where |
|---|---|---|
| `Iso3Sym` / `Iso4Sym` | `iso_flash_decode_symv` | `update_and_sdpa` iso-sym arm |
| `IsoKOnly3` / `IsoKOnly4` | `iso_flash_decode` | `update_and_sdpa` iso-K-only arm |
| `Rotor3Sym` / `Rotor4Sym` | `rotor_flash_decode_symv` | `update_and_sdpa` rotor-sym arm |
| `RotorKOnly3` / `RotorKOnly4` | `rotor_flash_decode` | `update_and_sdpa` rotor-K-only arm |
| `RotorK3Asym` / `RotorK4Asym` | `rotor_fused_qk` | fused-QK shadow path — **no flash arm exists**, so this is its only GPU decode kernel |

So the rotor-asym pair needs `--fused-qk on`. At the default (`auto`, which
resolves OFF) its decode reads the bf16 mirror, with correct output and no
rotor kernel. `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs` checks,
for each rotor codec, that the expected kernel ran and the other two did not.

### `head_dim` reachability — why fused-QK never fires on a Gemma4 model

The kernel shims accept `head_dim` 128 or 256 only. Gemma4 quantizes only its
global layers, and its sliding-window layers stay bf16. The global layers use
`global_head_dim`, which is 512 in every Gemma4 snapshot under test. So
`try_fused_qk_dispatch` rejects each Gemma4 decode step at the `head_dim`
gate. The rotor and iso flash-decode kernels accept `head_dim` up to 512, so
Gemma4 reaches those.

To check a model, run it with `--log verbose` and search the run's
`<RMLX_HOME>/logs/*.jsonl` for `fused_qk: skipped`. The `reason` field names
the gate. The `head_dim` field carries the rejected value.

### Storage shape

`KvCache` holds the shadow as `fused_qk_shadow: Option<FusedQkShadow>`:

| Buffer | Shape | Per-token payload |
|---|---|---|
| `k_codes` | `u32 [B, kv_h, max_seq, codes_per_token]` | codec-specific packed codes |
| `k_scales` | `f32 [B, kv_h, max_seq, scales_per_token]` | per-group f32 scales |
| `sideband_norms` | `f32 [B, kv_h, max_seq, 1]` | per-token L2 norm (rotor only) |
| `sideband_rotor_table` | `f32 [n_groups * 4]` | static per-layer rotor table (rotor only) |

`FusedQkLayout::for_codec(KvQuant, head_dim) -> Result<Option<Self>>` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_shadow.rs` computes the per-codec
layout:

| `KvQuant` | `codes_per_token` (u32) | `scales_per_token` (f32) | sidebands |
|---|---|---|---|
| K8V4, K8V8 | `head_dim/4` | `head_dim/128` | — |
| TurboSym3 | `head_dim*3/32` | `head_dim/32` | — |
| TurboSym4 | `head_dim/8` | `head_dim/32` | — |
| RotorK3Asym, RotorK4Asym | `row_words_for(head_dim, bits)`: `ceil(3 * ceil(head_dim/3) * bits / 32)` | `ceil(head_dim/3)` | per-token norm + rotor table |
| any other codec | — | — | `for_codec` returns `Ok(None)` |

`for_codec` returns an error when `head_dim` is not a multiple of the q8 group
(128) or of the turbo group (32).

The kernel shims read the codes and scales as flat 1-D inputs of length
`tok_count * payload_per_token`, where `tok_count = B * kv_h * kv_seq`. Each
dispatch slices the shadow from `[B, kv_h, max_seq, payload]` to
`[B, kv_h, kv_seq, payload]` and flattens it. The dim-2 slice is not
contiguous, so the flatten copies on each step (see KV_CACHE.md §9.5
"Per-step cost framing").

### Dispatch wire-in

`KvCache::try_fused_qk_dispatch` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch.rs` is called from
`update_and_sdpa` after the K8V4 TurboFlash branch and before the non-fused
bf16 SDPA fallback. `try_dispatch_shared_bf16` calls it on the cross-layer-KV
producer path. Its gates, in order:

1. `DispatchPolicy::fused_qk` is set. `--fused-qk on|off|auto` sets it. `auto`
   resolves OFF unless `RMLX_FUSED_QK=1` is set.
2. The device is `Device::Gpu`.
3. `q_seq == 1` and `new_k` has rank 4.
4. `head_dim ∈ {128, 256}` (kernel limit).
5. `offset + new_seq >= DispatchPolicy::fused_qk_min_kv_seq`. The default is
   512, and `RMLX_FUSED_QK_MIN` overrides it. A shorter cache goes to bf16
   SDPA.
6. The codec is in the `lookup_fused_qk_kernel` table.
7. The codec has a GPU encoder (`codec_has_gpu_encoder`).
8. For a rotor codec, the global `--rotor-qjl` toggle is off.
9. `decode_fp16_k` is seeded.
10. The storage variant carries a `max_seq` (`storage_max_seq_for_fused_qk`).
11. The step does not overflow it (`prev_offset + new_seq <= max_seq`). The
    shadow populate path has no out-of-range clamp.

Each fall-through emits `fused_qk: skipped` at `trace!`, with a `reason` field
that names the gate. The `head_dim` gate also logs the value. The overflow
gate also raises a one-shot `warn!`.

The first dispatch allocates the shadow and seeds it from the bf16 prefix in
`decode_fp16_k`. Each later decode step appends to it head-major with a 4-D
`slice_update` at `[:, :, prev_offset:prev_offset+new_seq, :]`. The bf16
`decode_fp16_k/v` mirror stays up to date, and the SV product reads its V.

### Codec coverage

| Codec family | GPU encoder | Test floor |
|---|---|---|
| q8 (K8V4, K8V8) | `q8_quantize_gpu` | cosine ≥ 0.99 against bf16 SDPA |
| TurboSym3 | `turbo_quantize_v3_gpu` | cosine ≥ 0.95 against bf16 SDPA |
| TurboSym4 | `turbo_quantize_v4_gpu` | cosine ≥ 0.99 against bf16 SDPA |
| RotorK3Asym / RotorK4Asym | `rotor_quantize_v3_gpu` / `rotor_quantize_v4_gpu` | routing only (`rotor_fused_qk_dispatch.rs`) |

`crates/rmlx-kv-quant/tests/fused_qk_dispatch.rs` holds the cosine floors. When
the dispatch counter does not move, it skips the cosine check, unless
`RMLX_FUSED_QK_STRICT=1` is set.

### Dispatch counter and trace

`rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count()` adds the per-family
counters. In-process tests read it as `after - before > 0`. Only tests call
it. For a real run, count the per-dispatch `trace!` events
(`q8_fused_qk_sdpa: dispatch`, `turbo_k{3,4}_fused_qk_sdpa: dispatch`,
`rotor_fused_qk_sdpa: dispatch`) in the run's `<RMLX_HOME>/logs/*.jsonl`,
with `--log verbose`.

### See also

* `crates/rmlx-kv-quant/tests/fused_qk_dispatch.rs` — GPU integration
  tests for q8, TurboSym3 and TurboSym4.
* `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs` — the rotor routing
  contract: which kernel each rotor codec reaches, and which it must not.

---

## Dispatch axis

`KvCache::update_and_sdpa` tries these paths in order:

1. SWA ring (`self.rotating`): `update()` then SDPA.
2. `Mixed` / `RotK` (`self.quant.uses_mixed_path()`):
   `update_and_sdpa_mixed`.
3. The fused fast paths, each of which returns `None` when not eligible:
   `PlanarK` fused QK, rotor K-only flash decode, iso K-only flash decode, iso
   symmetric flash decode, rotor symmetric flash decode, K8V4 TurboFlash, and
   the head-major fused-QK path.
4. The fallback: `update()` then `scaled_dot_product_attention`.

`KvCache::update` matches `&self.storage`:

```rust
match &self.storage {
    KvStorage::K8V4 { .. }                                   => update_k8v4
    KvStorage::K8V8 { .. }                                   => update_k8v8
    KvStorage::Planar { .. }                                 => update_planar
    KvStorage::None { .. }                                   => update_none
    KvStorage::Paged { .. }                                  => update_paged
    KvStorage::Mixed { .. }                                  => Err (contract violation)
    KvStorage::K8VTurbo3 | K8VTurbo2 | K8VTurbo3Tcq | K8VTurbo2Tcq => update_k8_turbo_v
    KvStorage::TurboSym3 | TurboSym4                         => update_tsym
    KvStorage::PlanarK { .. }                                => update_planar_k
    KvStorage::IsoV3 | IsoV4                                 => update_iso_v
    KvStorage::RotorV3 | RotorV4                             => update_rotor_v
    KvStorage::IsoSym3 | IsoSym4                             => update_iso_sym
    KvStorage::IsoKOnly3 | IsoKOnly4                         => update_iso_k_only
    KvStorage::RotorSym3 | RotorSym4                         => update_rotor_sym
    KvStorage::RotorKOnly3 | RotorKOnly4                     => update_rotor_k_only
    KvStorage::RotorKAsym3 | RotorKAsym4                     => update_rotor_k_asym
}
```

`self.quant` is the construction-time parameter. `self.storage` is the
dispatch key. Code that branches on the codec must match `storage`, not
`quant`. A cache rebuilt from an SSD spill can hold a storage that differs from
its `quant`: an SWA layer hydrates as `KvStorage::None` while `quant` is the
model's global codec.

Prefill is separate. `enter_prefill` switches to raw bf16 accumulation for
every codec. `exit_prefill` encodes the accumulated prefix into the storage
variant, when `KvQuant::materialises_packed_store()` is `true`. Each
`KvStorage` arm of `exit_prefill` is the bulk-init path of that codec.

`exit_prefill` runs on the request's `spawn_blocking` worker thread, the same
thread on which the prefill forward built its graph. The generate entry
points call `rmlx_mlx::ensure_cpu_default_stream()` to register the worker's
own streams. See `docs/KV_CACHE.md` §5.7.5 for the mechanism, the guard, and
its limitation.

**Warm-TTFT decode contract.** `exit_prefill` also seeds a bf16 K+V decode
mirror (`decode_fp16_k` / `decode_fp16_v`) for each axis whose decode reads it.
Every `update_<codec>` of the bf16-mirror family returns early to
`update_decode_fp16` while that mirror is live. Thus decode-phase K **and** V
are bf16 for those codecs. The K-only family (`IsoKOnly*`, `RotorKOnly*`) keeps
K quantized at decode and mirrors only V. The fused symmetric family
(`Iso*Sym`, `Rotor*Sym`) mirrors neither axis. `docs/KV_CACHE.md` §9.6 has the
per-codec table.

---

## Sparse attention

A two-phase MSL kernel pair in `crates/rmlx-kv-quant/src/sparse_attn/`. It
reads a PlanarQuant-packed K and a bf16 V.

| Kernel | Role |
|---|---|
| `phase1_score_msl::phase1_score` | Scores every KV slot per (query, head) and keeps the top `TOP_PER_TILE` (4) scores per tile. |
| `phase2_sparse_attend_msl::phase2_sparse_attend` | Attends only the slots at or above each head's threshold and writes per-tile partials. |
| `phase2_sparse_attend_msl::phase2_lse_merge` | Merges the per-tile partials by log-sum-exp into the attention output. |

Between the phases the host reads the per-tile top scores and computes each
head's threshold from its budget (`compute_head_threshold`).

The dispatcher is
`rmlx_models::kv_cache::attention_dispatch::sparse_attn_dispatch_if_enabled`.
It returns `None` unless `DispatchPolicy::sparse_attn` is set and head budgets
are given. An inner error logs a `warn!` and returns `None`.

**No production path calls the dispatcher.** `--sparse-attn {auto|on|off}`
sets `DispatchPolicy::sparse_attn`. The default, `auto`, resolves OFF unless
`RMLX_SPARSE_ATTN=1` is set. No decode path reads the policy, so
`--sparse-attn on` does not change a served output. Only tests reach the
kernels, through `sparse_attn_dispatch`.

`rmlx_kv_quant::sparse_attn::sparse_attn_total_dispatch_count` returns the
sum of the P1 and P2 dispatch counts. One `sparse_attn_dispatch` call adds 2.

Tests, in `crates/rmlx-models/tests/sparse_attn_dispatch.rs`:

* `sparse_attn_dormant_on_warm_ttft_update_and_sdpa` — `update_and_sdpa` on
  a warm PlanarK cache, under a `sparse_attn: true` policy, leaves the counter
  unchanged.
* `sparse_attn_dispatches_on_seedless_planar_k` — `sparse_attn_dispatch` over
  a seedless PlanarQuant-packed buffer adds exactly 2 to the counter. Its
  cosine against dense `planar_flash_decode_sdpa` is at least 0.99. When the
  counter does not move, the test skips its checks, unless
  `RMLX_SPARSE_ATTN_STRICT=1` is set.

### Head budgets (`head_budgets.json`)

The per-(layer, head) k-budget table for phase 2 is
`<MODEL>/head_budgets.json`. The reader accepts two schema versions.

**Schema v1** (K-norm² proxy):

```json
{
  "version": 1,
  "model_name": "<snapshot dirname>",
  "num_layers": 36,
  "num_heads": 32,
  "calibration": {
    "method": "softmax_mass",
    "prompt_set_sha256": "<hex>",
    "num_prompts": 8,
    "max_seq_len": 4096,
    "mass_threshold": 0.95
  },
  "per_layer_per_head_budget": [[<u32>...], ...]
}
```

**Schema v2** (true softmax-mass) adds four optional fields to `calibration`
and sets `version` to `2`:

```json
{
  "version": 2,
  "model_name": "<snapshot dirname>",
  "num_layers": 36,
  "num_heads": 32,
  "calibration": {
    "method": "softmax_mass",
    "prompt_set_sha256": "<hex>",
    "num_prompts": 15,
    "max_seq_len": 8192,
    "mass_threshold": 0.95,
    "recipe": "softmax_mass",
    "target_mass": 0.95,
    "target_mass_budget_floor": 16,
    "prompts_provenance": ["calibration_long_context.json"]
  },
  "per_layer_per_head_budget": [[<u32>...], ...]
}
```

- `recipe` — the measurement recipe. The writer sets `"softmax_mass"`.
- `target_mass` — the cumulative softmax-mass target.
- `target_mass_budget_floor` — the minimum per-(layer, head) budget. It stops
  a single dominant key from giving a 1-slot budget.
- `prompts_provenance` — the basenames of the calibration prompt files.

[`crates/rmlx-loader/src/head_budgets.rs`](../crates/rmlx-loader/src/head_budgets.rs)
holds the struct, the validator, the reader (`load_head_budgets`) and the
writer (`write_head_budgets`). Both ends fail on a version other than 1 or 2,
and on a zero budget. They also fail on a shape mismatch: `num_layers`
against the row count, or `num_heads` against the column count. A v1 load
logs a `warn!` that advises re-calibration with `softmax_mass`.

`rmlx serve` loads `head_budgets.json` from the snapshot and attaches it to
the loaded `kv_calib.json` (`crates/rmlx-cli/src/commands/serve.rs`). Without
a `kv_calib.json` it logs a `warn!` and ignores the budgets. A snapshot with no
`head_budgets.json` is the common case. No decode path reads the attached
budgets.

### Calibration recipes

CLI: see `docs/CLI.md`. Three `rmlx kv-calibrate --recipe` values write head
budgets. Each accepts only a snapshot that loads as `Architecture::Qwen3`.
Each takes the Metal claim while it loads the model on the GPU. It drops the
claim when the load returns, so the measurement runs without the claim.

| Recipe | Schema | Measurement |
|---|---|---|
| `head_budget` | v1 | K-norm² proxy (the H2O / StreamingLLM stand-in) |
| `k_norm_proxy` | v1 | Same as `head_budget` |
| `softmax_mass` | v2 | True Q@K^T → softmax → cumulative-mass top-k |

The v1 recipes write `method: "softmax_mass"`, the name of the target
concept. But they measure the K-norm² proxy. v2 adds the `recipe` field so
that a file names what was measured.

`--mass-threshold` sets the mass target, in [0.50, 1.00], with a default of
0.95. `--target-mass-budget-floor` sets the floor, with a default of 16. Only
`softmax_mass` uses the floor.

#### True softmax-mass calibration

For each calibration prompt, the recipe builds a fresh bf16 KV cache
(`KvQuant::None`) and runs `forward_seq_with_cache_calibrated` with a
`SoftmaxMassSink`. At each layer the sink scores the full K against the
last-position query of each query head, and applies the softmax. It then
finds the smallest top-k that covers `target_mass`. The budget is the maximum
over the query heads of a GQA group and over the prompts.
