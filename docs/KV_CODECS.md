# KV codec storage per variant

This doc gives the storage layout and the decode path of each `KvStorage`
variant, and the TurboQuant calibration file.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
CLI flags, the auto default, bit rates, byte accounting, hot-swap, codec
disposition, public API); [`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md) (which
codec each layer gets); [`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the
iso and rotor codecs); [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK,
fused flash-decode, the dispatch axis, sparse attention);
[`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md) (`truncate_to` per store);
[`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md) (measured codec fidelity).

---

## Per-variant deep dive

### `KvStorage::None` — unquantized bf16

**K codec**: bf16, shape `[B, kv_h, max_seq, head_dim]`.
**V codec**: same.

The buffers are `KvCache::decode_fp16_k` and `decode_fp16_v`, not a
`KvStorage` sub-struct. This is the same machinery as the bf16 mirror of the
quantized codecs. `KvStorage::None` records only `max_seq`.

`update()` calls `update_decode_fp16`, which issues a `slice_update` at the
current token offset. SDPA runs `scaled_dot_product_attention` on the bf16
arrays.

**Cache-boundary bf16 floor (model-agnostic f32-KV guard).** The store
boundary casts incoming K/V to bf16 (`cast_store_bf16`), whatever the inbound
dtype. Two store sites apply the floor:

- `update_prefill_raw` (the prefill buffer that `exit_prefill` slices into the
  decode mirror), and
- `update_decode_fp16` (the per-step decode append; the cast also sizes the
  `zeros(...)` allocation in bf16).

`update_decode_fp16_v_only` does not apply the floor. The K-only iso and rotor
codecs use it to write their bf16 V mirror, and the codec supplies bf16.

The cast is a no-op when the input is already bf16. The floor limits the
*memory* cost of an upstream f32 leak. It does not remove the *compute* cost:
upstream f32 arithmetic stays f32. The per-arch casts
(`docs/KV_LAYER_POLICY.md` §"Gemma4 global KV is bf16 at `--kv-quant none`",
`docs/KV_LAYER_POLICY.md` §"Qwen3 dense KV is bf16 at `--kv-quant none`") are the fix; the floor is
the guard.

`crates/rmlx-kv-quant/src/kvcache/resident_bytes_tests.rs` holds the
detector: an f32 K/V fed through the prefill and decode store paths must land
as bf16 (2 B/elem). `make model-check` runs it.

**Memory cost**: `2 · B · kv_h · max_seq · head_dim · 2` bytes per layer.

**CLI**: `--kv-quant none` (aliases: `bf16`, `f16`).

---

### `KvStorage::K8V8` — symmetric 8-bit both sides

> **INERT on this build** — `k8v8` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL `q8_0`, symmetric, `group_size=128`. The per-group
scale is `max(|x|) / 127`. There is no bias term.
**V codec**: the same codec as K.

Both sides use `QuantK`. `QuantK` has two storage paths:

- **CPU path**: `Vec<u8>` codes + `Vec<f32>` scales (`q8_quantize` /
  `q8_dequantize`).
- **GPU path**: a pre-allocated 1-D `Array` pair (`gpu_codes_buf` u32,
  `gpu_scales_buf` f32). The capacity grows in steps of `KV_PAGE_SIZE = 256`
  tokens. Each step quantizes the new slice and writes it with `slice_update`.

**Buffer layout (sequence-major).** `QuantK` stores the filled prefix
**sequence-major**: the logical `[B, kv_h, S, D]` cache is laid out as
`[B, S, kv_h, D]`. For each token, all heads are contiguous. Chunk `n` occupies
`[prev_seq * words_per_seq .. (prev_seq + new_seq) * words_per_seq]`, with
`words_per_seq = B * kv_h * D / 4`. Only this order keeps the active prefix
one contiguous slice after appends in any number of chunks.

`QuantK::append` transposes the incoming head-major chunk to
`[B, new_seq, kv_h, D]` before it quantizes. `QuantK::dequantize_choice`
reshapes the prefix to `[B, S, kv_h, D]` and transposes back. When
`head_dim % 128 == 0`, every q8 group stays inside one head. When `head_dim` is
not a multiple of 128, a q8 group spans a (head, token) boundary. The spill,
hydrate and paged-grow paths copy the contiguous prefix `[0 .. filled]`, so
they do not depend on the layout.

Store-backed `update_and_sdpa` path:
1. `QuantK::append` — quantize new K, write into the GPU buffer.
2. `QuantK::append` — same for V.
3. `QuantK::dequantize_choice` — dequantize the full K prefix to bf16.
4. `QuantK::dequantize_choice` — same for V.
5. `scaled_dot_product_attention` on the bf16 arrays.

**CLI**: `--kv-quant k8v8`.

---

### `KvStorage::K8V4` — q8_0 K, TurboQuant 4-bit V

> **INERT on this build** — `k8v4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL `q8_0`, `group_size=128` (the same as the K8V8 K side).
**V codec**: TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

The split is per axis (K versus V), not per layer.

The V side uses `QuantV { bits: 4 }`:

- CPU path: `Vec<TurboBlocks>`. Each block holds 4-bit packed codes
  (`Vec<u8>`) and f32 scales (`Vec<f32>`), one block per group of 32 elements.
- GPU path: a pre-allocated 1-D u32 codes buffer and an f32 scales buffer.
  `words_per_step = B * kv_h * D * 4 / GROUP_SIZE` (four u32 words per group
  of 32 at 4 bits). The capacity grows as in K8V8.

**Buffer layout (sequence-major).** `QuantV` also stores the prefix
sequence-major (`[B, S, kv_h, D]`) on both backends. `append` reorders the
head-major chunk before it quantizes. On the GPU this is `transpose` +
`contiguous`, because the TurboQuant MSL kernel reads raw linear indices.
`dequantize_choice` reshapes the prefix sequence-major and transposes back.
The same order applies to the K-side `QuantKTurbo3` / `QuantKTurbo4` stores
(`TurboSym3` / `TurboSym4`) and to `update_paged`, which reorders `new_k` /
`new_v` before it quantizes.

Store-backed `update_and_sdpa` path (without TurboFlash):
1. Append K via `QuantK::append`.
2. Append V via `QuantV::append` (`turbo_quantize_v4_gpu` on GPU).
3. Dequantize the full K prefix via `QuantK::dequantize_choice`.
4. Dequantize the full V prefix via `QuantV::dequantize_choice`
   (`turbo_dequantize_v4_gpu`).
5. `scaled_dot_product_attention` on bf16 arrays.

**TurboFlash path** (`KvCache::update_and_sdpa_k8v4_flash`). This path keeps
its own head-major buffers (`flash_k_codes`, `flash_k_scales`,
`flash_v_codes`, `flash_v_scales`), shaped `[B, kv_h, max_seq, D/.]`. The
first TurboFlash dispatch seeds them from the bf16 prefix. Each later
dispatch appends with a 4-D `slice_update` and also appends to the bf16
mirror. The `turbo_flash_sdpa` Metal kernel reads the flash buffers directly.
`turbo_flash_should_run` allows the kernel when all of these are true:
`DispatchPolicy::turbo_flash` is set, the smoke probe has not forced a
fallback (`turbo_flash_corrupted()` is `false`), `q_seq == 1`, and
`kv_seq > turbo_flash_min_kv_seq`. The minimum defaults to 4096;
`RMLX_TURBO_FLASH_MIN` sets it. The kernel supports `head_dim ∈ {128, 256}`
on the GPU. The flash buffers are resident in addition to the
bf16 mirror.

**TurboFlash is off by default.** `--turbo-flash` accepts
`{on, off, auto}`. The default is `auto`, and `auto` resolves **OFF on every
host**. The kernel decodes slower than the generic K8V4 path. It can also
change the generated tokens. The cause is the codec, not the kernel:
TurboFlash is the only `k8v4` configuration in which the 4-bit V codec runs at
decode. With the gate off, `k8v4` decodes from the bf16 mirror.

- `--turbo-flash on` turns the kernel on.
- `--turbo-flash off` is a hard override. An exported `RMLX_TURBO_FLASH=1`
  does not survive it.
- `auto` honours `RMLX_TURBO_FLASH=1`. In that case the kernel runs while the
  flag reads `auto`, and rMLX logs a `warn!` that names the cost.
- `--turbo-flash` is a **global** flag. Every subcommand resolves it the same
  way, so `rmlx bench` and `rmlx baseline` measure the configuration that
  `rmlx serve` runs.

**The kernel's own numerics.** `turbo_flash_reference_sdpa`
(`turbo_flash_msl.rs`, `#[cfg(test)]`) is a dequantize-then-SDPA arm over the
same `flash_*` buffers. It uses the same two codecs and the kernel's f32
working precision. Thus every quantization error cancels. What remains is the
kernel's block tiling, online softmax and two-pass rescale. The gate is cosine
≥ 0.999999 and ≤ 0.5 bf16 ULP per row (`KERNEL_VS_REFERENCE_MAX_ULPS`), on a
ring whose stride is wider than its fill and whose last block is partial.
Guards: `turbo_flash_matches_its_codec_reference_at_{bonsai_8b,qwen36_35b}_geometry`
and `..._with_an_additive_mask` (`#[ignore]`, GPU). Any comparison against a
bf16 attention measures the tq4-V codec, not the kernel.

### GQA divisibility at the TurboFlash kernel entry

The MSL maps `kv_head = q_head / n_repeats`, with
`n_repeats = n_q_heads / n_kv_heads`. A count that does not divide would read
past the KV base of the batch. `validate_flash_shapes` rejects such shapes,
and a zero `n_kv_heads`. The kernel and the reference arm both call it.
`reference_and_kernel_refuse_the_same_shapes_for_the_same_reason` covers the
GQA cells. `update_and_sdpa_k8v4_flash_inner` derives both counts from the
cache's own shapes, so no in-tree caller passes a non-multiple.

**Qwen MoE note**: K8V4 passes the Qwen MoE guard because K stays 8-bit.
`validate_resolved` rejects a codec with K below 8 bits on
`Qwen3_5MoeForConditionalGeneration` and `Qwen3VLMoeForConditionalGeneration`.
The guard keys off the K-side codec, not the variant name.

**CLI**: `--kv-quant k8v4`; or `--ctk q8_g128 --ctv tq4`.

---

### `KvStorage::Planar` — q8_0 K, PlanarQuant 4-bit V

> **INERT on this build** — `planar` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL `q8_0`, `group_size=128` (the same as K8V8 / K8V4).
**V codec**: PlanarQuant 4-bit with per-pair Givens rotation, `group_size=32`.

`QuantPlanarV` keeps three GPU buffers:

- `gpu_codes_buf` (u32): four u32 words per group of 32 elements.
- `gpu_scales_buf` (f32): one f32 scale per pair of elements, 16 per group.
- `gpu_rotations_buf` (u32): two u32 words per group, eight 4-bit Givens
  rotation indices per word.

The Givens rotation operates on pairs of V values before 4-bit quantization.
The CPU path uses `planar_quantize` / `planar_dequantize` from
`rmlx_kv_quant::planarquant`. The GPU path uses the
`planar_quantize_v4_gpu` / `planar_dequantize_v4_gpu` MSL kernels.

Store-backed `update_and_sdpa` path:
1. `QuantK::append` for K (as in K8V8).
2. `QuantPlanarV::append` (`planar_quantize_v4_gpu` on GPU).
3. `QuantK::dequantize_choice` for K.
4. `QuantPlanarV::dequantize_choice` (`planar_dequantize_v4_gpu`, with the
   codes, scales and rotations buffers).
5. `scaled_dot_product_attention`.

**The planar V store is larger than bf16.** The store spends **22.00 bits per
value**, against 16.0 for bf16, at every `head_dim` and at both bit widths
(planar3 and planar4 have the same storage). The split is codes 4.0 +
**per-pair scales 16.0** + rotation indices 2.0. `kv_rate_tests.rs` reads the
bytes that `planar_quantize` produces. `SideStore::Planar` models the same
22.0.

On a seeded cache `Planar` keeps no store, so the resident V is the bf16 mirror
at 16.0 bits per value. The 22.0-bit rate applies to a store-backed planar
cache: a hydrated cache, or a cache that did not go through a prefill bracket.

**CLI**: `--kv-quant planar`; or `--ctk q8_g128 --ctv planar4`.

---

### `KvStorage::Planar` (bits=3) — PlanarQuant 3-bit V

> **INERT on this build** — `planar3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**KvQuant variant**: `KvQuant::Planar3`. It uses `KvStorage::Planar { bits: 3 }`.

**Algorithm**: the same Givens rotation and per-pair scale as the 4-bit codec.
The differences:
- **Codebook**: 3-bit Lloyd-Max N(0,1), 8 centroids (`CODEBOOK_3BIT` in
  `turboquant.rs`).
- **Pack format**: 10 values per u32 (30 bits used, 2 unused). With
  GROUP_SIZE=32, `ceil(32/10) = 4` u32 words per group. This is the same word
  count as the 4-bit codec (8 values per u32 × 4).
- **Decision boundaries**: 7 midpoints (15 for 4-bit).
- **Mask**: `0x7u` (`0xFu` for 4-bit).

**Path-independent byte stream.** The CPU codec and the MSL kernels pack codes
in the same word convention: `word = elem / (32/bits)`,
`shift = (elem % (32/bits)) * bits`, little-endian u32 words. Thus the code
bytes cross the CPU/GPU boundary unchanged. SSD spill (CPU encode) → hydrate
(GPU read) needs this. For `bits=4` the convention equals a dense LSB-first
stream. For `bits=3` it is **not** dense: a dense 3-bit layout (12 bytes per
group) would be misread as 4 u32 words (16 bytes per group). iso3 and rotor3
use the same convention.

The GPU kernels are `planar_quantize_v3_gpu` / `planar_dequantize_v3_gpu` in
`planarquant_msl.rs`. The CPU path is `planar_quantize(bits=3)` /
`planar_dequantize` in `planarquant.rs`. The load-time precompile warms
`planar_quantize_v3_gpu` (`precompile::warm_v_side`). The GPU round-trip test
is `planar_v3_msl_roundtrip_within_tolerance` (`#[ignore]`, GPU).

**Cosine gate**: mean cosine ≥ 0.9989 on the LCG fixture.

**CLI**: `--kv-quant planar3`; or `--ctk q8_g128 --ctv planar_3`; or
`--kv-preset planar3`.

---

### `KvStorage::Mixed` — MLX affine at arbitrary (bits, group_size)

**K codec**: `mx.quantize(mode="affine", bits=k_bits, group_size=k_group_size)`.
**V codec**: `mx.quantize(mode="affine", bits=v_bits, group_size=v_group_size)`.

The affine codec stores a 3-tuple `(codes_u32, scales, biases)` per side.
Reconstruction is `x = scale * code + bias`. rMLX MSL `q8_0` is symmetric, has
no bias term and uses `Q8_GROUP_SIZE=128`. The two codecs are not
interchangeable.

`MixedKvState` (`mixed_quant/state.rs`) owns the state. Its buffers grow in
`STEP=256` token increments. Each decode step:
1. `MixedKvState::update_and_fetch` calls `mx.quantize` on the new K and V
   slices, writes them with `slice_update`, and returns views of the filled
   prefix as two `MixedTuple` structs.
2. `mixed_quantized_sdpa` runs two `mx.quantized_matmul` calls (queries @ K,
   then probs @ V) on the stored 3-tuples, with no dequantize step.

Prefill: the cache accumulates raw bf16. Then `exit_prefill` calls
`bulk_init_from_fp16`, which issues one batched `mx.quantize` per side.

`KvCache::update_and_sdpa` selects this path on `self.quant`
(`KvQuant::uses_mixed_path`). `KvCache::update` refuses a `Mixed` storage.

**`KvQuant::RotK`** uses `KvStorage::Mixed` with `rotate_k=true` on
`MixedKvState` (`MixedKvState::new_rotated`). K is fixed at 8 bits and
group_size 64. The state applies a Hadamard rotation to K before it
quantizes. `mixed_quantized_sdpa` applies the same rotation to Q before the
score matmul, so the rotations cancel. See rot_k below.

**CLI**: `--kv-quant mixed_k<kb>g<kg>_v<vb>g<vg>` (for example
`mixed_k8g64_v4g64`). `--ctk rot_k --ctv <affine-tag>` selects `RotK` (see
`docs/KV_QUANT.md` § "CLI flags").

---

### rot_k — K-side Hadamard rotation

**Math**: attention scores are `Q · Kᵀ`. Insert an orthogonal rotation `R`
(`Rᵀ R = I`) into the K basis and pre-rotate Q by the same `R`:

```
(Q Rᵀ) · (K Rᵀ)ᵀ = (Q Rᵀ) · (R Kᵀ) = Q (Rᵀ R) Kᵀ = Q Kᵀ
```

The scores stay equal to the unrotated scores, up to the quantization error
on `K_rot = K Rᵀ`. A Hadamard rotation decorrelates K channels and makes their
dynamic range equal. This reduces affine quantization error **only when the
channels were unequal**. On i.i.d. uniform data the transform raises the
peak-to-RMS ratio that sets the group scale, and the rotation loses bits. See
`docs/KV_CODEC_FIDELITY.md` § "Codec fidelity — measured".

K is never inverse-rotated; the rotation cancels. V-side rotation schemes
(PlanarQuant, TurboQuant) must un-rotate the output back to the value basis.

`R` is the normalized Walsh–Hadamard matrix `H_D / sqrt(D)`. It is orthogonal
and symmetric (`R = Rᵀ`), so the same matrix rotates K and Q. The construction
needs a power-of-two `head_dim` (Sylvester recurrence).

**Matmul path** (`rot_k.rs`): MLX `matmul` against a precomputed `[D, D]` matrix.
O(D²) arithmetic per step.

**Fused FWHT kernel** (`rot_k_msl.rs`; opt-in with `--rot-k-fused on` /
`RMLX_ROT_K_FUSED=1` → `DispatchPolicy::rot_k_fused`). A Fast Walsh-Hadamard
Transform in threadgroup shared memory, fused with the affine 8-bit quantize
in one kernel pass. O(D log₂ D) arithmetic and no DRAM allocation for
`K_rot`. The output *format* matches
`mx.quantize(mode="affine", bits=8, group_size=64)`: same shapes and same
dtypes. Thus it feeds `mixed_quantized_sdpa` unchanged. The scales and biases
come back at K's dtype, as with `mx.quantize`.

It is not bit-exact with `mx.quantize`. The cause is the affine
parameterisation, not a width. MLX's `affine_quantize` loads and reduces in
`float` and casts only at the store
(`mlx/backend/metal/kernels/quantized.h:2460-2489` in 0.31.2). MLX starts from
`w_max = 0`, takes `scale = max((w_max - w_min)/n_bins, eps)`, flips its sign
toward the larger end, then snaps the zero-point: `q0 = round(edge/scale)`,
`scale = edge/q0`, `bias = at_zero ? 0 : edge`.
`metal/rot_k_fwht_quantize_d128.metal:58-59` uses the plain unsigned form,
`scale = (gmax - gmin)/255` with `bias = gmin`. Thus a value can land one
level apart between the two arms.

Storing the scales at bf16 moves the reconstruction by at most 0.0156 against the
`mx.quantize` reference on the test fixture. `fwht_quantize_types_scales_like_mx_quantize`
gates it at 0.02 and asserts that the scale and bias dtypes match
`mx.quantize`.

A matching `rot_k_fwht_rotate_gpu` kernel applies the same FWHT to Q. It
replaces the `rotate_last_axis` matmul when the fused path is active.

**Storage**: `KvStorage::Mixed`. `MixedKvState` carries a
`k_rotation: Option<Array>` field with the precomputed `R` matrix.

**Requirements**: power-of-two `head_dim`. V must be an affine codec
(`q*_g*`).

**CLI**: `--ctk rot_k --ctv <affine-tag>`, or
`--kv-quant rot_k_v<vb>g<vg>`.

**Cosine gate**: K-side cosine ≥ 0.9970 (mean) and ≥ 0.9990 (min) on the LCG
fixture (head_dim=64, 8-bit affine, group_size=64). Test:
`rot_k_hadamard_8bit_cosine_gate` in `rot_k_tests.rs`. That gate measures the
quantizer, not the rotation: it passes with the Hadamard removed. Two tests in
`rotation_fidelity_tests.rs` gate the rotation:
`rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data` (at
least 1.5 bits gained on outlier data, a loss on i.i.d. data) and
`hadamard_incoherence_ratio_beats_every_block_local_rotation`.

---

### `rot_k_tq4v` is rejected

`--kv-quant rot_k_tq4v` is rejected at parse. `--ctk rot_k --ctv tq4` is
rejected at resolve (`combo_to_kv_quant`). Each error names the replacement:
`rot_k_v4g64` (`--ctv q4_g64`), the same rotated affine 8-bit K with an
MLX-affine 4-bit V.

---

### `KvStorage::K8VTurbo3` — q8_0 K, TurboQuant 3-bit V

> **INERT on this build** — `k8vturbo3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL q8_0, `group_size=128` (the same as the K8V8 K side).
**V codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

The 3-bit codebook has 8 centroids. Pack format: 32 × 3 bits = 96 bits =
three u32 words per group. The V encode and dequant run on the CPU: `QuantV`
refuses a GPU dispatch at 2 and 3 bits. `k8vturbo3_append_msl.rs` holds the
3-bit MSL kernels. The `TurboSym3` K side and the fused-QK path use them.

**CLI**: `--kv-quant k8vturbo3`. There is no `--ctk` / `--ctv` spelling.

---

### `KvStorage::K8VTurbo3Tcq` — q8_0 K, TurboQuant 3-bit V with Viterbi trellis

> **INERT on this build** — `k8vturbo3tcq` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL q8_0, `group_size=128` (the same K side as K8VTurbo3).
**V codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook, `group_size=32`. The
**codebook is the same as plain K8VTurbo3**. Only the encode-side assignment
changes: a 4-state Viterbi trellis (`TCQ_NUM_STATES = 4`) replaces the
nearest-centroid choice.

Transition rule: `next_state = ((state << 1) | (level & 1)) mod NUM_STATES`.
The forward pass and back-trace run over each 32-element group.

The **decoder is plain `turbo_dequantize`**. At the codes and scales level, a
TCQ pack has the same format as a `K8VTurbo3` pack. Only the `KvQuant`
discriminator and the SSD layout-key tag
(`K8VTURBO3_TCQ_LAYOUT_TAG = "k8vturbo3tcq"`) tell them apart on disk. The SSD
layer rejects a cross-codec hydrate, so a TCQ payload is never read as plain
turbo3.

**Trellis degeneracy.** The per-step Viterbi cost is
`dist(value, codebook[level])`. It depends only on the level, not on the
trellis state. Every level is reachable from every state, and the codebook
does not depend on the state. Thus the minimum-cost path equals the greedy
nearest-centroid assignment. A state-dependent codebook is necessary for a
shaping gain.

**Measured claw-back: 0.000 dB**, at both shipped widths, on i.i.d. Gaussian
data and on a dim-axis sweep.
`trellis_coded_quantization_claws_back_nothing` in `rate_distortion_tests.rs`
pins this as an equality, so a trellis with a real constraint turns it red.

**Cosine target**: ≥ 0.9807 on the LCG fixture. `tcq_tests.rs` also asserts
TCQ ≥ plain turbo3 cosine on a sinusoidal fixture. Equality satisfies that
gate.

**Calibration recipe**: `--recipe turbo3_tcq` in `rmlx kv-calibrate` maps to
the internal `turboquant35` recipe (the same as `turbo3` / `turbo4`). It emits
`high_precision_indices` only. It writes no codebook override.

**Implementation scope**: CPU Viterbi encode and CPU dequant. The MSL Viterbi
kernel
([`tcq_v_msl::tcq_quantize_v3_gpu`](../crates/rmlx-kv-quant/src/tcq_v_msl.rs))
has a CPU↔GPU parity test. No production path dispatches it.

**V-side only**: K stays `q8_0` (group=128). Thus the Qwen MoE K-bits guard
does not reject `K8VTurbo3Tcq` (K = 8).

**CLI**: `--kv-quant k8vturbo3tcq`; or `--ctv turbo3_tcq`
(`CacheType::Turbo3Tcq`, canonical tag `k8v_turbo_3_tcq`).

---

### `KvStorage::K8VTurbo2Tcq` — q8_0 K, TurboQuant 2-bit V with Viterbi trellis

> **INERT on this build** — `k8vturbo2tcq` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL q8_0, `group_size=128` (the same K side as K8VTurbo2).
**V codec**: TurboQuant 2-bit Lloyd-Max N(0,1) codebook (`CODEBOOK_2BIT`,
4 centroids), `group_size=32`. The codebook is the same as plain K8VTurbo2.
The encode uses the same 4-state trellis as K8VTurbo3Tcq, over 4 centroids.
The same degeneracy applies.

Pack format: 2-bit indices, 16 values per u32 (2 u32 words per 32-element
group). This is the same as plain `turbo_quantize_v` at `bits=2`. The decoder
is plain `turbo_dequantize`. The SSD layout-key tag
`K8VTURBO2_TCQ_LAYOUT_TAG = "k8vturbo2tcq"` prevents a cross-codec hydrate.

**Cosine target**: ≥ 0.957 on the LCG fixture. `tcq_tests.rs` also asserts
TCQ V2 ≥ plain turbo2 cosine on the sinusoidal fixture.

**V-side only**: K stays `q8_0` (group=128). There is no outlier mask. The
V encode runs on the CPU. There is no GPU Viterbi kernel for 2 bits.

**Calibration recipe**: `--recipe turbo2_tcq` in `rmlx kv-calibrate` maps to
the internal `turboquant25` recipe (the same as `turbo2`). It writes no
codebook override.

**CLI**: `--kv-quant k8vturbo2tcq`; or `--ctv turbo2_tcq`
(`CacheType::Turbo2Tcq`, canonical tag `k8v_turbo_2_tcq`).

---

### `KvStorage::TurboSym4` — symmetric TurboQuant 4-bit K + V

> **INERT on this build** — `tsym4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`.
**V codec**: the same.

This is the symmetric form of `K8V4`. Both axes use the same TurboQuant 4-bit
MSL kernels (`turboquant_msl::turbo_quantize_v4_gpu` /
`turbo_dequantize_v4_gpu`). The CPU and MSL codecs take a flat f32 buffer and
a 4-D shape, so K and V share the dispatch.

K and V are **separate types** (`QuantKTurbo<4>`, spelled `QuantKTurbo4`, and
`QuantV`) inside `KvStorage::TurboSym4 { k, v, max_seq }`. The SSD layout tag
is:

```
const TURBOSYM4_LAYOUT_TAG: &str = "tsym4_lloyd_4_4";
```

**Arch guard** — `validate_resolved` rejects `--kv-quant tsym4` on
`Qwen3_5MoeForConditionalGeneration` and `Qwen3VLMoeForConditionalGeneration`
with `ResolveError::QwenMoeKBitsTooLow(4)` (exit 78). `KvQuant::k_below_8bit()`
is `true` for this variant.

**Paged routing**: `KvStorage::new(KvQuant::TurboSym4, max_seq)` returns the
non-paged `TurboSym4` storage when `--paged-kv` is set. `PagedKStorage` is
q8-only.

**Head/tail layers** — `boundary_floor` promotes the head and tail layers to
`K8V8`.

**CLI**: `--kv-quant tsym4` (or `--kv-preset quality`).

---

### `KvStorage::TurboSym3` — symmetric turbo-3 K + turbo-3 V

> **INERT on this build** — `tsym3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook (8 centroids),
`group_size=32`. On GPU: `turbo_quantize_v3_gpu` / `turbo_dequantize_v3_gpu`
from `k8vturbo3_append_msl.rs`. On CPU: `turbo_quantize_v(bits=3)`.

**V codec**: the same codebook, `group_size=32`, through `QuantV { bits: 3 }`
as in `K8VTurbo3`. The V side runs on the CPU.

**No rotation is applied on either axis**, although the family has the name
TurboQuant. The layout tag names the Lloyd-Max codebook. See `docs/KV_CODEC_FIDELITY.md` §"The
turbo family's missing rotation — what it is worth, and where" for what the
transform would buy and on which axis.

The K buffer is `QuantKTurbo<3>`, spelled `QuantKTurbo3`: the same
const-generic store as the 4-bit spelling. The SSD layout tag is:

```
const TURBOSYM3_LAYOUT_TAG: &str = "tsym3_lloyd_3_3";
```

**Arch guard** — `validate_resolved` rejects `--kv-quant tsym3` and `--kv-preset speed` on
`Qwen3_5MoeForConditionalGeneration` and `Qwen3VLMoeForConditionalGeneration`
with `ResolveError::QwenMoeTurboKRejected { variant: "tsym3" }`.

**Paged routing**: `KvStorage::new(KvQuant::TurboSym3, max_seq)` returns the
non-paged `TurboSym3` storage when `--paged-kv` is set.

**Head/tail layers** — `boundary_floor` promotes the head and tail layers to
`K8V8`.

**CLI**: `--kv-quant tsym3`; or `--ctk tsym3 --ctv tsym3`; or
`--kv-preset speed`.

---

### `KvStorage::PlanarK` — K-axis PlanarQuant 4-bit

> **INERT on this build** — `planar_k` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: PlanarQuant 4-bit Givens-rotation codec (16-entry rotation
codebook, 4-bit code, per-pair scales). It uses the same
`planarquant::planar_quantize` and the same MSL kernels
(`planarquant_msl::planar_quantize_v4_gpu` / `planar_dequantize_v4_gpu`) as the
V side of `KvStorage::Planar`. The kernels take a flat `[B, kv_h, S, D]` input
with `D % 32 == 0`, so K and V share the dispatch.
**V codec**: bf16 in `KvCache::decode_fp16_v` (the same machinery as
`KvStorage::None`).

**Buffer layout (sequence-major).** `QuantPlanarK` and `QuantPlanarV` store
the prefix sequence-major (`[B, S, kv_h, D]`). `append` reorders the
head-major chunk before it quantizes (GPU: `transpose` then
`Array::contiguous`; CPU: `transpose_heads_seq`). `dequantize_choice` reshapes
the prefix and transposes back. With `head_dim % 32 == 0`, no group spans a
(head, token) boundary, so the reorder is bit-exact.

`QuantPlanarK` gives its packed codes to the GPU kernels through
`gpu_packed_view`. These kernels index K **sequence-major**: `planar_fused_qk`,
`planar_flash_decode` (P1) and the sparse-attention phase-1/2 score kernels
compute the K token base as `kv_tok = (b * kv_seq + s) * kv_h + kv_h_idx`. The
V offset in the flash and sparse kernels stays head-major, because V is the
separate bf16 mirror.

The K buffer is its own type (`QuantPlanarK`), with the same layout as
`QuantPlanarV`, inside `KvStorage::PlanarK { k, max_seq }`. The SSD layout tag
is:

```
const PLANARK4_LAYOUT_TAG: &str = "planar_k_4";
```

**Arch guard** — `validate_resolved` rejects
`--kv-quant planar_k` and `--ctk planar_k4 --ctv bf16` on
`Qwen3_5MoeForConditionalGeneration` and `Qwen3VLMoeForConditionalGeneration`
with `ResolveError::QwenMoePlanarKRejected`.

**Paged routing**: there is no `PagedPlanarKStorage`.
`KvStorage::new(KvQuant::PlanarK, max_seq)` returns the non-paged `PlanarK`
storage when `--paged-kv` is set.

**Head/tail layers** — `boundary_floor` promotes the head and tail layers to
`K8V8`.

**CLI**: `--kv-quant planar_k` or `--ctk planar_k4 --ctv bf16`
(or `--kv-preset k_only_planar`).

---

### `KvStorage::K8VTurbo2` — q8_0 K, TurboQuant 2-bit V

> **INERT on this build** — `k8vturbo2` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See `docs/KV_QUANT.md` § "Codec disposition — what every codec in the tree is for".

**K codec**: rMLX MSL q8_0, `group_size=128` (the same as the K8V8 K side).
**V codec**: TurboQuant 2-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

The 2-bit codebook has 4 centroids. Pack format: 32 × 2 bits = 64 bits = two
u32 words per group. The V encode runs on the CPU. `turbo2_v_msl.rs` holds a
2-bit MSL kernel with a CPU↔GPU equivalence test. No production path
dispatches it.

**Naïve 2-bit codec**: there is no outlier mask. The unit-test cosine gate on
the LCG-seeded uniform fixture is mean ≥ 0.956 and min ≥ 0.925
(`turbo2_v_msl_tests.rs::tq2_cosine_naive_baseline_floor`). The fixture is
uniform, not real V tensors.

**CLI**: `--kv-quant k8vturbo2`. There is no `--ctk` / `--ctv` spelling.

---

### `KvStorage::Paged` — vLLM-style block-table KV

PagedAttention allocation. It is opt-in with `--paged-kv` (default off).

With `--paged-kv`, `KvStorage::new` routes `K8V4`, `K8V8`, `Planar` and
`Planar3` to `Paged`. Other codecs keep their own storage. `--paged-kv` with a
resolved `none` codec (which includes `--kv-quant auto`) is refused at startup,
because bf16 has no packed store to page. `--paged-kv` with a
`--cache-type-k rot_k*` is also refused at startup.

`Paged` keeps:

1. A page pool of fixed-size GPU slabs. `--paged-kv-page-tokens` sets the
   page size (default 32 tokens). The value must be a positive integer; no
   other check applies.
2. A per-sequence block table (`Vec<usize>`) that maps a logical page index to
   a physical page ID.
3. Scatter/gather: writes go to `pool[phys_id][token_slot]`, and reads
   concatenate the active pages in order.

V storage follows the base `KvQuant`:
- `K8V4` → `PagedVStorage` (TurboQuant 4-bit).
- `K8V8` → `PagedVStorage` (q8_0 codes, `bits=8`).
- `Planar` → `PagedPlanarVStorage`.

`update_paged` has no `Planar3` arm. On the GPU, a paged `Planar3` cache with
no bf16 mirror appends the new K page and then returns an error for V. By then
`KvCache::update` has already advanced the offset. Thus the cache is left
inconsistent, not only refused. On the CPU the same cache falls back to the
bf16 mirror path.

`exit_prefill` seeds the bf16 mirror, and `update_paged` returns early to the
mirror while it is live. Thus a seeded paged cache decodes from the bf16
mirror, as the non-paged forms of these codecs do.

**CLI**: `--paged-kv --kv-quant <k8v4|k8v8|planar|planar3>`. The routing
accepts all four. `planar3` reaches the defect above.

---

## TurboQuant calibration (`kv_calib.json`)

`rmlx kv-calibrate` writes a set of high-precision channel indices per KV head
to `kv_calib.json`. `rmlx serve` reads the file at model load. **No codec
applies the loaded calibration.** Every codec encodes and decodes the same with
or without the file.

### Generation

```bash
rmlx kv-calibrate /path/to/model --recipe turbo3
# Writes /path/to/model/kv_calib.json
```

`--recipe` defaults to `turbo3`. `--out` sets another output path. The
weight-norm recipes (`turbo2`, `turbo2_tcq`, `turbo3`, `turbo3_tcq`, `turbo4`)
read the K/V projection weights. Float weights (F32, BF16, F16) are read as
they are. Quantized weights are dequantized to f32 first. The command
computes the L2 norm of each head across the input dimension. It keeps the
top-K indices per head, sorted ascending, as a `Vec<u32>`. These recipes run on
the CPU and take no Metal claim.

The `head_budget`, `softmax_mass` and `k_norm_proxy` recipes write
`head_budgets.json` instead (`docs/KV_FUSED_KERNELS.md` §"Sparse attention"). `head_budget` loads the
model on the GPU and needs the Metal claim.

### Recipe → outlier count

| Recipe | Internal | Ratio | head_dim=64 | head_dim=128 |
|---|---|---|---|---|
| `turbo2`, `turbo2_tcq` | `turboquant25` | 25% | 16 | 32 |
| `turbo3`, `turbo3_tcq`, `turbo4` | `turboquant35` | 50% | 32 | 64 |

Outlier count = `round(head_dim * ratio / 16) * 16`, where `round` rounds half
away from zero (`outlier_count_for`). For head_dim 64, 128 and 256 this equals
mtq. Python rounds half to even, so the two can differ only at an exact
midpoint on another head_dim. A count of 0, or a count of `head_dim` or more,
is an error.

### Schema

`version` is always `1`. The top-level keys are `version`, `recipe` (the
internal name), `head_size` (the `head_dim` from `config.json`), `model_name`,
`transform_version`, `codebook_version`, `layers` and `calibration`
(provenance). `layers` is a `BTreeMap<String, LayerCalib>`. Its key is the
attention module path, for example `"model.layers.0.self_attn"`.

`LayerCalib` holds `key_high_precision_indices`,
`value_high_precision_indices` and an optional `codebook`. A file without
`codebook` parses to `codebook = None` (`#[serde(default)]`). No struct sets
`deny_unknown_fields`, so a reader ignores fields it does not know.
`KvCalibration`, `LayerCalib` and `CodebookOverride` are `#[non_exhaustive]`:
construct them through the writer or from JSON.

### Runtime lifecycle

1. **Discover.** When `<model>/kv_calib.json` exists, the `rmlx serve`
   loader reads `head_dim` from `config.json`. It then calls
   `rmlx_loader::discover_kv_calibration(model_dir, head_dim)`.
2. **Reject.** The loader logs a `warn!` and continues without calibration in
   five cases:
   - `config.json` does not load;
   - `config.json` gives no `head_dim`;
   - the file does not parse;
   - `version != 1`;
   - `head_size` differs from the model's `head_dim`.
3. **Attach.** The result goes on `ModelLoadConfig::calibration`. A
   `head_budgets.json` beside it attaches as `KvCalibration::head_budgets`.
4. **Not applied.** Nothing reads `ModelLoadConfig::calibration` except two
   log fields in `rmlx serve`. `KvCacheBuilder::with_calibration` and
   `rmlx_models::kv_cache::lookup_layer_calibration` have no in-tree caller.
   No codec reads `QuantV::high_precision_indices`.

`lookup_layer_calibration(calib, layer_key)` tries an exact key first. Then it
compares the first three dot-separated components, without case. So
`"model.layers.0.self_attn.k_proj"` and `"model.layers.0"` both match the key
`"model.layers.0.self_attn"`. It returns `None` when nothing matches.

### Rust API

```rust
use rmlx_loader::{
    discover_kv_calibration,
    read_kv_calibration, write_kv_calibration,
    KvCalibration, LayerCalib,
};

let calib: Option<KvCalibration> =
    discover_kv_calibration(model_dir, head_dim as u32);

use rmlx_models::kv_cache::lookup_layer_calibration;
if let Some(calib) = &builder.calibration {
    if let Some(layer) = lookup_layer_calibration(calib, "model.layers.0.self_attn") {
        // layer.value_high_precision_indices[head_idx] → sorted u32 indices
    }
}
```

### Per-layer codebook override

`LayerCalib::codebook` is an optional `CodebookOverride`:

```json
{
  "layers": {
    "model.layers.7.self_attn": {
      "key_high_precision_indices": [[0, 1, 2]],
      "value_high_precision_indices": [[3, 4, 5]],
      "codebook": {
        "value": [-2.717667, -2.052138, ..., 2.717667]
      }
    }
  }
}
```

`codebook.value` holds `2^bits` V-side centroids in **strictly ascending
order**, shared by every KV head of the layer. An absent or `null` codebook
parses to `None`.

**No production path copies the override into the codec**, so every codec uses
its built-in Lloyd-Max N(0,1) codebook. The rest of this section applies only
when a caller sets `QuantV::value_codebook`:

| `value_codebook` | Semantics |
|---|---|
| `None` | The built-in Lloyd-Max N(0,1) codebook. |
| Empty | `Error::Quant` at the first encode. |
| `2^bits` centroids | Replaces the built-in codebook for the V encode. |

- The CPU encode passes it to `turbo_quantize_v_with_codebook`. 3-bit TCQ
  passes it to `tcq_quantize_v3_with_codebook`. 2-bit TCQ always uses the
  built-in 2-bit codebook.
- The GPU encode supports `bits == 4` only. With a codebook it uploads the 16
  centroids once into `QuantV::value_codebook_gpu`. It then dispatches
  `rmlx_tq4_quantize_codebook_buffer` and `rmlx_tq4_dequantize_codebook_buffer`.
  These kernels compute the 15 decision midpoints `(cb[i]+cb[i+1])*0.5f` at
  run time. Without a codebook it dispatches `rmlx_tq4_quantize` and
  `rmlx_tq4_dequantize`, which hold the Lloyd-Max codebook in the Metal source.
- The 2-bit and 3-bit V codecs dispatch on the CPU. A GPU append with
  `bits != 4` returns `Error::Quant`.
