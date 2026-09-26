# KV Cache

The codec layer lives in **`rmlx-kv-quant`**: storage enums, MSL kernels, the
per-layer `KvCache`, paged KV and the rotating SWA ring. The policy layer
(`KvQuant` resolution, `KvCacheBuilder`, `kv_quant_for_layer`, the SSD
spill/hydrate plumbing) and the per-arch entry points live in
**`rmlx-models::kv_cache`**. See `docs/KV_QUANT.md § Public API` for the
import paths.

This doc covers how a layer's cache is sized, grown, rolled back and read at
decode, and which K/V codec pairs the resolver accepts. The flag reference and
the codec classes are in `docs/KV_QUANT.md`.

---

## 1. Flags and asymmetric K/V

Four mutually exclusive ways name a codec. They apply to `rmlx serve`,
`chat`, `info`, `baseline`, `bench` and `eval ppl`.

| Flag | Selects |
|---|---|
| `--kv-quant <name>` | a codec by its `KvQuant` name (`k8v8`, `mixed_k8g64_v4g64`, …) |
| `--kv-preset <name>` | a named preset |
| `--cache-type-k <tag>` / `--ctk`, `--cache-type-v <tag>` / `--ctv` | the K and V codec, independently |
| `--kv-bits <b>` [`--kv-group-size <g>`] | an affine width alias |

With none of them, the codec is `DEFAULT_KV_QUANT`, unquantised bf16 (§6).
rMLX flags use a double dash (`--ctk`), not llama.cpp's single dash.

K and V are separate axes. Each has its own codec, its own storage slot and
its own decode feed (§9.6). The resolver never forces one width onto both:
`--ctk q8_g128 --ctv q2_g64` stores 8-bit K beside 2-bit V.

Clap rejects `--kv-quant` together with `--ctk` / `--ctv` (exit 2). A
resolver error exits 78 (`EX_CONFIG`) and names the rule (§9).

---

## 2. KV bytes

Every decode step reads each prior token's K and V:

```
kv_bytes = 2 · n_layers · n_kv_heads · head_dim · seq_len · bytes_per_value
```

The `2` is K plus V. `n_kv_heads`, not `n_q_heads`, sets the size under GQA.
Sliding-window layers are bounded by the window, not by `seq_len` (§4.6).

`bytes_per_value` is 2 for bf16. A quantized codec lowers it only if its
packed store is what serves decode. Most codecs decode off a bf16 mirror and
build no store, so they hold exactly bf16's bytes (§9.6). The classes are in
`docs/KV_QUANT.md` "Codec disposition". `rmlx info --list-cache-types` prints
the stored rate of the rotation codecs at `head_dim` 128.

---

## 3. Chunked prefill

Each architecture prefills the prompt in fixed chunks
(`rmlx_models::prefill_chunk`). The per-arch defaults:

| Arch key | Chunk (tokens) |
|---|---:|
| `qwen3` | 1024 |
| `qwen3_5_moe` (Qwen3.5 / Qwen3.6, MoE and dense) | 2048 |
| `gemma4` | 1024 |
| `qwen3_vl_moe` | 512 |
| `gemma3`, `qwen2`, `laguna` | 256 |
| `bitnet`, any unknown class | 64 |

Resolution order: the adaptive-admission runtime override, then
`RMLX_PREFILL_CHUNK_<ARCH>`, then `RMLX_PREFILL_CHUNK`, then the table. The
runtime override is clamped to `[32, 2048]`.

---

## 4. Supported types

`rmlx info --list-cache-types` prints every tag with its codec, bits, group
and side. The rotation and K-only tags, their aliases and the full K×V
combination table are in `docs/KV_QUANT.md` § "Per-side primitive interface".
The affine and bf16 tags:

| Tag | Codec | Bits | Group | Sides |
|---|---|---:|---:|---|
| `auto` | engine default (§6) | — | — | K, V |
| `bf16` (`f16`, `none`) | unquantized | 16 | — | K, V |
| `q8_g128` | rMLX MSL q8_0 or MLX affine (see below) | 8 | 128 | K, V |
| `q8_g64`, `q8_g32` | MLX affine (scale + bias) | 8 | 64 / 32 | K, V |
| `q6_g64`, `q5_g64` | MLX affine | 6 / 5 | 64 | K, V |
| `q4_g128`, `q4_g64`, `q4_g32` | MLX affine | 4 | 128 / 64 / 32 | K, V |
| `q3_g64` | MLX affine | 3 | 64 | K, V |
| `q2_g64` | MLX affine | 2 | 64 | V |

`q8_g128` names two codecs, chosen by its partner. Paired with `q8_g128`,
`tq4`, `planar*`, `iso_v_*`, `rotor_v_*` or a TCQ tag, it is the rMLX MSL q8_0
codec (symmetric, no bias), the K side of `k8v8`, `k8v4` and `planar`. Paired
with another affine V it is the K side of `Mixed`, MLX's `mx.quantize` 3-tuple
with scale and bias: `--ctk q8_g128 --ctv q4_g64` is `mixed_k8g128_v4g64`.
`q8_g64` and `q8_g32` always resolve to `Mixed`. `q2_g64` is V-only (§5.10).

A side left `auto` beside a quantized side becomes `q8_g128`. Both sides
`auto` give `DEFAULT_KV_QUANT`.

---

## 4.5 Dynamic prefill grow + hard cap

Prefill chunks accumulate into a raw per-layer buffer
(`KvCache::prefill_raw_k/v`, `[B, kv_h, max_seq, head_dim]`). `exit_prefill`
quantizes it once, at the end of prefill.

`update_prefill_raw` calls `ensure_prefill_capacity` on every chunk. When the
chunk needs more than the storage's `max_seq`, it:

1. allocates a buffer at `next_pow2_seq(needed)`, the next power of two,
   saturated at `2^30` and clamped to the ceiling (§4.6);
2. copies the filled prefix into it;
3. raises `max_seq` on the storage variant, so `exit_prefill` sizes its
   buffers to match.

A grow is legal only before the first `exit_prefill`. On a cache whose
quantized payload already exists, it fails with
`Error::Mlx("grow not legal after exit_prefill …")`. A resumed,
hydrated or branched cache must be built large enough for its prompt.

`RMLX_KV_MAX_SEQ_HARD_CAP` is an optional upper bound, read once per process.
A prefill needing more than it fails with
`Error::KvHardCapExceeded { requested, cap }` before any allocation. Unset,
there is no cap beyond the ceiling and unified memory.

## 4.6 `--max-ctx` is a virtual ceiling (lazy grow)

`--max-ctx N` is a ceiling, not an allocation. The Qwen3, Qwen3.5/3.6,
Qwen3-VL and Gemma4 generate paths start each layer's cache at
`KV_MAX_SEQ_DEFAULT` (4096), capped at the ceiling. The buffer then grows by
powers of two up to the ceiling as the prompt fills. A short request on a
server launched with a large `--max-ctx` holds only what it fills. Each
doubling costs one realloc and copy.

The speculative round loop is the exception: `round_common::cache_stack`
builds its stacks with `max_seq` at the ceiling.

`rmlx_models::context::resolve_context` produces every context bound. It
returns `{ positional_max, ceiling, initial_max_seq }`:

- `ceiling` is `--max-ctx` when given, else `min(positional capacity, 4096)`.
- A `--max-ctx` above the positional capacity is refused, not clamped. See
  `docs/CLI.md` § "Context ceiling".
- `initial_max_seq` is `min(4096, ceiling)`.

The server's `effective_max_ctx`, the per-request `max_ctx` and the default
`--max-prompt-tokens` all read it, so the cache ceiling and the request guard
cannot drift.

`KvCache::with_max_seq_ceiling(ceiling)` records the ceiling on each layer; a
value `<= 0` clears it. `ensure_prefill_capacity` rejects a prefill needing
more with `Error::KvCeilingExceeded { requested, ceiling }` before any
allocation. The server rejects an over-long prompt earlier
(`effective_max_ctx_for`); the engine check is the only guard on the CLI
`baseline` and `chat` paths.

### Windowed-layer ring sizing

A sliding-window layer uses the rotating ring (`rmlx-kv-quant::rotating`, a
port of mlx-lm `RotatingKVCache`). `KvCache::with_quant_max_seq_window`
takes it whenever `window > 0`, under every `KvQuant`. A quantized codec on an
SWA layer therefore runs the bf16 ring and never allocates its `KvStorage`.
mlx-lm does the same: `RotatingKVCache.to_quantized` raises.

The ring's buffer is bounded by the window, never by the context.
`update_in_place` grows only while `buf_len < max_size`. `update_concat` trims
to `max_size - 1` before appending a prefill chunk, so the buffer peaks at
`window - 1 + chunk` rows. The logical `offset` counts every token seen; it is
not the physical fill.

For one layer at `kv_h = 1`, `head_dim = 256`, window 512 and chunk 512, the
windowed ring holds 1,047,552 bytes at 4k, 16k and 64k context. A global
layer grows with `max_seq`. `windowed_ring_sizing_tests.rs` pins both: the
byte bound in `windowed_ring_stays_bounded_while_global_grows`, and the full
most-recent window in `windowed_ring_retains_full_swa_window`.

### Rolling the SWA ring back

`KvCache::truncate_to(n)` is the rollback a speculative round runs after a
partial acceptance. A rotating ring can serve it in exactly two states.

- **Before the first wrap** (`offset < max_size`), every position written is
  still in its own slot, so the rollback moves the write pointer. This is
  mlx-lm's `is_trimmable` (`cache.py:542-543`).
- **After a wrap, while the buffer is in temporal order.** `update_concat`
  leaves it that way, holding `max_size + s - 1` positions for a write of `s`,
  so every token in the block sees a full window. Dropping the last `n` is
  lossless while what is left still covers the window the rolled-back offset
  needs. A block-verify write can therefore always roll back its own rejected
  tail, which is at most `s - 1` long.

A ring left in **rotated** order by single-token writes past the wrap cannot
roll back: its newest slots hold the positions they overwrote. `truncate_to`
returns an error naming the layer and the distance.
`KvCache::can_truncate_to(n)` is the predicate it implements, so a caller's
gate and the operation cannot drift apart. The gemma4 partial-prefix path asks
first and re-prefills instead. A speculative round loop has no second route
and lets the failure surface.

mlx-lm's `trim_prompt_cache` returns 0 for the refused case, which is safe
only for a caller that re-prefills. A round loop would keep the rejected
drafts in the ring while the other layers roll back.

### `resident_bytes` counts the filled prefix

`KvCache::resident_bytes()` reports the K/V that serves decode: the filled
prefix of each buffer, not its capacity. The decode mirrors
(`decode_fp16_k/v`) and the `KvQuant::None` storage are counted by `offset`,
clamped to capacity, at each buffer's own shape and dtype. So the figure
depends on the prompt, not on `--max-ctx`, and per-layer `head_dim`
differences are picked up. Quantized storage is compacted at `exit_prefill`
and counted as-is. A prompt-cache snapshot is never part of the sum, so
`kv_cache_bytes` is live-inference KV on every arch.

### Per-request `max_ctx`

The OpenAI route accepts an optional `max_ctx` field. It overrides the launch
`--max-ctx` for that request, through the same `resolve_context`, so it is
refused with the same message. With the per-request `kv_quant` field, one
resident model can sweep codec × context cells without a reload. See
`docs/SERVER.md` § "Per-request KV config".

## 5. Hard invariants

The resolver enforces the `head_dim`, pairing and arch rules below. A
violation exits 78 with a message naming the input and an alternative. The
SWA, sequence-major and paged subsections are storage contracts.

### 5.1 `head_dim % group_size == 0` (affine codecs)

An MLX affine codec shares one scale and bias per group, so `head_dim` must
be a multiple of the group (`GroupSizeNotDivisible`). Example: `--ctv q4_g64`
at `head_dim` 80.

### 5.2 MLX bit-packing rule (`head_dim % (32 / bits) == 0`)

MLX packs `bits` into 32-bit words, so `head_dim` must be a multiple of
`32 / bits` (`MlxBitPackingViolation`). Example: `--ctv q3_g64` at
`head_dim` 64 (64 % 10 ≠ 0).

### 5.3 K-side rotation codecs

The V rotation codecs (`tq4`, `planar4`, `planar3`, `iso_v_*`, `rotor_v_*`,
the TCQ tags) are V-only. On K they fail with `KSideRotationCodec`.

**`rot_k`** is the K-side rotation codec in the Mixed family. K is 8-bit
affine, group 64, in a Hadamard-rotated basis. Q is pre-rotated by the same
`R` before the score matmul, so the rotations cancel:

```text
  (Q Rᵀ) · (K Rᵀ)ᵀ = Q (Rᵀ R) Kᵀ = Q Kᵀ      (R orthogonal)
```

K is never inverse-rotated; it stays quantized in the rotated basis.

- `--ctk rot_k` only. On V it fails with `RotKVSide`.
- V must be an affine tag and resolves to `rot_k_v<bits>g<group>`. A V of
  `auto` becomes `q8_g128`. A non-affine V is an `UnsupportedCombo`.
- `head_dim` must be a power of two (`RotKHeadDimNotPow2`).
- Opt-in only; `auto` never selects it.

`R` is applied by a plain MLX `matmul` against a `[D, D]` matrix by default.
`--rot-k-fused on` (or `RMLX_ROT_K_FUSED=1` under `auto`) selects a fused FWHT
+ quantize Metal kernel (`rot_k_msl.rs`) for `D ∈ {32, 64, 128, 256, 512}`.
Other `D` fall back to the matmul. The math is in
`crates/rmlx-kv-quant/src/rot_k.rs`.

### 5.4 Qwen MoE family requires K-bits ≥ 8

On `Qwen3_5MoeForConditionalGeneration` and
`Qwen3VLMoeForConditionalGeneration`, `validate_resolved` rejects every codec
whose K side is below 8 bits: `Mixed` with `k_bits < 8`
(`QwenMoeKBitsTooLow`), `planar_k` (`QwenMoePlanarKRejected`), the iso K
codecs (`QwenMoeIsoKRejected`), the rotor K codecs (`QwenMoeRotorKRejected`),
`tsym3` (`QwenMoeTurboKRejected`) and `tsym4` (`QwenMoeKBitsTooLow`). It runs
after `auto` is filled in and on the `--kv-quant` path too, so no spelling
bypasses it.

### 5.5 `tq4` requires `head_dim ∈ {128, 256}`

The TurboQuant 4-bit V codec fails otherwise (`Tq4UnsupportedHeadDim`).

The resolver checks the full-attention `head_dim`. For Gemma3 and Gemma4,
`ModelConfig::head_dim()` returns `text_config.global_head_dim`, which is 512
on Gemma4. So `--ctv tq4` on any Gemma4 fails at startup. The 256-wide
Gemma4 layers are SWA layers and stay bf16 anyway (§5.7).

At decode, `ensure_decode_capacity` grows the storage window past a
power-of-two boundary. The TurboFlash path calls it before appending, and
`grow_flash_buffers` re-sizes its own head-major buffers to match.

### 5.6 The planar encoder needs `head_dim % 32 == 0`

PlanarQuant groups are 32 wide. The resolver does not check the width.
`planarquant` refuses a row of another width when it encodes. `planar`,
`planar3` and `planar_k` decode off the bf16 mirror (§9.6), so the encoder
runs only on a cache with no mirror.

### 5.7 SWA layers always bf16

Gemma3 and Gemma4 sliding-window layers use the bf16 rotating ring under every
codec; quantization applies to full-attention layers only. A one-shot `info!`
at startup says so when a quantized codec is requested on such a model.

The ring is not serialised to the SSD tier, so SWA layers hydrate as empty
`KvStorage::None`. A hydrated entry whose length is not block-aligned then
degrades to a full re-prefill. See `docs/SSD_TIER.md` §"SWA layers are not
spilled".

### 5.7.1 Flat quantized stores are sequence-major

Every flat-buffer quantized store (`QuantK`, `QuantV`, `QuantKTurbo*`,
`QuantPlanarK`, `QuantPlanarV`) appends one chunk per call at
`prev_seq * words_per_seq`. The buffer is laid out sequence-major:
`[B, S, kv_h, D]`, all heads of a token contiguous. `append` reorders the
incoming head-major chunk to sequence-major before quantizing.
`dequantize_choice` reshapes the prefix as `[B, S, kv_h, D]` and transposes it
back to `[B, kv_h, S, D]`. The shared helpers are in `storage/seq_layout.rs`.

For one decode token (`new_seq == 1`) the reorder is the identity and is
skipped. For a single cold-prefill chunk the two reorders cancel. The GPU side
materialises each transpose with `Array::contiguous` before a custom kernel.

A flat store's prefix is one sequence-major run only at `B == 1`. Its readers
refuse `B != 1` rather than reorder.

### 5.7.2 Packed-K decode kernels read K sequence-major

The packed-K decode kernels (`planar_fused_qk`, `planar_flash_decode`,
`rotor_flash_decode`, the sparse-attention score kernel) index K as
`kv_tok = (b * kv_seq + s) * kv_h + kv_h_idx`. The bf16 mirror stays
head-major, `[B, kv_h, S, D]`, and a kernel reading V from it indexes it so.

### 5.7.3 The Iso / Rotor `Vec<Blocks>` codecs use the same sequence-major rule

`QuantIsoV<BITS>`, `QuantIsoK<BITS>`, `QuantRotorV<BITS>` and
`QuantRotorK<BITS>` keep one `*Blocks` entry per `append` and concatenate them
on `dequant`. Each `append` reorders the head-major chunk to sequence-major
before encoding; `dequant` reorders back. A multi-chunk decode at `B > 1`
uses `transpose_chunked_seq_heads`, which knows where each chunk ends.

These codecs are per-row positional, so their sidebands stay aligned:

- The iso per-(token, group) scale and norm permute with the value rows. The
  iso quaternion is the constant `FIXED_QUAT` for every group.
- The rotor table and the QJL projection are keyed by group position or by
  the projection, not by token. The per-token QJL codes and norms permute with
  the rows.

The GPU encode (`packed_k_chunk_seq_major`, `QuantIsoV::append_gpu`) makes the
same reorder and skips it for a one-token chunk. A `.kvb` SSD block stores the
rows in this sequence-major order.

### 5.7.4 The unquantised store boundary floors K/V to bf16

The bf16 decode mirror and `KvQuant::None` store bf16 by contract. Incoming
K/V carry whatever dtype the model's attention produced, and one f32 scalar
upstream would promote them to f32.

`cast_store_bf16` therefore casts incoming K/V to bf16 at the model-agnostic
store boundary: `update_prefill_raw` for the seed and `update_decode_fp16` for
decode. An input already bf16 costs one dtype check. The cast bounds memory
only; upstream f32 compute stays f32. `f32_prefill_seed_is_stored_bf16` and
`f32_decode_store_is_stored_bf16` (`resident_bytes_tests.rs`) pin it. See
`docs/KV_CODECS.md` §`KvStorage::None`.

### 5.7.5 `exit_prefill` runs on a worker thread — MLX stream affinity

`exit_prefill` evaluates on the thread that runs the request's generation, a
tokio blocking-pool worker under `rmlx serve`. MLX default streams are
per-thread. On the linked MLX 0.31.x the CPU command-encoder map is
process-global, so an array built on one thread evaluates on another.
`cross_thread_eval_resolves_through_the_process_global_encoder_map`
(`rmlx-mlx`) pins that. The map is filled without synchronisation, so
concurrent evaluation is serialised by the evaluation lock; see `docs/FFI.md`
§ "Evaluation (lazy graph)".

The generation entry points call `rmlx_mlx::ensure_cpu_default_stream()` and
`ensure_gpu_default_stream()` before building a graph, so each worker owns its
streams. `k8v8_q8_quantize_eval_on_worker_thread` (`rmlx-kv-quant`) covers the
`exit_prefill` quantize on a worker.

The limitation under MLX 0.32.0, where a cross-thread eval throws, is in
`docs/FFI.md` § "Per-thread CPU stream context — `ensure_cpu_default_stream`".

### 5.8 `head_dim` must be declarable

With no `head_dim` from `ModelConfig::head_dim()`, a `--ctk` / `--ctv` spec
fails with `HeadDimUnknown`. The resolver does not guess. `--kv-quant` skips
the affine validators.

### 5.9 Paged-KV append model

With `--paged-kv`, `KvStorage::new` routes `K8V4`, `K8V8`, `Planar` and
`Planar3` to `KvStorage::Paged`. Every other codec keeps its contiguous
storage. Pages hold `--paged-kv-page-tokens` tokens (default 32). All four
codecs decode off the bf16 mirror (§9.6), so a seeded cache builds no pages.

Paging is sequential append (`crates/rmlx-kv-quant/src/paged/ops.rs`). Each
`append` fills the next token slots and allocates a page when the current one
is full. `block_table: Vec<usize>` grows monotonically. There is no
`slot_mapping`, no scatter and no negative-index skip: chunked prefill splits
the prompt into contiguous chunks with no padding tokens. `update_paged`
reorders chunks to sequence-major before quantizing, since the page slabs are
token-major.

### 5.10 Pure 2-bit K is gated

`q2_g64` is V-only. `combo_to_kv_quant` rejects `q2_g64` on K with
`UnsupportedCombo`, because 2-bit K makes attention output incoherent. The
asymmetric forms are accepted:

- `--ctk q8_g128 --ctv q2_g64`, or
- `--kv-bits 2` (K stays 8-bit, V is 2-bit; group from `--kv-group-size`).

2-bit V runs through the `Mixed` `mx.quantize` path, 16 values per `u32`.

---

## 6. Default policy

With no codec flag, the codec is `rmlx_models::kv_cache::DEFAULT_KV_QUANT`,
unquantised bf16, on every architecture and prompt length. The CLI, the
server load path, the arch dispatcher and
`speculative::round_common::verifier_cache_stack` read that one constant.
The two-model draft stack uses the verifier's resolved codec.
`--kv-preset auto` resolves to the same constant.

Every codec stays selectable by name. `docs/KV_QUANT.md` "The auto default"
gives the reasons, and "Codec disposition" says which codecs change resident
KV at all.

---

## 7. Migration from llama.cpp

| llama.cpp | rMLX | Notes |
|---|---|---|
| `-ctk q8_0` | `--cache-type-k q8_g32` | Same block of 32. |
| `-ctk q4_0`, `-ctk q4_1` | `--cache-type-k q4_g32` | rMLX has no min/max codec. |
| `-ctk q5_0` | `--cache-type-k q5_g64` | No 5-bit group-32 codec. |
| `-ctk iq4_nl` | `--cache-type-k q4_g64` | No non-linear 4-bit codec. |
| `-ctk f16 -ctv f16` | default, or `--ctk bf16 --ctv bf16` | rMLX stores bf16. |
| asymmetric K8V4 | `--kv-quant k8v4` or `--ctk q8_g128 --ctv tq4` | Both give `KvQuant::K8V4`. |

The tags `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1` and `iq4_nl` are refused by
name, with the nearest rMLX tag in the error. They are llama.cpp's block-32
layout with an inline fp16 scale per block, which neither rMLX's `q8_g128` nor
MLX's affine 3-tuple matches.

---

## 8. Examples

```bash
rmlx baseline --model <snapshot>                          # bf16, the default
rmlx baseline --model <snapshot> --kv-quant mixed_k8g64_v4g64
rmlx baseline --model <snapshot> --ctk q8_g128 --ctv q4_g64
rmlx baseline --model <snapshot> --ctk rot_k --ctv q4_g64
```

On the Qwen MoE family keep K at 8 bits (§5.4).

---

## 9. Failure modes

| Exit | Source | Meaning | Example trigger |
|---|---|---|---|
| 2 | clap | Preset and primitive flags both set. | `--kv-quant k8v4 --ctk q8_g128` |
| 1 | CLI parse | Unknown tag. | `--ctv garbage` |
| 1 | CLI parse | Reserved llama.cpp tag. | `--ctv q8_0` |
| 78 | resolver | `HeadDimUnknown` | a spec on a model with no `head_dim` |
| 78 | resolver | `KSideRotationCodec` | `--ctk tq4` |
| 78 | resolver | `RotKVSide`, `RotKHeadDimNotPow2` | `--ctv rot_k` |
| 78 | resolver | `QwenMoeKBitsTooLow` and the other Qwen MoE guards | `--ctk q4_g64` on Qwen3.6 |
| 78 | resolver | `Tq4UnsupportedHeadDim` | `--ctv tq4` at `head_dim` 64 |
| 78 | resolver | `GroupSizeNotDivisible` | `--ctv q4_g64` at `head_dim` 80 |
| 78 | resolver | `MlxBitPackingViolation` | `--ctv q3_g64` at `head_dim` 64 |
| 78 | resolver | `UnsupportedCombo` | `--ctk q8_g64 --ctv tq4` |

A `--kv-quant` value runs the same `validate_resolved` guards and exits 78 on
a violation.

---

## 9.5 Head-major persistent K shadow

The fused-QK MSL kernels read K from a head-major shadow
(`KvCache::fused_qk_shadow`), not from the codec storage. It holds:

- `k_codes`: `u32 [B, kv_h, max_seq, codes_per_token]`;
- `k_scales`: `f32 [B, kv_h, max_seq, scales_per_token]`;
- for rotor-asym only, per-token `sideband_norms` and a static per-layer
  `sideband_rotor_table`.

`FusedQkLayout::for_codec` accepts `K8V4`, `K8V8`, `TurboSym3`, `TurboSym4`,
`RotorK3Asym` and `RotorK4Asym`. Every other codec gets `Ok(None)` and decodes
through the bf16 SDPA path.

The shadow is allocated on the first fused-QK decode dispatch, when the
cache's `DispatchPolicy::fused_qk` is set. `--fused-qk` defaults off; `auto`
honours `RMLX_FUSED_QK=1`. Its life:

1. **Allocate** zero-initialised at `[B, kv_h, max_seq, *]`.
2. **Seed** by encoding the bf16 K mirror's prefix. A cache with no mirror
   takes no fused-QK path.
3. **Grow** by encoding each decode token into its slot with `slice_update`.

The bf16 mirror stays maintained beside it as the fallback. A decode past
`max_seq`, or rotor with QJL on, skips fused-QK for that step.
`KvCache::reset` drops the shadow; `truncate_to` moves only its fill cursor.
Rotating caches never get one (`storage_max_seq_for_fused_qk` is `None`).
Per-codec shapes and the dispatch wire-in are in `docs/KV_FUSED_KERNELS.md`
§ "Fused-QK head-major K storage".

**Per-step cost framing.** The shadow is stored head-major, but the kernel
input is the `[B, kv_h, max_seq, payload]` buffer sliced to `kv_seq` on dim 2
and flattened. That slice is not contiguous, so each step copies
`B * kv_h * kv_seq * payload` bytes. Removing the copy would need every
fused-QK kernel to take an explicit `max_seq` row stride.

## 9.6 Warm-TTFT bf16-K decode contract

A codec's quantizer runs once, at `exit_prefill`. At decode, most codecs then
serve K and V from a bf16 mirror (`decode_fp16_k` / `decode_fp16_v`) and never
run the codec again. Three predicates on `KvQuant` classify every codec:

- `feeds_bf16_k_at_decode(shares_kv)` and `feeds_bf16_v_at_decode(shares_kv)`:
  decode reads the bf16 seed on that axis;
- `decode_reads_packed_store()`: some decode path reads the packed store.

`exit_prefill` stores a bf16 seed for each axis whose decode reads one. It
builds the packed store only when `materialises_packed_store()` holds:
`decode_reads_packed_store() || !feeds_bf16_k || !feeds_bf16_v`. A mirror-fed
codec therefore builds no store, and its resident KV equals `--kv-quant none`.

Each mirror-fed `update_<codec>` begins with
`if self.decode_fp16_k.is_some() { return self.update_decode_fp16(...); }`, so
one `slice_update` appends the decode token's K and V.

| Codec (`KvQuant`) | decode K | decode V | packed store |
|---|---|---|---|
| `None` | bf16 | bf16 | — |
| `K8V8`, `K8V4`, `Planar`, `Planar3`, `PlanarK` | bf16 | bf16 | not built |
| `K8VTurbo2/3` and their `Tcq` forms, `TurboSym3/4` | bf16 | bf16 | not built |
| `Iso3/4`, `Rotor3/4`, `RotorK{3,4}Asym` | bf16 | bf16 | not built |
| `Mixed`, `RotK` | quant | quant | built |
| `IsoKOnly3/4`, `RotorKOnly3/4` | quant | bf16 | built |
| `Iso3Sym`, `Iso4Sym`, `Rotor3Sym`, `Rotor4Sym` | quant | quant | built |

Notes:

- `Mixed` and `RotK` keep a bf16 mirror only when the cache's `shares_kv` is
  set: an arch whose consumer layers read another layer's K/V (Gemma4). There,
  `update_and_sdpa_shared_source` hands the mirror to the consumers. Elsewhere
  no mirror is built.
- The K-only family re-quantizes K every step and routes V through
  `update_decode_fp16_v_only`, which must not touch `decode_fp16_k`.
  `exit_prefill` builds no bf16 K seed for it.
  `warm_ttft_cross_codec_tests::iso_k_only3_quant_at_decode` pins the absent
  seed and the byte total.
- The fused symmetric family decodes with a flash kernel over both packed
  rings and keeps no mirror.
- TurboFlash (`K8V4`) and fused-QK keep their own head-major buffers,
  re-encoded from the mirror. Neither reads the packed store.
- `PlanarK`'s fused-QK and flash-decode arm runs only when the K mirror is
  absent, which a seeded cache never is.

`decode_reads_packed_store()` reads the one place the classification lives:
the codec's row in `KvQuant::descriptor` (`quant_descriptor.rs`). A codec that
gains a decode kernel over its own store flips `reads_packed_store` in that
row, and `exit_prefill` builds the store in the same change.

**Where the store is still read.** A cache with no mirror decodes through the
codec body: one rebuilt by `KvCache::from_storage`, or one that never
bracketed a prefill. The SSD tier spills a mirror-fed layer's bf16 mirror
under the `none_bf16` tag, and hydrate re-seeds the mirror, so a hit from disk
decodes off the same bytes as a hit from RAM.

The resolve-time `warn!` for an inert codec and the byte estimate
(`KvQuant::estimated_resident_bytes_per_layer`) follow the same predicates.
No production path calls the sparse-attention dispatcher
(`sparse_attn_dispatch_if_enabled`).

---

## 10. Further reading

- llama.cpp 4-bit KV cache discussion:
  <https://github.com/ggml-org/llama.cpp/pull/5932>
- MLX `mx.quantize`:
  <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html>
- IsoQuant (ParaMind2025): <https://github.com/ParaMind2025/isoquant>. SO(4)
  isoclinic rotation, stage-1 quantize/dequantize only. No cache and no decode
  path upstream, so rMLX's `iso*` KV codecs have no counterpart to port.
- `docs/KV_QUANT.md`, `docs/KV_CODECS.md`, `docs/KV_FUSED_KERNELS.md`.
