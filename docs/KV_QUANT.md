# KV Cache Quantization Reference

> The codec code lives in **`rmlx-kv-quant`** (storage enums, MSL kernels,
> per-layer `KvCache`, paged KV, mixed / rot-K codecs). The SSD tier lives in
> **`rmlx-kv-ssd`**. The policy / builder layer (`KvCacheBuilder`,
> `kv_quant_for_layer`, `DEFAULT_KV_QUANT`, `LAYER_ADAPTIVE_*`,
> `cache_type::*`) lives in **`rmlx-models::kv_cache`**. See § "Public API"
> below for the import paths.

Codec-level reference for every KV quantization variant in rMLX. Covers
storage layout, dispatch path, the auto default, and CLI flag surface.

For the flag-surface overview and per-command usage see `docs/KV_CACHE.md`.
For weight quantization see `docs/WEIGHT_QUANTS.md`. For the SSD spill tier
see `docs/SSD_TIER.md`.

---

## Public API

The `rmlx-kv-quant` crate owns these public items:

| Item                                       | Source                                       |
|--------------------------------------------|----------------------------------------------|
| `KvQuant`, `KvQuantParseError`             | `rmlx_kv_quant::quant`                       |
| `KV_MAX_SEQ_DEFAULT`                       | `rmlx_kv_quant::quant`                       |
| `KvCache`                                  | `rmlx_kv_quant::kvcache`                     |
| `LinearAttnCache`, `GdnTape`, `GdnTapeSegment` | `rmlx_kv_quant::linear_attn`             |
| `KvStorage`, `QuantK`, `QuantV`, `QuantPlanarV` | `rmlx_kv_quant::storage`                |
| `MixedKvState`, `MixedTuple`               | `rmlx_kv_quant::mixed_quant`                 |
| `PagedKStorage`, `PagedVStorage`, `PagedPlanarVStorage`, `install_paged_kv`, `resolve_paged_kv`, `resolve_paged_kv_page_tokens` | `rmlx_kv_quant::paged` |
| `q8_quantize`, `q8_dequantize`, `Q8_GROUP_SIZE` | `rmlx_kv_quant::q8`                     |
| `turboquant::{TurboBlocks, turbo_quantize_v, turbo_dequantize, GROUP_SIZE, …}` | `rmlx_kv_quant::turboquant` |
| `planarquant::{PlanarBlocks, planar_quantize, planar_dequantize, …}`           | `rmlx_kv_quant::planarquant` |
| MSL wrappers: `q8_msl::*`, `turboquant_msl::*`, `planarquant_msl::*`, `turbo_flash_msl::*`, `rot_k_msl::*`, `k8vturbo3_append_msl::*` | `rmlx_kv_quant::*` |
| SWA ring buffer: `rotating::*` | `rmlx_kv_quant::rotating` |

The `rmlx-kv-ssd` crate owns the SSD tier: `block_io`, `spill`, `hydrate`,
`ssd_index`, `ssd_tier`, and the hooks `set_ssd_event_recorder`,
`set_ssd_spill_prom_hook`, `set_ssd_hydrate_prom_hook`,
`set_ssd_bytes_used_hook`, `set_ssd_evict_total_hook`.

The policy / builder layer is in `rmlx-models::kv_cache`:

* `KvCacheBuilder`, `kv_quant_for_layer`, `DEFAULT_KV_QUANT`
* `LAYER_ADAPTIVE_TAIL_N`, `LAYER_ADAPTIVE_HEAD_N`
* `cache_type::*`

## Import paths

The import path of each public symbol:

```rust
// Codec layer — rmlx-kv-quant root + module re-exports:
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache, KV_MAX_SEQ_DEFAULT};
use rmlx_kv_quant::storage::{KvStorage, QuantK, QuantV, QuantPlanarV};
use rmlx_kv_quant::mixed_quant::{MixedKvState, MixedTuple};
use rmlx_kv_quant::paged::{PagedKStorage, PagedVStorage, PagedPlanarVStorage};
use rmlx_kv_quant::turboquant::{TurboBlocks, turbo_quantize_v, turbo_dequantize, GROUP_SIZE};
use rmlx_kv_quant::planarquant::{PlanarBlocks, planar_quantize, planar_dequantize};
use rmlx_kv_quant::{q8_msl, turboquant_msl, planarquant_msl, turbo_flash_msl};

// SSD-tier layer — rmlx-kv-ssd root + module re-exports:
use rmlx_kv_ssd::{
    write_caches, BlockIoError, KvBlockReader, KvBlockWriter, SsdKvIndex,
    SsdSpiller, SsdHydrator, SpillJob, HydratedBlock,
    set_ssd_event_recorder, set_ssd_spill_prom_hook, set_ssd_hydrate_prom_hook,
    set_ssd_bytes_used_hook, set_ssd_evict_total_hook,
};
use rmlx_kv_ssd::ssd_tier::{install_config, active, compute_layout_key, SsdTierConfig};
use rmlx_kv_ssd::{block_io, hydrate, spill, ssd_index};

// Builder / policy (rmlx-models):
use rmlx_models::kv_cache::{
    KvCacheBuilder,
    kv_quant_for_layer, DEFAULT_KV_QUANT,
    LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N,
    cache_type, CacheType, CacheTypeSpec,
    parse_cache_type_str, resolve_cache_type, validate_resolved_kv_quant,
    ResolverContext,
};

// Arch dispatch (Gemma4 / Qwen3 / Qwen3.5-MoE attach_at_load) stays in
// rmlx-models because its trait impls live there:
use rmlx_models::ssd_tier::attach_at_load;
```

---

## Overview

rMLX stores K and V tensors for each attention layer in a `KvCache` struct.
On each decode step the hot path appends the new K/V slice, then runs scaled
dot-product attention (SDPA) over the full accumulated prefix.

Two enums control the codec:

- `KvQuant` — the logical quantization mode. It is set at construction time
  and does not change for the request.
- `KvStorage` — the buffer variant that holds the data. `KvCache::update`
  matches `&self.storage`, not `self.quant`. See § "Dispatch axis".

`--kv-quant auto` resolves to `none` (bf16) on every arch
(`DEFAULT_KV_QUANT`). Every other codec is opt-in.

Sliding-window attention (SWA) layers do not use the codec. They use
`RotatingState`, a bf16 ring buffer, for every `KvQuant`.

### Per-layer net-benefit decision + net-negative warn

SWA layers always run the bf16 ring. Thus windowed layers are bf16 and global
(full-attention) layers carry the codec. The codec cannot make a windowed layer
larger.

A codec can make a global layer larger than bf16. This occurs when the layer
keeps a packed store **and** a bf16 mirror (`decode_fp16_k` / `decode_fp16_v`).
Two predicates decide the mirrors: `KvQuant::feeds_bf16_k_at_decode` and
`KvQuant::feeds_bf16_v_at_decode`. `KvQuant::materialises_packed_store` decides
the store.

- **bf16-mirror family** (`K8V4`, `K8V8`, `Planar*`, `PlanarK`, `K8VTurbo*`,
  `TurboSym*`, `Iso3/4`, `Rotor3/4`, `RotorK*Asym`). No decode path reads the
  store, so `exit_prefill` does not build it (`docs/KV_CACHE.md` §9.6 F3). The
  resident KV is the two bf16 mirrors. This is the same byte count as
  `--kv-quant none`.
- **`Mixed` / `RotK`.** Their decode reads the affine 3-tuples. They keep the
  mirrors only when the layer shares K/V across layers (`KvCache::shares_kv`,
  set by the arch builder). Gemma4 shares; Qwen3 and Qwen3.5-MoE do not. On a
  layer that does not share, the resident KV is the packed store alone. On a
  shared-KV arch, only the producer layer of a shared pair
  (`update_and_sdpa_shared_source`) appends to the mirror at decode. On every
  other layer the mirror stays at the prompt length, because `KvCache::update`
  refuses a `Mixed` storage and `update_decode_fp16` does not run.
  `boundary_floor` raises a promoted `Mixed` / `RotK` boundary layer to 8 bits
  in its own family when the layer does not share K/V. On a shared-KV layer
  the raised codec still reads both mirrors, so the result is `K8V8`. See
  § "Which codec the floor is".
- **Store-reading families** (`IsoKOnly*`, `RotorKOnly*`, `Iso*Sym`,
  `Rotor*Sym`). They keep the packed store. The K-only codecs also keep a bf16
  V mirror.

Size a Gemma4 layer from its **class**. Windowed layers are `head_dim` ×
`num_key_value_heads`. Global layers are `global_head_dim` ×
`num_global_key_value_heads`, which is a different shape (see
docs/MODELS.md, "Attention geometry splits by layer class").
`gemma4/generate/mod.rs` builds the per-layer `KvLayerShape` vector from these
two pairs.

At request build time, rMLX emits one structured `warn!` when the resolved
codec is estimated to hold more resident KV than bf16 on the active layer mix:

```
WARN KV codec increases resident KV vs bf16 on this layer mix — the
per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved
at this context; windowed layers already run bf16 and are unaffected. Read the
sign, not the magnitude: the estimator under-reports an iso codec while a layer
holds the CPU blocks the prefill encode built — until the first fused decode
step drops them, or for the whole request on a layer the fused path's shape gate
rejects (batch > 1, or a head_dim that is not a power of two at most 512), where
they are never dropped. Consider --kv-quant none if memory is the goal.
  kv_quant=<codec> eff_seq=<tokens> n_global=<n> n_windowed=<n> est_extra_bytes=<bytes>
```

`n_global` and `n_windowed` are the two layer classes. `est_extra_bytes` is
`estimated_net_saving_per_layer` summed over the mix, with the sign flipped. To
see the per-layer figures for a codec, run `rmlx info --list-cache-types`.

Only the **sign** of that number is exact. Do not size a buffer from its
magnitude.

The estimate sizes each side byte-for-byte against the store it writes.
`every_codec_byte_model_matches_the_store_it_writes` checks each `SideStore`
cadence against the store's own encoder at `head_dim` 64, 128 and 256. The
estimate does not model page rounding, the static per-layer rotation tables, or
GPU/CPU residual coexistence.

The estimate runs **low** for `k_iso*` and `iso*_sym`. It sizes an iso side
from the GPU ring. But `exit_prefill` encodes into CPU `IsoBlocks`, which are
3.98× the ring at `head_dim = 128`. The first fused decode step frees the
blocks (`drop_blocks_when_ring_live_iso_*`) on a layer the fused path serves.
The fused path's shape gate rejects batch > 1 and a `head_dim` that is not a
power of two at most 512 (`head_dim = 80` is one example). On such a layer,
`update_and_sdpa_iso_k_fused` returns before any mutation, the ring is not
allocated, and the layer keeps the blocks for the whole request.

The estimate is model-agnostic. It uses only layer geometry (`head_dim`,
`kv_heads`, `window`), the layer topology (`shares_kv`) and codec attributes:
`KvQuant::side_stores`, `KvQuant::approx_code_bits`, the two mirror predicates
and `KvQuant::materialises_packed_store`. The code is:

- `rmlx_kv_quant::KvQuant::estimated_resident_bytes_per_layer` /
  `estimated_net_saving_per_layer` (codec layer; the per-side byte model;
  windowed layers return saving 0).
- `rmlx_models::kv_cache::kv_codec_net_saving_total` /
  `warn_if_kv_codec_net_negative` (policy layer; sums the layer mix and emits
  the warn). The Gemma4, Qwen3 and Qwen3.5-MoE `generate` paths call it.

The warn is **advisory only**. It does not change the codec.

### Gemma4 global KV is bf16 at `--kv-quant none`

The Gemma4 residual stream is bf16 end-to-end, so the global `--kv-quant none`
K and V store as bf16. Three sites keep it bf16:

- the embed-scale (`hidden_size**0.5`) constant,
- the per-layer-input scales (embed / proj / inv-sqrt2),
- the fused GeGLU / PLI-GeGLU activations.

The scale constants adopt the operand dtype. The fused activation closures
restore the gate dtype on their output. Three unit tests in
`gemma4/layers/kernels_tests.rs` pin these sites at bf16:
`geglu_fused_bf16_gate_stays_bf16`, `pli_gelu_fused_bf16_gate_stays_bf16` and
`dtype_adopted_scale_keeps_bf16_operand_bf16`.

### Qwen3 dense KV is bf16 at `--kv-quant none`

The dense Qwen3 arch (`Qwen3ForCausalLM`) casts every float model parameter to
bf16 at load (`load_util::bf16_param`). This includes norm weights, quant
scales and biases, and embedding scales and biases. Some snapshots ship these
at fp16 (for example Bonsai-8B-2bit). Without the cast, MLX promotes a bf16
activation mixed with an fp16 parameter to f32, and K and V store at 4 B/elem.
The YARN mscale scalar is also stored as bf16 at load. Three unit tests in
`qwen3_tests.rs` pin the cast: `rms_norm_bf16_weight_keeps_output_bf16`,
`bf16_param_casts_fp16_to_bf16` and `yarn_mscale_dtype_adopted_keeps_bf16`.

**The chosen dtype is bf16. The reference does not do this.** mlx-lm applies
the same one-dtype rule but takes the dtype from the checkpoint. On
`prism-ml__Ternary-Bonsai-8B-mlx-2bit`, mlx-lm loads the float params as
float16 and keeps float16 logits and KV. bf16 has 3 fewer mantissa bits than
fp16, so rMLX decodes this checkpoint coarser than the weights on disk and the
reference. This can flip tokens at near-tie logits.

### Qwen3.6 MoE KV is bf16 at `--kv-quant none`

The Qwen3.5-MoE arch (`Qwen3_5MoeForConditionalGeneration`) uses the same
load-time cast. The `qwen3_5_moe` loader calls `load_util::bf16_param` on every
float param: FullAttention (q/k-norm weights, quant scales and biases,
embedding scales and biases) and the GDN recurrent layers (`conv1d_weight`,
`norm_weight`). Thus an fp16 repack also stays bf16 in compute. Two CPU tests
pin this: `moe_stream_stays_bf16_with_bf16_params` and
`bf16_param_casts_fp16_to_bf16` (both in `qwen3_5_moe/moe_tests.rs`).

### KV byte accounting

`KvCache::resident_bytes()` reports the KV-cache size. It reads the real
`Array` shape × `dtype.itemsize()` of every GPU buffer and the length of every
CPU codec block. This covers packed codes, scales, zero-points, rotation and
residual buffers, the GPU rings of the ring-backed K codecs, and the bf16
mirrors. It backs the `kv_cache_bytes` observations, the `kv_bytes` event,
prompt-cache eviction and `rmlx baseline`. **Cost is O(blocks).** Call it at
request boundaries, not per layer per decode step.

Each figure comes from the store that owns the buffers
(`KvStorage::resident_bytes` → per-codec `byte_size`). There is no second
bits-per-element formula. A nominal bit width is not the memory of a cache.

**One sample point on every arch: post-decode.** rMLX records
`kv_cache_bytes` after the decode loop, when every resident KV allocation
exists, including the decode-time GPU ring. A run that returns before the
decode loop (the first sampled token is EOS) does not refresh
`kv_cache_bytes`. Such a run allocates no ring, so the value it does not write
equals the prefill snapshot. A NaN prefill stops the request with an error.

`KvBytesCounter::store` requires a `PostDecode` witness. Only a completed
decode loop mints one: `pipelined_decode`, the per-arch decode loops and the
speculative round loop. If a change moves the store back to the prefill point
and reuses the loop's witness, the build fails, because that witness is not in
scope there. This is not an unforgeable guarantee: `PostDecode::seal()` is
`pub(crate)`, so a new arch can mint a witness at the prefill point. Review
and the `#[ignore]`d GPU test `kv_bytes_hit_equals_miss` are the backstop.
`make ci` does not run that test. The prompt-cache snapshot is
still cloned at the prefill point, because it stores the prompt's KV.

### Per-request hot-swap

The `KvQuant` of a request is not tied to the model load. A running
`rmlx serve` accepts a per-request `kv_quant` field (OpenAI route). The field
selects the codec for that request. The weights stay resident; only the KV
cache is rebuilt. If the field is absent, the launch `--kv-quant` applies.

The prompt and prefix cache is **partitioned by codec**, so a switch cannot
serve mismatched cached K/V. `KvQuant::cache_key_salt()` is XOR'd into the
block-hash seed with the SSD `layout_key`. See `docs/PROMPT_CACHE.md`
§ "Codec namespacing" and `docs/SERVER.md` § "Per-request KV-config hot-swap".

---

## Storage variants — summary table

The last column is the V-side cosine gate on the unit-test fixture.

| `KvStorage` variant | K codec | K group | V codec | V group | Dispatch path | Cosine gate (V mean ≥) |
|---|---|---|---|---|---|---|
| `None` | bf16 (no quant) | — | bf16 (no quant) | — | `decode_fp16_k/v` buffers | — |
| `K8V8` | rMLX MSL q8_0 | 128 | rMLX MSL q8_0 | 128 | `QuantK` + `QuantK` | 0.9990 |
| `K8V4` | rMLX MSL q8_0 | 128 | TurboQuant 4-bit | 32 | `QuantK` + `QuantV` | 0.9937 |
| `Planar` (bits=4) | rMLX MSL q8_0 | 128 | PlanarQuant 4-bit | 32 | `QuantK` + `QuantPlanarV` | 0.9942 |
| `Planar` (bits=3) | rMLX MSL q8_0 | 128 | PlanarQuant 3-bit | 32 | `QuantK` + `QuantPlanarV` | 0.9989 |
| `Mixed` | MLX affine `k_bits` | `k_group` | MLX affine `v_bits` | `v_group` | `MixedKvState` | 0.9937 (V4); 0.9990 (V8); 0.9000 (V2) |
| `Paged` | q8_0 per page | 128 | tq4 / q8_0 / planar per page | 32/128/32 | `PagedKStorage` + paged V | — |
| `TurboSym3` | TurboQuant 3-bit | 32 | TurboQuant 3-bit | 32 | `QuantKTurbo3` + `QuantV{bits:3}` | 0.9807 (K empirical floor) |

---

## Metal-vs-CPU hot path + load-time MSL precompile

Two codec attributes control startup behaviour. Both are exhaustive matches on
`KvQuant` (`crates/rmlx-kv-quant/src/quant.rs`). A new variant must be
classified, or the build fails.

* **`KvQuant::carries_msl()`** is `true` for every codec except `none`. For
  `Mixed` / `RotK` the kernel is MLX's own `mx.quantize`, a compiled Metal op,
  not a custom kernel. Every other codec can dispatch at least one custom Metal
  (MSL) kernel. MSL
  kernels compile **lazily**: `MetalKernel::new` only registers, and MLX
  compiles the pipeline on the first `apply()` dispatch (see `docs/FFI.md`
  § `MetalKernel`).

* **`KvQuant::cpu_hot_path_reason()`** is `Some(reason)` when the codec's
  encode and dequant run on the **CPU** on the default path:

  * `iso3` / `iso4`, `rotor3` / `rotor4`, `rotor_k_*_asym_*` → `Some`. Their
    `update_*` functions return early to the bf16 mirror at decode, so the GPU
    branch is shadowed. The encode that exists is CPU.
  * `iso3_sym` / `iso4_sym`, `k_iso3` / `k_iso4` → `None`. Decode is the iso
    flash-decode kernel over the packed ring (`iso_flash_decode`,
    `iso_flash_decode_symv`). Nothing restages through the host.
  * `rotor3_sym` / `rotor4_sym`, `k_rotor3` / `k_rotor4` → depends on QJL.
    The default is off (`--rotor-qjl off`). With QJL off, the K encode is the
    rotor MSL kernel (`rotor_gpu_append_into_k_blocks`) and decode is fused
    (see § `rotor_flash_decode` below), so the verdict is `None`. With QJL on,
    the 1-bit residual has no MSL kernel, so the verdict is `Some`. It forces
    the K append of `k_rotor*` onto the CPU every decode step, and it forces
    both K and V of `rotor*_sym` onto the CPU encode and dequant path. `update_rotor_k_only` reads
    the store's sticky `use_qjl()` flag, which is fixed at the first append.
  * Every other codec → `None`.

For the bf16-mirror family, a seeded cache never reaches the path that this
verdict describes: the codec is INERT (see § "Codec disposition — what every
codec in the tree is for").

### Per-codec verdict

| Codec family | `cpu_hot_path_reason()` | Notes |
|---|---|---|
| `none` | `None` | bf16, no kernel |
| `k8v4` / `k8v8` / `planar` / `planar3` / `planar_k` | `None` | q8_0 K + tq4 / planar V GPU kernels; INERT on a seeded cache |
| `mixed_*` / `rot_k_v*` | `None` | MLX-affine `mx.quantize` K and V (compiled Metal ops) |
| `k8vturbo3` / `k8vturbo2` / `*tcq` / `tsym3` / `tsym4` | `None` | q8_0 or turbo K on GPU; 2-bit and 3-bit turbo V is CPU-forced; INERT on a seeded cache |
| `iso3` / `iso4` | `Some` | bf16 mirror shadows the GPU iso branch; INERT on a seeded cache |
| `iso3_sym` / `iso4_sym` | `None` | `iso_flash_decode_symv` over both packed rings; no bf16 mirror |
| `k_iso3` / `k_iso4` | `None` | iso K MSL encode into the packed ring + `iso_flash_decode` |
| `rotor3` / `rotor4` / `rotor_k_*_asym_*` | `Some` | bf16 mirror shadows the GPU branch; INERT on a seeded cache |
| `rotor3_sym` / `rotor4_sym` / `k_rotor3` / `k_rotor4` | `None` with QJL off (default); `Some` with `--rotor-qjl on` | QJL off: rotor K MSL encode + `rotor_flash_decode` |

### Load-time precompile

`rmlx_kv_quant::precompile::precompile_kv_codec_msl(kq, head_dim, kv_heads,
device)` warms the kernels of a codec with one small GPU dispatch during model
load. Thus the first user request does not pay a cold compile. It is keyed off
codec attributes, never an arch name. It does nothing in these cases:

- the device is not the GPU,
- `head_dim` is unknown (`0`),
- `carries_msl()` is `false` (`none`),
- `cpu_hot_path_reason()` is `Some`,
- `is_k_only_iso_rotor()` is `true`. The K kernel of these codecs is the
  iso/rotor MSL kernel, not the q8_0 K kernel that this function warms. It
  compiles lazily on the first prefill.
- no small warm shape makes `kv_heads × head_dim × tokens` a multiple of the
  q8_0 group (128).

Otherwise it warms the q8_0 K kernels, plus the V kernel for `k8v4` (tq4),
`planar` (planar 4-bit) and `planar3` (planar 3-bit). A warm failure logs a
`warn!` and load continues; the kernel then compiles lazily on first use.
`ArchGenerator::from_snapshot_with_id`, the server-side generator factory for
every arch, calls it.

### CPU-codec classification at resolve time

`rmlx_models::kv_cache::validate_resolved` (alias `validate_resolved_kv_quant`)
runs the arch-agnostic Metal-vs-CPU check after the Qwen-MoE guards. When
`cpu_hot_path_reason()` is `Some`, it emits a structured `warn!` that names the
codec and the reason. It also warns when the codec is INERT
(`materialises_packed_store()` is `false`). Both warns go through
`warn_once_per_codec`, which uses one key, `(arch, codec)`, for both. Thus at
most one of the two warns fires for a pair. For `iso3`, `iso4`, `rotor3`,
`rotor4` and `rotor_k_*_asym_*`, both conditions are true: the CPU warn fires
and the INERT warn does not. The codec is not rejected.

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
(§"Gemma4 global KV is bf16 at `--kv-quant none`",
§"Qwen3 dense KV is bf16 at `--kv-quant none`") are the fix; the floor is
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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
`mixed_k8g64_v4g64`). `--ctk rot_k --ctv <affine-tag>` selects `RotK` (see the
CLI flags section below).

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
"Codec fidelity — measured" below.

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

**K codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook (8 centroids),
`group_size=32`. On GPU: `turbo_quantize_v3_gpu` / `turbo_dequantize_v3_gpu`
from `k8vturbo3_append_msl.rs`. On CPU: `turbo_quantize_v(bits=3)`.

**V codec**: the same codebook, `group_size=32`, through `QuantV { bits: 3 }`
as in `K8VTurbo3`. The V side runs on the CPU.

**No rotation is applied on either axis**, although the family has the name
TurboQuant. The layout tag names the Lloyd-Max codebook. See §"The
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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
> bf16. See § "Codec disposition — what every codec in the tree is for".

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
thread on which the prefill forward built its graph. MLX ≥0.31 streams are
thread-local: an `Array::eval()` on another thread throws
`There is no Stream(cpu, N) in current thread.` The generate entry points call
`rmlx_mlx::ensure_cpu_default_stream()` to register the worker's own streams.
See `docs/KV_CACHE.md` §5.7.5 for the mechanism, the guard, and its
limitation.

**Warm-TTFT decode contract.** `exit_prefill` also seeds a bf16 K+V decode
mirror (`decode_fp16_k` / `decode_fp16_v`) for each axis whose decode reads it.
Every `update_<codec>` of the bf16-mirror family returns early to
`update_decode_fp16` while that mirror is live. Thus decode-phase K **and** V
are bf16 for those codecs. The K-only family (`IsoKOnly*`, `RotorKOnly*`) keeps
K quantized at decode and mirrors only V. The fused symmetric family
(`Iso*Sym`, `Rotor*Sym`) mirrors neither axis. `docs/KV_CACHE.md` §9.6 has the
per-codec table.

---

## Layer-adaptive overrides

`kv_layer_quants` makes the per-layer codec vector of a model. It calls
`kv_quant_for_layer` once for each layer. The cache stack, the SSD layout key
and the prompt-cache seed all use this one vector.

A boundary layer gets the boundary floor in place of the requested codec, when
the requested codec quantizes. There are two sets of boundary layers:

- **Head layers**: the first `LAYER_ADAPTIVE_HEAD_N = 2` layers.
- **Tail layers**: the last `LAYER_ADAPTIVE_TAIL_N = 8` layers.

The rule uses only the layer index. It does not use the context length, the
architecture name or the codec name. A codec that quantizes neither side gets
no floor (see § "`--kv-quant none` is a bf16 control").

No measurement derives the default counts for this floor. They are a setting,
not a measured optimum.

**Set the counts.** `--kv-boundary-layers <head>,<tail>` (default `2,8`) sets
the counts on `rmlx serve`, `baseline`, `bench` and `eval ppl`. `0,0` turns the
promotion off. `eval ppl` refuses the flag when no KV codec is set, because its
default scorer has no per-layer cache. A run at a value that is not the default
records `decode_config = 'kv_boundary/head=<h>,kv_boundary/tail=<t>'`, so it
ranks as its own cell.

**Judge a codec by perplexity, not by a token digest.** A greedy token digest
shows whether a codec changes the output. It does not rank two codecs that both
change it. Use perplexity for that.

### Which codec the floor is

`boundary_floor` selects the target:

* `Mixed` and `RotK`, **on a stack that does not share K/V across layers**: the
  same codec with both axes raised to 8 bits. The store, the group sizes and
  the K rotation do not change (`mixed_k8g64_v4g64` → `mixed_k8g64_v8g64`,
  `rot_k_v4g64` → `rot_k_v8g64`).
* Every other quantizing codec: `K8V8`. These codecs have no 8-bit form in
  their own family. This includes `rotor_k_*_asym_*`: its V width is a
  parameter, but its K is a 3-bit or 4-bit rotor.

**The cross-layer-KV stack takes the fallback.** On a stack whose layers read
the K/V of other layers (Gemma4 sets `SHARES_KV_ACROSS_LAYERS = true`), `Mixed`
and `RotK` keep both bf16 mirrors, because the consumer layers read them. An
in-family target there holds the packed store **and** both mirrors: 8.50 +
16.00 = **24.50** bits per value at group 64, against 16.00 for `K8V8`. It also
changes the decode of the layer from bf16 to 8-bit affine. Thus
`boundary_floor` reads the `feeds_bf16_{k,v}_at_decode` predicates of the
target. When the target reads a bf16 mirror on both axes, `boundary_floor`
returns `K8V8`.

**A `K8V8` boundary layer is a bf16 layer.** `K8V8` builds no packed store. Its
decode reads the bf16 mirror on both axes. Thus a layer that falls back to
`K8V8` holds two bf16 buffers: 16.00 bits per value, the same bytes as `none`.
Ten codecs read their own packed store. Eight of them always take this
fallback: `iso3_sym`, `iso4_sym`, `k_iso3`, `k_iso4`, `rotor3_sym`,
`rotor4_sym`, `k_rotor3` and `k_rotor4`. On a shared-KV stack, `Mixed` and
`RotK` also take it, so all ten do. On those codecs, each promoted layer costs
the difference between the bf16 rate and the codec rate. An SO(4)-rotated or rotor
3-bit or 4-bit ring has no 8-bit form, so the only other choice is no floor on
those layers.

`store_bearing_boundary_promotion_never_costs_more` sweeps both topologies. It
makes sure that a boundary layer never costs more than the `K8V8` layer it
replaces. It names the eight fallback codecs, so a new store-bearing codec that
falls back makes it fail.

**The model sets how many layers the floor reaches.** Two conditions cancel the
promotion on a layer:

- A windowed (sliding-attention) layer runs the bf16 rotating ring for every
  codec (`KvCache::with_quant_max_seq_window`).
- A shared-KV consumer layer owns no cache. Gemma4 `num_kv_shared_layers`
  points each layer from `n_layers - num_kv_shared_layers` onward at the cache
  of an earlier layer (`gemma4/loader.rs::build_previous_kvs`).

On `gemma-4-e2b` and `gemma-4-e4b` (`num_kv_shared_layers` 20 and 18), every
tail layer is a consumer, and the two head layers are windowed. Thus the
promotion reaches no layer, and a boundary measurement on these models always
reads zero. `gemma-4-12B`, `gemma-4-26b-a4b` and `gemma-4-31b` have
`num_kv_shared_layers = 0`. On each of them, the promotion reaches the two
`full_attention` layers in the tail window (12B: 41, 47; 26b: 23, 29; 31b: 53,
59). The head layers are windowed. Under `Mixed` / `RotK`, these two layers take
the `K8V8` fallback. To measure the floor, use a dense model (for example
`Ternary-Bonsai-8B`, 10 of 36 layers promoted) together with one of these
Gemma4 models.

### Where the bytes of a `Mixed` layer stack go

`kv_cache_bytes` counts two kinds of layer in two different ways:

- A layer that holds the bf16 mirrors (a `K8V8` fallback layer):
  `KvCache::resident_bytes` counts the filled prefix.
- A layer that holds a `Mixed` store: `MixedKvState::byte_size` counts the full
  allocation. The store has the prompt length at `exit_prefill`. It then grows
  in blocks of `STEP = 256` rows.

For prompt `P` and generation `G`:

```
bf16 mirror layer:  bf16_rate  * (P + G - 1)
Mixed store layer:  layer_rate * C,   C = P + 256 * ceil(G / 256)
```

`bf16_rate = 2 * kv_h * head_dim * 2` B per token. One `Mixed` side is
`kv_h * head_dim * (bits/8 + 4/group)` B per token. An in-family boundary layer
is a `Mixed` store layer at 8 bits.

Thus a `kv_cache_bytes` value is not comparable across codecs at a short
generation: a store counted at capacity includes up to 255 unfilled rows.
Compare bits per value at a fixed `(P, G)`, not one raw byte count against
another.

### `--kv-quant none` is a bf16 control

`KvQuant::None` gets no boundary promotion. `kv_quant_for_layer` skips the
promotion when the codec keeps both sides at model dtype: `approx_code_bits`
reports 16 for such a side (`base_is_unquantized`). The decision uses this codec
property, not a codec name or an architecture. Thus no layer of a `none` run
holds a packed store, and its resident KV is the bf16 figure.

The K-only codecs (`planar_k`, `k_iso3`, `k_iso4`, `k_rotor3`, `k_rotor4`) keep
V at bf16, but they quantize K to 3 or 4 bits. They get the promotion.

---

## CLI flags

### Preset interface

`--kv-quant <spelling>` selects the codec by name. `KvQuant::from_str` parses
every spelling below except `auto` and `mixed`, which the CLI parser adds.

| Spelling | `KvQuant` |
|---|---|
| `auto` | `DEFAULT_KV_QUANT` (see § "The auto default") |
| `none` / `bf16` / `f16` | `None` |
| `k8v8` | `K8V8` |
| `k8v4` | `K8V4` |
| `planar` / `planar3` | `Planar` / `Planar3` |
| `planar_k` | `PlanarK` |
| `k8vturbo3` / `k8vturbo2` | `K8VTurbo3` / `K8VTurbo2` |
| `k8vturbo3tcq` / `k8vturbo2tcq` | `K8VTurbo3Tcq` / `K8VTurbo2Tcq` |
| `tsym3` / `tsym4` | `TurboSym3` / `TurboSym4` |
| `iso3` / `iso4` | `Iso3` / `Iso4` |
| `iso3_sym` / `iso4_sym` | `Iso3Sym` / `Iso4Sym` |
| `k_iso3` / `k_iso4` | `IsoKOnly3` / `IsoKOnly4` |
| `rotor3` (`rotor_v_3`) / `rotor4` (`rotor_v_4`) | `Rotor3` / `Rotor4` |
| `rotor3_sym` / `rotor4_sym` | `Rotor3Sym` / `Rotor4Sym` |
| `k_rotor3` / `k_rotor4` | `RotorKOnly3` / `RotorKOnly4` |
| `rotor_k_3_asym_v<vb>_g<vg>` / `rotor_k_4_asym_v<vb>_g<vg>` | `RotorK3Asym` / `RotorK4Asym` |
| `rot_k_v<vb>g<vg>` | `RotK` |
| `mixed_k<kb>g<kg>_v<vb>g<vg>` | `Mixed` |
| `mixed` | `Mixed`, the same as `mixed_k8g64_v4g64` |

Limits:

- A `mixed_*` side and the `rot_k_*` V side accept 2, 3, 4, 5, 6 or 8 bits and
  a group of 32, 64 or 128 (`validate_mixed_side`).
- The `rotor_k_*_asym_*` V side accepts `v4_g128`, `v4_g64`, `v4_g32`, `v3_g64`
  and `v2_g64` (`validate_rotor_k_asym_v`). The V codec is TurboQuant, which
  always uses 32-element groups. `v_group_size` only goes into the SSD layout
  tag.
- `rot_k_tq4v` is rejected by name. The error names `rot_k_v4g64` (see
  § "`rot_k_tq4v` is rejected").
- On Qwen MoE (`Qwen3_5MoeForConditionalGeneration`,
  `Qwen3VLMoeForConditionalGeneration`), every codec with K below 8 bits is
  rejected at resolve time, and the process exits with code 78.

### Named preset interface — `--kv-preset`

`--kv-preset <name>` selects a codec by a short name. A named preset resolves at
parse time. `auto` resolves to `DEFAULT_KV_QUANT`. The flag is on `serve`,
`chat`, `info`, `baseline`, `bench` and `eval ppl`.

**Conflict rule**: `--kv-preset` cannot go with `--kv-quant`,
`--cache-type-k`, `--cache-type-v` or `--kv-bits`. clap refuses the combination
before the subcommand body runs.

#### Preset table

| Name | `KvQuant` | Notes |
|---|---|---|
| `fp16` | `KvQuant::None` | bf16 on both sides (the `KvQuant` variant named `None`, not `Option::None`) |
| `q8` | `KvQuant::K8V8` | symmetric 8-bit K+V |
| `speed` | `KvQuant::TurboSym3` | symmetric 3-bit K+V, no rotation; rejected on Qwen MoE |
| `quality` | `KvQuant::TurboSym4` | symmetric 4-bit K + tq4 V, no rotation; rejected on Qwen MoE |
| `planar` | `KvQuant::Planar` | PlanarQuant 4-bit V |
| `planar3` | `KvQuant::Planar3` | PlanarQuant 3-bit V |
| `k_only_planar` | `KvQuant::PlanarK` | PlanarQuant 4-bit K, V bf16; rejected on Qwen MoE |

**None of the six non-`fp16` rows changes resident KV or output.** Each resolves
to a codec in the inert class (§"Codec disposition"): decode reads the bf16
mirror, so `exit_prefill` never builds the packed store, and the served request
holds the same bytes and emits the same token ids as `fp16`. A preset is a
codec name, not a memory setting. `no_preset_is_a_memory_lever` pins that claim
and fails the moment a preset's codec starts reading its own store.

#### Preset semantics — divergence from mtq

`speed` is `TurboSym3`: 3-bit K and 3-bit V, both with the 8-centroid Lloyd-Max
N(0,1) codebook. `quality` is `TurboSym4`: 4-bit K and a `tq4` V. These are the
K/V widths of the mtq (multi-turboquant) presets with the same names.

The divergence: the rMLX turbo encoder applies no rotation on either axis.
Where the upstream name implies one, §"The turbo family's missing
rotation — what it is worth, and where" states what a rotation would be worth.

Examples:

```
rmlx serve --model <path> --kv-preset fp16
rmlx serve --model <path> --kv-preset q8
rmlx baseline --model <path> --kv-preset speed
rmlx info --model <path> --kv-preset planar
rmlx baseline --model <path> --kv-preset auto    # == --kv-quant auto
```

### `--kv-preset auto`

`--kv-preset auto` resolves to `rmlx_models::kv_cache::DEFAULT_KV_QUANT`. This
is the same constant that `--kv-quant auto` resolves to. It does not read the
preset table or the hardware. `preset_auto_is_the_same_default_as_kv_quant_auto`
makes sure that the two stay the same.

### Per-side primitive interface

`--cache-type-k <tag>` (`--ctk`) sets the K codec. `--cache-type-v <tag>`
(`--ctv`) sets the V codec. They cannot go with `--kv-quant`, `--kv-preset` or
`--kv-bits`.

| Tag (aliases) | Side | Codec |
|---|---|---|
| `auto` | K, V | see the `auto` rule below |
| `bf16` (`f16`, `none`) | K, V | unquantized bf16 |
| `q8_g128`, `q8_g64`, `q8_g32` | K, V | 8-bit, group 128 / 64 / 32 |
| `q6_g64`, `q5_g64`, `q4_g128`, `q4_g64`, `q4_g32`, `q3_g64` | K, V | MLX affine at that width and group |
| `q2_g64` | V | MLX affine 2-bit, group 64 |
| `rot_k` | K | Hadamard-rotated MLX affine 8-bit, group 64 |
| `planar_k4` | K | PlanarQuant 4-bit K |
| `iso_k_3` (`k_iso3`), `iso_k_4` (`k_iso4`) | K | IsoQuant K |
| `rotor_k_3` (`k_rotor3`), `rotor_k_4` (`k_rotor4`) | K | rotor K |
| `tsym3` | K and V | symmetric TurboQuant 3-bit; both sides must name it |
| `tq4` (`turbo4`) | V | TurboQuant 4-bit |
| `planar4`, `planar3` (`planar_3`) | V | PlanarQuant 4-bit / 3-bit |
| `iso_v_3` (`iso3`), `iso_v_4` (`iso4`) | V | IsoQuant V |
| `rotor_v_3` (`rotor3`), `rotor_v_4` (`rotor4`) | V | rotor V |
| `k8v_turbo_3_tcq` (`turbo3_tcq`), `k8v_turbo_2_tcq` (`turbo2_tcq`) | V | TurboQuant TCQ 3-bit / 2-bit |

`combo_to_kv_quant` maps the pair to one codec:

| K | V | Codec |
|---|---|---|
| `bf16` | `bf16` | `none` |
| `q8_g128` | `q8_g128` | `k8v8` |
| `q8_g128` | `tq4` | `k8v4` |
| `q8_g128` | `planar4` / `planar3` | `planar` / `planar3` |
| `q8_g128` | `iso_v_3` / `iso_v_4` | `iso3` / `iso4` |
| `q8_g128` | `rotor_v_3` / `rotor_v_4` | `rotor3` / `rotor4` |
| `q8_g128` | `turbo3_tcq` / `turbo2_tcq` | `k8vturbo3tcq` / `k8vturbo2tcq` |
| other affine (not `q2_g64`) | affine | `mixed_k<kb>g<kg>_v<vb>g<vg>` |
| `rot_k` | affine | `rot_k_v<vb>g<vg>` |
| `planar_k4` | `bf16` | `planar_k` |
| `iso_k_N` | `iso_v_N` / `bf16` | `isoN_sym` / `k_isoN` |
| `rotor_k_N` | `rotor_v_N` / `bf16` | `rotorN_sym` / `k_rotorN` |
| `rotor_k_N` | `q4_g128`, `q4_g64`, `q4_g32`, `q3_g64`, `q2_g64` | `rotor_k_N_asym_v<vb>_g<vg>` |
| `tsym3` | `tsym3` | `tsym3` |

The `auto` rule: when both sides are `auto`, the result is `DEFAULT_KV_QUANT`.
When one side is `auto` and the other side is quantized, the `auto` side
becomes `q8_g128`.

The resolver rejects:

- any spec when the model config gives no `head_dim` (`HeadDimUnknown`);
- the llama.cpp tags `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1` and `iq4_nl`, by
  name, with the nearest rMLX tag in the error;
- a V-only codec on K, a K-only codec on V, and `rot_k` on V;
- `q2_g64` on K (2-bit K makes attention incoherent);
- `bf16` on one side and a quantized codec on the other, except the K-only
  pairs above;
- a V rotation codec (`tq4`, `planar*`, `iso_v_*`, `rotor_v_*`, TCQ) with a K
  that is not `q8_g128`. There are two exceptions: `iso_v_N` with `iso_k_N`
  gives `isoN_sym`, and `rotor_v_N` with `rotor_k_N` gives `rotorN_sym`;
- `tq4` when `head_dim` is not 128 or 256, and `rot_k` when `head_dim` is not
  a power of two;
- an affine tag whose group does not divide `head_dim`, or whose `bits` pack
  does not divide it (`head_dim % (32 / bits) != 0`).

Notes:

- SWA layers use the bf16 rotating ring for every `--ctk` / `--ctv` value.
  This is the mlx-lm behavior.
- `rmlx serve --paged-kv` refuses a `--ctk` value that starts with `rot_k`.

### Canonical combo examples

```
rmlx serve --model <path> --kv-quant k8v8
rmlx serve --model <path> --kv-quant k8v4
rmlx serve --model <path> --kv-quant planar
rmlx serve --model <path> --ctk q8_g128 --ctv tq4      # equivalent to k8v4
rmlx serve --model <path> --ctk rot_k   --ctv q4_g64   # RotK affine V
rmlx serve --model <path> --kv-quant mixed_k8g64_v4g64
rmlx serve --model <path> --paged-kv --kv-quant k8v4
```

---

### iso3 codec

> **INERT on this build** — `iso3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See § "Codec disposition — what every codec in the tree is for".

**Algorithm — quaternion SO(4) isoclinic rotation.** iso3 applies a
left-isoclinic SO(4) rotation to each group of 4 elements, then quantizes to 3
bits:

```
T(v) = q_L * v     (one quaternion per group)
```

`*` is the **Hamilton product**, and `v ∈ ℝ⁴` is a quaternion
`v = (v₀, v₁, v₂, v₃)`. The inverse is `T⁻¹(r) = q̄_L * r`.

**Encode (per token):**

1. Compute the L2 norm of the vector and store it. Divide the vector by it.
2. Split the vector into `head_dim / 4` quaternion groups.
3. Apply `r = q_L * v` with the scalar Hamilton product.
4. Compute a per-group scale: `max(|r_i|) / max_centroid`.
5. Find the nearest centroid of the 3-bit Lloyd-Max codebook.
6. Write the codes into the dense code plane of the row, 3 bits each
   (`crate::code_plane`).

**Decode:** unpack → centroid lookup → rescale → inverse rotation → multiply by
the norm.

**Iso memory truth.** iso spends, per 4-element group, four codes in the row's
dense code plane (`bits` each — see `crate::code_plane`) **and** one scale at
the ring's sideband dtype (`bf16`), plus one `bf16` norm per token. At
head_dim=128 that is 114 B per token per kv_head for iso3 against bf16's 256 B
(**7.125 bits per value, 0.445× bf16**) and 130 B for iso4 (**8.125**, 0.508×).

The rate changes with `head_dim`. The allocation is
`ceil(D·bits/32)·4 B` codes + `(D/4)·2 B` scales + `2 B` norm, thus:

```
iso stored bits/value = bits + 4 + 16/head_dim
```

For iso3 this is 7.125 at D=128, 7.0625 at D=256 and 7.03125 at D=512. It comes
from `QuantKGpuRing::alloc` (`Dtype::U32` codes, `KV_SIDEBAND_DTYPE` scales and
norms, one scale per group and one norm per token per KV head).
`ring_bytes_match_independent_geometry` and `kv_rate_tests` measure it against
the allocation.

**The scale is the largest term after the codes:** 4.000 bits per value,
against 3.000 bits of codes at iso3. The CPU encoders round each scale and norm
to the stored width before they select the codes. Thus the encoder and the
store agree, and a ring seeded from CPU blocks decodes the same bits as those
blocks.

The CPU `IsoBlocks` form adds a 4×f32 quaternion per group and holds its scale
and norm planes at `f32`. This makes ≈692 B per token (≈43.25 bits per value,
2.7× bf16). The quaternion is the constant `FIXED_QUAT` repeated per group, not
data. The GPU ring that the K-only and symmetric codecs decode from does not
hold it. Three cases:

- The V-only `iso3` / `iso4` codecs decode from the bf16 mirror. `exit_prefill`
  builds no store for them, and they measure the same bytes as `none`
  (§"Codec disposition", Class 2).
- The store-reading members, `k_iso3/4` and `iso3_sym/4_sym`, hold the block
  form for one window: `exit_prefill` encodes them on the CPU, into blocks. The
  first fused decode step seeds the ring from those blocks and frees them
  (`drop_blocks_when_ring_live_iso_*` in `kvcache/update_iso.rs`).
- After that step, the ring is the only resident copy.

`estimated_resident_bytes_per_layer` sizes the four store-reading members from
the ring. The estimate is low in two cases. The first case is the window
between `exit_prefill` and the first fused decode step. The second case is the
full request on a layer whose shape the fused path rejects: batch > 1, or a
`head_dim` that is not a power of two at most 512. In the second case the ring
is never allocated and the blocks stay.

**Both rings are under bf16 at every head_dim, and iso is under it by more.**
The code plane costs `bits` per value for iso and `bits·⌈D/3⌉·3/D` for rotor:
3.000 and 3.250 at `bits = 3`, `D = 128`. The sideband makes the difference:
one scale per group at `KV_SIDEBAND_DTYPE` is 4.000 bits per value for a group
of 4 values, and 5.375 for a group of 3.
`every_store_family_is_at_or_below_the_bf16_floor_or_exempt` and
`exempt_families_actually_exceed_the_floor`
(`crates/rmlx-kv-quant/src/kv_rate_tests.rs`) pin both signs from real encoder
bytes. The operator sees the same sign in the resolve-time net-negative warn
(§"Per-layer net-benefit decision + net-negative warn") and in
`rmlx info --list-cache-types`.

**Crate-wide rate ceiling.** `crates/rmlx-kv-quant/src/kv_rate_tests.rs`
reports the bits per value of each store family. It fails a family above
bf16's 16.0 that has no written exemption. It also fails an exempt family that
does not measure above the floor. The `KvQuant` → family map is an exhaustive
`match`, so a new variant does not compile until it names its families. The
list of measured representatives is kept by hand: a variant with no
representative is not measured, and the gate stays green. Review must catch
this. Table at `head_dim = 128`:

| Family | Stored bits / value | Provenance | Verdict |
|---|---|---|---|
| turbo2 / tcq2 | 3.00 | measured | under |
| turbo3 / tcq3 | 4.00 | measured | under |
| turbo4 | 5.00 | measured | under |
| q8 (group 128) | 8.25 | measured | under |
| affine (`CacheType::Q8G32`) | 9.00 | layout formula, measured sideband | under |
| **iso3** (GPU ring) | **7.12** | measured | under |
| **iso4** (GPU ring) | **8.12** | measured | under |
| **rotor3** (GPU ring) | **8.75** | measured | under |
| **rotor4** (GPU ring) | **9.75** | measured | under |
| bf16 | 16.00 | by definition | the floor |
| **planar3 / planar4** | **22.00** | measured | exempt |

Planar is exempt because its scale is per *pair*. The `IsoBlocks` host form
measures 43.25; no served request stays in that form.

"Measured" means the byte total that the store reports for the buffers its own
encoder made from a shared fixture. For iso and rotor that is
`QuantKGpuRing::byte_size`, read from a real seeded ring. bf16 and MLX affine
have no CPU encoder in this crate. bf16 is two bytes per value. MLX affine is
`bits + 32/group`: the code bits plus one scale and one bias per group, each at
the dtype of the KV stream. **That sideband is 32 bits, measured**: the stream
at the store is bf16 (`cast_store_bf16` floors it at the store boundary), and
`mx.quantize(mode = "affine")` writes both scalars at the input dtype.
`affine_sideband_is_thirty_two_bits_per_group`
(`crates/rmlx-kv-quant/src/quant_tests.rs`) reads the figure from a real
`MixedTuple`.

**The affine row covers the whole `mixed_*` grammar.** The row uses the widest
cadence the grid accepts (`q8_g32`, 9.00). `parse_kv_side` output goes through
`validate_mixed_side` (2/3/4/5/6/8 bits, group 32/64/128), so this table has a
rate for every spelling that parses.
`mixed_grammar_no_longer_admits_unbounded_affine_rates` pins this.

**`head_dim % 4 == 0` constraint.** iso3 uses groups of 4. The encoder and the
decoder reject a `head_dim` that is not a multiple of 4 with
`IsoQuantError::HeadDimNotMultipleOf4`.

**Fixed quaternion.** The codec applies the golden-ratio unit quaternion
`q = (1, φ, φ−1, 1) / ‖(1, φ, φ−1, 1)‖` (`φ = (1+√5)/2`) to every group
(`FIXED_QUAT`). This is the quaternion of `multi_turboquant/methods/isoquant.py`.
It needs no calibration.

**Codebook — Gaussian Lloyd-Max, not Beta Lloyd-Max.** The Python references
(`rotorquant/turboquant/lloyd_max.py`) derive a Lloyd-Max codebook for the Beta
distribution that a random rotation of a unit vector gives. rMLX uses
`turboquant::lloyd_gaussian_codebook(3)` (Lloyd-Max for N(0,1)), the same
codebook family as TurboQuant and PlanarQuant. For `head_dim ≥ 64`, Beta(d) →
N(0, 1/d), and the per-group scale normalizes to N(0,1) before the centroid
lookup.

**GPU dispatch.** When `device == Device::Gpu`, `update_iso_v`,
`update_iso_sym` and `update_iso_k_only` send the encode and the dequant through
`isoquant_msl_dispatch`, which selects the kernel from the `BITS` of the store.
`QuantIsoV::dequant_gpu` / `QuantIsoK::dequant_gpu` put the CPU block payloads
(codes, scales, quaternions, per-token norms) into single byte buffers, upload
them once with `Array::from_bytes`, run the dequant kernel and reshape the
output to `[B, kv_h, S, D]`. On `Device::Cpu`, the CPU codec runs. The
GPU-resident `QuantIsoV` mirror is off in production:
`gpu_resident_iso_enabled()` returns `false`.

**CPU ↔ GPU parity.** `iso_v3_dequant_gpu_matches_dequant_cpu` and
`iso_k3_dequant_gpu_matches_dequant_cpu` in
`crates/rmlx-kv-quant/src/isoquant_msl_tests.rs` (`#[ignore]`-gated).
Observed `max|cpu-gpu| ≤ 2.4e-7` on the LCG fixture
(a few f32 ULPs from a different summation order between CPU `iso_decode_fast`
and the MSL kernel). The tests assert 5e-3 per element (codebook tolerance)
and a strict ≤ 1e-6 bound.

**Cosine gates.** `iso3_cosine_gate` in
`crates/rmlx-kv-quant/src/isoquant_tests.rs`. `quant_iso_v_roundtrip_dequant`
asserts that the `QuantIsoV3` round-trip matches `iso_decode_fast` within
`max_abs_err < 1e-3`.

**Sequence-major buffer layout (all Iso / Rotor block stores).**
`QuantIsoV<BITS>`, `QuantIsoK<BITS>`, `QuantRotorV<BITS>` and `QuantRotorK<BITS>`
add one `*Blocks` entry per `append` and concatenate them on `dequant`. The
caller reshapes the concatenation head-major `[B, kv_h, S, D]`. Thus each
`append` reorders the head-major chunk heads↔seq (`[B, new_seq, kv_h, D]`)
before it encodes. `dequant` reorders **each block back at its own sequence
offset** (`seq_layout::transpose_chunked_seq_heads`). One reorder over the
whole concatenation gives the same result only at `B == 1`.

The codec is positional per token row, so the sidebands stay with their rows:

- The Iso per-(token, group) scale and norm and the constant `FIXED_QUAT` move
  with the rows.
- The Rotor static rotor table and the QJL projection are keyed by group or
  projection. They do not move.
- The per-token QJL `qjl_codes` / `qjl_norms` move with the rows.

A `.kvb` SSD block stores its token rows in this sequence-major order. See
`docs/KV_CACHE.md` §5.7.3.

---

### iso4 codec

> **INERT on this build** — `iso4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See § "Codec disposition — what every codec in the tree is for".

**Algorithm — quaternion SO(4) isoclinic rotation, 4-bit codebook.**

iso4 is the 4-bit form of [iso3](#iso3-codec). The rotation, the group
geometry and the fixed quaternion are the same. The differences are the
codebook (16 centroids, not 8) and the code width (4 bits, not 3).

| Property | iso3 | iso4 |
|---|---|---|
| Code bits / element | 3 | 4 |
| Delivered bits / element, ring-resident (`k_iso*`, `*_sym`) | **7.125** (114 B/token at head\_dim=128, see "Iso memory truth" in the iso3 section) | **8.125** (130 B/token) |
| Delivered bits / element, CPU-blocks form | ≈43.25 (≈692 B/token at head\_dim=128, with the constant quaternion sideband). The ring-backed members hold it between `exit_prefill` and the first fused decode step. `iso3` / `iso4` build no store (§"Codec disposition", Class 2) | ≈44.25 |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids) | `lloyd_gaussian_codebook(4)` (16 centroids) |
| Pack density | dense code plane, 3 bits per code | dense code plane, 4 bits per code |
| Rotation | golden-ratio fixed quaternion (`FIXED_QUAT`) | same |
| Group size | 4 elements (one quaternion block) | same |
| `head_dim` constraint | `% 4 == 0` | same |
| MSL kernel | `iso_quantize_v3_gpu` / `iso_dequantize_v3_gpu` in `crates/rmlx-kv-quant/src/isoquant_msl.rs` | `iso_quantize_v4_gpu` / `iso_dequantize_v4_gpu` in `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` |
| SSD layout tag (V-only / symmetric) | `iso_v_3` / `iso_sym_3` | `iso_v_4_v2` / `iso_sym_4_v2` |

**Codebook** — the same as iso3: Gaussian Lloyd-Max `lloyd_gaussian_codebook(4)`,
not Beta Lloyd-Max.

**MSL kernel.** `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs` dispatches the
4-bit kernel pair `iso_quantize_v4_gpu` / `iso_dequantize_v4_gpu`: one thread
per (token, group), atomic-OR pack with `(idx & 0xF) << shift`, and a boundary
table of 15 mid-points from `lloyd_gaussian_codebook(4)`. The bodies are in
`src/metal/isoquant_quantize_iso4.metal` and
`src/metal/isoquant_dequantize_iso4.metal`. `make check-metal-compiles` compiles
them against the header snapshot `src/metal/probes/isoquant_iso4.hdr.metal`.
The three iso update entries reach this width through `isoquant_msl_dispatch`
when `device == Device::Gpu`. The CPU codec runs on `Device::Cpu`.

**CPU ↔ GPU parity.** Three tests in
`crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs`, all `#[ignore]`-gated (run
`cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1`).
`iso_v4_msl_matches_cpu_within_eps` asserts that the CPU codec
(`iso_encode_fast` + `iso_decode_fast`, `bits=4`) and the MSL kernels agree
within 5e-3 on a 32×128 LCG fixture. `iso_v4_dequant_gpu_matches_dequant_cpu`
and `iso_k4_dequant_gpu_matches_dequant_cpu` assert the same on the store
entries that a decode step calls, at the two bounds of the 3-bit pair: 5e-3 per
element and a strict `max|cpu-gpu| ≤ 1e-6`.

**Cosine gate.** `iso4_cosine_gate` in
`crates/rmlx-kv-quant/src/isoquant_tests.rs`. SSD round-trip: `roundtrip_iso4`
in `crates/rmlx-kv-ssd/src/block_io_tests.rs` asserts that all four V buffers
(codes_packed, scales, quaternions, norms) are bit-identical after hydrate.

**One store type, two widths.** The CPU encode and decode functions take
`bits ∈ {3, 4}` (the shared dense code plane, `crate::code_plane`), and so does
the storage struct: one `QuantIsoV<BITS>` and one `QuantIsoK<BITS>`, with
`QuantIsoV3` / `QuantIsoV4` / `QuantIsoK3` / `QuantIsoK4` as type aliases.
`IsoBlocks` is shared.

---

### rotor3 codec

> **INERT on this build** — `rotor3` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See § "Codec disposition — what every codec in the tree is for".

**Algorithm — Cl(3,0) Clifford rotor sandwich, 3-bit codebook.**

rotor3 is the 3-bit member of the Clifford rotation family of KV codecs. The
codec puts each `head_dim`-element V vector into Cl(3,0) (the 8-dimensional
multivector algebra of 3D Euclidean space) in groups of 3 grade-1 elements. It
applies a per-(layer, head) static rotor `R_g` as a sandwich, then quantizes to
3 bits with the Lloyd-Max N(0,1) codebook. The codec stores the static rotor
once per layer, for all tokens.

| Property | rotor3 |
|---|---|
| Delivered bits / element | **8.75** at head\_dim=128: 140 B/token/kv\_head against 256 B for bf16 (**0.547× bf16**). The split is **3.25 codes + 5.375 scales + 0.125 norm**. 43 groups cover a 128-element row. Each group stores its three grade-1 codes in the dense code plane of the row. Each group also stores one scale at `KV_SIDEBAND_DTYPE`. The allocation gives `rotor stored bits/value = (32·⌈⌈D/3⌉·3·bits/32⌉ + 16·⌈D/3⌉ + 16) / D`: 8.75 at D=128, 8.5625 at D=256. This comes from `QuantKGpuRing::alloc`. `rotor_rate_splits_into_documented_code_scale_and_norm_bits` measures it against a seeded ring. **The scale is the largest term.** Rotor shares one scale across 3 values, iso across 4. Thus rotor is the wider of the two families at every width and every head\_dim (`iso_and_rotor_k_codecs_are_under_the_floor_at_every_geometry`). |
| Code budget | **Only the 3 grade-1 codes per group are stored.** A rotor sandwich keeps the grade. Thus 3 values put in as the grade-1 part leave the scalar, the three bivector and the pseudoscalar slots at zero on encode. On decode, the inverse sandwich keeps every part that is not grade 1 out of the vector. `clifford_tests::sandwich_of_grade1_in_3d_stays_grade1` (encode side) and `clifford_tests::inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1` (decode side) pin this. |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids), one codebook for all 8 multivector components |
| Pack density | dense code plane, 3 bits per code, 3 codes per group |
| Rotation | static per-(layer, head) rotor table `[n_groups, 4]` in `[s, b12, b13, b23]` form, seeded from `ROTORQUANT_GLOBAL_SEED ^ (layer << 32) ^ (head << 16) + group` (`crate::clifford::rotor_seed`) |
| Group size | 3 elements (one Cl(3,0) grade-1 group) |
| `head_dim` constraint | none: the encoder pads a `head_dim % 3 != 0` tail with zeros and the decoder masks it off |
| MSL kernel | `rotorquant_msl.rs`: V-side encode, and K-side when QJL is off |
| SSD layout tag | `rotor_v_3` |

**One codebook.** The Python reference (`rotorquant/turboquant/rotorquant.py`)
uses a grade-aware codebook split (separate `vector` and `trivector` codebooks
at different bit budgets). rMLX uses **one 8-centroid codebook** for all 8
multivector components.

**Per-layer rotor tables.** `KvCache` has a `layer_idx: usize` field. Each arch
builder sets it with `KvCache::with_quant_max_seq(…).with_layer_idx(i)`. The
rotor3/rotor4 codec constructors (`QuantRotorV3::new`, `QuantRotorV4::new`,
`QuantRotorK3::new`, `QuantRotorK4::new`) get `self.layer_idx as u32` at each
`exit_prefill` and decode-time creation site. The `(layer << 32)` term in
`crate::clifford::rotor_seed` gives each layer a different rotor table.

**No QJL residual (V-only codec).** The Python reference has an optional 1-bit
QJL sign-quantization residual on the K side. The V-only rotor3 codec does not
apply it (see § "rotor K-side variants").

**Sign errors in the references.** The Python `clifford.py::geometric_product`
and the `rotor_fused.metal::gp_rotor_mv` kernel have sign errors in the grade-2
and grade-3 component formulas (for example, `e23 * e1 = +e123` in the Cl(3,0)
multiplication table, but the Python formula gives `-e123`). The Rust port
computes a table-driven dense geometric product at compile time from the
algebra rules. The algebra tests in `clifford_tests.rs` (known-answer 90°
rotation, unit rotor identity, sandwich of grade 1 stays grade 1) validate it.

**MSL kernel.** `crates/rmlx-kv-quant/src/rotorquant_msl.rs` has GPU encode and
decode kernels for rotor3 and rotor4. The kernel applies the Cl(3,0) sandwich
as a closed-form 3×3 SO(3) rotation matrix `M(R)` from `R * mv * R̃` (for
grade-1 input the grade-2 and grade-3 components cancel). The per-(layer, head,
group) rotor table is a buffer argument (`rotors_in : f32 [n_groups, 4]`).
`update_rotor_v`, `update_rotor_sym`, `update_rotor_k_only` and
`update_rotor_k_asym` (one entry per family for both code widths) dispatch it
when `device == Device::Gpu`. The CPU encoder runs on `Device::Cpu`.

**K-side QJL.** The K-side rotor codecs can carry a 1-bit QJL residual that
needs the JL projection matrix `S` at dequant. The GPU dequant kernels in
`rotorquant_msl.rs` do not implement QJL. When
`crate::rotor_qjl::rotor_qjl_enabled()` is `true` (`--rotor-qjl on`), the K-side
append and decode use the CPU `rotor3_k_encode` / `rotor3_k_decode` path. With
QJL off (**the default**), the GPU K-side kernel runs.

**CPU ↔ GPU parity tests.** `crates/rmlx-kv-quant/src/rotorquant_msl_tests.rs`
asserts max-abs-error ≤ 5e-3 between the CPU `rotor3_encode` / `rotor4_encode`
round-trip and the MSL round-trip (the same tolerance as iso3 / iso4). The tests
are `#[ignore]`-gated:
`cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1`.

**Cosine gate.** `rotor3_cosine_gate` in
`crates/rmlx-kv-quant/src/rotorquant_tests.rs`.

**SSD round-trip.** `roundtrip_rotor3` in
`crates/rmlx-kv-ssd/src/block_io_tests.rs` asserts that all four V buffers
(`codes_packed`, `scales`, `norms`, `rotors`) are bit-identical after hydrate.
The block stores the rotor table with the per-token payload, so a change to
`ROTORQUANT_GLOBAL_SEED` does not change a hydrated cache.

**Paged-KV routing.** rotor3 does not use the PagedAttention block-table path.
`PagedKStorage` is q8 only and `PagedPlanarVStorage` is PlanarQuant only.

---

### rotor4 codec

> **INERT on this build** — `rotor4` decodes from the bf16 mirror on both
> axes, so `exit_prefill` never builds the packed store described below and the
> codec math never runs. Resident KV and generated tokens measure identical to
> bf16. See § "Codec disposition — what every codec in the tree is for".

**Algorithm — Cl(3,0) Clifford rotor sandwich, 4-bit codebook.**

rotor4 is the 4-bit member of the Clifford rotation family. The algebra and the
rotor sandwich are the same as rotor3. Only the codebook and the code width are
different:

| Property | rotor4 |
|---|---|
| Delivered bits / element | **9.75** at head\_dim=128 (156 B/token/kv\_head, **0.609× bf16**): 4.25 codes + 5.375 scales + 0.125 norm |
| Codebook | `lloyd_gaussian_codebook(4)` (16 centroids), one codebook for all 8 multivector components |
| Pack density | dense code plane, 4 bits per code, 3 codes per group |
| Rotation | the same static per-(layer, head) rotor table as rotor3 (`[n_groups, 4]`), from the same `ROTORQUANT_GLOBAL_SEED` formula |
| Group size | 3 elements (the same Cl(3,0) grade-1 group as rotor3) |
| `head_dim` constraint | none: the same tail padding as rotor3 |
| MSL kernel | `rotorquant_msl.rs`, shared with rotor3 through `rotor_quantize_v{3,4}_gpu` / `rotor_dequantize_v{3,4}_gpu` |
| SSD layout tag | `rotor_v_4` |

**One type, two widths.** `QuantRotorV4` and `QuantRotorV3` are aliases of
`QuantRotorV<4>` and `QuantRotorV<3>`: one const-generic store that encodes and
decodes through `rotor_encode` / `rotor_decode` at its own `BITS`. `RotorBlocks`
is shared.

**Tests.** `rotor4_cosine_gate` in
`crates/rmlx-kv-quant/src/rotorquant_tests.rs`;
`rotorquant_msl_tests.rs::rotor_v4_msl_matches_cpu_within_eps`;
`roundtrip_rotor4` in `crates/rmlx-kv-ssd/src/block_io_tests.rs` (all four V
buffers bit-identical after hydrate, rotor table stored with the payload).

**Paged-KV routing:** the same as rotor3. `RotorV4` does not use PagedAttention.

---

### iso K-side variants

Four variants apply the iso3 / iso4 codec to the K axis. The IsoQuant codec
(`iso_encode_fast` / `iso_decode_fast`) is **axis-agnostic**: the encoder takes
a flat `[B, kv_h, S, D]` row buffer and a per-row `head_dim`. Only the role on
the SDPA path and the SSD tensor names (`l{idx}.k.*` against `l{idx}.v.*`) make
it K or V.

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD layout tag |
|---|---|---|---|---|
| `Iso3Sym` | iso3 (3-bit quaternion SO(4)) | iso3 (3-bit) | `(IsoK3, Iso3)` | `iso_sym_3` |
| `Iso4Sym` | iso4 (4-bit quaternion SO(4)) | iso4 (4-bit) | `(IsoK4, Iso4)` | `iso_sym_4_v2` |
| `IsoKOnly3` | iso3 (3-bit) | **bf16** (parent `decode_fp16_v`) | `(IsoK3, Bf16)` | `iso_k_only_3` |
| `IsoKOnly4` | iso4 (4-bit) | **bf16** | `(IsoK4, Bf16)` | `iso_k_only_4` |

**Qwen MoE arch guard.** `cache_type::validate_resolved` rejects all four
variants on Qwen MoE with `ResolveError::QwenMoeIsoKRejected { variant }`. The
error names the variant, and the process exits with code 78. `auto` never
selects any of the four on any arch.

**IsoKOnly bf16-V layout.** The V buffer is the parent
`KvCache::decode_fp16_v`, as for `KvStorage::None` and `KvStorage::PlanarK`.
The SSD writer writes only the K-side tensors
(`l{idx}.k.codes_packed/scales/quaternions/norms`).

**Status.** GPU-resident on the hot path. `QuantIsoK3` / `QuantIsoK4` each hold
a `QuantKGpuRing`. The K encode writes the packed ring on the GPU, and
`iso_flash_decode` reads that ring (see § `iso_flash_decode`).

**Cosine gates.** On the LCG fixture at `head_dim=128, n_rows=16, TEST_SEED`
(`storage/quant_iso_k_tests.rs`): `iso_k_3` gates at 0.97, `iso_k_4` at 0.99
(minimum cosine).

**SSD round-trip tests.** Four tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`: `roundtrip_iso_sym_3`,
`roundtrip_iso_sym_4`, `roundtrip_iso_k_only_3` and `roundtrip_iso_k_only_4`.
They assert that the K-side codes are bit-identical after hydrate and that the
K dequant matches within 1e-3.

### rotor K-side variants

> **INERT on this build** — the two asymmetric members,
> `rotor_k_3_asym_v<vb>_g<vg>` and `rotor_k_4_asym_v<vb>_g<vg>`, decode from the
> bf16 mirror on both axes, so `exit_prefill` never builds the packed store
> described below and their codec math never runs. Resident KV and generated
> tokens measure identical to bf16. The two K-only members of this section are
> not in that class. See § "Codec disposition — what every codec in the tree
> is for".

Six variants apply the rotor3 / rotor4 codec to the K axis. They can add a
**1-bit QJL residual sideband** (a Johnson–Lindenstrauss sketch of the
post-rotor MSE residual) with `--rotor-qjl on`. QJL is **off** by default. QJL
has no MSL kernel, so QJL on moves the rotor K encode and dequant to the CPU.
`rotor_qjl_enabled()` in `rmlx-kv-quant::rotor_qjl` holds the toggle.

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD tag (QJL off / on) |
|---|---|---|---|---|
| `Rotor3Sym` | rotor3 (+ QJL) | rotor3 | `(RotorK3, Rotor3)` | `rotor_sym_3` / `rotor_sym_3_qjl` |
| `Rotor4Sym` | rotor4 (+ QJL) | rotor4 | `(RotorK4, Rotor4)` | `rotor_sym_4` / `rotor_sym_4_qjl` |
| `RotorKOnly3` | rotor3 (+ QJL) | **bf16** (parent `decode_fp16_v`) | `(RotorK3, Bf16)` | `rotor_k_only_3` / `rotor_k_only_3_qjl` |
| `RotorKOnly4` | rotor4 (+ QJL) | **bf16** | `(RotorK4, Bf16)` | `rotor_k_only_4` / `rotor_k_only_4_qjl` |
| `RotorK3Asym { v_bits, v_group_size }` | rotor3 (+ QJL) | **TurboQuant V** at `v_bits` (the K8V4 / K8VTurbo3 / K8VTurbo2 V codec) | `(RotorK3, Q*G*)` | `rotor_k_asym_3_v{vb}_g{vg}` / `rotor_k_asym_3_qjl_v{vb}_g{vg}` |
| `RotorK4Asym { v_bits, v_group_size }` | rotor4 (+ QJL) | **TurboQuant V** at `v_bits` | `(RotorK4, Q*G*)` | `rotor_k_asym_4_v{vb}_g{vg}` / `rotor_k_asym_4_qjl_v{vb}_g{vg}` |

**Asymmetric rotor-K variants.** `RotorK{3,4}Asym` carry a TurboQuant V codec at
`v_bits ∈ {2, 3, 4}`. The V slot uses the `QuantV` codec of `K8V4` /
`K8VTurbo3` / `K8VTurbo2` (Lloyd-Max N(0,1) codebook, fixed internal group of
32). The `v_group_size` field goes into the SSD layout key only. The parser and
the composer reject `v_bits = 8`, because TurboQuant has no 8-bit path. For
bf16 V use `--ctv bf16` (`RotorKOnly{3,4}`); for rotor V use
`--ctv rotor_v_{3,4}` (`Rotor{3,4}Sym`).

Display form: `rotor_k_3_asym_v{v_bits}_g{v_group_size}` (and `_4_`).
Compose forms:

- `--ctk k_rotor3 --ctv q4_g64` → `RotorK3Asym { v_bits: 4, v_group_size: 64 }`
- `--ctk k_rotor3 --ctv q4_g128` → `RotorK3Asym { v_bits: 4, v_group_size: 128 }`
- `--ctk k_rotor4 --ctv q3_g64` → `RotorK4Asym { v_bits: 3, v_group_size: 64 }`
- `--ctk k_rotor4 --ctv q2_g64` → `RotorK4Asym { v_bits: 2, v_group_size: 64 }`
- `--ctk k_rotor3 --ctv rotor_v_3` → `Rotor3Sym`
- `--ctk k_rotor3 --ctv bf16` → `RotorKOnly3`

**Qwen MoE arch guard.** `cache_type::validate_resolved` rejects all six
variants on Qwen MoE with `ResolveError::QwenMoeRotorKRejected { variant }`.
The `variant` field holds the full Display form (for example
`rotor_k_3_asym_v4_g64`). The process exits with code 78.

**Decode cost with QJL on.** With `--rotor-qjl on`, the rotor K-side codecs
have no GPU-resident code mirror. Each decode step decodes the full K prefix on
the CPU, applies an O(head_dim²) QJL score correction per cached token, and
uploads the K prefix again. This cost grows with `kv_seq`. With QJL off, the
rotor K encode and `rotor_flash_decode` run on Metal.

**Fused-QK.** Of the rotor codecs, the head-major fused-QK path has kernels
for `RotorK3Asym` and `RotorK4Asym` only. `--fused-qk` is off by default. The
path runs only when all of these conditions are true
(`try_fused_qk_dispatch`):

- `--fused-qk on`;
- the device is GPU;
- the step is a decode step (`q_seq == 1`) and `new_k` has rank 4;
- `head_dim` is 128 or 256;
- `kv_seq` is at least `fused_qk_min_kv_seq` (default 512, env
  `RMLX_FUSED_QK_MIN`);
- the codec has a fused-QK kernel and a GPU K encoder;
- QJL is off (the kernel does not use the QJL residual);
- the bf16 K mirror is live to seed the shadow;
- the storage variant carries a `max_seq`, and the step does not overflow it.

See § "Fused-QK head-major K storage".

**QJL residual — storage format.** When QJL is on at the first `append`, the
codec stores one extra sign bit per `head_dim` element per token with the rotor
codes. Wire format: packed `u8`, row-major, shape
`[B, kv_h, max_seq, ceil(head_dim/8)]`. Bit order: LSB = element 0, MSB =
element 7 (the same as the Python `rotorquant/turboquant/rotorquant.py`
reference). The QJL projection matrix `S` (`[head_dim, head_dim]` f32,
row-major) is made once at the first append and written to the SSD block
(`l{idx}.k.qjl_s`). The layout tag (`*_qjl`) tells the reader to hydrate the
projection matrix.

**QJL toggle.** CLI: `--rotor-qjl on|off` (default `off`). The `rmlx` binary
always installs the flag value at startup, so the environment variable
`RMLX_ROTOR_QJL` has no effect on it. `RMLX_ROTOR_QJL=1` (or `on`, `true`,
`yes`) enables QJL only in a process that does not install the flag, for
example a test binary. The toggle is read at each construction (not cached).

**Score-time QJL correction.** `apply_qjl_correction` (called from
`rotor3_k_decode` / `rotor4_k_decode`) applies the correction **at decode time**
as a per-token K-side residual-add:

```
Δk[t, j] = ||r_t|| · sqrt(π/2)/m · sum_i ( S[i, j] · signs[t, i] )
K_corrected[t] = K_rotor[t] + Δk[t]
```

The result `Q · K_corrected` is equal to the score-time `term1 + term2` of the
Python reference (`RotorQuantProd.inner_product` in
`rotorquant/turboquant/rotorquant.py:246-263`), because `term2` is linear in
`Q`. Thus the correction stays inside `rmlx-kv-quant`, and every caller of
`rotor3_k_decode` / `rotor4_k_decode` gets it.

Validation (both in `crates/rmlx-kv-quant/src/rotorquant_tests.rs`, run by
`make ci`):

- `qjl_correction_score_estimator_unbiased_rotor3` and `_rotor4` reproduce the
  Python
  `test_inner_product_unbiased` (n=1024 unit-normalized pairs). They assert
  `|bias| < 0.05` with QJL on and off, through `apply_qjl_correction`.
- `qjl_residual_add_matches_score_time_correction` asserts, per token,
  `Q · K_on == Q · K_off + Python_term2` within 1e-4 absolute. This proves that
  the dequant-side residual-add is the same as the score-time `term2` of the
  reference for the same rotor MSE codes.

Per-K cosine on the LCG fixture is slightly lower with QJL on. This is
expected: the JL sketch gives an unbiased inner product, not a closer K vector.

**SSD round-trip tests.** Eight tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`:
`roundtrip_rotor_sym_3_{qjl,no_qjl}`,
`roundtrip_rotor_sym_4_{qjl,no_qjl}`,
`roundtrip_rotor_k_only_{3,4}_{qjl,no_qjl}`. Each asserts that the K codes are
bit-identical after hydrate and that the `use_qjl()` flag matches the tag. The
tests hold `ROTOR_QJL_ENV_LOCK` (a process-wide mutex) to prevent env-var races
under parallel `cargo test`.

---

## The auto default

`--kv-quant auto` is the value when no codec flag is given. It resolves to
**unquantised bf16** (`KvQuant::None`) on every architecture, checkpoint and
prompt length. `rmlx_models::kv_cache::DEFAULT_KV_QUANT` is the only producer.
These read it and nothing else:

- the CLI resolver (`crates/rmlx-cli/src/commands/parse.rs`);
- the server engine (`crates/rmlx-server/src/engine/helpers.rs`);
- the arch dispatcher, text and image entries
  (`crates/rmlx-models/src/arch/mod.rs`);
- `speculative::round_common::verifier_cache_stack`
  (`crates/rmlx-models/src/speculative/round_common.rs`).

Every speculative verifier stack comes from `verifier_cache_stack`. The
two-model draft stack uses the codec that the verifier resolved. No per-arch
table and no per-context selection sits behind the constant.
`--kv-preset auto` resolves to the same constant (§"`--kv-preset auto`").

Every codec stays selectable by name: `--kv-quant`, `--cache-type-k` /
`--cache-type-v`, `--kv-bits` and `--kv-preset`. Only what `auto` resolves to
is fixed.

### Why bf16

- The 17 Class 2 codecs build no packed store under the default flags. Their
  decode reads the bf16 mirror, so their resident KV and their greedy token
  ids equal `none`'s (§"Codec disposition", Class 2).
- A Class 3 codec that holds less resident KV than bf16 is an opt-in memory
  setting. At `heads_per_kv ≥ 4`, the iso and rotor codecs cannot beat bf16
  at decode (§"Why ε is small — grid geometry").

---

## Memory and bit-rate summary

Resident KV depends on the codec class and on the cache topology. The figures
below are stored bits per value per axis at `head_dim = 128`; bf16 is 16.0.

- **`none` and every Class 2 codec**, on a cache that went through a prefill
  bracket: 16.0 on both axes. `exit_prefill` builds no packed store. Two flags
  add a buffer beside the mirror. `--fused-qk on` gives `k8v4`, `k8v8`,
  `tsym3`, `tsym4` and `rotor_k_*_asym` a head-major K shadow
  (§"Fused-QK head-major K storage"). `--turbo-flash on` gives `k8v4` the
  TurboFlash buffers. Both flags default to `auto`, which resolves OFF.
- **A store-backed cache of any codec** holds its packed store. This is an SSD
  hydrate, or a cache that did not go through a prefill bracket. The rate of
  each store family is in the table under §"Crate-wide rate ceiling".
- **Class 3 codecs** hold their packed store on the axes their decode reads:

| Codec | K | V |
|---|---:|---:|
| `iso3_sym` / `iso4_sym` | 7.125 / 8.125 | 7.125 / 8.125 |
| `k_iso3` / `k_iso4` | 7.125 / 8.125 | 16.0 |
| `rotor3_sym` / `rotor4_sym` | 8.75 / 9.75 | 8.75 / 9.75 |
| `k_rotor3` / `k_rotor4` | 8.75 / 9.75 | 16.0 |
| `mixed_k<kb>g<kg>_v<vb>g<vg>` | `kb + 32/kg` | `vb + 32/vg` |
| `rot_k_v<vb>g<vg>` | 8.5 | `vb + 32/vg` |

The iso and rotor rates are the GPU ring after the first fused decode step.
The ring exists only inside the fused shape gate: batch 1 and a power-of-two
`head_dim` of at most 512. Outside the gate the CPU block form stays: iso
holds 43.25 bits per value there (§"Iso memory truth"). The rotor rates are
with `--rotor-qjl off`. An MLX
affine group spends 32 bits on its scale and bias, hence the `32/group` term.

`mixed_*` and `rot_k_*` also keep a bf16 K and V mirror on a stack whose layers
share KV (`KvCache::shares_kv`). There the mirror adds 16.0 per axis, and the
codec holds more than `none`. On a stack without shared KV they hold the store
alone. `global_layer_store_plus_mirror_codec_sign_follows_shared_kv`
(`crates/rmlx-kv-quant/src/quant_tests.rs`) pins both signs.

`rmlx info --list-cache-types` prints the resident KV of each codec on one
global layer at `head_dim` 128, for a dense and a shared-KV stack. Its producer
is `KvQuant::estimated_resident_bytes_per_layer`. It does not count the
layer-adaptive boundary, which gives some layers another codec
(§"Layer-adaptive overrides").

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
`head_budgets.json` instead (§"Sparse attention"). `head_budget` loads the
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

## Fused flash-decode over a quant store — the break-even condition

The fused flash-decode kernels in the sections below read a packed KV store at
decode instead of a bf16 mirror. So does TurboFlash
(§"TurboFlash is off by default"). A smaller store moves fewer bytes per
decode step. This section states when that pays.

### The condition

Two quantities, both measurable with `rmlx bench`:

* **ρ**: the bytes the fused kernel reads, over the bytes MLX `sdpa_vector`
  reads from the bf16 mirror for the same cell.
* **ε**: the fraction of `sdpa_vector`'s per-byte throughput that the kernel
  achieves. `ε = ρ / (marginal-slope ratio against none)`, where the slope is
  `b` in `ms/step = a + b·(KV tokens/1000)`.

A fused decode beats bf16 only when **ρ < ε**. ε is a property of the kernel
shell and ρ of the store, so a good codec and a good kernel can still lose as
a pair. The store must hold fewer than `16·ε` bits per value per axis.

### Why ε is small — grid geometry

Each P1 kernel indexes its grid by **query** head and reads the KV head
`hq / heads_per_kv`: `turbo_flash_p1.metal`, `iso_flash_decode_p1.metal`,
`iso_flash_decode_symv_p1.metal`, `rotor_flash_decode_p1.metal`,
`rotor_flash_decode_symv_p1.metal` and `planar_flash_decode_p1.metal` under
`crates/rmlx-kv-quant/src/metal/`. So `heads_per_kv` threadgroups read the
same KV bytes. That caps the shell at **ε ≤ 1/heads_per_kv** before any cost in
the kernel body.

The densest store in the tree is `tsym3` at 4.00 bits per value per axis
(§"Crate-wide rate ceiling"). So its ρ is 0.25, and no kernel decodes over it. The
iso and rotor rings have ρ between 0.445 (`iso3_sym`, 7.125 / 16) and 0.80
(`k_rotor4`). At `heads_per_kv ≥ 4` the ceiling is 0.25 or less. So on such an
arch no fused decode over a current store can beat bf16, even with a perfect
kernel body.

### The decode ceiling — deleting the codec's arithmetic does not reach bf16

The ceiling `1/heads_per_kv` comes from the grid, not from the decode math. A
change to the math cannot lift ε above it: not a rotation hoist, a narrower K
store or a better codebook. At `heads_per_kv ≥ 4` the kernels would not beat
bf16 with no decode math at all, because ρ ≥ 0.25 for every current store. Only
a grid that reads each KV byte once per KV head removes this ceiling.

### Codec disposition — what every codec in the tree is for

The KV enum spells **28 codecs**, all listed in `ALL_KV_QUANTS`. Each has one
of the three dispositions below. `KvQuant::decode_reads_packed_store` decides
the class. The tests that hold it, all in
`crates/rmlx-kv-quant/src/quant_tests.rs`:

- `DISPOSITIONS` writes the class of each codec by hand.
  `every_codec_carries_a_disposition` fails when a row and the predicate
  disagree.
- `disposition_table_covers_every_variant_once` fails when a codec in
  `ALL_KV_QUANTS` has no row, or two rows.
- `disposition_is_a_property_of_the_family_not_its_parameters` pins that the
  four parameterised families keep their class at every parameter set.

`scripts/bench/codec_inertness_probe.sh` measures `kv_cache_bytes` and a greedy
token-id digest for each codec against `none`. Generate at least 200 tokens:
at 32 tokens, codecs that differ can give the same digest.

#### Class 1 — the baseline (1 codec)

`none`: bf16 on both axes, the `auto` default, and the reference for the other
classes. `exactly_one_codec_is_the_baseline` keeps it the only one.

**Disposition: keep.**

#### Class 2 — inert, mirror-fed (17 codecs)

`k8v4`, `k8v8`, `planar`, `planar3`, `planar_k`, `k8vturbo3`, `k8vturbo3tcq`,
`k8vturbo2`, `k8vturbo2tcq`, `tsym3`, `tsym4`, `iso3`, `iso4`, `rotor3`,
`rotor4`, `rotor_k_3_asym_v*_g*`, `rotor_k_4_asym_v*_g*`.

Decode reads the bf16 mirror on both axes. So `exit_prefill` builds no packed
store, and the codec math does not run on a cache that went through a prefill
bracket. `exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`
(`crates/rmlx-kv-quant/src/kvcache/warm_ttft_cross_codec_tests.rs`) pins this.
With `--fused-qk` and `--turbo-flash` at their `auto` default (OFF),
resident KV and greedy token ids equal `none`'s. With `--fused-qk on`, the
q8, TurboSym and rotor-asym codecs build a head-major K shadow
(§"Fused-QK head-major K storage"). With `--turbo-flash on`, `k8v4` holds the
TurboFlash buffers and decodes its 4-bit V. The store is still the
authority for a cache with no mirror: an SSD hydrate, or a cache that did not
go through a prefill bracket.

Every codec here carries an INERT banner at the head of its section above. The
`--kv-quant`, `--kv-bits` and `--kv-preset` help says the same.
`make check-kv-codec-disposition` derives the class from `ALL_KV_QUANTS` and the
three decode predicates. It fails on a banner or a help entry that disagrees
with the class. `make check-kv-codec-disposition-fixtures` measures that gate's
recall. At resolve time, `validate_resolved` logs a `warn!` for every codec
other than `none` whose packed store is never built.

`k8vturbo3tcq` and `k8vturbo3` share one decoder, `turbo_dequantize`. Only the
encode differs, and the encode does not run on a cache with a mirror. The same
holds for `k8vturbo2tcq` and `k8vturbo2`.

**Disposition: keep parseable and selectable.** The reasons:

1. **The store is the re-enable path.** A codec that gains a decode kernel over
   its own store changes its arm in `decode_reads_packed_store`. `exit_prefill`
   then builds the store that the kernel reads.
2. **Recorded rows must stay readable.** `observations` is append-only, and
   its rows name these codecs.
3. **The widest-matrix goal.** `CLAUDE.md` names the rotation KV families as a
   differentiator.

A withdrawn codec name keeps failing. `KvQuant::from_str` returns
`KvQuantParseError::Retired`, which names the replacement. It is never an
alias (§"`rot_k_tq4v` is rejected").

#### Class 3 — reads its own packed store (10 codecs)

`mixed_k<kb>g<kg>_v<vb>g<vg>`, `rot_k_v<vb>g<vg>`, `iso3_sym`, `iso4_sym`,
`k_iso3`, `k_iso4`, `rotor3_sym`, `rotor4_sym`, `k_rotor3`, `k_rotor4`.

These are the only codecs whose quantization a served request uses:

- `mixed_*` and `rot_k_*` append to the MLX affine 3-tuples and read them at
  every decode step (`mixed_quantized_sdpa`).
- The K-only family (`k_iso*`, `k_rotor*`) appends K to the packed store at
  every step, and the flash-decode arm reads it back.
- The symmetric family (`iso*_sym`, `rotor*_sym`) decodes with a flash kernel
  over both packed rings.

Residency (§"Memory and bit-rate summary"):

- The eight iso and rotor codecs hold less than bf16 inside the fused shape
  gate: batch 1 and a power-of-two `head_dim` of at most 512. There the ring
  is the only resident copy after the first decode step. Outside the gate the
  CPU block form stays, and iso holds 43.25 bits per value
  (§"Iso memory truth"). The result does not depend on the topology, because
  their `feeds_bf16_*` arms are constants that do not read `shares_kv`.
  `iso_and_rotor_k_codecs_are_under_the_floor_at_every_geometry` pins the ring
  rates under bf16.
- `mixed_*` and `rot_k_*` hold less than bf16 on a stack without shared KV.
  They hold more on a shared-KV stack, where the bf16 mirror stays.

Decode: at `heads_per_kv ≥ 4` no iso or rotor codec can beat bf16
(§"Why ε is small — grid geometry").

**Disposition: keep.** Each is a memory setting where it holds less than bf16.
They are also the only codecs that decode over a packed store, so any
fused-decode work builds on them.

### `kv_frac` bounds a codec claim — and is not a statement about context

`kv_frac` is the KV share of the bytes a decode step reads:
`kv_bytes_step / (weight_bytes_step + kv_bytes_step)`. It is the ceiling on the
part of decode that a KV codec can change. `scripts/perf_ceiling.py` prints it
in the last column of every row. It reads `config.json` and the safetensors
index and runs no model. `make check-kv-byte-model-parity` holds its KV byte
model to the engine's.

**It is a property of the (model, context) pair, not of the context.** At a
4 096-token prompt with `--kv-quant none`, the script gives 0.010 for
Qwen3.8-27B-mxfp8 and 0.221 for Ternary-Bonsai-8B-2bit. So "measured at 4k"
does not mean "measured where the codec axis is near zero".

A large `kv_frac` is necessary, not sufficient. ε decides how much of the
bound a fused kernel collects. State `kv_frac` and the **K bit width** next to
every codec cell. Do not infer an effect from `kv_frac` alone.

Before you design a long-context cell, check the model's positional capacity:
`max_position_embeddings`, extended by a `rope_scaling` in `config.json` or by
`--yarn-factor`. `rmlx baseline` refuses a `--max-ctx` above that capacity.
On `--device gpu` it also refuses a prompt longer than the context ceiling,
unless `--allow-truncate` or `--max-prompt-tokens` is given.

#### The null was a bit-width result, not a context result

A decode result for one K bit width is not a result for another. The K bit
width sets the bytes of the store: `kb + 32/kg` bits per value for `mixed_*`.
`kv_frac`, from `scripts/perf_ceiling.py --kv-quant <codec>`, bounds the share
of a decode step those bytes can change. Both depend on the model and the
context. So state the codec, its K bit width and `kv_frac` with every cell.

---

## `rotor_flash_decode` — fused MSL flash-decode over rotor-quant K

This is the decode kernel for `k_rotor3` and `k_rotor4`
(`KvStorage::RotorKOnly3` / `RotorKOnly4`). It computes QK over the packed
rotor K ring, an online softmax, and SV over the bf16 V mirror, in two Metal
dispatches per decode step. The Cl(3,0) K decode runs inside the attention
loop. So no bf16 or f32 K is built, and no K data goes through the host.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_msl.rs` — Rust dispatcher,
  header builder, dispatch counters.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_p1.metal` — pass-1 body,
  one body for both bit widths.
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — pass-2
  log-sum-exp merge. It is codec-agnostic. Every flash-decode kernel in this
  document uses it.
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
these conditions are true. Otherwise the step takes the CPU dequant path.

- The cache is not a sliding-window ring. `update_and_sdpa` sends a rotating
  cache to the legacy bf16 path first.
- The device is GPU.
- The storage is `RotorKOnly3` or `RotorKOnly4`.
- The store carries no QJL sideband.
- `q_seq == 1`.
- `b == 1`.
- `head_dim` is a power of two and at most `ROTOR_FLASH_HEAD_DIM_MAX` (512).

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
| `Rotor{3,4}Sym` | NO | Decodes through `rotor_flash_decode_symv`, which reads V from its own ring. |
| `RotorK{3,4}Asym` | NO | V is TurboQuant, not rotor. Its only GPU decode kernel is `rotor_fused_qk`, with `--fused-qk on` (§"Fused-QK head-major K storage"). |

### Ring eligibility is passed down, not inferred

The caller of the rotor or iso GPU encode decides whether the ring is fed. It
passes a `RingFeed`:

- **`Maintain`** — feed the ring and push a CPU block. The legacy rotor K-only
  entries use it, because they dequantize the whole prefix on the same step.
- **`MaintainRingOnly`** — feed the ring and push no CPU block. This is a
  ring-only tail. The fused decode entries use it. `shape[2]` still advances.
- **`Skip`** — drop the ring. The legacy rotor symmetric and asymmetric
  entries and the legacy iso entries use it.

A ring-only feed at `b > 1` takes the block path instead
(`is_ring_only_append`). A ring grows with the context:
`ring_bits_per_value` gives its bits per value.

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

`truncate_to` keeps the ring. It lowers `shape[2]` to `n`, and the next append
writes the ring from `n`. So a speculative rollback keeps a ring-only tail up
to `n`.

A store whose blocks fall short of `shape[2]` with no ring to supply the rest
is an error, never a zero-padded read. `synced_rotor_k_blocks` refuses it at
the codec. `ensure_rotor_k_blocks_cover_shape` refuses it at SSD
serialization. Both return `Err` and are not `debug_assert`s, so they hold
under `release-perf`.

**Row vs. sequence units.** A rotor or iso block's `n_tokens` counts rows
(`b * kv_h * seq_of_block`). `truncate_to(n)` takes a sequence position. So
the planner compares cumulative rows with `n * b * kv_h`.

**A cut inside a block splits it.** A block spans one append. A speculative
partial accept cuts inside the verifier's `K + 1`-token chunk. The planner
splits the trailing block and cuts every per-row buffer to the kept rows:
codes, per-group scales, per-group quaternions, per-token norms and the QJL
sideband.

**The split is `b == 1` only.** A block's rows run `[B, S_block, kv_h, D]`, so
at `b > 1` a row prefix is not a sequence prefix. At `b > 1` the planner drops
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

* The flat GPU buffers of the turbo, planar and affine stores (`QuantV`,
  `QuantKTurbo3/4`, `QuantPlanarK/V`, `QuantK`). Their prefix records no
  chunk boundary.
* `QuantK`'s CPU `codes` / `scales`, one flat pair with no per-append
  boundary. It refuses even a single-append `b > 1` store.

The eight rotor and iso K and V stores share one planner, `truncate_plan` in
`crates/rmlx-kv-quant/src/storage/mod.rs`, with one `BlockRows` implementation
per block type. Tests: `storage/truncate_plan_tests.rs`, and one store-level
round trip per block type in `quant_rotor_k_tests.rs`, `quant_rotor_v_tests.rs`
and `quant_iso_v_tests.rs`.

**Scope — every CPU-side store cuts, and every one of them is loud.** The same
planner drives the turbo and planar blocks (`TurboBlocks`, `PlanarBlocks`).
`QuantV`, `QuantKTurbo3/4`, `QuantPlanarK` and `QuantPlanarV` implement
`truncate_to`. `QuantK` cuts its flat `codes` / `scales` pair to the first `n`
positions. Every arm of `KvStorage::truncate_to` and `KvStorage::reset`
delegates to the store's own `truncate_to` or `reset`. So no arm lowers
`shape[2]` and leaves the payload in place.

Every CPU dequant path checks that its blocks decode to `prod(shape)`
elements. On a mismatch in either direction it returns `"CPU blocks decode to
N elems but shape [...] implies M — refusing to zero-pad / truncate"`. The
stores refuse some cuts, and this check reports each one. They refuse a cut
at `b > 1`, a block that is not a whole number of positions, and, for
`QuantK`, a target inside a 128-element q8 group.

`TurboBlocks` and `PlanarBlocks` carry no `n_tokens`. Their row count is the
product of the first three axes of `original_shape` (`storage::block_rows`).
The CPU append and the SSD hydrate write those axes in different orders, and
the product is the same for both. A split writes `[1, 1, rows, width]`.
`cpu_block_truncate_tests::quant_v_truncate_reads_rows_from_the_shape_product`
pins it.

**When a cut changes an answer.** A Class 2 codec builds no packed store on a
cache that went through a prefill bracket (§"Class 2"). Its decode reads the
bf16 mirror, so a served request cannot tell a correct cut from a no-op. Its
store is read only on a cache with no mirror: an SSD hydrate
(`KvCache::from_storage` leaves `decode_fp16_k` at `None`), or a cache that
never went through a prefill bracket. The device does not change this. The
store-reading codecs also have their store read by the SSD spill
(`write_quant_k` / `write_quant_v`) and the prompt-cache snapshot
(`try_deep_clone`).

**Truncation is monotone-decreasing.** The turbo, planar and affine stores
clamp the target to their own `shape[2]` (`storage::clamp_truncate_target`). A
target past `shape[2]` is reachable. A store-backed codec that also keeps a
bf16 mirror advances `KvCache::offset` on paths that the store does not
follow. A `shape[2]` raised to meet the target would claim tokens that no
payload holds.

The rotor and iso stores do not clamp. A ring-only tail lies below
`shape[2]`, and the ring readback returns `Err` on an over-long target. So for
`n > shape[2]` the mixed arms leave the two axes of one codec at different
lengths. These are `IsoV3`, `IsoV4`, `RotorV3` and `RotorV4`, where the affine
K clamps, and `RotorKAsym3/4`, where the affine V clamps. The guard on the
unclamped side reports it at spill.

**The `Mixed` arm truncates to its fill marker.** `MixedKvState` is a capacity
buffer that grows in `STEP` increments, with `offset` as its fill marker.
`truncate_to(n)` sets the marker to `n`, and the next append writes over the
rows from `n`. The bf16 mirror of a shared-KV producer follows the same offset.
For `n > offset` the state keeps its fill and emits an `error!` that names
both numbers. `offset` is the coverage, so there is nothing to clamp down to.
`kvcache/shared_source_tests.rs::mixed_truncate_to_keeps_the_prefix_it_was_told_to_keep`
compares the cut cache with a cache prefilled to the kept length.

Tests: `storage/cpu_block_truncate_tests.rs` covers the partial-accept round
trip per store, the `b > 1` and q8-group refusals, the zero, exact and
past-the-end targets, `KvStorage::reset`, and the `KvStorage::truncate_to`
dispatch. Each oracle is a reference store built from the kept tokens only.
`crates/rmlx-kv-ssd/src/hydrate_truncate_tests.rs` truncates a hydrated cache
inside a block, appends a correction and checks the decoded V.

In production, `KvCache::truncate_to` is called by the prompt-cache prefix
trim (`PromptCacheEntry::truncate_kv_to`) and by the speculative
partial-accept rollback. `truncate_plan` has no arch or head-count branch.
`kv_cache_truncate_iso3_kv_h_gt_1_path` (`rmlx-models/src/kv_cache/tests.rs`)
drives the full `KvCache` dispatch at `kv_h = 4`.

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
The shared decode helpers declare `norms` in the `device` address space, and
that mismatch fails the MSL compile at the first dispatch. So both
dispatchers call `flash_decode_common::pad_norms_to_device_floor`. It
zero-pads `norms` to `NORMS_DEVICE_MIN` (16) elements when
`b * kv_h * kv_seq` is below it. The kernel loop stops at the real `kv_seq`,
so no pad element is read. On the pinned MLX the compile does not fail without
the pad. The pad stays as a guard against another MLX build.

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
these conditions are true. Otherwise the step takes the CPU dequant path.

- The cache is not a sliding-window ring.
- The device is GPU.
- The storage is `IsoKOnly3` or `IsoKOnly4`.
- `q_seq == 1`.
- `b == 1`.
- `head_dim` is a power of two, a multiple of 4, and at most
  `ISO_FLASH_HEAD_DIM_MAX` (512).

The iso codecs have no QJL residual. So `k_iso3` and `k_iso4` reach the kernel
at the default flags.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `IsoKOnly3` / `IsoKOnly4`, `b == 1` | **YES** | GPU ring + `iso_flash_decode_sdpa`. |
| `IsoKOnly{3,4}`, `b > 1` | NO | The ring stride does not interleave batch. |
| `Iso{3,4}Sym` | NO (this kernel) | Decodes through `iso_flash_decode_symv` (`iso_flash_decode_symv_msl.rs`, `metal/iso_flash_decode_symv_p1.metal`), which reads V from its own ring. Its gate is this gate, with `IsoSym3` or `IsoSym4` as the storage. |

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
Its fixtures are seeded. It prints these counts:

| `kv_h` × `heads_per_kv` | `head_dim` | `kv_seq` | f32 accumulator differs | max abs err | **bf16 output differs** |
|---|---:|---:|---:|---:|---:|
| 8 × 4 | 128 | 64 | 3569 / 4096 | 8.94e-8 | **0 / 4096** |
| 8 × 4 | 128 | 512 | 3643 / 4096 | 2.98e-8 | **0 / 4096** |
| 8 × 4 | 128 | 4096 | 3863 / 4096 | 2.05e-8 | **3 / 4096** |
| 1 × 8 | 256 | 64 | 2048 / 2048 | 1.13e-4 | **273 / 2048** |
| 1 × 8 | 256 | 512 | 2048 / 2048 | 3.55e-5 | **280 / 2048** |
| 1 × 8 | 256 | 4096 | 2048 / 2048 | 1.46e-5 | **298 / 2048** |

The bf16 outputs differ in 4 of 6 cells and agree in 2. The clean cells are
`head_dim = 128` with `kv_seq <= 512`. So a check at one short Bonsai-shaped
cell would wrongly confirm byte-identity. The f32 errors stay inside a bf16
ULP at these magnitudes, which is what a summation-order difference gives.
The kernel computes the same attention, but it is not lossless.

The test asserts three things. At least one cell differs at bf16, which is the
claim. At least one cell differs at f32, which shows that both arms ran. Every
cell's f32 error is below 1e-3, which separates rounding from a defect.

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
`KvQuant::feeds_bf16_k_at_decode()` is true. Eight codecs return false:
`Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4`, `Rotor3Sym`, `Rotor4Sym`,
`RotorKOnly3` and `RotorKOnly4`. Each of them decodes through its own
flash-decode kernel over the packed ring.

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
`update_and_sdpa` after the K8V4 TurboFlash branch and before the legacy bf16
SDPA fallback. `try_dispatch_shared_bf16` calls it on the cross-layer-KV
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
  cosine against dense `planar_flash_decode_sdpa` is at least 0.99.

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
on a shape mismatch (`num_layers` against the row count, `num_heads` against
the column count), and on a zero budget. A v1 load logs a `warn!` that advises
re-calibration with `softmax_mass`.

`rmlx serve` loads `head_budgets.json` from the snapshot and attaches it to
the loaded `kv_calib.json` (`crates/rmlx-cli/src/commands/serve.rs`). Without
a `kv_calib.json` it logs a `warn!` and ignores the budgets. A snapshot with no
`head_budgets.json` is the common case. No decode path reads the attached
budgets.

### Calibration recipes

CLI: see `docs/CLI.md`. Three `rmlx kv-calibrate --recipe` values write head
budgets. Each loads the model on the GPU and holds the Metal claim. Each
accepts only a snapshot that loads as `Architecture::Qwen3`.

| Recipe | Schema | Measurement |
|---|---|---|
| `head_budget` | v1 | K-norm² proxy (the H2O / StreamingLLM stand-in) |
| `k_norm_proxy` | v1 | Same as `head_budget` |
| `softmax_mass` | v2 | True Q@K^T → softmax → cumulative-mass top-k |

The v1 recipes write `method: "softmax_mass"`, the name of the target concept,
but they measure the K-norm² proxy. v2 adds the `recipe` field so that a file
names what was measured.

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

---

## Retired: the per-arch default table (composite-score audit)

Between 2026-05 and this change, `auto` resolved through a per-arch table
scored by a 3-term composite (0.571 x decode TPS + 0.286 x cosine + 0.143 x
1/mem_bits). The table and the audit behind it are **gone**: `auto` is
unquantised bf16 on every arch (see "The auto default").

The audit is not merely superseded, it was scoring a quantity that no longer
exists. Its `mem_norm` term ranked codecs by packed-store bit width, and the
bf16-mirror codecs it ranked build no packed store - their resident KV is
bf16's, byte for byte. Its `decode_tps` term was recorded before the store
elision and the f32-leak fixes moved both arms. Re-running it would not restore
a table; it would have to be designed against what the codecs cost today.

**Operator-visible consequence.** An operator who passed no `--kv-quant` and
relied on the table now gets bf16. Output is byte-identical at temp=0 for every
arch whose table entry was a bf16-mirror codec (`K8V8`, `K8V4`, `Planar`),
because those codecs already decoded off the bf16 mirror. It is **not**
byte-identical for the one entry that read its store - `Qwen3ForCausalLM` at
`weight_bits == 2`, which defaulted to `Mixed{k8g64,v4g64}` - where the old
default was lossy and bf16 is the reference. Pass
`--kv-quant mixed_k8g64_v4g64` to reproduce the old bits. `k8vturbo3`, which
the table briefly selected for Gemma4 small, is likewise still available by
name and simply never automatic.

The Qwen-MoE K-width rejection table below is independent of all this and
still stands.

---

### A.y guard re-verification

`validate_resolved` (in `crates/rmlx-models/src/kv_cache/cache_type.rs`) was
inspected and confirmed to reject K-side ≤4-bit codecs on Qwen MoE arches:

- `TurboSym4` → `QwenMoeKBitsTooLow(4)`
- `PlanarK` → `PlanarKOnQwenMoe`
- `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4` → `IsoKOnQwenMoe`
- `Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4` → `RotorKOnQwenMoe`
- `TurboSym3` → `TurboSym3KOnQwenMoe`

None of these K-side codecs is ever selected by `auto`, on Qwen MoE or
anywhere else - `auto` is bf16. The rejection table itself has not been
weakened by the codec adds.

#### What the guard keys off

`validate_resolved` takes an architecture *string*, and which string it is
handed decides whether the guard can fire at all.

Both Qwen3.5 arch strings (`Qwen3_5MoeForConditionalGeneration` and the dense
`Qwen3_5ForConditionalGeneration`) load through one loader into one
`Architecture` variant. The loader does **not** believe the declaration: it
selects dense-vs-sparse-MoE per layer from the tensor witness
`mlp.switch_mlp.gate_proj.weight`. So `architectures[0]` and the model that
actually gets built can disagree, and a checkpoint declaring the dense name
while shipping MoE tensors used to run every codec in the list above to
completion — no error, only wrong output.

The enforcing check is therefore keyed on the **resolved** architecture:

- `Architecture::arch_class()` reports what the loader built. For the Qwen3.5
  variant it asks `has_sparse_moe_layers()` rather than echoing a fixed string.
- `Architecture::validate_kv_quant()` re-runs the table against that resolved
  class, after load and before any KV cache exists.
- `load_model` emits a `warn!` naming `declared_arch` and `resolved_arch` when
  they differ, because that mismatch invalidates every predicate still keyed on
  the declared name. Deliberate aliases (`registry::is_declared_arch_alias` —
  today only Gemma4-unified) are exempt and log at `debug!`, so the warning
  stays meaningful.

The enforcement has to sit on **every path that builds a KV cache**, not on the
one that reads the architecture. There are two such families, and they do not
share a call graph:

| Path | Cache built by | Enforced at |
|---|---|---|
| Non-speculative | per-arch `generate_greedy` (`gemma4`, `gemma3`, `qwen2`, `qwen3`, `qwen3_5_moe`, `qwen3_vl_moe`, `laguna`, `bitnet`), reached only from `Architecture::generate_greedy` / `generate_image` | those two methods, plus `ArchGenerator::new` at startup |
| Speculative | `speculative::{mtp,dflash,eagle3,gemma4_assistant,mod}` build the verifier's caches directly — they never call `Architecture::generate_greedy` | `SpeculativeGenerator::new` at startup, and the per-request seam in its `generate` |

Drafter-side caches are constructed with a hardcoded `KvQuant::None`, which
passes every arch invariant by construction and needs no check.

Startup checks are the fast-feedback copy (`exit 78` / a failed load rather than
a per-request failure after a successful launch); the per-request checks are the
enforcing copy, because a `kv_quant` field on a request arrives after startup and
the server's resolver does not validate it.

The startup resolvers (`rmlx-cli` `resolve_kv_quant`, the server's
`resolve_kv_quant_for_load`) still read `architectures[0]` — they run before
the model is loaded, so it is the only value available. They stay as the
fast-feedback path (`exit 78` at launch); they are no longer the only check.

Empirical positive test: `qwen_moe_low_k_bits_rejected_post_decompose`
in `crates/rmlx-models/src/kv_cache/cache_type_tests.rs` verifies the
runtime rejection path. The declared-vs-resolved bypass is covered by
`crates/rmlx-models/tests/resolved_arch_class.rs`, which builds a snapshot
that declares dense while shipping MoE tensors and asserts the guard fires.

#### The guard's stated premise does not reproduce on 8:1 GQA

The K-side bit floor is justified upstream by a claim that GQA *amplifies*
K-side quantization error — one K head serving many query heads, so its error
is said to compound with the ratio. That prediction was tested directly, as a
needle-in-a-haystack retrieval screen across the rotation-based K codecs at two
architectures (an 8:1 shared-KV Gemma4 and a 4:1 dense Qwen3), two contexts and
five needle depths, against a criterion declared before any cell ran.

**It does not fire.** Retrieval is perfect in every treatment cell, on both
ratios, and the precondition arm (`none`) passed, so the run is valid rather
than void. The 8:1 arm was in fact *less* perturbed than the 4:1 arm by greedy
digest — the opposite of the amplification prediction, and the direction the
rotation predicts, since these codecs decorrelate before quantizing. That is
reported as direction only: the two arms are not coverage-matched and their
head widths differ, so it is not a ratio.

Codec liveness was confirmed on device rather than inferred from the enum —
`decode_reads_packed_store` true for the K-only iso variant, no bf16 K seed
allocated, and per-cell dispatch counters equal to decode steps × the arch's
quantized-layer count exactly, against zero for `none` and `k8v8`.

**No guard was added and none is warranted**: a geometry-keyed screen built on
this would gate a variable with no measured effect. The upstream perplexity
figure behind the bit floor remains an unbacked external claim, and byte-exact
needle retrieval at 8:1 is incompatible with catastrophic degradation. The
floor itself stays — this measurement removes a *rationale*, not a rejection
rule, and re-litigating the rule needs a perplexity cell, which NIAH is not.

One limit, stated because it bounds the claim: NIAH excludes catastrophic
degradation outright but cannot resolve *subtle* degradation on the shared-KV
arch, where the greedy digest separated in too few cells to carry a null.
The verdict rests on retrieval rate and the dispatch counter, never on digest
divergence — the digest was not leaned on where it was measured blind.

**Two premises this section used to imply are false.** Sliding layers
short-circuit to bf16 before any K-codec arm, so **only full-attention layers
carry the codec** on a Gemma4 stack, and boundary promotion protects none of
the long-range ones — it lands on cacheless consumers and sliding layers. The
treatment on those layers is therefore total, not diluted; an argument that
reasons from a whole-stack layer count will get the dilution backwards. And
digest identity does not imply a void cell: see §"Class 3" for the same trap
measured from the other side.

---


## Codec fidelity — measured

Two CPU-only measurement surfaces in `rmlx-kv-quant`, both deterministic from
`TEST_SEED`, both inside `make model-check`. Neither needs a model snapshot or
the GPU. See `docs/TESTING.md` for how to run them and for the helper list.

### Incoherence — does the rotation do anything

`crates/rmlx-kv-quant/src/rotation_fidelity_tests.rs`.

The per-codec cosine gates all run on the i.i.d.-uniform LCG fixture, which is
already close to maximally incoherent (mean `mu = sqrt(d)·max|x_i|/||x||_2` of
1.72 at `head_dim = 128`, against a minimum of 1). A decorrelating rotation
cannot improve that and in fact makes it slightly worse — the Hadamard pushes
uniform toward Gaussian, whose `mu` is 2.87. **The LCG cosine gates therefore
carry no information about rotation quality: an identity rotation passes every
one of them.**

The outlier fixture is i.i.d. Gaussian with 4 of 128 channels scaled 20x, a
model of the persistent per-channel Key outliers reported by KIVI
(arXiv:2402.02750) and KVQuant (arXiv:2401.18079) at the magnitude ratio
reported for emergent outlier features (arXiv:2208.07339). Mean `mu` = 8.37
(p99 10.72).

The channel **count** is not from the literature — 4 of 128 is 3.1%, some 30x
denser than the reported emergent-outlier fraction. It is chosen so that every
affine group of 64 contains an outlier, the condition under which a
full-dimension rotation has something to recover across the whole row. Both
fixture parameters are swept rather than asserted: `mu` against the ratio is
monotone, and `mu` against the channel count rises, peaks near 2 channels, then
decays back to exactly the i.i.d. value once every channel is scaled (a pure
change of units, which `mu` is invariant to).

A block-`b` orthogonal transform can reduce `mu` by at most `sqrt(b)`: the peak
coordinate's block preserves its L2 norm, and a `b`-vector's max is at least
its norm over `sqrt(b)`. Measured on the outlier fixture at `head_dim = 128`:

| Family | Transform | Block | `mu` ceiling | `mu` reduction |
|---|---|---|---:|---:|
| `rot_k` / `RotK` | Walsh-Hadamard, full `head_dim` | 128 | 11.31x | **3.89x** |
| `iso3` / `iso4` | isoclinic SO(4), fixed quaternion | 4 | 2.00x | 1.38x |
| `planar3` | Givens, 16-entry codebook, per-pair search | 2 | 1.41x | 1.19x |
| `planar4` | Givens, 16-entry codebook, per-pair search | 2 | 1.41x | 1.15x |
| `rotor3` / `rotor4` | Cl(3,0) rotor sandwich, static per (layer, head) | 3 | 1.73x | 1.08–1.21x |

The rotor row is a range, not a point. Only the groups holding outlier channels
move `mu` — four rotors of 43 at `head_dim = 128` — so a single `(layer, head)`
table is a four-sample estimate. Across eight draws the reduction spans
1.0815x–1.2089x; the gate pins the weakest, so it describes the family rather
than one layer.

Only `rot_k` applies a full-dimension transform. The block-local families
deliver between 1.15x and 1.38x, well under their own ceilings, because their
rotations are fixed (iso, rotor) or fitted to reconstruction error rather than
to incoherence (planar). **This is not a defect in them** — they buy packing
efficiency, a different axis — but the "rotation" naming implies a capability
only one family has.

`rot_k` end to end, against the identical `affine q8 group=64` quantizer with
the Hadamard deleted:

| Fixture | rotated | unrotated | delta |
|---|---:|---:|---:|
| outlier channels | 46.95 dB | 36.06 dB | **+1.81 bits** |
| i.i.d. uniform (LCG) | 44.53 dB | 48.30 dB | **−0.63 bits** |

The rotation is worth most of two bits where it matters and costs two thirds of
a bit where it does not. Both directions are gated.

The gain is exactly `log2(peak_plain / peak_rotated)` over the affine group,
which is the same quantity the block ceiling bounds: a block-`b` transform can
buy at most `0.5·log2(b)` bits. That is what makes the 1.5-bit gate a
separation rather than a fitted number — it demands an effective block of 8 or
more. Substitutes measure 0.91 bits (the same Hadamard truncated to blocks of
4) and 0.47 (the iso quaternion), both rejected.

### The turbo family's missing rotation — what it is worth, and where

TurboQuant is named for a rotation this tree does not apply. The shipped codec
quantizes raw KV against a Lloyd-Max codebook with no decorrelating transform,
at any width, on either axis: `crates/rmlx-kv-quant/src/turboquant.rs` contains
no Hadamard or Walsh-Hadamard code at all. The on-disk layout tags say so too:
`TURBOSYM3_LAYOUT_TAG` / `TURBOSYM4_LAYOUT_TAG` are `tsym3_lloyd_3_3` /
`tsym4_lloyd_4_4`, naming the codebook the encoder applies. Because hydrate
dispatches on exact string equality, a tag is a stored format and any change to
one needs a `SCHEMA_VERSION` bump.

Whether adding the transform is worth its implementation cost was measured
*before* any transform code was written, by
`crates/rmlx-kv-quant/src/turbo_rotation_fidelity_tests.rs`. It holds the turbo
codec, the width and the group size fixed and moves only a full-`head_dim`
normalized FWHT in and out around the shipped CPU encoder. A test-side reference
rotation is a controlled ablation and needs no codec change — which is why the
gate could precede the implementation, and why "there cannot be an ablation
until the rotation exists" was wrong.

**The result inverts the family's cost structure.** Run the gate for the
figures; the ordering is what matters here:

| data shape | rotation is worth | implementing it costs |
|---|---|---|
| K-shaped (outlier channels) | order of a bit to two bits | reuses `maybe_pre_rotate_q_gpu` — close to wiring |
| V-shaped (i.i.d. Gaussian) | order of a hundredth of a bit | an explicit inverse transform after SV accumulation: the P2 kernel, four dequant kernels, both fused-QK kernels |

Turbo is primarily a **V** codec. So the measured payoff lands on the smaller
half of the family, and the expensive half is the half where the measurement
says there is nothing to recover. The payoff also shrinks monotonically as the
codebook widens, which puts what value there is at the *narrow* spellings, not
the wide ones. Anyone scoping the transform should read that as: K axis, narrow
widths, and a separate justification demanded for the V-side kernel work.

This also settles a confound the older per-codec cosine comparison could not:
holding group size fixed, the absent rotation — not the difference in scale
cadence — is the dominant term in turbo's reconstruction-error deficit against
a rotated sibling at the same width.

Scope it honestly: this is measured on a **model** of K-cache structure, not on
a K/V tensor captured from a forward pass. Read it as an estimate for a real
checkpoint; the honest check is a serving cell and has not been run. The
threshold is imported from `ROT_K_MIN_OUTLIER_GAIN_BITS` rather than retyped,
and the gate carries its own mutation check.

**Read the gate's verdict, not only its magnitudes.** It records a FAIL of the
magnitude criterion together with a PASS of the ordering, and both halves are
the result: the rotation helps on exactly the data it was predicted to help on,
and by less than the imported threshold at the widest codebook. An earlier draft
reported PASS from these same numbers by moving the verdict into a friendlier
bucket with the threshold untouched. Committing the criterion first protects the
numbers; it does not protect the conclusion.

### Rate-distortion — is the bit width delivering

`crates/rmlx-kv-quant/src/rate_distortion_tests.rs`.

Measured against the fixed-rate Lloyd-Max SQNR for the standard normal (Max
1960, Table I): 4.396 / 9.300 / 14.616 / 20.224 dB at 1–4 bits. That anchor is
**not** the rate-distortion bound (`6.02·b` dB) and assumes a quantizer matched
to the source, spending no rate on its scale. Every codec here stores a
per-group scale, so it can legitimately land above the anchor; the rate column
is what makes the dB interpretable. i.i.d. Gaussian fixture, 256 x 128:

| Codec | bits | measured | anchor | wasted bits | stored bits/value |
|---|---:|---:|---:|---:|---:|
| turbo | 2 | 7.232 dB | 9.300 dB | +0.344 | 3.00 |
| turbo | 3 | 14.956 dB | 14.616 dB | −0.056 | 4.00 |
| turbo | 4 | 21.630 dB | 20.224 dB | −0.233 | 5.00 |
| tcq | 2 | 7.232 dB | 9.300 dB | +0.344 | 3.00 |
| tcq | 3 | 14.956 dB | 14.616 dB | −0.056 | 4.00 |
| planar | 3 | 40.604 dB | 14.616 dB | −4.316 | 22.00 |
| planar | 4 | 36.724 dB | 20.224 dB | −2.741 | 22.00 |
| iso | 3 | 19.292 dB | 14.616 dB | −0.777 | 43.25 † |
| iso | 4 | 25.391 dB | 20.224 dB | −0.858 | 44.25 † |
| rotor | 3 | 20.432 dB | 14.616 dB | −0.966 | 8.75 |
| rotor | 4 | 26.551 dB | 20.224 dB | −1.051 | 9.75 |

No shipped cell is short of its anchor by more than 0.35 bits.

† **The iso rate is path-specific, and neither path is resident today.** 43.25
(iso3) / 44.25 (iso4) is the CPU `IsoBlocks` figure, which carries a per-group
quaternion sideband at `f32`. It
is the rate the V-only `iso3` / `iso4` stores *would* cost — those codecs decode
from the bf16 mirror, so `exit_prefill` builds them no store at all and they
measure byte-identical to `none` (§"Codec disposition", Class 2). `k_iso3/4` and
`iso3_sym/4_sym` do build a store: a GPU ring that does not carry the quaternion
— it is the constant `FIXED_QUAT` replicated per group, not data — and they sit
at **7.125** (iso3) / **8.125** (iso4) bits/value. See § iso3 "Memory truth".
Read against rotor's 8.75 without that distinction the table inverts the
comparison: on the ring path iso is the cheaper of the two. Distortion is
identical on both paths, so only the rate column is affected.

**The small-group-scale mismatch is not a loss.** Deriving the scale from the
maximum of 3 or 4 samples presents the codebook with data of standard deviation
≈ 1.47 rather than 1, but the small groups come out *ahead* of the anchor, not
behind it: the group maximum is a strong conditioning statistic, it is
reconstructed near-exactly by construction, and the two or three remaining
elements are then known to be smaller than it. The shortfall is at the other
end of the range — large groups at low bit widths (`turbo`/`tcq` at 2 bits,
one f32 scale per 32 values, +0.34 bits).

**The 3-bit and 4-bit widths of iso, rotor and planar cost byte-identical
storage** — already stated for iso under § iso3 "Memory truth" and for rotor in
the `rotorquant` module docs; what is new here is the consequence. All three
pack under the shared `32 / bits` vals-per-word convention, and at every shipped group size the word count is the same for
`bits = 3` and `bits = 4`. Each family therefore has one strictly dominated
width — same bytes, worse quality:

| Family | 3-bit | 4-bit | Dominated |
|---|---:|---:|---|
| iso | 19.29 dB | 25.40 dB | `iso3` loses 6.10 dB for nothing |
| rotor | 20.43 dB | 26.56 dB | `rotor3` loses 6.12 dB for nothing |
| planar | 40.60 dB | 36.72 dB | **`planar4` loses 3.88 dB for nothing** |

The planar direction is the surprising one, and it reproduces on the LCG
fixture: measured mean cosine 0.999956 for planar3 against 0.999901 for
planar4. Do **not** read the committed cosine floors (`planar_v3` 0.9989 against
`planar_v4` 0.9942) as corroboration — they are not commensurable. The v3 floor
is a local measurement minus 0.001; the v4 floor is an upstream README anchor
minus 0.001 and is not a measurement of this code at all. The real signal is the
5.5e-5 gap between the measured means, not the 4.7e-3 gap between those floors.
Per pair the larger element is pinned to the outermost
centroid, leaving the smaller on the grid `centroid / max_centroid`, whose
outermost gap is `(2.152 − 1.344)/2.152 = 0.375` at 3 bits and
`(2.718 − 2.052)/2.718 = 0.245` at 4 bits — only 1.5x finer, while the 16-angle
Givens search that must land *both* elements on centroids gets no larger. The
extra bit does not pay for itself. All three are pinned by
`byte_identical_bit_widths_leave_one_width_dominated`; fixing the packing or
the codebook is a separate change.

**TCQ's claw-back measures 0.000 dB.** See the trellis degeneracy note under
`K8VTurbo3Tcq` below.

---

## See also

- `docs/KV_CACHE.md` — flag surface, Qwen MoE PPL disaster, codec matrix.
- `docs/WEIGHT_QUANTS.md` — weight quantization families (separate from KV).
- `docs/SSD_TIER.md` — SSD spill / hydrate for long-context eviction.
- `docs/TESTING.md` — cosine, incoherence and rate-distortion gates; helpers.
