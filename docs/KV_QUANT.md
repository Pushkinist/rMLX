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

- The 17 Class 2 codecs build no packed store. Their decode reads the bf16
  mirror, so their resident KV and their greedy token ids equal `none`'s
  (§"Codec disposition", Class 2).
- No Class 3 codec decodes faster than bf16 on every architecture and context
  (§"Fused flash-decode over a quant store"). A 4-bit-K `mixed_*` codec wins
  on a dense stack and loses on others (§"The null was a bit-width result, not
  a context result").
- A Class 3 codec that holds less resident KV than bf16 is an opt-in memory
  setting. The iso and rotor codecs among them decode slower than bf16.

---

## Memory and bit-rate summary

Resident KV depends on the codec class and on the cache topology. The figures
below are stored bits per value per axis at `head_dim = 128`; bf16 is 16.0.

- **`none` and every Class 2 codec**, on a cache that went through a prefill
  bracket: 16.0 on both axes. `exit_prefill` builds no packed store.
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

The iso and rotor rates are the GPU ring after the first fused decode step
(§"Iso memory truth"). The rotor rates are with `--rotor-qjl off`. An MLX
affine group spends 32 bits on its scale and bias, hence the `32/group` term.

`mixed_*` and `rot_k_*` also keep a bf16 K and V mirror on a stack whose layers
share KV (`KvCache::shares_kv`). There the mirror adds 16.0 per axis, and the
codec holds more than `none`. On a stack without shared KV they hold the store
alone. `global_layer_store_plus_mirror_codec_sign_follows_shared_kv`
(`crates/rmlx-kv-quant/src/quant_tests.rs`) pins both signs.

The layer-adaptive boundary gives some layers a different codec
(§"Layer-adaptive overrides"). `rmlx info --list-cache-types` prints the
whole-stack figure per codec and per topology. Its producer is
`KvQuant::estimated_resident_bytes_per_layer`.

---

## TurboQuant calibration (`kv_calib.json`)

`rmlx kv-calibrate` writes a set of high-precision channel indices per KV head
to `kv_calib.json`. `rmlx serve` reads the file at model load. **No codec reads
it**: every codec encodes and decodes the same with or without the file.

### Generation

```bash
rmlx kv-calibrate /path/to/model --recipe turbo3
# Writes /path/to/model/kv_calib.json
```

`--recipe` defaults to `turbo3`. `--out` sets another output path. The
command reads the K/V projection weight tensors (F32, BF16 or F16). It
computes the L2 norm of each head across the input dimension. It keeps the
top-K indices per head, sorted ascending, as a `Vec<u32>`. The command runs on
the CPU and takes no Metal claim. The `head_budget`, `softmax_mass` and
`k_norm_proxy` recipes write `head_budgets.json` instead (§"Sparse attention").

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

1. **Discover.** When `<model>/kv_calib.json` exists, the `rmlx serve` loader
   reads `head_dim` from `config.json` and calls
   `rmlx_loader::discover_kv_calibration(model_dir, head_dim)`. It logs a
   `warn!` and continues without calibration in four cases: the file does not
   parse, `version != 1`, `head_size` differs from the model's `head_dim`, or
   `config.json` gives no `head_dim`.
2. **Attach.** The result goes on `ModelLoadConfig::calibration`. A
   `head_budgets.json` beside it attaches as `KvCalibration::head_budgets`.
3. **Not consumed.** `KvCacheBuilder::with_calibration` and
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

| Field | Semantics |
|---|---|
| `codebook.value` | V-side codebook for the layer: `2^bits` centroids in **strictly ascending order**, shared by every KV head of the layer. |
| `codebook` absent or `null` | The built-in Lloyd-Max N(0,1) codebook. |
| `codebook.value = []` | Parses, and returns `Error::Quant` at the first encode for the layer. |

No production path copies the override into the codec. `QuantV` honours a
codebook only when `QuantV::value_codebook` is set:

- The CPU encode passes it to `turbo_quantize_v_with_codebook`. 3-bit TCQ
  honours it; 2-bit TCQ always uses the built-in 2-bit codebook.
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

The default decode path dequantizes K to bf16, then runs
`scaled_dot_product_attention` over it. A fused-QK kernel instead reads the
packed K (codes, scales, rotation indices) directly. It writes pre-softmax
scores `[B, n_q_heads, 1, S_kv]`, so no dequantized K is written to memory.
After the softmax, V takes the split path: a matmul with the bf16 V.

### PlanarK fused-QK scope

`KvStorage::PlanarK` is the only storage with a fused-QK kernel.

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
(§"TurboFlash is off by default"). A smaller store moves fewer bytes per decode step. This section
states when that pays.

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

### Measured ε

| kernel | arch | `heads_per_kv` | ρ | slope ratio | ε | ceiling `1/heads_per_kv` |
|---|---|---|---|---|---|---|
| TurboFlash (`k8v4`) | Bonsai-8B | 4 | 0.262 | 6.37× | **0.041** | 0.250 |

At ε = 0.041 the store must hold fewer than 0.66 bits per value per axis. The
densest store in the tree is `tsym3` at 4.00 bits per value per axis
(ρ = 0.25), and no kernel decodes over it. Decode through these kernels is
**not bandwidth-bound**, so a smaller store does not buy time.

### Why ε is small — grid geometry

Each P1 kernel indexes its grid by **query** head and reads the KV head
`hq / heads_per_kv`: `turbo_flash_p1.metal`, `iso_flash_decode_p1.metal`,
`iso_flash_decode_symv_p1.metal`, `rotor_flash_decode_p1.metal`,
`rotor_flash_decode_symv_p1.metal` and `planar_flash_decode_p1.metal` under
`crates/rmlx-kv-quant/src/metal/`. So `heads_per_kv` threadgroups read the
same KV bytes. That caps the shell at **ε ≤ 1/heads_per_kv** before any cost in
the kernel body.

At `heads_per_kv ≥ 4` this ceiling is at or below ρ = 0.25, the densest store
in the tree. So on such an arch no fused decode over a current store can beat
bf16, even with a perfect kernel body.

The ceiling is necessary, not sufficient. The one kernel with limiter
counters, `turbo_flash_p1`, is **issue-bound, not memory-bound**: Integer and Conditional
Limiter 50.45% and Instruction Throughput 45.66%, against Last Level Cache
10.66%. The bf16 `sdpa_vector` encoders in the same capture show LLC 42.16%.
A grid indexed by KV head would read each KV byte once. It removes no issue
cost, so it does not reach parity (§"The decode ceiling").

### The decode ceiling — deleting the codec's arithmetic does not reach bf16

A fused packed-store kernel with **all** per-step decode math and all K-store
reads removed still decodes well short of `none`. Most of the gap is the
kernel shell: dispatch, grid geometry, barriers and the fixed per-layer cost of
a custom kernel. So no change to the decode math reaches parity: not a rotation
hoist, a narrower K store or a better codebook, alone or together.

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
Resident KV and greedy token ids equal `none`'s. The store is still the
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

- The eight iso and rotor codecs hold less than bf16 on every topology and at
  every `head_dim`. Their `feeds_bf16_*` arms are constants that do not read
  `shares_kv`. `iso_and_rotor_k_codecs_are_under_the_floor_at_every_geometry`
  pins the sign.
- `mixed_*` and `rot_k_*` hold less than bf16 on a stack without shared KV.
  They hold more on a shared-KV stack, where the bf16 mirror stays.

Decode: no codec in this class decodes faster than bf16 on every architecture
and context (§"Fused flash-decode over a quant store").

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
4 096-token prompt the script gives 0.010 for Qwen3.8-27B-mxfp8 and 0.221 for
Ternary-Bonsai-8B-2bit. So "measured at 4k" does not mean "measured where the
codec axis is near zero".

A large `kv_frac` is necessary, not sufficient. ε decides how much of the
bound a fused kernel collects. State `kv_frac` and the **K bit width** next to
every codec cell. Do not infer an effect from `kv_frac` alone.

Before you design a long-context cell, check the model's positional capacity:
`max_position_embeddings`, extended by a `rope_scaling` in `config.json` or by
`--yarn-factor`. `rmlx baseline` refuses a `--max-ctx` above that capacity.
On `--device gpu` it also refuses a prompt longer than the context ceiling,
unless `--allow-truncate` or `--max-prompt-tokens` is given.

#### The null was a bit-width result, not a context result

At 8-bit K, `mixed_k8g64_v4g64` does not decode faster than `none`, even at
the highest `kv_frac` of the release set. On Qwen3.8-27B at a 130 848-token
prompt it decodes slower. At 4-bit K, `mixed_k4g64_v4g64` decodes faster than
`none` on Ternary-Bonsai-8B at 32k and 63k prompts, with disjoint per-slot
ranges.

- The win is on the `Mixed` path, through MLX `quantized_matmul`. It does not
  transfer to a fused flash-decode kernel over the iso, rotor or planar ring
  (§"The decode ceiling").
- It is not a long-context effect: it holds at 32k and at 63k.
- On a low-`kv_frac` model the `Mixed` path loses at 4-bit K too, and it loses
  more at longer context. So a per-step cost that grows with `kv_seq` sits on
  that path and does not shrink with the store. No code site for it is known.

---

## `rotor_flash_decode` — fused MSL flash-decode over rotor-quant K

Fused flash-decode for `KvStorage::RotorKOnly3` / `RotorKOnly4`: QK over the
packed rotor K store + online softmax + bf16-V SV, in two Metal dispatches per
decode step. The rotor codec's Cl(3,0) K-decode runs **inside** the attention
inner loop, so no bf16 / f32 K is materialised and nothing restages through the
host.

**What it replaced.** `update_rotor_k_only` called
`QuantRotorK{3,4}::dequant()` on every decode step — a full-prefix **CPU** rotor
decode into a `Vec<f32>` plus a re-upload. That is O(seq) host work per token
with the GPU idle, and it is what pinned the K-only rotor family in the
"Tier 3 — CPU-bound" bucket (0.05–8.8 TPS, see `docs/models/bonsai/27B/rMLX.md`).
The store is now GPU-resident (`storage::QuantKGpuRing`) and the kernel reads it
directly.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_msl.rs` — Rust dispatcher,
  header builder, dispatch counters.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_p1.metal` — pass-1 body
  (one body for **both** bit widths).
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — codec-agnostic
  pass-2 log-sum-exp merge, shared with `planar_flash_decode`.
* `crates/rmlx-kv-quant/src/storage/quant_k_gpu_ring.rs` — `QuantKGpuRing`, the
  GPU-resident packed ring (codes / per-group scales / per-token L2 norms) with
  paged growth and CPU-prefix seeding. Codec-agnostic: it is told `n_groups`
  rather than deriving it, and is shared with the iso K stores.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_rotor_k_fused` —
  dispatch site.

### Bit width is a header parameter

`bits ∈ {3, 4}` arrives via the header (`RF_BITS` / `RF_MASK`) alongside the
matching Lloyd-Max codebook, so one `.metal` body serves both variants — the
3-bit codes unpack at `shift = e*3, mask = 0x7`, the 4-bit at `e*4, 0xF`,
matching `rotorquant::{unpack_group, unpack_group_4bit}`. Selection is explicit;
any other `bits` is an `Err`, never a silent fallback to the wrong unpack width.

### Reusable K-decode half

The per-lane rotor decode is emitted into the **header** as the MSL function
`rf_decode_k_lane(codes, scales, norms, rotors, tok_idx, n_groups, lane)` rather
than inlined into the body. A quantized-V flash kernel needs the identical
K-side decode and can call it unchanged. (Bodies in this repo are statement
sequences spliced inside a generated kernel signature, so a body cannot define
functions — the header is the only place a shared function can live.)

### Gate

No env var and no CLI flag: the path is on whenever it is applicable. Gates, in
order — device is GPU, storage is a rotor K-only variant, the store does **not**
carry QJL, `q_seq == 1`, `b == 1`, `head_dim` is a power of two and
`<= ROTOR_FLASH_HEAD_DIM_MAX` (512). Any miss falls through to the legacy CPU
dequant path.

**QJL.** The optional 1-bit QJL residual (`--rotor-qjl on`, opt-in) is a
per-token back-projection through a dense `[head_dim, head_dim]` matrix.
Reproducing it in the flash inner loop would mean reading that whole matrix per
token per threadgroup — far more bandwidth than the kernel saves — so a
QJL-carrying store keeps the CPU dequant path. **`--rotor-qjl off` is required
to reach the kernel.** The gate reads the *store's* sticky QJL decision
(`use_qjl()`), not the live global toggle: the codec fixes QJL at first append
and never adds or drops the sideband mid-stream, so a toggle flipped afterwards
must not change how existing bytes are read.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `KvStorage::RotorKOnly3` / `RotorKOnly4`, QJL off, `b == 1` | **YES** | GPU ring + `rotor_flash_decode_sdpa`. |
| `KvStorage::RotorKOnly{3,4}`, QJL on | NO | Kernel cannot reproduce the QJL residual. |
| `KvStorage::RotorKOnly{3,4}`, `b > 1` | NO | Ring stride does not interleave batch — see below. |
| `Rotor{3,4}Sym` | NO (this kernel) | Both axes are rotor-quantized; they decode through the all-quant sibling `rotor_flash_decode_symv` instead (see below), which reads V from its own packed ring rather than a bf16 mirror. |
| `RotorK{3,4}Asym` | NO | V is affine-quantized (TurboQuant), not rotor; no fused kernel yet, so it keeps the bf16 decode path. |

### Ring eligibility is passed down, not inferred

`QuantKGpuRing` is only built for the codecs that can actually read it. The rotor K
GPU encode takes a `RingFeed` from its caller, one of three modes:

- **`Maintain`** — feed the ring **and** push a CPU block. Used by prefill
  (`update_rotor_k_only`) and the non-fused decode fallback, which `dequant()`
  the whole prefix on the same step and so need the block immediately.
- **`MaintainRingOnly`** — feed the ring **without** pushing a CPU block: a
  **ring-only tail**. Used by the fused decode entry
  (`rotor_k_only_gpu_append`). The flash kernel reads the ring, never the
  block, so skipping the per-step host download (`rotor_gpu_outputs_to_cpu`) is
  the win; `shape[2]` still advances so the ring and the attention length stay
  in lockstep.
- **`Skip`** — clear the ring. Used by the sym/asym mirrors, and as the `b > 1`
  fallback of a ring-only feed (which reverts to the block path).

A ring for a non-eligible codec is not free — one `u32` code word plus one
`KV_SIDEBAND_DTYPE` scale per group and one such norm per token, so
`capacity * kv_h * (n_groups * (4 + sideband) + sideband)` bytes per layer,
growing with context (order of a few hundred MB across a 36-layer model at 4k)
— and nothing would ever read it.

**Invariant: the CPU `blocks` track `shape[2]` exactly, or the GPU ring holds
the tail and the blocks are rebuilt from it on demand — never a silent gap.**
Two regimes satisfy it:

- *Blocks-authoritative* (Maintain / CPU `append` / SSD hydrate): `blocks` cover
  the full `shape[2]`; the ring, when live, mirrors them and re-seeds from
  `blocks` (`seed_from_cpu`) after a drop (`reset()`, a CPU `append()`).
- *Ring-only tail* (fused decode): `blocks` freeze at the prefill prefix while
  the ring carries the decode tail, so `blocks` trail `shape[2]`. The blocks are
  rebuilt from the ring on demand — `synced_rotor_k_blocks` reconciles them at
  every consumer boundary (`dequant`, and the SSD-spill / prompt-cache clone
  `try_deep_clone`).

`truncate_to` **keeps** the ring (it does not `clear()`): it lowers `shape[2]`
to `n` and leaves the ring's `[n, prev)` capacity to be overwritten by the next
append, exactly like the flat GPU-buffer codecs (`QuantK` / K8V4 just lower
`shape[2]`). This preserves a ring-only decode tail up to `n` across the
speculative-decode partial-accept rollback; clearing it there would discard the
tail (the only copy of `[frozen_prefix, n)`) and abort the next `dequant`. Both
mutators reconcile: the block-path append (`materialize_*_ring_tail`)
materialises any pre-existing ring-only tail into `blocks` before pushing, so
`blocks` stay a contiguous prefix. That block path is **live at `b == 1`**, not
just a `b > 1` fallback: the fused decode entry is gated on `q_seq == 1`, so any
multi-token append on a cache that has already run a fused decode step — a
speculative verify chunk, or a continuation turn's prompt tokens against a warm
cache — falls through to the legacy `update_*` entries, which pass
`Maintain` / `Skip`. The readback it pays is not additive: every block-path call
site dequantizes the whole prefix on the same step, so `dequant` would take the
identical readback if `blocks` were left short.

**How reachable is the divergence?** Narrowly, and it is worth knowing why
before writing a repro. The state needs a cache whose ring is live *and* whose
CPU blocks were dropped — which only the fused decode path creates — followed by
a decode-mode `update()` with `q_seq > 1` on **that same cache**. A warm
prompt-cache continuation looks like it should qualify (gemma4's `is_prefix`
flush appends the tail through decode-mode `update` with no enter/exit
brackets), but it does not: that tail runs against a prompt-cache *clone*, and
`try_deep_clone` materialises any ring-only tail into blocks and hands back a
store with no ring at all, so there is nothing to diverge. What does qualify is a
speculative verify chunk — a multi-token decode step on a live cache. So the
codec can serve a normal single-request generate loop indefinitely without
meeting it, which is why it surfaced from a truncation proof matrix rather than
from serving, and why the guards above are the gate rather than a serve-time
smoke test.

**Every block push reconciles, and every reader derives its count from the same
place.** Two holes in that used to be reachable and are now closed. The iso-V
GPU-encode append (`QuantIsoV::append_gpu`, the V side of the legacy
`update_iso_v` / `update_iso_sym` entries) pushed a CPU block without touching a
live ring, leaving the ring stale *and* the blocks short — it now calls
`QuantIsoV::reconcile_ring(device, RingDisposition::Drop)`, which takes the
ring's prefix back and then drops the ring, matching what the CPU `append` does
by clearing. That is one body, shared with the `kvcache` append helpers, which
pass `RingDisposition::Keep` because their `sync_ring` decides the
ring's fate immediately after — the disposition is a parameter precisely because
it is the only thing the two callers disagree on. The 4-bit V
side had the same hole in a separate ring-unaware helper; it went with the
storage collapse, and the fused 4-bit caller now goes through the ring-aware
`iso_gpu_append_into_v_blocks` (it also stored its block head-major, unlike
every other iso append). On the read side `QuantIsoK::dequant_gpu` and
`QuantIsoV::dequant_gpu` counted `self.blocks` directly while their CPU siblings
counted the ring-reconciled list, so a legitimate ring-only tail was rejected as
a blocks-vs-shape disagreement (`dequant_gpu: actual_total=... !=
declared_total=...`); both now start from `synced_iso_v_blocks`, which borrows
(costs nothing) whenever the blocks already cover `shape[2]`.

**Row vs. sequence units.** Every rotor/iso K and V store's per-append
`RotorBlocks` / `IsoBlocks` carries `n_tokens` counting **rows**
(`b * kv_h * seq_of_block`), not sequence positions, but `truncate_to(n)`
takes `n` as a **sequence** target. Deciding which leading blocks to keep must
therefore compare cumulative `n_tokens` against `n * b * kv_h`, not against
`n` directly — at `kv_h > 1` (or `b > 1`) a raw comparison undercounts and
drops blocks that should have been kept, landing the store in exactly the
forbidden gap described below. This was invisible at `b * kv_h == 1` (rows and
sequence positions coincide), which is how the bug shipped and how it stayed
latent until a `kv_h > 1` truncation path was exercised (#284).

**Blocks are not a truncation alignment.** A block spans one whole append, and a
speculative partial accept cuts *inside* the verifier's `K + 1`-token chunk.
Keeping only the blocks that fit whole throws the accepted prefix away with the
rejected tail and leaves `blocks` covering fewer rows than `shape[2]` — the
forbidden gap below, recoverable only when a ring happens to hold the same
prefix. It is not recoverable on the CPU append path (a QJL-carrying rotor K
store, or a `Device::Cpu` run) nor after a `Skip` feed cleared the ring, which is
what `update_rotor{3,4}_sym` and the asym entries do on every append; the store
then aborts the request with
`"rotor K store: CPU blocks cover N tokens but shape[2] needs M"`. The planner
therefore **splits** the trailing block, cutting every per-row buffer — codes,
per-group scales, per-group quaternions, per-token norms, and the rotor QJL
sideband — to the accepted row count.

**The split is `b == 1` only, and that is a correctness bound.** The bound is
*inside* one block. A block's rows run `[B, S_block, kv_h, D]`, so batch element
1's rows all sit after batch element 0's, and `BlockRows::retain_rows` keeps a
**row prefix**. At `b > 1` a row prefix is not a sequence prefix: a cut to
`keep_seq` positions would keep every one of batch 0's rows and none of batch
1's, silently dropping one batch element's tail instead of cutting both at the
same position. So the planner drops the block there and lets the reconciliation
guard report the gap. `sdpa::rotor_flash_shape_ok` refuses `b != 1` separately,
because the GPU ring's per-step stride does not interleave batch — which is also
why a `b > 1` store never has a ring to rebuild from. Pinned by
`quant_rotor_v_tests::quant_rotor_v3_truncate_at_b_gt_1_stays_loud`.

Reading the *concatenation* of the blocks used to be a second, independent bound
and is no longer one. Every store ended `dequant` with
`seq_layout::transpose_seq_heads` over the concatenation, reading it as one
`[B, S_total, kv_h, D]` run — but each block is only `[B, S_block, kv_h, D]`, so
at `b > 1` the concatenation interleaves batch elements and any store holding
more than one block decoded scrambled (measured on the `b = 2`, `kv_h = 2`,
`head_dim = 96`, 5-position fixture: **960 of 1920 elements** disagreed with a
one-block store, while the `b = 1` control matched to the last bit). Every
block-accumulating CPU store now calls
`seq_layout::transpose_chunked_seq_heads`, which reorders each block at its own
sequence offset and is exactly the old whole-buffer reorder when `B == 1`. The
per-store proof is `*_two_block_decode_matches_one_block_at_b_gt_1` — one per
store, all thirteen, each over `(b, kv_h) ∈ {1,2} × {1,2}` — with the index-math
oracle in `seq_layout_tests`.

The **GPU** readers took the same reorder on the way in rather than the way out.
`QuantIsoV3::dequant_gpu` / `QuantIsoK3::dequant_gpu` had the identical defect
(they reshaped the kernel's flat output as one `[B, S, kv_h, D]` run), and it is
the multi-block case that reaches them: every `*_sync_ring` clears the ring at
`b != 1`, so `synced_iso_v_blocks` returns a borrowed multi-block list there.
Both now build their kernel inputs through `iso_kernel_inputs_head_major`, which
places each token row at its head-major position via
`seq_layout::head_major_token_order`; the iso dequant kernel is per-token
positional, so the flat result is already `[B, kv_h, S, D]` and the trailing
reshape/transpose is gone.

**Where the bound still stands.** Two readers refuse `b != 1` (with `S > 1`)
rather than reorder:

* The **flat GPU buffers** of the turbo / planar / affine stores (`QuantV`,
  `QuantKTurbo3/4`, `QuantPlanarK/V`, `QuantK`) and the gated `QuantIsoV3` GPU
  mirror. Each is a run of `[B, S_chunk, kv_h, D]` chunks written at
  `prev_seq * words_per_step` with `b` folded into the stride, so the prefix
  carries no chunk boundary to partition on.
* `QuantK`'s **CPU** `codes` / `scales`, which are one flat append-only pair with
  no per-append boundary recorded. The refusal is deliberately wider than the
  defect — a single-append `b > 1` store would read correctly and is refused
  anyway — because `b > 1` reaches no production path today and the boundaries
  are not recoverable after the fact. Lifting it needs a recorded per-append
  sequence length on the store.

All eight rotor/iso K and V codecs share one crate-internal planner,
`truncate_plan` in `rmlx-kv-quant/src/storage/mod.rs`, plus a `BlockRows`
implementation per block type, so the unit conversion and the split are defined
once rather than re-derived per codec. Tests: `storage/truncate_plan_tests.rs`
(planner + a payload-carrying fake block, including the `b > 1` refusal and the
non-row-divisible refusal), and one store-level round trip per block type in
`quant_rotor_k_tests.rs` (`RotorKBlocks`, QJL sideband on),
`quant_rotor_v_tests.rs` (`RotorBlocks`) and `quant_iso_v_tests.rs`
(`IsoBlocks`, quaternion sideband).

**Scope — every CPU-side store now cuts, and every one of them is loud.** The
same planner drives the turbo, planar and affine stores. `TurboBlocks` and
`PlanarBlocks` gained `BlockRows` implementations, `QuantV`, `QuantKTurbo3/4`,
`QuantPlanarK` and `QuantPlanarV` gained `truncate_to`, and `QuantK` — whose CPU
payload is one flat append-only `codes`/`scales` pair, not a block list — cuts
that buffer to the leading `n` sequence positions. `KvStorage::truncate_to` no
longer contains a bare `shape[2] = n` in any arm: `K8V4`, `K8V8`, `Planar`,
`PlanarK`, `TurboSym3/4`, `K8VTurbo2/3`, `K8VTurbo3Tcq`, `K8VTurbo2Tcq`,
`IsoV3/4`, `RotorV3/4` and both axes of `RotorKAsym3/4` all delegate to a
store-level `truncate_to`, so a codec no longer truncates its two axes with
different semantics.

Before that, those stores' blocks **over**-covered `shape[2]` after a
truncation, the next append stacked on top, and
`QuantV::dequantize_choice`'s `out.resize(total, 0.0)` (and
`QuantPlanarK::dequantize_choice` via `transpose_seq_heads`, which reads only the
first `b * s * kv_h * d` elements) silently kept the **rejected** speculative
tokens while discarding the correction — wrong attention, no error.

Those silent fix-ups are gone. Every CPU dequant path now compares what its
blocks decoded to against `prod(shape)` and returns
`"CPU blocks decode to N elems but shape [...] implies M — refusing to zero-pad /
truncate"` on any mismatch, in **both** directions. That was not optional: the
planner deliberately refuses some cuts (`b > 1`, a block whose row count is not a
whole number of sequence positions, and for `QuantK` a target landing inside a
128-element q8 group, where one scale covers the whole group and the f32 source is
gone), and each refusal leaves the store deliberately inconsistent. Without the
check, a refusal on the turbo stores would have zero-padded and on the planar
stores would have panicked with an out-of-range index inside
`seq_layout::transpose_seq_heads`.

**Where this is actually observable — the bf16 decode seed gates it.** Two
independent things have to be true for the cut to change an answer, and getting
only the first one right leads to the wrong conclusion.

*First*, the store's CPU payload has to be live rather than its flat GPU mirror.
The GPU half needs no cut and gets none: its dequant slices `[0, shape[2])` and
its next `append` writes at `prev_seq == shape[2]`, so lowering `shape[2]`
already makes the rejected region overwritable.

*Second — and this is the binding constraint — the codec store has to **exist and
be read** after the truncate.* On a normal serve it does neither. `exit_prefill`
materialises the bf16 `decode_fp16_{k,v}` seed for every quant whose
`feeds_bf16_k_at_decode()` is true (`quant.rs`), which covers `K8V4`, `K8V8`,
`Planar`, `Planar3`, `PlanarK`, `K8VTurbo2/3`, both TCQ variants and
`TurboSym3/4` — i.e. every store this section is about. From then on each
quantized `update_<codec>` early-returns into `update_decode_fp16` at its first
line, so the store is not consulted at decode-read time; and because it is not,
`exit_prefill` no longer builds it at all for those codecs
(`KvQuant::materialises_packed_store()`, `docs/KV_CACHE.md` §9.6 F3). A plain
GPU serve therefore **cannot distinguish a correct cut from a no-op cut** —
there is nothing there to cut — including with `--kv-quant k8vturbo3`, whose
forced-CPU `QuantV::append` sits below that same early return.

So the live paths are the ones with **no** bf16 seed:

- **A hydrated cache.** `KvCache::from_storage` leaves `decode_fp16_k: None`, so
  the codec arm runs on every decode step — the store's blocks *are* the cache.
  The hydrated prefix arrives as a single block, so any trim inside it is a
  mid-block cut. This is the path the round-trip tests in
  `rmlx-kv-ssd/src/hydrate_tests.rs` drive.
- **Any cache that never bracketed a prefill**, and so never reached
  `exit_prefill` to be seeded.

The device is not one of them. The `exit_prefill` gate has no device arm and
neither do the `feeds_bf16_*` predicates, so a `Device::Cpu` run that brackets a
prefill lands on the same mirrors and the same absent store as a GPU one — which
is what the CPU-device sweep in
`warm_ttft_cross_codec_tests::exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`
asserts over every codec.

The store is also still read *without* a decode step in two places, and they
matter for the codecs that still have one — the K-only and fused-symmetric
families, `Mixed` / `RotK`, and any hydrated store-backed cache: the
SSD spill (`write_quant_k` / `write_quant_v` serialise `blocks` and report
`shape[2]`) and the prompt-cache snapshot (`try_deep_clone`). An uncut store
spills a header claiming more tokens than its bytes hold, which is how the defect
propagates from a serve into the hydrated cache that later reads it. For the
bf16-mirror family that route is closed at the source: there is no store on the
seeded path to spill.

An earlier revision of this section named `--kv-quant k8vturbo3` on a plain serve
as the cheapest observable cell. That was wrong for the reason above — device
routing is not the same question as whether the store is read — and is corrected
here rather than left for a reader to re-derive.

`TurboBlocks` and `PlanarBlocks` carry no `n_tokens`, so their row count comes
from `original_shape` — and that field's axis order is **not** consistent across
producers: the CPU append paths record the sequence-major chunk shape
`[B, S_block, kv_h, D]` while the SSD hydrate paths record the store's head-major
`[B, kv_h, S, D]` over the same sequence-major bytes. Only the product is ever
read back (`turbo_dequantize` / `planar_dequantize` use it purely as an element
count), so `storage::block_rows` multiplies the leading three axes rather than
naming one, and a split records the geometry it actually produced —
`[1, 1, rows, width]` — instead of guessing which axis the caller meant. Pinned
by `cpu_block_truncate_tests::quant_v_truncate_reads_rows_from_the_shape_product`.

**Truncation is monotone-decreasing.** All six clamp the target to the store's
current `shape[2]` (`storage::clamp_truncate_target`). `n > shape[2]` is
reachable, not hypothetical: a store-backed cache whose codec also keeps a bf16
mirror (`Mixed` and the K-only families) advances `KvCache::offset` on paths the
store does not follow, so a speculative rollback into the decode window arrives
with a target past the store's own fill. Raising `shape[2]` to meet it invents coverage no payload
backs — the dequant reads past the blocks and the SSD spill persists a header
claiming more tokens than its bytes hold.

The rotor / iso stores deliberately do **not** clamp, and the reason is that they
do not need to, not that clamping would cost them anything: a ring-only tail
spans `[blocks_coverage, shape[2])`, strictly below `shape[2]`, so a
`min(n, shape[2])` could never discard it. What makes the asymmetry safe is that
those stores already abort loudly on an over-long target —
`synced_rotor_v_blocks` / `synced_iso_v_blocks` size their ring readback from
`shape[2]` and return `Err` when the ring cannot cover it. These six have no ring
and no such guard, so the clamp is the only reading that keeps
`shape[2] == payload coverage` true.

One consequence, named rather than hidden: for `n > shape[2]` the mixed arms now
leave the two axes of one codec at different lengths — `IsoV3`, `IsoV4`,
`RotorV3`, `RotorV4` (affine K clamps, codec V does not) and `RotorKAsym3/4`
(rotor K does not, affine V does). It surfaces on spill, where the layer geometry
comes from the K shape while the V payload is written raw; the reconciliation
guard on the unclamped side is what reports it.

`KvStorage::reset` carried the same defect one screen above `truncate_to` — a
bare `shape[2] = 0` on exactly these six store types, leaving the payload
covering the sequence just discarded. Every arm now delegates to the store's own
`truncate_to(0)` / `reset()`; the GPU buffers are still kept for reuse.

**The `Mixed` arm truncates, it no longer resets.** `Mixed` was the one storage
whose `truncate_to` arm answered by dropping the whole quant state
(`state.reset()`), on the reading that mlx-lm-tq's `is_trimmable` returns
`False` for it. That is not the same contract: `KvCache::truncate_to` goes on to
set `self.offset = n`, so the cache reported `n` positions and held none. The
store does not need the reset — it is a capacity buffer that grows in `STEP`
increments, with `MixedKvState::offset` as the fill marker
(`update_and_fetch` writes the new rows at `[offset .. offset + seq)` via
`write_at` and hands back `slice_seq_to(offset)`), so rolling the marker back to
`n` **is** the truncation: rows `[n..]` become dead capacity the next append
overwrites. The bf16 K/V mirror a shared-KV producer keeps
(`decode_fp16_{k,v}`) is capacity-allocated against the same offset and follows
for free.

The reset surfaced two ways, and only one of them was loud. A multi-token
forward — a speculative verify block — attends `seq` keys against a mask the
caller sized from the reported offset, which is the opaque
`add: [broadcast_shapes] Shapes (1,kv,rep,seq,seq) and (1,1,seq,n+seq) cannot be
broadcast` that Gemma4-assistant MTP hit under `--kv-quant mixed_k8g64_v4g64` on
its first partial-accept round. A single-token decode needs no mask, so it
silently attended the current token alone with no error anywhere. Pinned by
`kvcache/shared_source_tests.rs::mixed_truncate_to_keeps_the_prefix_it_was_told_to_keep`,
which asserts through the store (an additive mask sized to the kept prefix) and
not through the surfaced share — the bf16 mirror is rebuilt from
`KvCache::offset` and spans the kept prefix either way, so an assertion on it
cannot tell the two behaviours apart. Its oracle is a second cache prefilled to
exactly the kept length and never over-filled: the two must decode the next token
to the same output. Every prefix row carries a distinct position-dependent value,
so a store that kept the right *number* of rows with the wrong contents fails it
too — with a constant fill the same defect moves the output by 2.4e-7, at the
f32 noise floor, against 9.7e-4 with the ramp.

**And an over-long target is loud.** `n > offset` is the same defect in the other
direction and this store cannot absorb it: `offset` *is* the coverage, so unlike
the turbo / planar / affine stores there is no larger `shape[2]` to clamp down to.
`KvCache::truncate_to` sets `offset = n` regardless of what the store did, and its
`debug_assert` is compiled out of `release-perf`, so accepting it quietly would
leave the cache reporting positions no payload backs. `MixedKvState::truncate_to`
keeps its fill and emits an `error!` naming both numbers — loud like the rotor /
iso stores, but at the truncate rather than at the next read, because nothing
downstream of it would notice.

Tests: `storage/cpu_block_truncate_tests.rs` — the partial-accept round trip per
store (`QuantV`, `QuantKTurbo3`, `QuantKTurbo4`, `QuantPlanarK`, `QuantPlanarV`,
`QuantK`) at `kv_h` 1 and 3; the `b > 1` and q8-group refusals; the zero,
exact-length and past-the-end targets; `KvStorage::reset`; and a five-arm
`KvStorage::truncate_to` dispatch case (`K8V4`,
`TurboSym3`, `TurboSym4`, `Planar`, `PlanarK`) that decodes both axes and
compares against reference stores.

Enum-arm coverage is **partial and stated as such**. Those five arms cover all
six *store types*, so a regression in any store's `truncate_to` is caught — but
ten arms are unpinned because nothing in the workspace drives
`KvStorage::truncate_to` on them: `K8V8`, `K8VTurbo2`, `K8VTurbo3Tcq`,
`K8VTurbo2Tcq`, the K axis of `IsoV3` / `IsoV4` / `RotorV3` / `RotorV4`, and the
V axis of `RotorKAsym3` / `RotorKAsym4`. `reset` is thinner still: only its
`K8V4` arm is pinned, out of 28 (one per `KvStorage` variant). Reverting any of the rest to a bare `shape[2] = n` leaves
the suite green; catching that is review's job, not the suite's.

Every oracle is a reference store
built from only the retained tokens, sharing no arithmetic with the truncation
logic — deliberately not a recomputation of the cut via `block_rows`, which
would pass any mutation scaling the cut and the reading together.

End-to-end on the live path: `rmlx-kv-ssd/src/hydrate_tests.rs` spills a
256-token cache, hydrates it, truncates to 200 (mid-block), appends a 2-token
correction and checks the decoded V — for `K8VTurbo3` and `Planar`, at `kv_h` 1
and 2. It asserts as a **premise** that the hydrated cache carries no bf16 decode
seed, so it cannot pass vacuously on the frozen-store path. The retained prefix
is compared against a decode of the same store taken before the cut; the
correction against its own raw f32 source.

**Real-serve reachability audit (per #284).** `KvCache::truncate_to` has three
production callers: prompt-cache partial-prefix trim
(`PromptCacheEntry::truncate_kv_to`), SWA context handling, and
speculative-decode partial-accept rollback (MTP / DFlash / Eagle3 / the
gemma4-assistant self-speculative path). Whether any of them can reach a
`kv_h > 1` rotor/iso store depends on the arch:
- **Bonsai (`Qwen3ForCausalLM`, `kv_h = 8`)** — **reachable.** Its prompt-cache
  `ReusePolicy` is `ExactOnly`, so the partial-prefix trim never fires, but the
  speculative path does. Two reasons, both arch-generic:
  1. `SpeculativeGenerator::from_snapshots_with_id` takes the `two_model`
     branch — `SpeculativeDispatcher::load_speculative` — for any
     `--draft-model` whose `config.json` declares a registered architecture
     (`docs/SPECULATIVE.md` § "Which drafter a snapshot is"), so a bare
     `--draft-model <full model>` or a `profiles.toml` `draft_model` reaches
     it. That branch calls plain `load_model` on both sides and
     `spec_generate_greedy_cached` builds caches from `num_hidden_layers()`
     with **no arch check**, rolling back through `KvCache::truncate_to`.
  2. No drafter gates the **verifier** arch at all. The
     `"Qwen3_5MoeForConditionalGeneration"` strings in `speculative/mtp.rs` and
     `speculative/dflash/mod.rs` are error-message text, not architecture
     guards. The KV-quant fallback those sites take when `kv_quant_override`
     is `None` is `DEFAULT_KV_QUANT`, which consults no arch at all.
- **Gemma4 (e.g. e4b, `kv_h = 2`)** — reachable twice over: its
  `ReusePolicy::Partial` performs the trim on a real partial-prefix cache hit,
  and gemma4-assistant self-speculative decode also rolls back via
  `truncate_to`.

The fix is proven at the codec level (all eight `*_tests.rs` files, `kv_h`
values 1 and 4) and at the full `KvCache::update` / `KvCache::truncate_to`
dispatch level (`kv_cache_truncate_iso3_kv_h_gt_1_path` in
`rmlx-models/src/kv_cache/tests.rs`, `kv_h = 4`) rather than via a live HTTP
trigger. `truncate_plan` reads `shape[1]` directly with no per-arch or
hardcoded head-count branch, so `kv_h = 4` and Bonsai's real `kv_h = 8`
exercise the identical code path — no arch-specific behavior exists to miss.

The mid-block split has a **wider** reachability than the unit-conversion bug it
sits next to: it fires at `kv_h == 1` too, since a cut inside a block is about
block boundaries, not head counts. Any rotor or iso codec with speculative
decoding on reaches it on the first partial accept — including the Bonsai cell
above, where a `--kv-quant rotor3_sym` verifier takes the `Skip`-feed legacy
append (no ring to rebuild from) on every `q_seq > 1` verifier forward.

The forbidden state is `blocks` short of `shape[2]` with **no** ring to supply
the tail: `dequant()` would zero-pad the gap (silently wrong attention) and an
SSD spill would persist a truncated store. That state is rejected **loudly**
(an `Error`, never a `debug_assert` — those compile out under `release-perf`):
`synced_rotor_k_blocks` at the codec and `ensure_rotor_k_blocks_cover_shape` at
the SSD serialization boundary both refuse it rather than fabricate zeros.

**`b > 1` skips.** The ring's per-step stride is `kv_h * n_groups` and does not
interleave batch, so a batched chunk cannot be laid into it (the encode carries
`b` × the span). A `MaintainRingOnly` feed with `b > 1` therefore falls back to
the block path (which handles `b > 1` correctly and keeps the CPU blocks the
source of truth) — it must not error, since a batched rotor cache worked before
this kernel existed. Per request the batch dim is fixed, so a `b > 1` cache
never builds a ring-only tail to lose. Both the append (`rotor{3,4}_sync_ring`)
and the dispatcher (`rotor_flash_shape_ok`) gate on it.

### Arch reachability

Keyed off codec + shape (`head_dim`, `kv_heads`, `bits`), never an arch name —
so any arch that routes a rotor K-only cache through `KvCache::update_and_sdpa`
reaches it.

| Arch | Routing | Reachable? | Why |
|---|---|---|---|
| Bonsai (`Qwen3ForCausalLM`) | `update_and_sdpa` | **YES** | head_dim 128. Measured 78 dispatches / 8 tokens. |
| medgemma (`Gemma3ForConditionalGeneration`) | `update_and_sdpa` | **YES** | head_dim 256, no cross-layer KV share. Measured 28 dispatches / 8 tokens. |
| Qwen2 / Laguna / bitnet / Qwen3-VL-MoE | `update_and_sdpa` | **YES** (by shape) | Same entry point; subject to the shape gates. |
| Any arch with cross-layer KV sharing (e.g. `Gemma4ForConditionalGeneration`) | `update_and_sdpa_shared_source` (cross-layer KV share) | **YES** | The producer runs the same fused arm a non-sharing model runs and reports `SharedKv::Store`; each consumer layer re-enters the same kernel over that store via `KvCache::sdpa_shared`. No bf16 K is materialised. Previously this path had no fused arm at all and every shared-KV model fell back to the O(seq) CPU dequant. |
| Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`) | rejected at `cache_type::validate_resolved` | NO | Contract A.y — sub-4-bit K on Qwen MoE is a PPL disaster; the cache is never built. |

### Performance posture

4k prompt, `release-perf`, `--rotor-qjl off`, decode TPS (median of 3+ runs).
"Before" is the same binary minus this change, so the delta is the kernel alone
(the QJL flag is held constant across the pair).

| Model | Codec | Before | After | Gain |
|---|---|---|---|---|
| Bonsai-8B (Qwen3, D=128) | `k_rotor3` | 1.34 | **17.0** | 12.7× |
| Bonsai-8B | `k_rotor4` | 1.36 | **15.9** | 11.7× |
| medgemma-4B (Gemma3, D=256) | `k_rotor3` | 7.37 | **51.8** | 7.0× |
| medgemma-4B | `k_rotor4` | 7.34 | **52.1** | 7.1× |

Against the `--rotor-qjl on` baseline — which was the default when these cells
were taken and is now opt-in — the same cells move 0.66 → 17.0 (Bonsai, 26×)
and 2.35 → 51.8 (medgemma, 22×).

Bonsai is a noisy measurement target at this prompt size (k_rotor4 spans
14.0–17.1 across 5 runs); medgemma is stable to ~±3%. Treat a single Bonsai run
as indicative only.

The QJL-on path is unchanged (medgemma `k_rotor3`: 2.37–2.40 before,
2.34–2.36 after — the kernel is dormant and adds no work), as is Gemma4, where
the kernel does not fire.

This makes the K-only rotor family **usable** rather than fast: it is still
below `none` (Bonsai bf16 ≈ 110 TPS). The rotor sandwich is ~64 FMAs per group
per lane and each of a group's 3 lanes redoes it, so the inner loop is
compute-bound, not KV-bandwidth-bound. Narrowing that gap (sparse geometric
product, one decode per group instead of per lane) is future work.

---

## `rotor_flash_decode_symv` — fused flash-decode over rotor-quant K **and** V

The all-quant sibling of `rotor_flash_decode`, for `KvStorage::RotorSym3` /
`RotorSym4`. It reads **both** axes straight from their packed rotor rings —
there is no bf16 K or V mirror at all — so the symmetric rotor codecs finally
carry only their advertised ~3-bits-per-axis cost.

**What it replaced.** `Rotor{3,4}Sym` quantized both axes at `exit_prefill` and
then decoded from a full bf16 K+V mirror (`decode_fp16_k` / `decode_fp16_v`),
which `update_rotor{3,4}_sym` short-circuited to on its first line: the packed
store was written and never read. The codec was dormant, and a codec advertising
~3 bits/axis actually carried bf16 K + bf16 V *plus* its codes — i.e. **more**
resident KV than plain bf16. Dropping the mirror turns the advertised
compression into a real resident-byte win (measured ≈ −34% resident KV: Bonsai-8B
590.0 → 390.8 MB, gemma-4-e2b 36.1 → 23.5 MB on a 1838-token prompt).

### Reuse of the K-decode half

The header ([`build_rotor_flash_header`], shared verbatim with the bf16-V
sibling) emits the Cl(3,0) block decode as the MSL function `rf_decode_k_group`.
This kernel calls it **twice per token** — once over the K ring, once over the
V ring — because the rotor codec is axis-agnostic (`rotor{3,4}_encode` (V) and
`rotor{3,4}_k_encode` (K) are the same function; the K fork only adds the
optional QJL sideband, and the dispatcher fires only with QJL off). Both axes
share one bit width by construction, so a single `RF_BITS` covers the K and the
V unpack, and both probe against the existing header snapshots. Following the
one-decode-per-group shell, each block's leader stages its group's grade-1 lanes
into threadgroup memory (separate `k_shared` / `v_shared`) so the ~64-FMA
sandwich runs once per Cl(3,0) block per axis rather than once per lane.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_symv_msl.rs` — dispatcher +
  counters; reuses the sibling's header builder.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_symv_p1.metal` — pass-1
  body (one body for both bit widths). Pass-2 is the shared LSE merge.
* `crates/rmlx-kv-quant/src/storage/quant_rotor_v.rs` — the V store gains
  the same `QuantKGpuRing` the K stores carry, fed via `RingFeed::Maintain` from
  the symmetric append.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_rotor_sym_fused` —
  dispatch site (main + shared-KV producer + consumer paths).

### Gate

Same shape as the K-only path: device is GPU, storage is `RotorSym{3,4}`, the
store does **not** carry QJL, `q_seq == 1`, `b == 1`, `head_dim` a power of two
`<= 512`. A QJL-carrying store keeps the CPU dequant path on both axes (the QJL
residual is not reproducible in the flash inner loop). `feeds_bf16_k_at_decode`
**and** `feeds_bf16_v_at_decode` are both false for these variants, so
`exit_prefill` allocates neither seed and the resident-byte estimate reads the
same two predicates — it cannot drift from what is materialised.

### Speed vs. the mirror (honest)

Dropping the mirror is **neither** a memory win nor a decode-speed win. It is
not a memory win because the mirror is not what the decode reads (see
"Memory truth" above — the store is 8.75 bits/value for rotor3 and 7.125 for
iso3, and dropping the mirror moves neither), so there is no bandwidth prize to
collect at the decode. It is
not a speed win because the two-pass flash-decode shell — a per-token
threadgroup barrier pair with a thread-0-only softmax section, one threadgroup
per *query* head (so `heads_per_kv` threadgroups re-read the same KV stream),
and an f32 `partial_o` round trip between P1 and P2 — costs more per token than
MLX's bf16 flash attention over the same bytes.

What is **no longer** part of that gap: until the dispatchers were made lazy,
every one of these kernels forced `Array::eval()` on its inputs immediately
before dispatch, which blocked the host on the GPU once per attention layer per
decode step. That alone was worth 1.2–2.9× decode across iso and rotor, K-only
and `_sym`, on both `kv_h = 8` and `kv_h = 1` architectures — the `_sym` pair at
a 4k prompt on Ternary-Bonsai-8B went 19.1 → 55.1 TPS with an unchanged token
digest.

**What that invalidates, and what it does not.** The eval was a *fixed* cost per
decode step — one host↔GPU round trip per attention layer, the same count
whatever the KV length. It therefore moves the **intercept** of per-step decode
time and leaves the **slope** alone. Fitting `ms/step = a + b × (KV tokens/1000)`
across the binary pair: `a` 41.14 → **7.01 ms/step (−83%)**, `b` 2.437 → **2.449
ms/1k KV tokens (+0.5%)** — ≈34 ms/step recovered, which over the layer count is
≈0.16 ms per eval, a textbook round trip.

So: every **absolute decode-TPS** cell for these codecs recorded before this
change measures the dispatcher and must be re-recorded — including the tables in
`docs/models/bonsai/8B/rMLX.md` §2 and issue #292. **Marginal-cost figures
(ms/1k KV tokens) survive**: a slope cancels a fixed per-step cost by
construction, so a published ms/1k table is still valid and should not be
discarded. `make check-no-kernel-input-eval` keeps the defect from coming back.

So these codecs remain opt-in (`--kv-quant rotor3_sym` / `rotor4_sym`, and the
iso pair), and remain research codecs for quality experiments and kernel work —
not memory or throughput candidates. Closing the remaining speed gap needs a
flash-decode shell that stops re-reading KV once per query head, drops the f32
P1→P2 partial round trip, and stops serialising the online softmax on one lane;
closing the memory gap needs a repacked store, which is a redesign, not a kernel
change. Neither alone is enough — see "Fused flash-decode over a quant store —
the break-even condition" above for the measured `ρ < ε` arithmetic and why the
shell, not the store, is the binding constraint.

### Short-prompt abort at `kv_h == 1` — small-`norms`-buffer device floor

Both symv kernels — this one and `iso_flash_decode_symv` below — bind a
per-token `norms` array as a kernel input. MLX's custom-kernel builder binds
a small input array's outer-kernel parameter in the **`constant`** address
space instead of `device` (an internal size heuristic — see
`docs/FFI.md` § "MSL source conventions"), but the shared decode helpers each
kernel calls (`if_decode_k_lane` for iso, `rf_decode_k_group` for rotor)
declare their `norms` parameter `device const float*` — an address-space
mismatch that fails the MSL compile at first dispatch
(`cannot pass pointer to address space 'constant' as a pointer to address
space 'device'`). Measured trip point: `b * kv_h * kv_seq < 8` aborts, `>= 8`
does not, for every `head_dim`.

This is reachable on a **normal short chat prompt** against a single-KV-head
model — Gemma4 global layers are `kv_h == 1`, so a 2-token prompt reaches
`kv_seq == 2` on the very first decode step, well below the trip point.

**Fix (general, both codecs).**
`rmlx_kv_quant::flash_decode_common::pad_norms_to_device_floor` zero-pads the
flat `norms` array up to `NORMS_DEVICE_MIN` (16, a 2× margin over the
measured 8-element trip point) before dispatch whenever
`b * kv_h * kv_seq` is below it. Both kernels' per-tile decode loop is
bounded by the real `kv_seq` carried in their `dims` buffer, not by the
`norms` buffer's allocated length, so the padding is allocated but never
read — correctness is unaffected. `iso_flash_decode_symv_sdpa` and
`rotor_flash_decode_symv_sdpa` both call this one shared helper; there is no
per-codec copy. This keeps both fused kernels on the GPU at **every**
`kv_seq >= 1` with **no CPU dequant fallback** (hard rule 10) — an earlier,
superseded version of this fix routed small-`kv_seq` steps to a CPU dequant
SDPA (`RING_NORMS_DEVICE_MIN` gate + `iso_sym_cpu_sdpa_fallback`); that gate
and fallback function are gone, replaced by the padding above.

Regression coverage (hard rule 6): `iso_sym_short_kv_seq_kv_h1_stays_on_gpu`
and the continuity tests `iso_sym_transition_across_ring_norms_floor` /
`rotor_sym_transition_across_ring_norms_floor` in
`crates/rmlx-kv-ssd/src/block_io_tests.rs` drive `kv_h == 1` decode across
and through the padding floor, on both codecs, checked against an
**independent** `KvStorage::None` (bf16/f32) reference cache fed the
identical per-step tokens — not a scalar reference rebuilt from the same
ring the kernel just read, which a ring corruption both reads see
identically would pass silently. Mutation: disabling
`pad_norms_to_device_floor` reproduces the `kv_seq == 2` abort on both
codecs (`if_decode_k_lane` / `rf_decode_k_group`, `constant` vs `device`).

---

## `iso_flash_decode` — fused MSL flash-decode over iso-quant K

Sibling of `rotor_flash_decode` for `KvStorage::IsoKOnly3` / `IsoKOnly4`: QK over
the packed iso K store + online softmax + bf16-V SV, in two Metal dispatches per
decode step. Same two-pass shell, same shared
`metal/flash_decode_merge_p2.metal`; only the K-decode differs.

**What it replaced.** `update_iso_k_only` called `QuantIsoK::dequant()`
on every decode step — a full-prefix **CPU** iso decode into a `Vec<f32>` plus a
re-upload. That is O(seq) host work per token with the GPU idle, and it is what
pinned the K-only iso family in the "Tier 3 — CPU-bound" bucket. The store is now
GPU-resident (`storage::QuantKGpuRing`, shared with rotor) and the kernel reads
it directly.

### Files

* `crates/rmlx-kv-quant/src/iso_flash_decode_msl.rs` — Rust dispatcher, header
  builder, dispatch counters, `assert_fixed_quat_blocks`.
* `crates/rmlx-kv-quant/src/metal/iso_flash_decode_p1.metal` — pass-1 body (one
  body for **both** bit widths).
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — codec-agnostic
  pass-2 log-sum-exp merge, shared with `rotor_flash_decode` / `planar_flash_decode`.
* `crates/rmlx-kv-quant/src/storage/quant_iso_k.rs` — the iso K store
  (`QuantIsoK<BITS>`), embedding a `QuantKGpuRing`.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_iso_k_fused` —
  dispatch site (plus `try_dispatch_shared_store` / `sdpa_shared` for shared-KV
  models).

### Decode: one left Hamilton product, not a sandwich

The iso codec encodes `r = q * v_unit` with the single fixed golden-ratio unit
quaternion `FIXED_QUAT`, so the decode is `q̄ * r` — **one** left Hamilton
product. Do not carry the rotor codec's `R̃ * mv * R` sandwich across; they are
different algebras. This is also why iso's inner loop is much cheaper than
rotor's: ~16 FMAs per group, not ~64.

The decode is **self-contained per lane** — a group's four codes all live in one
u32, so `if_decode_k_lane` unpacks them and runs the Hamilton product in
registers with no threadgroup staging and no barrier — a barrier per token
inside a flash inner loop would serialise the tile.

### Fixed quaternion is baked into the header

`iso_encode_fast` writes the one `FIXED_QUAT` constant into every slot of its
per-group `quaternions` array, so the kernel bakes `q̄` in and the ring does not
carry the quaternion table at all — storing `n_tokens * n_groups * 4` copies of
one constant would be pure bandwidth.

That is a real coupling, not an assumption. If the encoder ever emits per-group
quaternions (its own docs float that as future work) this kernel would be
silently wrong rather than merely stale, so `assert_fixed_quat_blocks` rejects a
store whose quaternions are not `FIXED_QUAT`.

### Bit width is a header parameter

`bits ∈ {3, 4}` arrives via the header (`IF_BITS` / `IF_MASK`) alongside the
matching Lloyd-Max codebook, so one `.metal` body serves both variants. Both
widths pack one group of 4 into a single u32
(`words_per_group = ceil(4 / (32 / BITS)) = 1`); element `e` sits at
`[e*BITS, e*BITS + BITS)`. Selection is explicit; any other `bits` is an `Err`,
never a silent fallback to the wrong unpack width.

### Reusable K-decode half

The per-lane iso decode is emitted into the **header** as the MSL function
`if_decode_k_lane(codes, scales, norms, tok_idx, n_groups, lane)` rather than
inlined into the body. A quantized-V flash kernel (the `iso*_sym` follow-up)
needs the identical decode against the V store's `(codes, scales, norms)` triple
and can call it unchanged.

### Gate

No env var and no CLI flag: the path is on whenever it is applicable. Gates, in
order — device is GPU, storage is an iso K-only variant, `q_seq == 1`, `b == 1`,
`head_dim` is a power of two, a multiple of the quaternion block size (4), and
`<= ISO_FLASH_HEAD_DIM_MAX` (512). Any miss falls through to the legacy CPU
dequant path.

**No QJL analogue.** The 1-bit QJL residual is rotor-only, so unlike
`k_rotor*` — which needs `--rotor-qjl off` to reach its kernel at all — `k_iso3`
/ `k_iso4` reach this kernel at **stock defaults**.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `KvStorage::IsoKOnly3` / `IsoKOnly4`, `b == 1` | **YES** | GPU ring + `iso_flash_decode_sdpa`. |
| `KvStorage::IsoKOnly{3,4}`, `b > 1` | NO | Ring stride does not interleave batch. |
| `Iso{3,4}Sym` | NO (this kernel) | Both axes are iso-quantized; they decode through the all-quant sibling `iso_flash_decode_symv` instead, which reads V from its own packed ring rather than a bf16 mirror (ring-as-sole-store). Shares the per-lane K-decode half (`if_decode_k_lane`). |

### Measured

`release-perf`, M-series, decode TPS, 3 measured runs per cell, median. Both A/B
binaries verified by kernel-name string (`main` = 0 hits, `fix` = 1) and distinct
sha256. Every `after` cell carries a positive dispatch witness
(`rmlx_iso_flash_decode_p1_b{3,4}` in the log); every `before` cell has none.

| Model | Codec | ctx (real tok) | Before | After | Gain |
|---|---|---|---|---|---|
| Bonsai-8B (Qwen3, D=128) | `k_iso3` | 4k (4085) | 4.24 | **18.9–19.9** | ~4.5× |
| Bonsai-8B | `k_iso4` | 4k (4085) | 1.89 | **17.8–19.9** | ~9.9× |
| Bonsai-8B | `k_iso3` | 16k (16913) | 0.96 | **10.59** | 11.0× |
| Bonsai-8B | `k_iso3` | 32k (33612) | 0.59 | **6.63** | 11.2× |
| gemma-4-e2b (Gemma4, D=256, shared-KV) | `k_iso3` | 4k | 44.65 | **64.80** | 1.45× |
| medgemma-4B (Gemma3, D=256) | `k_iso3` | 4k | 21.96 | **51.96** | 2.4× |

Bonsai at 4k is a noisy target (individual runs span 15.4–20.1 across repeats of
the same binary), so its 4k cells are given as a range over two independent
3-run medians; the 16k/32k cells and the other two models are stable to a few
percent. Treat a single Bonsai 4k run as indicative only.

The gain grows with context because the cost removed is O(seq) host work per
token. Fitting `itl = a + b·kv_seq` over Bonsai `k_iso3` at 4k/16k/32k:

| path | `a` (fixed) | `b` (per KV token) |
|---|---|---|
| before — CPU dequant | 101 ms | **48.5 µs** |
| after — `iso_flash_decode` | 37 ms | **3.40 µs** |

`b` is what decides whether a codec can win at long context, and it drops 14.3×.

`gemma-4-e2b` gains least because only its global layers are iso-quantized (its
SWA layers stay bf16), so the CPU dequant removed was a smaller share of the
step. It dispatches via the shared-KV **store** path; without that wiring the
kernel would be dead on every shared-KV model.

This makes the K-only iso family **usable** rather than fast: Bonsai is still
below `none` (bf16 ≈ 110 TPS). The residual 3.40 µs/KV-token is the flash-decode
*shell*, not the iso decode — a barrier-tree reduction per token with most lanes
idle. That is shared with `rotor_flash_decode` and is where further work belongs.

**Memory.** The GPU ring is additional resident memory on top of the CPU blocks
(~8.1 MB/layer vs 23.9 MB of blocks at Bonsai 4k — ~34% on top of the blocks).
It **is** counted: `KvStorage::resident_bytes` delegates to
`QuantIsoK3::byte_size`, which sums the CPU blocks and the ring. Same for rotor.

This was not always so. The byte total used to route through a per-codec
bits-per-element formula that never read the store, so the ring was invisible
and `k_iso3` and `k_rotor3` reported byte-identical KV. Measured on Bonsai-8B at
4k, the decode-time `kv_bytes` for `k_iso3` went from 25.3 MB/layer to
42.9 MB/layer once the total was derived from the allocations — the old figure
accounted for only 60% of the process's RSS growth, the new one for 99%.

---

## `planar_flash_decode` — single-pass MSL flash-decode for PlanarK

Single-pass MSL flash-decode for `KvStorage::PlanarK`: keeps QK + softmax
+ SV in one threadgroup over the decode-step (q_seq == 1) PlanarK K and
bf16 V buffers. Two-pass tile structure mirrors TurboFlash
(`turbo_flash_msl::TILE_SIZE = 64`).

### Files

* `crates/rmlx-kv-quant/src/planar_flash_decode_msl.rs` — MSL kernel,
  Rust dispatcher, dispatch counter for NIAH.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused`
  — dispatch site (when the cache's `DispatchPolicy::planar_flash_decode` is
  set, replaces the split fused-QK chain).
* `crates/rmlx-cli/src/commands/serve.rs::resolve_planar_flash_decode`
  — `--planar-flash-decode {on|off|auto}` CLI flag. Auto resolves OFF on
  every host (see below).

### Gate

`DispatchPolicy::planar_flash_decode` enables the kernel. CLI flag
`--planar-flash-decode {on|off|auto}` (default `auto`) is the production
switch; `RMLX_PLANAR_FLASH_DECODE=1` is the `auto` fallback. Default OFF on
every host as of 2026-05-31 — see "Auto-flip status" below.

### Storage applicability

| Variant | Eligible? | Notes |
|---|---|---|
| `KvStorage::PlanarK { k: QuantPlanarK, .. }` | **YES** | Sole route through `update_and_sdpa_planar_k_fused` → `planar_flash_decode_sdpa`. Requires power-of-two `head_dim`. |
| Any other `KvStorage` variant | NO | Routed through `mixed_quantized_sdpa` or `update_and_sdpa_shared_source`. |

### Arch reachability

| Arch | Routing | Reachable? | Why |
|---|---|---|---|
| Bonsai (`Qwen3ForCausalLM`) | `update_and_sdpa` → `sdpa_dispatch` → `update_and_sdpa_planar_k_fused` | **YES** | The only arch that both (a) routes through the fused-QK chain and (b) does not reject PlanarK at validate_resolved. |
| Qwen3.6 (`Qwen3_5MoeForConditionalGeneration`) | rejected at `cache_type::validate_resolved` | NO | Contract A.y `QwenMoePlanarKRejected` — pre-existing PPL-disaster guard. The cache is never built; the kernel can never dispatch. |
| Any arch with cross-layer KV sharing (e.g. `Gemma4ForConditionalGeneration`) | `update_and_sdpa_shared_source` (cross-layer KV share) | YES | The shared-source chain mirrors `update_and_sdpa` arm for arm, so `sdpa_dispatch` is reached exactly as on a non-sharing model. |

NIAH cells covering all three routes ship in
`crates/rmlx-models/tests/niah_long_context.rs` (`niah_pflash_*`) and
assert `dispatch_delta > 0` on Reachable+ON cells and `== 0` on Unreachable
or OFF cells.

### Performance posture (Bonsai canary)

| Shape | OFF (split chain) | ON (flash kernel) | Delta | StdDev OFF | StdDev ON |
|---|---:|---:|---:|---:|---:|
| 4k prompt × 100 decode | 96.648 TPS | 96.460 TPS | -0.19% | 1.764 | 0.278 |
| 8k prompt × 100 decode (smoke) | 75.833 | 75.060 | -1.0% | n=1 | n=1 |

The flash-decode kernel shows **6× lower stddev** at the 4k canary shape but
does not beat the split-chain mean at this decode-token budget. The fused
single-kernel save is balanced by the loss of the upstream MLX flash kernel's
tuning. See `docs/PERF_BASELINE.md` for full data.

### Numerical relationship to the split chain — measured, not bit-exact

The kernel is **not** byte-identical to the split chain. Both arms decode the
same packed K, but the flash kernel folds the softmax into a per-tile online
log-sum-exp reduction while the split chain materialises the whole score row
and calls `softmax_precise`; the summation orders differ, so the f32
accumulators differ in the low mantissa bits.

Whether that survives the closing `astype(queries.dtype())` — the bf16 both
arms actually return — depends on how close each exact value sits to a bf16
rounding boundary, which is a property of the data. So **some cells are clean
and some are not**, and a single-cell check proves nothing either way.

Measured by `planar_flash_decode_is_not_bit_exact_vs_split_chain`
(`crates/rmlx-kv-quant/src/planar_flash_decode_msl_tests.rs`), running both
arms over one packed store with production dtypes throughout — bf16 Q as the
model streams it, bf16 V, and the output cast the dispatcher applies to both
returns:

| `kv_h` × `heads_per_kv` | `head_dim` | `kv_seq` | f32 accumulator differs | max abs err | **bf16 output differs** |
|---|---:|---:|---:|---:|---:|
| 8 × 4 | 128 | 64 | 3569 / 4096 | 8.94e-8 | **0 / 4096** |
| 8 × 4 | 128 | 512 | 3643 / 4096 | 2.98e-8 | **0 / 4096** |
| 8 × 4 | 128 | 4096 | 3863 / 4096 | 2.05e-8 | **3 / 4096** |
| 1 × 8 | 256 | 64 | 2048 / 2048 | 1.13e-4 | **273 / 2048** |
| 1 × 8 | 256 | 512 | 2048 / 2048 | 3.55e-5 | **280 / 2048** |
| 1 × 8 | 256 | 4096 | 2048 / 2048 | 1.46e-5 | **298 / 2048** |

Read the last column: the two arms are observably different to a caller in
4 of 6 cells, and identical in 2. The clean pair is exactly
`head_dim=128, kv_seq<=512` — so a check run only at the Bonsai shape and a
short context would have "confirmed" byte-identity outright. That is the same
failure the TurboFlash claim was retracted for.

The f32 spread (~2e-8 at `head_dim=128`, ~1.2e-4 at `head_dim=256`) stays well
inside a bf16 ULP at these output magnitudes, and the fraction of elements that
flip after rounding tracks the ratio of that error to the ULP — consistent with
summation order alone, not a correctness defect. The kernel is a faithful
implementation of the same attention; it is simply not lossless, and must not
be described as such.

**Measure at the dtype the dispatcher returns.** Comparing the f32
accumulators alone overstates the difference — it reports thousands of
differing elements in cells where the shipped bf16 is bit-identical. Comparing
only the bf16 understates it — two arms that never ran would also agree. The
test asserts both: at least one cell differs at bf16 (the claim), and every
cell differs at f32 (the null control that both arms ran).

**The serve-path A/B cannot settle this.** `--planar-flash-decode on|off` on a
normal generate flow compares two runs in which the kernel never dispatches at
all: the warm-TTFT bf16-K seed is live for the whole post-prefill decode window
and the PlanarK dispatcher bypasses both the fused-QK chain and the flash
kernel (see "Correctness gap" below). Measured on Ternary-Bonsai-8B,
`--kv-quant planar_k`, 4096-token prompt, 32 generated, 1 warmup + 2 measured
runs, `--log verbose`:

| Arm | `planar_flash_decode_sdpa: dispatch` | `planar_fused_qk: dispatch` | `warm_ttft_bypass` | token digest |
|---|---:|---:|---:|---|
| `--planar-flash-decode on` | 0 | 0 | 2418 | `0x8d52921f8217bb27` |
| `--planar-flash-decode off` | 0 | 0 | 2418 | `0x8d52921f8217bb27` |

The digests match because both arms took the same branch. An A/B in which
neither arm dispatches the kernel under test confirms any equivalence put to
it; count the per-dispatch `trace!` events before drawing a conclusion from
one.

### Correctness gap — RESOLVED (warm-TTFT bf16-K shortcut)

The initial NIAH tests reported retrieval failures on every Bonsai PlanarK
cell, OFF and ON alike (both producing the same incoherent decoded output
`"9. The secret. The grass. ..."`). Investigation found the bug was NOT in
PlanarK's chunked-prefill broadcast or in the GPU codec at scale — both
`planar_v4_msl_roundtrip_8k_bonsai_shape` and
`quant_planar_k_single_append_8k_bonsai_shape` confirmed the codec
is bit-exact at 8k Bonsai shape, and
`quant_planar_k_oneshot_vs_chunked_append_parity` confirmed
one-shot vs chunked append are byte-identical.

The real root cause: `KvCache::update_planar_k` was the **only**
quantised `update_<arch>` that lacked the warm-TTFT bf16-K seed
shortcut. Every other codec (K8V4 / K8V8 / Planar / Mixed / K8VTurbo* /
Iso* / Rotor* / TurboSym*) returns early to `update_decode_fp16` when
`decode_fp16_k` is `Some(_)` (set by `exit_prefill`), so the bf16
prefill K is reused for the whole post-prefill decode window. PlanarK
uniquely re-encoded K through the lossy 4-bit Lloyd-Max + Givens
rotation kernel on every decode step. The resulting per-position drift
compounded across the 8k softmax tail and broke needle retrieval — the
K8V4 reference cell `niah_bonsai_8k_d50` "passes" because it silently
ran bf16 K, not because K8V4's codec was somehow more faithful.

Fix landed in `KvCache::update_planar_k`
(`crates/rmlx-kv-quant/src/kvcache/update.rs`) + the fused-QK dispatcher
gate at `crates/rmlx-kv-quant/src/kvcache/sdpa.rs`. Both now route through
bf16 SDPA whenever `decode_fp16_k.is_some()`, matching every other quant.
Side effects:

* `niah_pflash_bonsai_{8k,16k}_d{10,30,50,70,90}` now retrieve
  `AX7-PURPLE-FOX-9421` correctly under both `RMLX_PLANAR_FLASH_DECODE=0`
  and `=1`.
* The PlanarK fused-QK and `planar_flash_decode` kernels intentionally do NOT
  fire during a request's post-prefill decode loop (the bf16 seed is live).
  Both remain reachable on a fresh `KvCache` with no seed (e.g. PPL eval
  fixtures that bypass `exit_prefill`), so the kernels are not dead code, just
  dormant for normal generate flows.
* Decode TPS on Bonsai PlanarK improves: 4k canary mean 101.19 TPS
  (vs flash-decode baseline 96.65, +4.7%); 8k smoke 77.17 TPS (vs 75.83,
  +1.8%). The fused-QK kernels' theoretical wins were balanced by the
  loss of MLX's tuned `scaled_dot_product_attention`; routing through
  bf16 SDPA wins back the upstream kernel's tuning.

### Auto-flip status: OFF (HOLD)

The brief gated the Auto-on flip on a clean Bonsai NIAH **and** ≥10% TPS
gain. Neither lands:

- **NIAH correctness**: blocked by the pre-existing PlanarK +
  chunked-prefill bug (see "Correctness gap" above) — not a flash-decode defect.
- **Perf gain**: -0.19% at the 4k canary (well below 10% gate).

`PlanarFlashDecodeMode::Auto` therefore resolves OFF on every host. The
existing `--planar-flash-decode on` opt-in is preserved for ablation
benches.

---

## Fused-QK head-major K storage

The fused-QK MSL kernels compute pre-softmax `QK` straight off a head-major
packed K shadow, skipping the K dequant round-trip. They are reached from the
production decode path by q8 (`K8V4` / `K8V8`), `TurboSym3`, `TurboSym4`, and
the two rotor-asym codecs (`RotorK3Asym` / `RotorK4Asym`).

### Which codecs can reach this path, and why the rest cannot

The shadow is built by **re-encoding the bf16 K mirror** (`decode_fp16_k`)
that `exit_prefill` materialises. `exit_prefill` only materialises that mirror
for codecs whose `KvQuant::feeds_bf16_k_at_decode()` is true. Eight codecs
return false there — `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4`,
`Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4` — because each decodes
through its own flash-decode-over-quant kernel reading the packed ring
directly, which is the point of not keeping a second bf16 copy of K.

So those eight can **never** reach the fused-QK path: not at any `head_dim`,
not at any batch size, not on any architecture. It is a codec property, not an
arm-ordering accident. They were listed in the dispatch table anyway until the
tables were pruned to the reachable set; the iso fused-QK kernel, whose only
possible callers were four of those eight, was retired with them.

Decode routing for the rotation-KV families, at `b = 1`:

| Codec | Decode kernel | Where |
|---|---|---|
| `Iso3Sym` / `Iso4Sym` | `iso_flash_decode_symv` | `update_and_sdpa` iso-sym arm |
| `IsoKOnly3` / `IsoKOnly4` | `iso_flash_decode` | `update_and_sdpa` iso-K-only arm |
| `Rotor3Sym` / `Rotor4Sym` | `rotor_flash_decode_symv` | `update_and_sdpa` rotor-sym arm |
| `RotorKOnly3` / `RotorKOnly4` | `rotor_flash_decode` | `update_and_sdpa` rotor-K-only arm |
| `RotorK3Asym` / `RotorK4Asym` | `rotor_fused_qk` | fused-QK shadow path — **no flash arm exists**, so this is its only GPU decode kernel |

The rotor-asym pair therefore depends on `--fused-qk on`. With the shipped
default (`auto`, which resolves OFF) their decode serves from the warm bf16
mirror instead — correct output, no rotor kernel. Pinned by
`crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs`, which asserts for
each rotor codec both that the expected kernel fired and that the other two
did not.

### `head_dim` reachability — why fused-QK never fires on a Gemma4 model

The kernel shims are hard-gated on `head_dim ∈ {128, 256}`. That excludes
**every Gemma4 model**, at every size, with every fused-QK codec.

Gemma4 quantises only its full-attention (global) layers — the SWA layers stay
bf16 — and the global layers use `global_head_dim = 512`, not the
`head_dim = 256` the SWA layers use. 512 is outside the shims' supported set,
so `try_fused_qk_dispatch` rejects at gate 4 on every decode step. Measured on
gemma-4-e2b with `--ctk rotor_k_3 --ctv q4_g64 --fused-qk on`: zero
`rotor_fused_qk_sdpa: dispatch` events, and 63 `fused_qk: skipped` events
carrying `reason = "head_dim not in {128, 256}"` with `head_dim = 512`.

This is not a defect and not a Gemma4-specific gate — it is the kernel's shape
support meeting the arch's shape. The rotor and iso **flash-decode** kernels
accept `head_dim` up to 512, so Gemma4 does reach those: the same model on
`--ctk rotor_k_3 --ctv bf16` dispatches `rotor_flash_decode` 147 times over
the same workload. If you want a GPU-side quantised K decode on Gemma4, that
is the family to use.

To confirm it on your own model, run with `--log verbose` and search the run's
`<RMLX_HOME>/logs/*.jsonl` for `fused_qk: skipped`; the `reason` field names
the gate and the `head_dim` field carries the value that was rejected.

### Storage shape

Added on `KvCache` as `fused_qk_shadow: Option<FusedQkShadow>`:

| Buffer | Shape | Per-token payload |
|---|---|---|
| `k_codes` | `u32 [B, kv_h, max_seq, codes_per_token]` | codec-specific packed codes |
| `k_scales` | `f32 [B, kv_h, max_seq, scales_per_token]` | per-group f32 scales |
| `sideband_norms` | `f32 [B, kv_h, max_seq, 1]` | per-token L2 norm (rotor only) |
| `sideband_rotor_table` | `f32 [n_groups * 4]` | static per-layer rotor table (rotor only) |

The per-codec layout is computed by
`FusedQkLayout::for_codec(KvQuant, head_dim) -> Result<Option<Self>>` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_shadow.rs`:

| `KvQuant` | `codes_per_token` (u32) | `scales_per_token` (f32) | sidebands |
|---|---|---|---|
| K8V4, K8V8 | `head_dim/4` | `head_dim/128` | — |
| TurboSym3 | `head_dim*3/32` | `head_dim/32` | — |
| TurboSym4 | `head_dim/8` | `head_dim/32` | — |
| RotorK3Asym, RotorK4Asym | `ceil(head_dim/3)` | `ceil(head_dim/3)` | per-token norm + rotor table |
| any codec with no bf16 K mirror | — | — | `for_codec` returns `Ok(None)` — unreachable, see above |

The kernel shims read the codes / scales buffers as flat 1-D inputs of length
`tok_count * payload_per_token` where `tok_count = B * kv_h * kv_seq`. The
shadow is sliced `[B, kv_h, max_seq, payload] → [B, kv_h, kv_seq, payload]`
and flattened on every dispatch — the dim-2 slice is non-contiguous, so the
flatten forces a per-step materialisation (see KV_CACHE.md §9.5 "Per-step cost
framing").

### Dispatch wire-in

`KvCache::try_fused_qk_dispatch` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch.rs` is called from
`update_and_sdpa` right after the K8V4-TurboFlash branch and before the legacy
bf16 SDPA fallback, and from `try_dispatch_shared_bf16` on the cross-layer-KV
producer path. Gates (in order):

1. `DispatchPolicy::fused_qk` on the cache's policy (CLI flag
   `--fused-qk on|off|auto`, `auto` fallback `RMLX_FUSED_QK=1`).
2. `Device::Gpu`.
3. `q_seq == 1` (decode-only).
4. `head_dim ∈ {128, 256}` (kernel hard gate).
5. `kv_seq ≥ DispatchPolicy::fused_qk_min_kv_seq` (default 512, override
   `RMLX_FUSED_QK_MIN`; sub-threshold caches go to bf16 SDPA where the launch
   overhead is not amortised).
6. Codec is in the `lookup_fused_qk_kernel` table. This is the only such
   table: `rmlx-models` used to carry a public mirror of it with no caller,
   which was removed rather than hand-synced against the one that runs.
7. The codec has a GPU encoder wired in (`codec_has_gpu_encoder`).
8. Rotor only: `--rotor-qjl` is off (the kernel does not reproduce the 1-bit
   K-side residual).
9. `decode_fp16_k` is seeded.
10. The storage variant carries a `max_seq`
    (`storage_max_seq_for_fused_qk`).
11. The step does not overflow it (`prev_offset + new_seq <= max_seq`) — the
    shadow populate path has no out-of-range clamp, so an overflowing step
    falls back rather than encoding a 0-length chunk.

Every fall-through emits `fused_qk: skipped` at `trace!` with a `reason`
field naming the gate, so `--log verbose` distinguishes "gate rejected" from
"codec has no kernel" without reading the dispatcher. The `head_dim` gate also
logs the observed value; the overflow gate additionally raises a one-shot
`warn!`, and the per-step trace is what shows the fall-through continuing
after that single warning.

Gate 1 is traced like the rest on purpose. `--fused-qk` resolves OFF by
default, so it is the rejection an operator hits first — a dispatcher that
logged nothing there would answer "why doesn't the kernel fire?" with an empty
log, which reads as "the dispatcher was never called".

The shadow is allocated lazily on the first dispatch (seeded by quantising the
prefill bf16 prefix in `decode_fp16_k`) then appended head-major every
subsequent decode step via 4-D `slice_update` at
`[:, :, prev_offset:prev_offset+new_seq, :]`. Bf16 `decode_fp16_k/v` stay
maintained as the fallback path.

### Codec coverage

| Codec family | GPU encoder | Status |
|---|---|---|
| q8 (K8V4, K8V8) | `q8_quantize_gpu` | Wired — cosine ≥ 0.999 vs bf16; dispatch delta proven |
| TurboSym3 | `turbo_quantize_v3_gpu` (axis-agnostic) | Wired — cosine ≥ 0.998 vs bf16 |
| TurboSym4 | `turbo_quantize_v4_gpu` (axis-agnostic) | Wired — cosine ≥ 0.999 vs bf16 |
| RotorK3Asym / RotorK4Asym | `rotor_quantize_v3_gpu` / `rotor_quantize_v4_gpu` | Wired — sole GPU decode path for these codecs |

### Dispatch counter and trace

`rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count()` aggregates the
per-family counters; in-process tests use `delta = after - before > 0`. That
counter has no caller outside tests and is unreachable from a shipped binary —
from a real run, count the per-dispatch `trace!` events instead
(`q8_fused_qk_sdpa: dispatch`, `turbo_k{3,4}_fused_qk_sdpa: dispatch`,
`rotor_fused_qk_sdpa: dispatch`) in the run's `<RMLX_HOME>/logs/*.jsonl`
under `--log verbose`.

### See also

* `crates/rmlx-kv-quant/tests/fused_qk_dispatch.rs` — GPU integration
  tests for q8 + TurboSym3 + TurboSym4.
* `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs` — the rotor
  routing contract (which kernel each rotor codec reaches, and which it
  must not).

---

## Sparse attention

Two-phase MSL kernel pair in `crates/rmlx-kv-quant/src/sparse_attn/`:

| Kernel | Role |
|---|---|
| `phase1_score_msl::phase1_score` | Per-(q, head) cheap-inner-product score against every KV slot; emits sorted partial top-K (`TOP_PER_TILE` slots per tile). |
| `phase2_sparse_attend_msl::phase2_sparse_attend` | Runs SDPA only on the phase-1 selected slots; per-tile partials are LSE-merged into the final attention output. |

Dispatcher: `rmlx_models::kv_cache::attention_dispatch::sparse_attn_dispatch_if_enabled`.
Gate: `DispatchPolicy::sparse_attn`, passed to the dispatcher by its caller.
CLI flag: `--sparse-attn {auto|on|off}` (default `auto` → OFF on every host;
`auto` fallback `RMLX_SPARSE_ATTN=1`; see `docs/CLI.md`).

### Head budgets (`head_budgets.json`)

The per-(layer, head) k-budget table consumed by phase-2 lives in
`<MODEL>/head_budgets.json`. Two schema versions are supported.

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

**Schema v2** (true softmax-mass) adds four optional fields to
`calibration` and bumps `version` to `2`:

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

- `recipe` — `"softmax_mass"` (current default) or `"k_norm_proxy"`
  (legacy K-norm² alias).
- `target_mass` — cumulative softmax-mass coverage target.
- `target_mass_budget_floor` — minimum per-(layer, head) budget; guards
  against pathological single-mass distributions producing a 1-slot
  budget.
- `prompts_provenance` — basenames of calibration prompt files.

See [`crates/rmlx-loader/src/head_budgets.rs`](../crates/rmlx-loader/src/head_budgets.rs)
for the canonical struct, validator, reader (`load_head_budgets`), and
writer (`write_head_budgets`). Both ends fail on shape mismatch
(`num_layers` vs row count, `num_heads` vs column count) or zero budgets
(every (layer, head) must attend to ≥1 slot). The reader accepts both
versions; a v1 load emits a `tracing::warn!` advising softmax-mass
re-calibration.

`head_budgets.json` is loaded at model-load time alongside
`kv_calib.json` — see the `discover_kv_calibration` site in
`crates/rmlx-cli/src/commands/serve.rs`. A snapshot without the file is
the common case; consumers treat missing budgets as "no sparse path
enabled for this snapshot".

### Calibration recipes

CLI: see `docs/CLI.md`.

Three head-budget family recipes are supported:

| Recipe | Schema | Measurement | Default |
|---|---|---|---|
| `head_budget` | v1 | K-norm² proxy (H2O / StreamingLLM stand-in) | legacy |
| `k_norm_proxy` | v1 | Explicit alias for the K-norm² proxy | — |
| `softmax_mass` | v2 | True Q@K^T → softmax → cumulative-mass top-K | **current default** |

#### True softmax-mass calibration

Algorithm: load model → for each calibration prompt → fresh bf16 KV
cache (`KvQuant::None`) → run `forward_seq_with_cache_calibrated` with a
`SoftmaxMassSink` → at each layer's post-RoPE / pre-SDPA boundary, the
sink reads the last-position Q (mean-folded over the q_per_kv group for
GQA) and the full accumulated K → computes per-kv-head softmax scores →
finds smallest top-K covering `target_mass` → max-aggregates across
prompts. GQA-expands the per-kv-head budget table to per-q-head rows
for the v2 schema.

Per-prompt host-side cost is O(n_layers × n_kv_heads × S_kv × head_dim)
in pure-Rust f32 arithmetic (no extra Metal kernels). On a 36-layer
Bonsai-2bit run with 15 prompts × ~400-600 tokens, calibration
completes in ~2.5 s on M2 Max.

#### Legacy — K-norm² proxy

`multi-turboquant`'s reference calibration writes
`calibration.num_prompts: 0` and stamps `method = "weight_norm"` — it
ships a *placeholder* head_budget hint rather than a real measurement.
rMLX's `head_budget` / `k_norm_proxy` recipes replace this with a real,
prompt-driven measurement under the K-norm² ranking proxy (H2O,
StreamingLLM). The schema's `method = "softmax_mass"` label named the concept
(per-(layer, head) cumulative mass coverage); the v1 implementation used
K-norm² as a stand-in. v2 lifts the recipe to true softmax-mass and adds
`recipe` as an explicit field. v1 files are still loaded transparently; the
runtime dispatcher consumes both shapes identically.

### Production dispatch — warm-TTFT dormant by design

Sparse-attn is intentionally dormant on the normal generate flow. Every
quantised KV codec routes its decode-window through the bf16-K seed
materialised by `exit_prefill` (`decode_fp16_k`), so
`KvCache::update_and_sdpa` never reaches the PlanarK fused-QK /
flash-decode / sparse-attn kernels when the seed is live (warm-TTFT shortcut
at `crates/rmlx-kv-quant/src/kvcache/sdpa.rs:617-655`). The kernels remain
reachable for **seedless workloads** (synthetic PlanarK caches in tests, PPL
eval, future prompt-cache hits that skip prefill) via the public production
entry point
`rmlx_models::kv_cache::attention_dispatch::sparse_attn_dispatch`.

Aggregated dispatch counter
[`rmlx_kv_quant::sparse_attn::sparse_attn_total_dispatch_count`]
returns the process-lifetime sum of P1 + P2 enqueues; one
`sparse_attn_dispatch` call increments the counter by exactly 2.

Auto-policy: `resolve_sparse_attn` on `Auto` resolves OFF on every host
(same posture as `PlanarFlashDecodeMode::Auto`). The On override sets
`DispatchPolicy::sparse_attn` but does NOT cause the kernels to fire on a
warm-TTFT decode — that contract is structural, not gated.

Invariant tests:

* `crates/rmlx-models/tests/sparse_attn_dispatch.rs::sparse_attn_dormant_on_warm_ttft_update_and_sdpa`
  — warm PlanarK cache through `update_and_sdpa` under a `sparse_attn: true` policy keeps the counter flat.
* `crates/rmlx-models/tests/sparse_attn_dispatch.rs::sparse_attn_dispatches_on_seedless_planar_k`
  — seedless PlanarQuant-packed buffer through `sparse_attn_dispatch` increments the counter by exactly 2 and cosine ≥ 0.99 vs dense `planar_flash_decode_sdpa`.

**GPU-resident iso/rotor V mirror — dormant by design:** the GPU-resident
iso/rotor V mirror is hardcoded OFF on the normal decode path for the same
structural reason: every iso and rotor update path short-circuits at
`decode_fp16_k.is_some()` (warm-TTFT bf16 seed) before reaching the
GPU-resident mirror branch. The 7-codec phase-2 extension (iso3 K, iso4 V/K,
rotor3/4 V/K) was evaluated and declined. A/B bench on Bonsai 8B
(8k prompt, `--ctk q8_g128 --ctv iso_v_3`, 3 runs per arm) showed Δ decode-TPS
= −0.73% and Δ TTFT = −0.46% (both inside ±2σ noise). The gate
(`gpu_resident_iso_enabled()`) is hardcoded `false` in production; it is only
controllable in tests. Re-open condition: a production path where
`decode_fp16_k.is_none()` during steady-state decode. Full numbers:
`docs/PERF_BASELINE.md`.

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
