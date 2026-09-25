# KV Cache Quantization Reference

> The codec code lives in **`rmlx-kv-quant`** (storage enums, MSL kernels,
> per-layer `KvCache`, paged KV, mixed / rot-K codecs). The SSD tier lives in
> **`rmlx-kv-ssd`**. The policy / builder layer (`KvCacheBuilder`,
> `kv_quant_for_layer`, `DEFAULT_KV_QUANT`, `LAYER_ADAPTIVE_*`,
> `cache_type::*`) lives in **`rmlx-models::kv_cache`**. See § "Public API"
> below for the import paths.

This doc gives the KV quantization contract: the public API and import
paths, the storage summary, the Metal-vs-CPU hot path, the CLI flags and
presets, the auto default, the memory and bit-rate summary, and the break-even
condition with the disposition of each codec.

The other KV quantization docs: [`KV_LAYER_POLICY.md`](KV_LAYER_POLICY.md)
(which codec each layer gets); [`KV_CODECS.md`](KV_CODECS.md) (storage per
`KvStorage` variant, TurboQuant calibration);
[`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the iso and rotor codecs);
[`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK, fused flash-decode,
sparse attention); [`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md)
(`truncate_to` per store); [`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md)
(measured codec fidelity).

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
  matches `&self.storage`, not `self.quant`. See `docs/KV_LAYER_POLICY.md` § "Dispatch axis".

`--kv-quant auto` resolves to `none` (bf16) on every arch
(`DEFAULT_KV_QUANT`). Every other codec is opt-in.

Sliding-window attention (SWA) layers do not use the codec. They use
`RotatingState`, a bf16 ring buffer, for every `KvQuant`.

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
| `Mixed` | MLX affine `k_bits` | `k_group` | MLX affine `v_bits` | `v_group` | `MixedKvState` | 0.9937 (V4 g64); 0.9990 (V8 g128); 0.9000 (V2 g32) |
| `Paged` | q8_0 per page | 128 | tq4 / q8_0 / planar per page | 32/128/32 | `PagedKStorage` + paged V | — |
| `TurboSym3` | TurboQuant 3-bit | 32 | TurboQuant 3-bit | 32 | `QuantKTurbo3` + `QuantV{bits:3}` | 0.9807 |

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
    (see `docs/KV_FUSED_KERNELS.md` § `rotor_flash_decode`), so the verdict is `None`. With QJL on,
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
  `docs/KV_CODECS.md` § "`rot_k_tq4v` is rejected").
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
Where the upstream name implies one, `docs/KV_CODEC_FIDELITY.md` §"The turbo family's missing
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
  (`docs/KV_FUSED_KERNELS.md` §"Fused-QK head-major K storage"). `--turbo-flash on` gives `k8v4` the
  TurboFlash buffers. Both flags default to `auto`, which resolves OFF.
- **A store-backed cache of any codec** holds its packed store. This is an SSD
  hydrate, or a cache that did not go through a prefill bracket. The rate of
  each store family is in the table under `docs/KV_ROTATION_CODECS.md` §"Crate-wide rate ceiling".
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
holds 43.25 bits per value there (`docs/KV_ROTATION_CODECS.md` §"Iso memory truth"). The rotor rates are
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
(`docs/KV_LAYER_POLICY.md` §"Layer-adaptive overrides").

---

## Fused flash-decode over a quant store — the break-even condition

The fused flash-decode kernels in `docs/KV_FUSED_KERNELS.md` read a packed KV store at
decode instead of a bf16 mirror. So does TurboFlash
(`docs/KV_CODECS.md` §"TurboFlash is off by default"). A smaller store moves fewer bytes per
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
(`docs/KV_ROTATION_CODECS.md` §"Crate-wide rate ceiling"). So its ρ is 0.25, and no kernel decodes over it. The
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
(`docs/KV_FUSED_KERNELS.md` §"Fused-QK head-major K storage"). With `--turbo-flash on`, `k8v4` holds the
TurboFlash buffers and decodes its 4-bit V. The store is still the
authority for a cache with no mirror: an SSD hydrate, or a cache that did not
go through a prefill bracket.

Every codec here carries an INERT banner at the head of its section in
`docs/KV_CODECS.md` or `docs/KV_ROTATION_CODECS.md`. The
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
alias (`docs/KV_CODECS.md` §"`rot_k_tq4v` is rejected").

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
  (`docs/KV_ROTATION_CODECS.md` §"Iso memory truth"). The result does not depend on the topology, because
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

## See also

- `docs/KV_CACHE.md` — flag surface, supported types, hard invariants.
- `docs/WEIGHT_QUANTS.md` — weight quantization families (separate from KV).
- `docs/SSD_TIER.md` — SSD spill / hydrate for long-context eviction.
- `docs/TESTING.md` — cosine, incoherence and rate-distortion gates; helpers.
