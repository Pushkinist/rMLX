# KV Cache Quantization Reference

> The codec impl lives in **`rmlx-kv-quant`**
> (storage enums, MSL kernels, per-layer `KvCache`, paged-KV, mixed/rot-K
> codecs). The policy / builder layer (`KvQuant` resolution, `KvCacheBuilder`,
> `kv_quant_for_layer`, the SSD spill/hydrate plumbing) and the per-arch
> entry points live in **`rmlx-models::kv_cache`**. See the `## Public API`
> section below for the canonical import paths.
>
> The `rmlx_models::kv_cache::*` re-export shim
> for codec-layer items (`KvCache`, `KvQuant`, `LinearAttnCache`, …) and
> for SSD-tier items (`write_caches`, `set_ssd_*_hook`, …) was dropped.
> Callers now import directly from `rmlx_kv_quant::*` / `rmlx_kv_ssd::*`.
> Only the **policy / builder** items (`KvCacheBuilder`, `ResolverSignals`,
> `kv_quant_for_layer`, `kv_quant_for_ctx`, `LAYER_ADAPTIVE_*`,
> `cache_type::*`) remain under `rmlx_models::kv_cache::*`.

Codec-level reference for every KV quantization variant in rMLX. Covers
storage layout, dispatch path, per-arch defaults, and CLI flag surface.

For the flag-surface overview and per-command usage see `docs/KV_CACHE.md`.
For weight quantization see `docs/WEIGHT_QUANTS.md`. For the SSD spill tier
see `docs/SSD_TIER.md`.

---

## Public API

The `rmlx-kv-quant` crate owns these public items.
The `rmlx_models::kv_cache::*` re-export shim was
dropped; callers import the items directly from `rmlx_kv_quant`:

| Item                                       | Source                                       |
|--------------------------------------------|----------------------------------------------|
| `KvQuant`, `KvQuantParseError`             | `rmlx_kv_quant::quant`                       |
| `KV_MAX_SEQ_DEFAULT`                       | `rmlx_kv_quant::quant`                       |
| `KvCache`                                  | `rmlx_kv_quant::kvcache`                     |
| `LinearAttnCache`                          | `rmlx_kv_quant::linear_attn`                 |
| `KvStorage`, `QuantK`, `QuantV`, `QuantPlanarV` | `rmlx_kv_quant::storage`                |
| `MixedKvState`, `MixedTuple`               | `rmlx_kv_quant::mixed_quant`                 |
| `PagedKStorage`, `PagedVStorage`, `PagedPlanarVStorage`, `install_paged_kv`, `resolve_paged_kv`, `resolve_paged_kv_page_tokens` | `rmlx_kv_quant::paged` |
| `q8_quantize`, `q8_dequantize`, `Q8_GROUP_SIZE` | `rmlx_kv_quant::q8`                     |
| `turboquant::{TurboBlocks, turbo_quantize_v, turbo_dequantize, GROUP_SIZE, …}` | `rmlx_kv_quant::turboquant` |
| `planarquant::{PlanarBlocks, planar_quantize, planar_dequantize, …}`           | `rmlx_kv_quant::planarquant` |
| MSL wrappers: `q8_msl::*`, `turboquant_msl::*`, `planarquant_msl::*`, `turbo_flash_msl::*`, `sparse_v_msl::*`, `rot_k_msl::*`, `k8vturbo3_append_msl::*` | `rmlx_kv_quant::*` |
| Rotation helpers: `rot_k::{hadamard_rotation, rotate_last_axis, …}` | `rmlx_kv_quant::rot_k` |
| SWA ring buffer: `rotating::*` | `rmlx_kv_quant::rotating` |

The policy / builder layer stays in `rmlx-models::kv_cache`:

* `KvCacheBuilder::for_arch_default`, `KvCacheBuilder::resolve_default`
* `ResolverSignals`
* `kv_quant_for_layer`, `kv_quant_for_ctx`
* `LAYER_ADAPTIVE_TAIL_N`, `LAYER_ADAPTIVE_HEAD_N`
* SSD: `block_io::*`, `spill::*`, `hydrate::*`, `ssd_index::*`,
  `set_ssd_event_recorder`, `set_ssd_spill_prom_hook`,
  `set_ssd_hydrate_prom_hook`, `set_ssd_bytes_used_hook`,
  `set_ssd_evict_total_hook`
* `cache_type::*`

## Import paths

Canonical import per public symbol. The
`rmlx_models::kv_cache::*` shim re-exports were dropped — every caller
imports directly from the owning crate:

```rust
// Codec layer — rmlx-kv-quant root + module re-exports:
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache, KV_MAX_SEQ_DEFAULT};
use rmlx_kv_quant::storage::{KvStorage, QuantK, QuantV, QuantPlanarV};
use rmlx_kv_quant::mixed_quant::{MixedKvState, MixedTuple};
use rmlx_kv_quant::paged::{PagedKStorage, PagedVStorage, PagedPlanarVStorage};
use rmlx_kv_quant::turboquant::{TurboBlocks, turbo_quantize_v, turbo_dequantize, GROUP_SIZE};
use rmlx_kv_quant::planarquant::{PlanarBlocks, planar_quantize, planar_dequantize};
use rmlx_kv_quant::{q8_msl, turboquant_msl, planarquant_msl, turbo_flash_msl, sparse_v_msl};

// SSD-tier layer — rmlx-kv-ssd root + module re-exports:
use rmlx_kv_ssd::{
    write_caches, BlockIoError, KvBlockReader, KvBlockWriter, SsdKvIndex,
    SsdSpiller, SsdHydrator, SpillJob, HydratedBlock,
    set_ssd_event_recorder, set_ssd_spill_prom_hook, set_ssd_hydrate_prom_hook,
    set_ssd_bytes_used_hook, set_ssd_evict_total_hook,
};
use rmlx_kv_ssd::ssd_tier::{install_config, active, compute_layout_key, SsdTierConfig};
use rmlx_kv_ssd::{block_io, hydrate, spill, ssd_index};

// Builder / policy stay on the rmlx-models side:
use rmlx_models::kv_cache::{
    KvCacheBuilder, ResolverSignals,
    kv_quant_for_layer, kv_quant_for_ctx,
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

The active codec is controlled by two orthogonal enums:

- `KvQuant` — the logical quantization mode. Set at construction time, stays
  fixed for the lifetime of the request.
- `KvStorage` — the on-device buffer variant that actually holds the data.
  Dispatch inside `KvCache::update_and_sdpa` matches `&self.storage`, not
  `self.quant`. The storage variant is the canonical dispatch axis.

Sliding-window attention (SWA) layers are exempt from quantization regardless
of the active `KvQuant`: they use `RotatingState` (a ring-buffer bf16 path
ported from mlx-lm's `RotatingKVCache`). `mlx-lm.to_quantized` raises
`NotImplementedError` for rotating caches; rMLX matches that behaviour.

### Per-layer net-benefit decision + net-negative warn

Because SWA layers already run the bf16 ring (above), the per-layer
quantization decision is implicit: **windowed layers are bf16, global
(full-attention) layers are quantized**. The codec is a no-op on windowed
layers and can never make them larger — the "skip quant on tiny windowed
layers" condition is therefore already satisfied by the rotating-ring
exemption, not by an extra gate.

The residual net-negative is on the **global** layers. A quantized global
layer keeps a warm-TTFT bf16 decode seed (`decode_fp16_k` / `decode_fp16_v`,
gated by [`KvQuant::feeds_bf16_k_at_decode`]) *alongside* its packed codes and
per-group scales. At small effective context the codes + scales are pure
overhead on top of a buffer the same size as bf16, so the codec is strictly
larger than `--kv-quant none`. Measured on Gemma4 e2b (7 global + 28 windowed
layers, head_dim 256, 1 KV head) at a 4096-token prompt: `k8v4` ≈ 125.0 MB vs
`none` ≈ 113.2 MB (`kv_cache_bytes`) — `k8v4` is +11.7 MB on the global layers,
zero delta on the windowed layers. (These figures predate the global-`none`
bf16 fix below, where the global `none` K/V were resident as f32; the +11.7 MB
codec-vs-`none` delta is the same conclusion — the codec adds scales + seed on
top of the global buffer — and the warn math is unaffected.)

rMLX emits one structured `warn!` at request build time when the resolved codec
is estimated to increase resident KV vs bf16 on the active layer mix:

```
WARN KV codec increases resident KV vs bf16 on this layer mix — the
per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved
at this context; windowed layers already run bf16 and are unaffected. Consider
--kv-quant none if memory is the goal.
  kv_quant=k8v4 eff_seq=8192 n_global=7 n_windowed=28 est_extra_bytes=51380224
```

The estimate is model-agnostic — keyed only on layer geometry (`head_dim`,
`kv_heads`, `window`) and codec attributes (`KvQuant::approx_code_bits`,
the per-group scale cadence, and whether the codec retains a bf16 seed). The
decision lives in:

- `rmlx_kv_quant::KvQuant::estimated_resident_bytes_per_layer` /
  `estimated_net_saving_per_layer` (codec layer — the per-side byte model;
  windowed layers return saving 0).
- `rmlx_models::kv_cache::kv_codec_net_saving_total` /
  `warn_if_kv_codec_net_negative` (policy layer — sums the layer mix and emits
  the warn). Wired into the Gemma4 `generate` path; any arch interleaving
  windowed + global attention can call it with its own `KvLayerShape` vector.

The warn is **advisory only** — the codec is not changed. Keeping the resolved
codec is the operator's explicit choice (and forcing bf16 globally would change
numerics); the warn just surfaces the byte math so `--kv-quant none` is an
informed option when memory is the goal. Seed-free codecs (the K-only
re-quantize families, `feeds_bf16_k_at_decode == false`) cross over to
net-positive at large context and do not warn.

### Gemma4 global `--kv-quant none` KV is bf16 (was f32)

On the mxfp8 path Gemma4 previously ran its whole attention + FFN stream in
f32, so the global (full-attention) `--kv-quant none` K **and** V were resident
as f32 (4 B/elem) — roughly **2× the bf16 expectation**. Only the global layers
grow KV with context (windowed layers are ring-bounded, already bf16), so this
dominated KV residency on long prompts.

Root cause was **model-dtype discipline**, not the codec, the RoPE freqs table,
or RMSNorm: strong-F32 scalar constants meeting bf16 activations promoted the
residual stream to f32, which then propagated through the Q/K/V projections,
attention, and the global KV cache. Three sources, all matching mlx-lm's
weak-typed Python floats now:

- the embed-scale (`hidden_size**0.5`) constant,
- the per-layer-input scales (embed / proj / inv-sqrt2),
- the fused GeGLU / PLI-GeGLU activations, whose internal `gelu_tanh`
  arithmetic constants are f32 and silently widen a bf16 gate.

The scale constants now adopt the operand dtype, and the fused activation
closures restore the gate dtype on their output (the cast folds into the
compiled program, no extra launch). The stream is bf16 end-to-end, so **both**
global K and V store as bf16.

Measured (e4b mxfp8, `--kv-quant none`, ~18.5 k context): `kv_cache_bytes`
≈ 325 MB, exactly half the prior f32 ≈ 649 MB. Decode TPS is unchanged-to-faster
(no mixed-dtype SDPA). A unit-level dtype-lock regression test pins the fused
activations and scale sites at bf16, so a future re-promotion of the Gemma4
stream to f32 fails CI.

### Qwen3 dense `--kv-quant none` KV is bf16 (was f32)

The same class hit the dense Qwen3 arch (`Qwen3ForCausalLM`). Some snapshots
ship norm weights and quant scales/biases at **fp16** (e.g. Bonsai-8B-2bit).
rMLX runs the residual stream in **bf16** (the embedding dequant is forced to
bf16). When `rms_norm` mixes a bf16 activation with an fp16 norm weight — and
when `quantized_matmul` mixes a bf16 activation with fp16 scales/biases — MLX
promotes the result to **f32**. That f32 carried through the Q/K/V projections,
attention, and the global `--kv-quant none` KV cache, which then stored K and V
at f32 (4 B/elem) — roughly **2× the bf16 expectation**. (The YARN mscale scalar
is *not* the cause: q/k/v arrive at the YARN branch already f32 from the
projection.)

Fix, matching mlx-lm's "uniform model dtype" discipline (`w.astype(model_dtype)`
at load): every float model parameter — norm weights, quant scales/biases, and
embedding scales/biases — adopts the bf16 activation dtype at load. The
projection and norm outputs then stay bf16, so K and V store as bf16. The YARN
mscale scalar is also stored as bf16 at load (defense-in-depth; the scalar was
never the root cause, but prebuilding it as bf16 is cheaper than a per-step
cast and keeps the multiply unambiguous. Two unit-level dtype-lock tests pin
the `rms_norm` and `bf16_param` call paths.

Measured (Bonsai-8B-2bit, `--kv-quant none`): decode-time K/V resident dtype
flips f32→bf16 (4→2 B/elem), halving KV residency. Decode TPS gains widen with
context as KV bandwidth dominates: ~+34 % at 4 k, ~+73 % at 16 k, ~+100 % at
64 k — recovering the prior loss vs the mlx-lm champion on this model.

### Qwen3.6 MoE `--kv-quant none` KV is bf16 — audited clean AND hardened

The Qwen3.5-MoE arch (`Qwen3_5MoeForConditionalGeneration`) was audited for the
same f32-KV leak class. **Verdict: clean AND structurally hardened.** The
`qwen3_5_moe` loader casts every float param to bf16 at load via
`load_util::bf16_param`, covering FullAttention (q/k-norm weights, quant
scales/biases, embedding scales/biases) and GDN recurrent layers (`conv1d_weight`
and `norm_weight`). This is identical to the dense Qwen3 loader — so **any future
Qwen3.6 snapshot, including an fp16 repack, stays bf16-clean in compute**. The
compute stream is bf16 end-to-end: `rms_norm` and `quantized_matmul` stay bf16
(no fp16→f32 promotion), the MoE router `softmax(bf16 logits)` keeps
`routing_weights` bf16 (no Gemma4-MoE-class router leak — the router gate is a
plain quantized `Linear`, not a strong-f32-scaled RMSNorm), GDN
`conv1d(bf16 qkv, bf16 conv1d_weight)` stays bf16 through `v4`/`y_bf16`, and
`rms_norm(&y_bf16, bf16 norm_weight)` stays bf16 at the GDN RMSNormGated site.
This arch's attention has no YARN mscale on the q/k path.

Decisive measurement: at `--kv-quant none`, every K/V tensor arrives at the
cache-store boundary as **bf16 (400+/400+ prefill+decode store calls, zero f32)**
— so the compute is genuinely clean, not merely capped by the model-agnostic
`cast_store_bf16` floor. Two CPU dtype-lock tests pin this:
`moe_stream_stays_bf16_with_bf16_params` (q/k-norm + router promotion semantics)
and `bf16_param_casts_fp16_to_bf16` (helper-contract gate — RED if `bf16_param`
stops casting fp16→bf16; loader call sites verified by real-model load proof).

**Byte accounting.** Two methods report KV-cache size:

- `KvCache::approx_bytes()` — formula-based estimate using stored shape fields.
  Used for SWA ring sizing and internal shape guards. Can return 0 when shape
  fields are absent (e.g. before the first prefill).
- `KvCache::resident_bytes()` — actual on-device allocation: reads the real
  `Array` shape × `dtype.itemsize()` for every GPU buffer, and `Vec.len()`
  for CPU codec blocks. Covers packed codes, scales, zero-points, and optional
  rotation/residual buffers. This is the value recorded in `kv_cache_bytes`
  metrics observations and returned by `rmlx baseline`.

### Per-request hot-swap

The `KvQuant` for a request is **not** tied to the model load. A running
`rmlx serve` accepts a per-request `kv_quant` field (OpenAI route) that selects
the codec for that one request — weights stay resident, only the KV cache is
rebuilt. This means one resident model can serve `none`, `k8v4`, `k8v8`, … back
to back with no reload. The override threads down to the same per-request cache
builder the per-ctx `auto` policy already used; absent → launch `--kv-quant`.

The prompt/prefix cache is **partitioned by codec** so a switch can never serve
mismatched cached K/V — `KvQuant::cache_key_salt()` is XOR'd into the
block-hash seed alongside the SSD `layout_key`. See `docs/PROMPT_CACHE.md`
§ "Codec namespacing" and `docs/SERVER.md` § "Per-request KV-config hot-swap".

---

## Storage variants — summary table

| `KvStorage` variant | K codec | K group | V codec | V group | Dispatch path | Cosine gate (V mean ≥) |
|---|---|---|---|---|---|---|
| `None` | bf16 (no quant) | — | bf16 (no quant) | — | `decode_fp16_k/v` buffers | — |
| `K8V8` | rMLX MSL q8_0 | 128 | rMLX MSL q8_0 | 128 | `QuantK` + `QuantK` | 0.9990 |
| `K8V4` | rMLX MSL q8_0 | 128 | TurboQuant 4-bit | 32 | `QuantK` + `QuantV` | 0.9937 |
| `Planar` (bits=4) | rMLX MSL q8_0 | 128 | PlanarQuant 4-bit | 32 | `QuantK` + `QuantPlanarV` | 0.9942 |
| `Planar` (bits=3) | rMLX MSL q8_0 | 128 | PlanarQuant 3-bit | 32 | `QuantK` + `QuantPlanarV` | 0.9989 |
| `Mixed` | MLX affine `k_bits` | `k_group` | MLX affine `v_bits` | `v_group` | `MixedKvState` | 0.9937 (V4); 0.9990 (V8); 0.9000 (V2) |
| `RotKTq4V` | MLX affine 8-bit (rotated) | 64 | TurboQuant 4-bit | 32 | `MixedKvState` (K) + `QuantV` (V) | 0.9937 (V) |
| `Paged` | q8_0 per page | 128 | tq4 / q8_0 / planar per page | 32/128/32 | `PagedKStorage` + paged V | — |
| `TurboSym3` | TurboQuant 3-bit | 32 | TurboQuant 3-bit | 32 | `QuantKTurbo3` + `QuantV{bits:3}` | 0.9807 (K empirical floor) |

---

## Metal-vs-CPU hot path + load-time MSL precompile

Two orthogonal codec attributes drive startup behaviour. Both are exhaustive
matches on `KvQuant` (`crates/rmlx-kv-quant/src/quant.rs`) — a new variant must
be classified or the build fails.

* **`KvQuant::carries_msl()`** — `true` when the codec dispatches at least one
  custom Metal (MSL) kernel on its hot path (every codec except `none`, whose K
  is q8_0 MSL or MLX-affine `mx.quantize`). MSL kernels in this crate compile
  **lazily** — `MetalKernel::new` only registers; MLX compiles the pipeline on
  the *first* `apply()` dispatch (see `docs/FFI.md` § `MetalKernel`). For a
  shader-heavy codec that first dispatch lands inside the first user request, so
  the first long-prompt forward pays a one-time shader cold-compile (a 1-token
  `"hi"` warmup does not trigger it — the codec kernel only fires on a real
  prefill encode).

* **`KvQuant::cpu_hot_path_reason()`** — `Some(reason)` when the codec's KV
  encode + dequant run on the **CPU** on the default hot path. Grounded in the
  actual decode/prefill dispatch in `crates/rmlx-kv-quant/src/kvcache/update.rs`,
  not in assumptions (CLAUDE.md hard rule 7):

  * **V-only iso / rotor** (`iso3/4(/sym)`, `rotor3/4(/sym)`,
    `rotor_k_*_asym_*`) → **`Some`**. At decode, `update_iso3*` / `update_rotor3*`
    early-return to the warm-TTFT bf16 decode seed (`decode_fp16_k.is_some()`),
    so the GPU iso/rotor branch is shadowed; the codec encode that runs (at
    prefill) is CPU. The rotor family's GPU fused-QK encoder is gated OFF by
    default (`RMLX_FUSED_QK`).
  * **K-only iso** (`k_iso3` / `k_iso4`) → **`None`** (Metal). No bf16
    early-return: `update_iso_k_only_{3,4}` dispatches the `iso{3,4}` MSL encode
    kernel every decode step on GPU (`k_iso3` also runs the iso3 MSL dequant
    kernel). Hybrid — the dequant restages the growing prefix host-side and
    re-uploads via `Array::from_bytes` each step (a real, growing CPU cost) — but
    a Metal kernel demonstrably dispatches, so it is **not** "no Metal kernel",
    and must not be hard-rejected by the CPU-path classifier.
  * **K-only rotor** (`k_rotor3` / `k_rotor4`) → **QJL-dependent**. No bf16
    early-return; `update_rotor_k_only_{3,4}` gates the GPU K encode on
    `device == Gpu && !rotor_qjl_enabled()`. QJL **on** (default) → CPU
    (`Some`); QJL **off** (`--rotor-qjl off`) →
    `rotor{3,4}_gpu_append_into_k_blocks` Metal MSL encode (`None`). The verdict
    reads the live `rotor_qjl_enabled()` gate so it tracks the dispatcher.
    With QJL off the **decode** side is fused too — see
    § `rotor_flash_decode` below — so the QJL-off path is fully GPU-resident,
    not a hybrid.

  The `Some` cases are the source of the 30–60× first-forward slowdown and the
  monotonic decode decay as KV grows.

### Per-codec verdict

| Codec family | Hot-path verdict | Notes |
|---|---|---|
| `none` | bf16, no kernel | nothing to compile |
| `k8v4` / `k8v8` / `planar` / `planar3` / `planar_k` | **Metal** | q8_0 K + tq4 / planar V GPU kernels |
| `mixed_*` / `rot_k_v*` / `rot_k_tq4v` | **Metal** | MLX-affine `mx.quantize` K + tq4/affine V (compiled Metal ops) |
| `k8vturbo3` / `k8vturbo2` / `*tcq` / `tsym3` / `tsym4` | **Metal K**, CPU V (bounded) | K=q8_0 GPU; V CPU-forced by the −1 %/−2 % TPS gate, cost small |
| `iso3` / `iso4` / `iso3_sym` / `iso4_sym` | **CPU** | bf16 decode seed shadows the GPU iso branch; prefill V-encode on host |
| `k_iso3` / `k_iso4` | **Metal (hybrid)** | iso K MSL encode every decode step (`k_iso3` also MSL dequant); dequant restages prefix host-side per step. `cpu_hot_path_reason() == None` |
| `rotor3` / `rotor4` / `rotor*_sym` / `rotor_k_*_asym_*` | **CPU** | bf16 decode seed shadows the GPU branch; GPU fused-QK encoder is `RMLX_FUSED_QK`-only |
| `k_rotor3` / `k_rotor4` | **QJL-dependent** | QJL on (default) → CPU; QJL off (`--rotor-qjl off`) → **fully Metal**: rotor K MSL encode + `rotor_flash_decode` fused decode. Verdict reads `rotor_qjl_enabled()` |

### Load-time precompile

`rmlx_kv_quant::precompile::precompile_kv_codec_msl(kq, head_dim, kv_heads,
device)` warms the kernels a codec carries with one representative GPU dispatch
during model load (the eager-preload window), so the first user request is
steady-state instead of paying a cold compile. It is **general per-codec**
(keyed off `carries_msl()`, never an arch name): a no-op on CPU device, when
`head_dim` is unknown (`0`), for `none`, for the CPU-hot-path V-only iso/rotor
families (nothing to warm), and for the K-only iso/rotor families
(`is_k_only_iso_rotor()`) — those are Metal on the hot path but their K kernel is
the iso/rotor MSL kernel, **not** the shared q8_0 K kernel this warm compiles, so
warming q8 for them would compile the wrong shader; their K kernel compiles
lazily on first prefill. It warms the shared q8_0 K-side kernels for every q8-K
MSL codec, plus the tq4 / planar V kernel for `k8v4` / `rot_k_tq4v` / `planar`.
Best-effort — a warm failure logs
`warn!` and proceeds (the kernel then compiles lazily on first use, the
previous lazy-compile behaviour). Wired into `ArchGenerator::from_snapshot_with_id` (the
single server-side generator factory all archs route through).

### CPU-codec classification at resolve time

`rmlx_models::kv_cache::validate_resolved` (alias `validate_resolved_kv_quant`)
runs the arch-agnostic Metal-vs-CPU check after the Qwen-MoE guards. When the
resolved codec is CPU-hot-path (`cpu_hot_path_reason()` is `Some`) it emits a
loud structured `warn!` naming the codec + reason so the cost is never silent.
These codecs still produce correct output — the classifier is warn-and-proceed
only. The K-only iso (`k_iso3/4`) and QJL-off rotor (`k_rotor3/4`) codecs have
`cpu_hot_path_reason() == None` (Metal on the hot path) and are unaffected by the
warn.

---

The `KvQuant::RotK { v_bits, v_group_size }` variant uses `KvStorage::Mixed`
(same `MixedKvState` machinery) with the `rotate_k=true` flag set. It is
listed under Mixed below.

`KvQuant::K8VTurbo3` is available via `--kv-quant k8vturbo3`. It is no longer
the auto default for Gemma4 small (reverted to K8V8 per the composite-score
audit). It reuses the `K8V4` storage path with `bits=3` for the
V side. See the per-variant section below.

`KvQuant::K8VTurbo2` is the native 2-bit Lloyd-Max V codec; it reuses the
`K8V4` storage path with `bits=2` for the V side. Ships **naïve** (no
outlier-mask); outlier-mask wiring is deferred. See per-variant section below for
the gap-vs-mtq quantification.

---

## Per-variant deep dive

### `KvStorage::None` — unquantized bf16

**K codec**: stored as bf16, shape `[B, kv_h, max_seq, head_dim]`.
**V codec**: same.

Buffers live in `KvCache::decode_fp16_k` and `decode_fp16_v`, not inside a
`KvStorage` sub-struct. This reuses the same machinery as the warm-TTFT
fp16 seed path used by quantized variants during prefill. `KvStorage::None`
records only `max_seq`; the actual arrays are owned by `KvCache` directly.

`update()` calls `update_decode_fp16`, which issues a `slice_update` at the
current token offset into the pre-allocated buffer. SDPA uses
`scaled_dot_product_attention` on the raw bf16 arrays.

**Cache-boundary bf16 floor (model-agnostic f32-KV guard).** The store boundary
casts incoming K/V to bf16 **independent of the inbound dtype**, so the resident
buffer is bf16 regardless of what the model's attention stream produced. Both
store sites that funnel into `decode_fp16_k/v` apply the floor:

- `update_prefill_raw` (the warm-TTFT seed buffer that `exit_prefill` slices
  into the decode mirror), and
- `update_decode_fp16` (the per-step decode append; the cast also sizes the
  resident `zeros(...)` allocation in bf16).

The K-only / V-only decode helper `update_decode_fp16_v_only` (used by the
IsoKOnly and RotorKOnly asymmetric codecs to write their bf16 V mirror without
disturbing the quantized K store) writes V in whatever dtype the codec provides,
which is bf16 by codec contract — it is **not** floored here because it is a
quantized-codec path, not the `KvQuant::None` / warm-TTFT path, and touching it
would violate the hard rule that the floor must not reach into quantized codec
internals.

The cast is **idempotent** — a cheap `dtype == Bf16` check returns the input
untouched (no `astype` launch) in the steady state that the per-arch source
fixes already produce, so it is pure insurance with negligible hot-path cost.
This is **defense-in-depth, not a substitute for the per-arch fix**: it caps the
*memory* damage of an upstream f32 leak (it cannot store f32) but does not fix
the *compute* slowdown — any upstream f32 arithmetic (RoPE / SDPA) stays f32.
The per-arch source fixes (Gemma4 §"Gemma4 global `--kv-quant none` KV is bf16",
Qwen3 §"Qwen3 dense `--kv-quant none` KV is bf16") remain the real fix; this
floor is the structural guard that makes the leak class impossible to re-create
silently.

The detector is a bytes-per-element invariant in
`crates/rmlx-kv-quant/src/kvcache/resident_bytes_tests.rs`: an f32 K/V fed
through the prefill-seed and decode-store paths must land as bf16 (2 B/elem). It
is wired into `make model-check` (which now runs `-p rmlx-kv-quant`), so a future
arch that leaks f32 into the unquantised KV store trips CI at integration instead
of being found months later in a bench.

**Memory cost**: `2 · B · kv_h · max_seq · head_dim · 2 bytes` per layer.
At 128K context on a 35B-A3B model this is tens of gigabytes. Reserve for
short-context parity benches only.

**CLI**: `--kv-quant none` (aliases: `bf16`, `f16`).

**Arch defaults**: `Qwen3VLMoeForConditionalGeneration` defaults to `None`
because quantized KV produces incoherent output on that checkpoint.

**Smoke-probe status**: validated across all primary test-target families.

---

### `KvStorage::K8V8` — symmetric 8-bit both sides

**K codec**: rMLX MSL `q8_0` — symmetric affine, `group_size=128`. Per-group
scale equals `max(|x|) / 127`; no bias term.
**V codec**: identical codec to K.

Both sides use the `QuantK` struct. `QuantK` maintains two parallel storage
paths:

- **CPU path**: `Vec<u8>` codes + `Vec<f32>` scales, filled by scalar Rust
  `q8_quantize` / `q8_dequantize`.
- **GPU path**: pre-allocated 1-D `Array` pair (`gpu_codes_buf` u32,
  `gpu_scales_buf` f32). Sized in multiples of `KV_PAGE_SIZE = 256` tokens,
  growing by one page when the filled sequence would exceed current capacity
  (paged growth path). Each step issues `q8_quantize_gpu` on the new slice
  then a `slice_update` into the buffer at the current offset. This avoids
  the `O(n²)` lazy-concat tree that `concatenate` would produce.

**Buffer layout (sequence-major).** The flat `QuantK` buffer (both the GPU
`Array` pair and the CPU `Vec`s) stores the filled prefix **sequence-major**:
the logical `[B, kv_h, S, D]` cache is laid out as `[B, S, kv_h, D]`, so for a
given token all heads are contiguous, and chunk `n` occupies
`[prev_seq * words_per_seq .. (prev_seq + new_seq) * words_per_seq]` with
`words_per_seq = B * kv_h * D / 4`. The per-step write places one chunk at a
sequence offset, so this is the only ordering under which appending in *any*
number of chunks keeps the active prefix readable as one contiguous slice.

`QuantK::append` therefore transposes the incoming head-major chunk
(`[B, kv_h, new_seq, D]`) to `[B, new_seq, kv_h, D]` before quantizing, and
`QuantK::dequantize_choice` reshapes the flat active prefix to `[B, S, kv_h, D]`
and transposes heads↔seq back to the logical `[B, kv_h, S, D]`. For a
single-chunk cold prefill (`prev_seq == 0`) the two transposes cancel at the
logical-mapping level, so the common path stays correct. The cold output is
**byte-identical** to the pre-fix head-major grouping only when
`head_dim % 128 == 0` (every q8 group of 128 stays inside one head) — which
holds for every current QuantK-routed target arch (Qwen3.5-MoE linear
`head_dim=128`, Gemma3 and Gemma4 text KV `head_dim=256`), so the cold path is
byte-identical in practice. When `head_dim` is not a multiple of 128 (no current
target arch, but exercised directly by the `d=64` cross-head round-trip test) a
q8 group of 128 spans a (head,token) boundary and its per-group `abs_max` scale
differs from the old grouping, so the cold path is logically correct and within
q8 noise but not bit-identical to the base commit. Without this transpose, a head-major chunk written at a
sequence offset and read with a `[B, kv_h, S, D]` reshape transposed one head's
new-token slot onto another head's prefix whenever `kv_h > 1` and the cache was
appended in more than one chunk (the multi-append-after-SSD-hydrate decode
path) — silent K corruption. The spill / hydrate / paged-grow paths copy the
contiguous active prefix `[0 .. filled]` and are layout-agnostic, so they
remain correct and the on-disk `.kvb` payload is unchanged by this ordering.

`update_and_sdpa` path:
1. `QuantK::append` — quantize new K, write into GPU buffer.
2. `QuantK::append` — same for V.
3. `QuantK::dequantize_choice` — dequantize full K prefix to bf16.
4. `QuantK::dequantize_choice` — same for V.
5. `scaled_dot_product_attention` on the recovered bf16 arrays.

**Perf characterization**: Fastest path for full-attention MoE models
(`Qwen3_5MoeForConditionalGeneration`). The per-step dequantize is bounded by
memory bandwidth; on GQA-light archs (25% FA layers) the overhead is small
relative to routing computation.

**Arch defaults**: universal fallback; every unknown arch defaults to K8V8.
Explicit defaults: `Qwen3_5MoeForConditionalGeneration`, `Qwen3ForCausalLM`
(non-ternary), `Qwen2ForCausalLM`, `Gemma4ForConditionalGeneration` (MoE and
small hidden-size variants).

**CLI**: `--kv-quant k8v8`.

**Smoke-probe status**: green on all 11 Open Models at 4K context.

---

### `KvStorage::K8V4` — q8_0 K, TurboQuant 4-bit V

**K codec**: rMLX MSL `q8_0`, `group_size=128` (identical to K8V8 K-side).
**V codec**: TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

This is an asymmetric split — K and V use different codecs. The split is
per-axis (K versus V tensor), not per-layer-index. The Python fork's
`"8,4"` flag applies different widths by layer; rMLX applies them by axis.

The V side uses `QuantV { bits: 4 }`. Layout:

- CPU path: `Vec<TurboBlocks>` — each block holds 4-bit packed codes (`Vec<u8>`)
  and per-group f32 scales (`Vec<f32>`), one block per group of 32 elements.
- GPU path: pre-allocated 1-D u32 codes buffer and f32 scales buffer.
  `words_per_step = B * kv_h * D * 4 / GROUP_SIZE` (four u32 words per group
  of 32 at 4 bits = 128 bits = 4 u32). Paged growth as in K8V8.

**Buffer layout (sequence-major).** Like `QuantK`, the `QuantV` buffer stores
the filled prefix **sequence-major** (`[B, S, kv_h, D]`) on both backends:
`append` reorders the head-major chunk heads↔seq before quantizing (GPU
`transpose` + `contiguous` so the raw-linear-index TurboQuant MSL kernel sees
the permuted bytes; CPU reorders `f32_data` and passes the seq-major shape to
the positional `turbo_quantize_v` / TCQ codec), and `dequantize_choice`
reshapes the prefix seq-major then transposes back to the logical
`[B, kv_h, S, D]` (GPU output `contiguous` for raw byte-readers / SSD spill).
Single-chunk cold prefill is the identity; byte-identical at `head_dim % 32 ==
0` (every TurboQuant group of 32 stays inside one head). Without this reorder,
a head-major store read with a `[B, kv_h, S, D]` reshape transposed heads
whenever `kv_h > 1` and the cache was appended in more than one chunk (the
post-SSD-hydrate decode-append path) — silent V corruption. The same
sequence-major ordering applies to the K-side `QuantKTurbo3` / `QuantKTurbo4`
structs (`TurboSym3` / `TurboSym4`) and to the paged K/V handoff in
`update_paged` (which reorders `new_k`/`new_v` to seq-major before quantizing,
since the page slabs are physically token-major).

`update_and_sdpa` path (without TurboFlash):
1. Append K via `QuantK::append`.
2. Append V via `QuantV::append` (calls `turbo_quantize_v4_gpu` on GPU).
3. Dequantize full K prefix via `QuantK::dequantize_choice`.
4. Dequantize full V prefix via `QuantV::dequantize_choice` (calls
   `turbo_dequantize_v4_gpu`).
5. `scaled_dot_product_attention` on bf16 arrays.

**TurboFlash path** (`KvCache::update_and_sdpa_k8v4_flash`): maintains a
parallel set of head-major buffers (`flash_k_codes`, `flash_k_scales`,
`flash_v_codes`, `flash_v_scales`) shaped `[B, kv_h, max_seq, D/.]`. These
are seeded once from the prefill bf16 prefix on the first decode step, then
appended per-token via 4-D `slice_update`. The `turbo_flash_sdpa` Metal
kernel reads these buffers directly — no dequantize round-trip. Enabled by
`RMLX_TURBO_FLASH=1` or `turbo_flash_lock_enabled()`.

**TurboFlash default-on policy**: `--turbo-flash`
accepts `{on, off, auto}` with `auto` as the default. `auto` resolves at
startup via `rmlx_core::apple_gpu::apple_silicon_generation()`:
- Apple ≤9 (M1/M2/M3/M4) → ON (validated: 32k NIAH × 3 models on M3,
  commit fcb2e894ccc4; 100% needle retrieval at 5 depths × 3 ctx tiers).
- Apple10 (M5+) → ON: the historical `head_dim = 256`
  hazard was re-driven on M5 Max via
  `crates/rmlx-kv-quant/tests/apple10_head_dim_256.rs` (synthetic
  K8V4, smoke + 16-step decode stress + TF=0 control). No SIGSEGV,
  dispatch fired, cosine min 0.997 vs bf16.
- Apple11+ (M6 +) → ON with an `info`-level log noting the family has not
  been hw-validated yet. Operators that hit a regression can force OFF
  with `--turbo-flash off`.
- Unknown / non-Apple-Silicon hosts (sysctl probe failed) → OFF
  (conservative).

`--turbo-flash on` is an explicit force-ON. `--turbo-flash off` is a hard
override that removes `RMLX_TURBO_FLASH` from the environment so a stale
shell value cannot latch the OnceLock back to true.

**head_dim coverage (TurboFlash kernel)**: `head_dim ∈ {128, 256}`. The
P1 kernel's register arrays (`q_vals`, `o_state`, `v_decoded`) are sized
for the larger dim (8 entries = 256/32). `head_dim = 256` is
hw-validated on Apple10 (M5 Max) — historical SIGSEGV
hazard at this size does not reproduce.

**Qwen MoE note**: K8V4 is safe for Qwen MoE because K stays 8-bit. Running
K below 8-bit on a 7:1 GQA model amplifies quantization error through
softmax and produces catastrophic PPL degradation (218 → 8641 observed).
The K-side codec is the safeguard — not the variant name.

**Arch defaults**: `Qwen3_5ForConditionalGeneration` (PARO checkpoints);
`Gemma4ForConditionalGeneration` (small + PARO); auto-by-ctx at ≤8192 tokens.

**CLI**: `--kv-quant k8v4`; or `--ctk q8_g128 --ctv tq4`.

**Smoke-probe status**: green on all primary test-target families.

---

### `KvStorage::Planar` — q8_0 K, PlanarQuant 4-bit V

**K codec**: rMLX MSL `q8_0`, `group_size=128` (same as K8V8 / K8V4).
**V codec**: PlanarQuant 4-bit with per-pair Givens rotation, `group_size=32`.

PlanarQuant stores three parallel buffers per V side:

- `gpu_codes_buf` (u32): four u32 words per group of 32 elements.
- `gpu_scales_buf` (f32): one f32 scale per pair of elements — 16 per group
  (16× more fine-grained than TurboQuant's one scale per group of 32).
- `gpu_rotations_buf` (u32): two u32 words per group, encoding eight 4-bit
  Givens rotation indices per word.

The Givens rotation operates on pairs of V values before 4-bit quantization.
This per-pair micro-rotation decorrelates adjacent channels and reduces
per-element reconstruction error versus TurboQuant V4 on Gaussian-distributed
KV vectors by approximately 2–3×.

Storage struct: `QuantPlanarV`. CPU path uses scalar `planar_quantize` /
`planar_dequantize` from `rmlx_quant::planarquant`. GPU path calls
`planar_quantize_v4_gpu` / `planar_dequantize_v4_gpu` MSL kernels.

`update_and_sdpa` path:
1. `QuantK::append` for K (identical to K8V8).
2. `QuantPlanarV::append` — calls `planar_quantize_v4_gpu` on GPU.
3. `QuantK::dequantize_choice` for K.
4. `QuantPlanarV::dequantize_choice` — calls `planar_dequantize_v4_gpu`,
   passing all three buffer arrays (codes, scales, rotations).
5. `scaled_dot_product_attention`.

**Perf characterization**: wins TPS outright vs K8V8 at long context (≥32K)
on dense full-attention archs. Verified at 64K context on Qwen3.6-35B-A3B
(71.53 TPS Planar vs 65.2 TPS K8V8). The per-pair scale buffers give PlanarQuant
V ≈4.4× the resident memory of tq4 V at head\_dim=128 (≈352 B vs ≈80 B per
token per kv\_head: codes 64 B + 16 scales/group × 4 B × 4 groups = 256 B +
rotations 32 B). The quality gain from finer scales and per-pair rotation
justifies this for dense full-attention archs at long context.

**Arch defaults**: `Gemma3ForConditionalGeneration`;
`Gemma4ForConditionalGeneration` (dense, hidden_size ≥ 5376); auto-by-ctx at
>32K tokens.

**CLI**: `--kv-quant planar`; or `--ctk q8_g128 --ctv planar4`.

**Smoke-probe status**: green on primary test targets.

---

### `KvStorage::Planar` (bits=3) — PlanarQuant 3-bit V

**KvQuant variant**: `KvQuant::Planar3`. Routes to `KvStorage::Planar { bits: 3 }` — no new storage variant.

**Algorithm**: same Givens-rotation + per-pair scale as the 4-bit codec. Diverges at quantization:
- **Codebook**: 3-bit Lloyd-Max N(0,1) — 8 centroids (from `CODEBOOK_3BIT` in `turboquant.rs`).
- **Pack format**: 10 vals/u32 (3 × 10 = 30 bits used, 2 wasted per u32). With GROUP_SIZE=32:
  `ceil(32/10) = 4` u32 words per group — **identical word count** to the 4-bit codec (8 vals/u32 × 4).
  ForgeAttention-compatible buffer shape.
- **Decision boundaries**: 7 midpoints (vs 15 for 4-bit).
- **Mask**: `0x7u` (3 bits, vs `0xFu` for 4-bit).

**Path-independent byte stream.** Both the CPU codec and the MSL kernels pack
codes in this same word convention (`word = elem / (32/bits)`,
`shift = (elem % (32/bits)) * bits`, little-endian u32 words), so the code bytes
round-trip across the CPU/GPU boundary unchanged — required for SSD spill (CPU
encode) → hydrate (GPU read). For `bits=4` (8 vals/u32) the word convention is
byte-identical to a dense LSB-first stream; for `bits=3` (10 vals/u32) it is
**not** dense. A dense 3-bit layout (12 bytes/group) would be misread by the GPU
as 4 u32 words (16 bytes/group) and silently corrupt the V cache. iso3 and
rotor3 share this convention; PlanarQuant 3-bit now does too.

The GPU kernels are `planar_quantize_v3_gpu` / `planar_dequantize_v3_gpu` in `planarquant_msl.rs`;
CPU path uses `planar_quantize(bits=3)` / `planar_dequantize` from `planarquant.rs`.

**Cosine gate**: mean cosine ≥ 0.9989 on LCG fixture (measured 0.999956 — very high
because per-pair rotation+scale compresses correlated pairs extremely well even at 3 bits).

**Memory**: same codes buffer size as 4-bit (4 words/group), with per-pair scales and rotation arrays.

**CLI**: `--kv-quant planar3`; or `--ctk q8_g128 --ctv planar_3`; or `--kv-preset planar3`.

**Smoke-probe status**: CPU codec verified; the V3 **quantize** kernel
(`planar_quantize_v3_gpu`) is **precompiled at load** via
`precompile::warm_v_side`, so a first `--kv-quant planar3` request pays no
Metal cold-compile stall on the prefill V-encode path (the separate
`planar_dequantize_v3_gpu` kernel still cold-compiles lazily on first cache
read). The GPU round-trip test `planar_v3_msl_roundtrip_within_tolerance`
exists and passes but is `#[ignore]`-gated (needs a local Metal context — run
with `cargo test -p rmlx-kv-quant --release -- --ignored`).

---

### `KvStorage::Mixed` — MLX affine at arbitrary (bits, group_size)

**K codec**: `mx.quantize(mode="affine", bits=k_bits, group_size=k_group_size)`.
**V codec**: `mx.quantize(mode="affine", bits=v_bits, group_size=v_group_size)`.

Default parameters: `k_bits=8, v_bits=4, k_group_size=64, v_group_size=64`.
These match `mlx-lm-turboquant`'s `MixedQuantKVCache` defaults exactly.

The affine codec stores a 3-tuple `(codes_u32, scales_f32, biases_f32)` per
side. Reconstruction: `x = scale * code + bias`. This differs from rMLX MSL
`q8_0` (symmetric, no bias term; `Q8_GROUP_SIZE=128`) despite both being
nominally "8-bit affine". The two codecs are not interchangeable.

State is owned by `MixedKvState` in `mixed_quant.rs`. Six pre-allocated
`[B, kv_h, max_seq, D/.]` buffers grow in `STEP=256` token increments. Each
decode step:
1. `MixedKvState::update_and_fetch` — calls `mx.quantize` on new K and V
   slices, writes into the six buffers via `slice_update`, returns views of
   the filled prefix as two `MixedTuple` structs.
2. `mixed_quantized_sdpa` — runs two `mx.quantized_matmul` calls (queries @ K
   then probs @ V) directly on the stored 3-tuples without a dequantize
   round-trip.

Prefill bulk path (`exit_prefill`): accumulates raw bf16 during prefill, then
`bulk_init_from_fp16` issues a single batched `mx.quantize` per side (no
per-step quantize overhead during prompt processing).

**Key distinction vs K8V4/K8V8**: Mixed uses the portable MLX affine
quantizer with `(scale, bias)` per group. The K8V4/K8V8/Planar K-side uses
rMLX MSL q8_0 with symmetric `scale = max(|x|)/127` and no bias.

**`KvQuant::RotK`**: reuses `KvStorage::Mixed` with `rotate_k=true` on
`MixedKvState`. K bits are fixed at 8, group_size fixed at 64. The storage
and SDPA machinery is identical to plain Mixed; the only difference is that
`MixedKvState::update_k_and_fetch` applies a Hadamard rotation to K before
quantization, and `mixed_quantized_sdpa` applies the same rotation to Q
before the score matmul so the rotations cancel. See rot_k below.

**Perf characterization**: +24% decode TPS vs K8V4 on Bonsai (36/36
full-attention layers), because `quantized_matmul` eliminates the per-step
full dequantize that dominates the rMLX K8V4 hot path at long sequences.
On GQA-light MoE archs (25% FA layers) Mixed K8V4 regresses by 11–28% vs
K8V8 — the `quantize` + `quantized_matmul` overhead amortises poorly when
most layers are not full-attention.

**Arch defaults**: `Qwen3ForCausalLM` at `weight_bits=2` (Bonsai ternary).

**CLI**: `--kv-quant mixed_k<kb>g<kg>_v<vb>g<vg>` (e.g.
`mixed_k8g64_v4g64`). The `RotK` variant is reached via `--ctk rot_k` (see
CLI flags section below).

**Smoke-probe status**: green on Bonsai (Qwen3ForCausalLM, bits=2).

---

### rot_k — K-side Hadamard rotation

**Math**: attention scores are `Q · Kᵀ`. Insert orthogonal rotation `R`
(`Rᵀ R = I`) into the K basis and pre-rotate Q by the same `R`:

```
(Q Rᵀ) · (K Rᵀ)ᵀ = (Q Rᵀ) · (R Kᵀ) = Q (Rᵀ R) Kᵀ = Q Kᵀ
```

Storing rotated K (`K_rot = K Rᵀ`) and pre-rotating queries (`Q_rot = Q Rᵀ`)
before the score matmul leaves attention scores identical to the unrotated
computation up to quantization error on `K_rot`. A Hadamard rotation
decorrelates K channels and equalizes their dynamic range, reducing affine
quantization error in the rotated basis.

K is never inverse-rotated — the rotation cancels algebraically. This
distinguishes rot_k from V-side rotation schemes (PlanarQuant, TurboQuant)
where the output must be un-rotated back to the value basis.

`R` is the normalized Walsh–Hadamard matrix `H_D / sqrt(D)`. It is orthogonal
and symmetric (`R = Rᵀ`), so the same matrix rotates both K and Q.
Construction requires a power-of-two head_dim (Sylvester recurrence).

**v1 path** (`rot_k.rs`): plain MLX `matmul` against a precomputed `[D, D]`
matrix. Correct and coherent; O(D²) arithmetic per step.

**Fused FWHT kernel** (`rot_k_msl.rs`, opt-in via `RMLX_ROT_K_FUSED=1`):
Fast Walsh-Hadamard Transform in Metal threadgroup shared memory, fused with
affine-8-bit quantize in a single kernel pass. O(D log₂ D) arithmetic and no
intermediate DRAM allocation for `K_rot`. For D=128 (Bonsai): 896 arithmetic
ops vs 16 384 for the matmul (~18×). Output format is bit-exact with
`mx.quantize(mode="affine", bits=8, group_size=64)` so it feeds directly into
`mixed_quantized_sdpa` unchanged.

A matching `rot_k_fwht_rotate_gpu` kernel applies the same FWHT to Q,
replacing the `rotate_last_axis` matmul when the fused path is active.

**Storage**: `KvStorage::Mixed` (or `KvStorage::RotKTq4V` for the hybrid).
`MixedKvState` carries a `k_rotation: Option<Array>` field with the
precomputed `R` matrix.

**Requirements**: power-of-two `head_dim`. Pair with any affine V codec
(default V = `q4_g64`).

**CLI**: `--ctk rot_k [--ctv <affine-tag>]`.

**Cosine gate**: K-side cosine similarity ≥ 0.9970 (mean), ≥ 0.9950 (min)
on LCG fixture data (head_dim=64, 8-bit affine group_size=64).
Test: `rot_k_hadamard_8bit_cosine_gate` in `rot_k_tests.rs`.

---

### `KvStorage::RotKTq4V` — rotated K + TurboQuant 4-bit V

A hybrid: K uses the rotated affine 8-bit codec (same as `RotK`), V uses
TurboQuant 4-bit (same V codec as `K8V4`).

Storage is split between two sub-structs:
- `k_state: MixedKvState` with `rotate_k=true` — holds the K-side affine
  3-tuple and the `R` matrix.
- `v: Option<QuantV>` — holds TurboQuant 4-bit V codes and scales.

SDPA path (`rot_k_tq4v_sdpa`):
1. `k_state.update_k_and_fetch` — rotate K, quantize, `slice_update`, return
   the full prefix 3-tuple.
2. `QuantV::append_uncapped` — TurboQuant-encode the new V token.
3. `QuantV::dequantize_choice` — dequantize full V prefix to bf16.
4. Dequantize K from its affine 3-tuple to bf16 via `mx.dequantize`.
5. Pre-rotate Q by `R` (`rotate_last_axis` or fused FWHT kernel).
6. `scaled_dot_product_attention` on the recovered bf16 arrays.

This is a dequant-then-SDPA path (not fused `quantized_matmul`). The V-side
still benefits from the full 4× memory reduction of tq4 versus bf16.

**Requirements**: power-of-two `head_dim` (Hadamard); `head_dim ∈ {128, 256}`
(TurboQuant kernel constraint).

**K group_size**: fixed at 64 (matches RotK). V group_size: 32 (TurboQuant).

**CLI**: `--ctk rot_k --ctv tq4`.

---

### `KvStorage::K8VTurbo3` — q8_0 K, TurboQuant 3-bit V

**Status**: opt-in for non-Gemma4-small archs; no longer the auto default for
Gemma4 small (reverted to K8V8 per the composite-score audit; see
per-arch defaults below). Available via `--kv-quant k8vturbo3`.

**K codec**: rMLX MSL q8_0, `group_size=128` (same as K8V8 K-side).
**V codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

The 3-bit codebook has 8 centroids. Pack format: 32 × 3 bits = 96 bits = three
u32 words per group. This gives 3/4 the memory of 4-bit V versus approximately
the same decode complexity.

**Promotion bench** (canary shape 4096 prompt tokens, release-perf binary):

| Model | K8V4 median TPS | K8VTurbo3 median TPS | Delta |
|---|---:|---:|---:|
| Gemma4-e4b | 74.670 | 74.370 | −0.40% |
| Qwen3.6-35B | 97.869 | 95.958 | −1.95% (opt-in only) |
| Bonsai 8B | 91.235 | 99.055 | +8.6% (not arch target) |

Gemma4-e4b −0.40% is within the <1% promote gate. Cosine gate ≥ 0.9807
passes. Smoke probe green on all 3 models.

An earlier bench at 17K context showed −3.5% (e4b) vs `Mixed{v_bits:3}`,
which failed the −2% gate. That shape had thermal crosstalk between back-to-back
long-prefill runs. The canary 4K shape shows the codec is within noise.

The CPU dequant path is canonical; the MSL module (`k8vturbo3_append_msl.rs`)
is retained as a future-reference hook.

**CLI**: `--kv-quant k8vturbo3`.

---

### `KvStorage::K8VTurbo3Tcq` — q8_0 K, TurboQuant 3-bit V with Viterbi trellis

**Status**: opt-in via `--kv-quant k8vturbo3tcq`. Never an auto baseline.
Turbo3-equivalent quality (same Lloyd-Max codebook, degenerate trellis — see
note below).

**K codec**: rMLX MSL q8_0, `group_size=128` (same K-side as K8VTurbo3).
**V codec**: TurboQuant 3-bit Lloyd-Max N(0,1) codebook, `group_size=32`. The
**codebook is unchanged from plain K8VTurbo3** — quality comes purely from
smarter encode-side assignment: a 4-state Viterbi trellis (rate-1/2
convolutional code, `TCQ_NUM_STATES = 4`) replaces nearest-centroid.

Transition rule: `next_state = ((state << 1) | (level & 1)) mod NUM_STATES`.
Per-block forward + back-trace runs over the 32-element group; back-pointer
table is `32 × 4 × 2 bytes` per block.

The **decoder is bit-identical to plain `turbo_dequantize`** — TCQ output is
byte-for-byte indistinguishable from a `K8VTurbo3` pack at the codes / scales
level. The two codecs share `k8vturbo3_append_msl::turbo_dequantize_v3_gpu`.
Only the `KvQuant` discriminator and the SSD layout-key tag
(`K8VTURBO3_TCQ_LAYOUT_TAG = "k8vturbo3tcq"`) distinguish them on disk; the
SSD layer hard-rejects cross-codec hydrate to prevent a TCQ payload from
silently being treated as plain turbo3 (and then mixed with
nearest-centroid indices on the next decode append).

**Cosine target**: ≥ 0.9807 on the canonical LCG fixture (mtq `turbo3_tcq`
row 0.9817 − 0.001 empirical floor). The load-bearing quality test in
`tcq_tests.rs` asserts TCQ ≥ plain turbo3 cosine on a non-Gaussian
(sinusoidal) fixture — a non-regression gate satisfied trivially by equality,
not a demonstration of a strict quality win.

**Trellis degeneracy note.** The per-step Viterbi cost is
`dist(value, codebook[level])`, which depends only on the chosen level, not
on the trellis state. Because every level is reachable from every state and
the codebook is state-independent, the minimum-cost Viterbi path equals the
greedy nearest-centroid assignment. TCQ output is therefore bit-identical to
plain turbo3 on unstructured data with the same codebook. A state-dependent
(grade-aware) codebook would be required to obtain a shaping gain; that
follow-up is deferred. The `>=` quality gate in `tcq_tests.rs` is satisfied
by equality and does not demonstrate a strict improvement over plain turbo3.

**Bench** (canary shape 4096 prompt, 100 decode tokens, release-perf binary, 3-run mean):

| Model | k8vturbo3 (TPS) | k8vturbo3tcq (TPS) | Delta |
|---|---:|---:|---:|
| Bonsai 8B | 98.95 | 95.11 | −3.9% |
| Gemma4-e4b | 73.16 | 73.54 | +0.5% |
| Qwen3.6-35B | 97.17 | 94.57 | −2.7% |

All three within the −10% gate. The Bonsai overhead reflects the
sequential per-token Viterbi loop (4 states × 8 levels × 32 dims per block);
Gemma4 wider attention amortises it.

**Calibration recipe**: `--recipe turbo3_tcq` in `rmlx kv-calibrate` maps to
the internal `turboquant35` recipe (same as plain `turbo3` / `turbo4`):
emits `high_precision_indices` only; **no codebook override** is written
because TCQ reuses the standard Lloyd-Max codebook. Calibration runtime is
identical to plain `turbo3` (~30 s on a 7B model).

**Implementation scope**: CPU Viterbi encode + CPU dequant on the hot
path. The MSL Viterbi kernel
([`tcq_v_msl::tcq_quantize_v3_gpu`](../crates/rmlx-kv-quant/src/tcq_v_msl.rs))
is parity-tested CPU↔GPU (bit-identical codes + scales) but ships as a
future-reference hook (precedent: K8VTurbo3 / K8VTurbo2 MSL hooks both
regressed the −2 % TPS gate when wired on the hot path).

**V-side only**: TCQ is V-side only. K stays `q8_0` (group=128, no Viterbi).
The Viterbi trellis is not applied to `QuantK`. `K8VTurbo3Tcq` therefore keeps
the asymmetric K8/V3.25 shape and is never rejected by the Qwen MoE K-bits
guard (K = 8 ≥ 8).

**CLI**: `--kv-quant k8vturbo3tcq` (also surfaced as `CacheType::Turbo3Tcq`
with canonical tag `k8v_turbo_3_tcq`, alias `turbo3_tcq` in
`--ctv turbo3_tcq`).

---

### `KvStorage::K8VTurbo2Tcq` — q8_0 K, TurboQuant 2-bit V with Viterbi trellis

**Status**: opt-in via `--kv-quant k8vturbo2tcq`. Never an auto baseline.
Turbo2-equivalent quality (same Lloyd-Max codebook, degenerate trellis — same
caveat as K8VTurbo3Tcq above).

**K codec**: rMLX MSL q8_0, `group_size=128` (same K-side as K8VTurbo2).
**V codec**: TurboQuant 2-bit Lloyd-Max N(0,1) codebook (`CODEBOOK_2BIT`,
4 centroids), `group_size=32`. **Codebook unchanged from plain K8VTurbo2** —
quality comes from Viterbi-optimal encode assignment (same 4-state trellis as
K8VTurbo3Tcq, but over 4 centroids instead of 8).

Pack format: 2-bit indices, 16 values per u32 (2 u32 words per 32-element
block = 64 bits) — identical to plain `turbo_quantize_v` at `bits=2`. The
decoder is `turbo_dequantize` with no TCQ-specific path.

The **decoder is bit-identical to plain `turbo_dequantize`** — TCQ output at
2-bit is byte-for-byte indistinguishable from a `K8VTurbo2` pack. The SSD
layout-key tag `K8VTURBO2_TCQ_LAYOUT_TAG = "k8vturbo2tcq"` prevents silent
cross-codec hydrate (TCQ payload must not be treated as plain turbo2 and then
mixed with nearest-centroid indices on the next decode append).

**Cosine target**: ≥ 0.957 on the canonical LCG fixture (empirical measured
value ~0.9579; floor = measured − 0.001 ≈ 0.957). The load-bearing quality
test in `tcq_tests.rs` further asserts TCQ V2 ≥ plain turbo2 cosine on the
sinusoidal fixture.

**V-side only**: TCQ is V-side only. K stays `q8_0` (group=128, no Viterbi).
`K8VTurbo2Tcq` therefore keeps the asymmetric K8/V2.25 shape.

**Outlier-mask deferred**: The `high_precision_indices` outlier-mask wiring
(present in the 3-bit path) is deferred. Ships the naïve Viterbi path over
the unmasked 2-bit codebook.

**MSL hook**: `tcq_v2_msl::tcq_quantize_v2_gpu` is parity-tested CPU↔GPU
(bit-identical codes + scales at 2-bit) but ships as a future-reference hook.
The hot path forces `Device::Cpu`.

**Calibration recipe**: `--recipe turbo2_tcq` in `rmlx kv-calibrate` maps to
the internal `turboquant25` recipe (same as plain `turbo2`). No codebook
override written.

**CLI**: `--kv-quant k8vturbo2tcq` (also surfaced as `CacheType::Turbo2Tcq`
with canonical tag `k8v_turbo_2_tcq`, alias `turbo2_tcq` in
`--ctv turbo2_tcq`).

---

### `KvStorage::TurboSym4` — symmetric TurboQuant 4-bit K + V

**Status**: opt-in via `--kv-quant tsym4` (or the `quality` preset). Never an
auto baseline.

**K codec**: TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`
(K-axis use of the axis-agnostic V codec).
**V codec**: same — TurboQuant 4-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

This is the symmetric counterpart of `K8V4`: both axes use the **same**
TurboQuant 4-bit MSL kernel (`turboquant_msl::turbo_quantize_v4_gpu` /
`turbo_dequantize_v4_gpu`). The CPU + MSL codecs are axis-agnostic — they
take a flat f32 buffer plus a 4-D shape and produce flat codes/scales —
so the K side and V side share dispatch, **no kernel fork** (shared dispatch).

The K and V buffers are kept as **independent types** (`QuantKTurbo4` and
`QuantV`), not a renamed wrapper, so the two append paths stay decoupled
inside `KvStorage::TurboSym4 { k, v, max_seq }`. Layout tag (single source
of truth for the SSD geometry header):

```
const TURBOSYM4_LAYOUT_TAG: &str = "tsym4_wht_4_4";
```

**Arch guard (CLAUDE.md hard rule 6)** — symmetric 4-bit K is the PPL-218→8641
disaster path on Qwen MoE. `--kv-quant tsym4` on a
`Qwen3_5MoeForConditionalGeneration` checkpoint is rejected at resolve-time
by `validate_resolved` with `ResolveError::QwenMoeKBitsTooLow(4)` (exit 78,
same surface as the existing Mixed K<8 rejection). The helper
`KvQuant::k_below_8bit()` returns `true` for this variant — extend the
helper when adding any future sub-8-bit-K codec.

`KvCacheBuilder::resolve_default` never returns `TurboSym4` for any arch.

**Paged routing**: `PagedKStorage` is q8-only; adding a TurboQuant-K paged
variant requires a separate page allocator and gather kernel.
`KvStorage::new(KvQuant::TurboSym4, max_seq)` therefore returns the
**non-paged** `TurboSym4` storage even when `--paged-kv` is set.

**Tail/head adaptive fallback** — `kv_quant_for_layer` falls back to `K8V8`
(8-bit K) on the head / tail layers. `TurboSym4` is **not** added to the
tail/head candidate set; the fallback to `K8V8` is the correct safety net.

**Closes** the asymmetric-K8V4 gap for mtq's `quality` / `agents_*` presets.

**CLI**: `--kv-quant tsym4` (or `--kv-preset quality`).

---

### `KvStorage::TurboSym3` — symmetric WHT-3 K + turbo3 V

**Status**: opt-in via `--kv-quant tsym3` (or `--kv-preset speed`).
Never an auto baseline.

**K codec**: TurboQuant 3-bit Lloyd-Max N(0,1) 8-centroid codebook, `group_size=32`.
On GPU: `turbo_quantize_v3_gpu` / `turbo_dequantize_v3_gpu` MSL kernel from
`k8vturbo3_append_msl.rs` (same kernel as V-side turbo3 — axis-agnostic,
no fork needed). On CPU: `turbo_quantize_v(bits=3)`.

**V codec**: TurboQuant 3-bit Lloyd-Max N(0,1) 8-centroid codebook,
`group_size=32` — same codec as V in `K8VTurbo3`, dispatched via `QuantV { bits:3 }`.
V-side is CPU-forced (same as K8VTurbo3 precedent — GPU V-side dispatch
regressed −2% TPS; see K8VTurbo3 finding).

Both K and V use the **same codebook** — the symmetric designation is
literal: the codec treats K and V identically.

The K buffer is `QuantKTurbo3` (independent type from `QuantK` and
`QuantKTurbo4`), decoupled from V to keep append paths separate.
Layout tag (single source of truth for SSD geometry header):

```
const TURBOSYM3_LAYOUT_TAG: &str = "tsym3_wht_3_3";
```

**Arch guard (Contract A.y — mandatory)** — K-side 3-bit on Qwen MoE is the
PPL-disaster zone. `--kv-quant tsym3` and `--kv-preset speed` on
`Qwen3_5MoeForConditionalGeneration` or `Qwen3VLMoeForConditionalGeneration`
are rejected at resolve-time by `validate_resolved` with the dedicated
`ResolveError::QwenMoeTurboKRejected { variant: "tsym3" }`.

**Paged routing**: `KvStorage::new(KvQuant::TurboSym3, max_seq)` returns
non-paged `TurboSym3` storage even when `--paged-kv` is set.

**Tail/head adaptive fallback** — `kv_quant_for_layer` falls back to `K8V8`
on head/tail layers. `TurboSym3` is not added to the tail/head candidate set.

**Matches** mtq `speed` preset (`TurboSym3` = `turbo3_symm` in paroquant
nomenclature).

**CLI**: `--kv-quant tsym3` (or `--kv-preset speed`).

---

### `KvStorage::PlanarK` — K-axis PlanarQuant 4-bit

**Status**: opt-in via `--kv-quant planar_k` (or the `k_only_planar` preset).
Never an auto baseline.

**K codec**: PlanarQuant 4-bit Givens-rotation codec (16-entry rotation
codebook + 4-bit code, per-pair scales) — the same scalar
`planarquant::planar_quantize` and the same MSL kernel
(`planarquant_msl::planar_quantize_v4_gpu` / `planar_dequantize_v4_gpu`)
already used by `KvStorage::Planar` on the V side. PlanarQuant is
axis-agnostic at the kernel input level (flat `[B, kv_h, S, D]` with
`D % 32 == 0`), so the K side and V side **share dispatch** — shared kernel, no fork.
**V codec**: unquantised bf16 (lives on `KvCache::decode_fp16_v`, same
machinery as `KvStorage::None` for the V buffer).

**Buffer layout (sequence-major).** Like every other flat-buffer quantized KV
storage, the `QuantPlanarK` (and `QuantPlanarV`) buffer stores the filled
prefix **sequence-major** (`[B, S, kv_h, D]` element order): per token, all
heads are contiguous. `append` reorders the incoming head-major chunk heads↔seq
before quantizing (GPU: `transpose` then `Array::contiguous`, since the
raw-linear-index MSL kernel ignores lazy-transpose strides; CPU: the
`transpose_heads_seq` mirror), and `dequantize_choice` reshapes the prefix
`[B, S, kv_h, D]` and transposes back to the logical `[B, kv_h, S, D]`. For a
single decode token the transpose is the identity (hot path byte-unchanged);
for a single cold-prefill chunk the two transposes cancel. PlanarQuant is
layout-agnostic (group-by-group over the flat stream, `head_dim % 32 == 0`, so
no group spans a (head, token) boundary), so the reorder is **bit-exact** —
planar3 / planar4 packing untouched. This closes the multi-append head-scramble
class (the SSD-hydrate-then-reprefill path) for the whole codec family.

Because `QuantPlanarK` also feeds its packed codes to the GPU kernels via
`gpu_packed_view`, those kernels index K **sequence-major** to match:
`planar_fused_qk`, `planar_flash_decode` (P1), and the sparse-attn phase-1/2
score kernels compute the K token base as
`kv_tok = (b * kv_seq + s) * kv_h + kv_h_idx`. The V offset in the flash /
sparse kernels stays head-major — V is the separate bf16 decode mirror, not the
planar-packed buffer.

The K buffer is kept as an independent type (`QuantPlanarK`, layout-identical
to `QuantPlanarV` but distinct so K and V append paths stay decoupled)
inside `KvStorage::PlanarK { k, max_seq }`. Layout tag (single source of
truth for the SSD geometry header):

```
const PLANARK4_LAYOUT_TAG: &str = "planar_k_4";
```

**Arch guard (Contract A.y — mandatory, CLAUDE.md hard rule 6)** — K-side
4-bit on Qwen MoE is the PPL-218→8641 disaster path (7:1 GQA amplifies
K-head error through softmax). `--kv-quant planar_k` and
`--ctk planar_k4 --ctv bf16` on `Qwen3_5MoeForConditionalGeneration` or
`Qwen3VLMoeForConditionalGeneration` are rejected at resolve-time by
`validate_resolved` with the dedicated `ResolveError::QwenMoePlanarKRejected`
(distinct from `QwenMoeKBitsTooLow` so the K-side disaster surface is
preserved in the diagnostic). The helper `KvQuant::k_below_8bit()`
returns `true` for this variant — extend the helper when adding any future
sub-8-bit-K codec.

`KvCacheBuilder::resolve_default` never returns `PlanarK` for any arch.

**Paged routing**: there is no `PagedPlanarKStorage`.
`KvStorage::new(KvQuant::PlanarK, max_seq)` returns the non-paged `PlanarK`
storage even when `--paged-kv` is set.

**Tail/head adaptive fallback** — `kv_quant_for_layer` falls back to `K8V8`
on the head / tail layers. `PlanarK` is **not** added to the tail/head
candidate set; the fallback to `K8V8` is the correct safety net.

**Mirrors** mtq's `k_only_planar` preset
(`../multi-turboquant/multi_turboquant/presets.py`).

**CLI**: `--kv-quant planar_k` or `--ctk planar_k4 --ctv bf16`
(or `--kv-preset k_only_planar`).
### `KvStorage::K8VTurbo2` — q8_0 K, TurboQuant 2-bit V

**Status**: native 2-bit V codec, ships **naïve** (no outlier-mask). Not a
production default for any arch.

**K codec**: rMLX MSL q8_0, `group_size=128` (same as K8V8 K-side).
**V codec**: TurboQuant 2-bit Lloyd-Max N(0,1) codebook, `group_size=32`.

The 2-bit codebook has 4 centroids. Pack format: 32 × 2 bits = 64 bits = two
u32 words per group. This gives 1/2 the memory of 4-bit V versus approximately
the same decode complexity — same compression target as multi-turboquant's
`turbo2` row (~5.8–7× over bf16 V when combined with q8_0 K).

A Metal kernel (`turbo2_v_msl.rs`) is wired as a future-reference hook,
mirroring `k8vturbo3_append_msl.rs`. Following the K8VTurbo3 finding (Metal
3-bit kernel regressed Gemma4-e4b/26b by 3.5%/6.9%, failing the −2% gate),
the V-side is kept on CPU on the hot update path. The MSL module is
unit-tested for bit-exact CPU↔GPU equivalence so that re-wiring it later
(once a bench shows a TPS win) is a one-line dispatch-site change.

**Naïve 2-bit caveat**: ships the **naïve** Lloyd-Max 2-bit codec without
outlier-mask. multi-turboquant's published GPU cosine (`README.md` method row
1: 0.9420, 5.8× compression) is *with* its `build_outlier_masks` + offline
calibration. rMLX empirical cosine on the LCG-seeded uniform fixture is mean =
0.9579, min = 0.9269 (n_rows = 512; see
`turbo2_v_msl_tests.rs::tq2_cosine_naive_baseline_floor`) — but the fixture is
uniform, not real V tensors, so the numbers are not directly comparable to
mtq's bench. The expected production gap on real model V tensors comes from the
missing heavy-tail residual that outlier-mask handles. Outlier-mask + calibration
wiring is a deferred follow-up.

**Deferred outlier-mask plan**:

- Port `build_outlier_masks` from `multi-turboquant/multi_turboquant/methods/turboquant.py`.
- Wire calibration-derived per-channel outlier masks through the QuantV bits=2
  encode + dequant paths.
- Re-measure cosine against the calibrated fixture; lift the cosine floor
  in `tq2_cosine_naive_baseline_floor` once the gap closes.

**CLI**: `--kv-quant k8vturbo2`. Like K8VTurbo3 the codec has **no**
`--ctk`/`--ctv` axis entry: it is accessible only via the preset flag.
This matches the K8VTurbo3 convention (a single `KvQuant` enum variant
without a `CacheType` registration), keeping the per-side axis reserved
for standard affine + rotation codecs.

---

### `KvStorage::Paged` — vLLM-style block-table KV

PagedAttention allocation (opt-in via `--paged-kv` flag; default OFF).

Instead of a single contiguous buffer grown in `KV_PAGE_SIZE=256` token
increments (contiguous-growth path), `Paged` maintains:

1. A page pool — pre-allocated slab of N fixed-size GPU arrays, controlled by
   `RMLX_KV_PAGE_SIZE` (default 32 tokens per page).
2. A per-sequence block table — `Vec<usize>` mapping logical page index to
   physical page ID in the pool.
3. Scatter/gather — writes land into `pool[phys_id][token_slot]`; reads
   concatenate the active pages in order.

For single-request decoding the block table is monotonically appended (no
sharing, no eviction), degenerating to contiguous-growth behaviour with the same
peak memory and TPS. The value is future continuous-batching support where
different requests can share a pool and return pages on completion.

V codec is determined by the base `KvQuant`:
- `K8V4` → `PagedVStorage` (TurboQuant 4-bit).
- `K8V8` → `PagedVStorage` (q8_0 codes, same struct, `bits=8`).
- `Planar` → `PagedPlanarVStorage`.

Page size must be a multiple of the quantizer group size (32 for TurboQuant,
128 for q8_0 K) to avoid straddled groups at page boundaries.

**Restrictions**: `--paged-kv` is rejected for `KvQuant::None` (bf16 paged
is not implemented) and for `RotK` / `RotKTq4V` (rotation codecs are not
paged-compatible).

**CLI**: `--paged-kv [--kv-quant <k8v4|k8v8|planar>]`.

---

## Dispatch axis

`KvCache::update_and_sdpa` matches `&self.storage`:

```rust
match &self.storage {
    KvStorage::None { .. }      => update_none / update_decode_fp16
    KvStorage::K8V8 { .. }      => update_k8v8
    KvStorage::K8V4 { .. }      => update_k8v4 / update_and_sdpa_k8v4_flash
    KvStorage::Planar { .. }    => update_planar
    KvStorage::Mixed { state }  => update_and_sdpa_mixed (MixedKvState)
    KvStorage::RotKTq4V { .. }  => update_and_sdpa_rot_k_tq4v
    KvStorage::Paged { .. }     => update_paged
}
```

`self.quant` is the construction-time parameter; `self.storage` is the
canonical dispatch key. The two are normally consistent, but code that needs
to branch on codec must match `storage`, not `quant`. Matching on the
storage axis prevents silent misrouting when a cache is reconstructed from
an SSD spill.

Prefill is handled separately: `enter_prefill` switches to raw bf16 accumulation
regardless of the active codec; `exit_prefill` bulk-quantizes the accumulated
prefix into the correct storage variant. Each `KvStorage` arm of `exit_prefill`
is the codec-specific bulk-init path.

`exit_prefill` runs on the request's `spawn_blocking` worker thread — the same
thread the prefill forward built its graph on. That co-location matters because
MLX ≥0.31 streams are thread-local: a cross-thread `Array::eval()` throws
`There is no Stream(cpu, N) in current thread.` The generate entry points call
`rmlx_mlx::ensure_cpu_default_stream()` to register the worker's own streams up
front. See `docs/KV_CACHE.md` §5.7.5 for the mechanism, the guard, and its
limitation.

**Warm-TTFT decode contract.** `exit_prefill` also seeds a bf16 K+V
decode mirror (`decode_fp16_k`/`decode_fp16_v`). Every quantized
`update_<codec>` early-returns to `update_decode_fp16` while that seed is live
(always, post-prefill), so decode-phase K **and** V are bf16 and the codec is
quiescent — it runs only at `exit_prefill`. Full per-codec audit table + the
keep-universal decision (with real-model parity numbers) live in
`docs/KV_CACHE.md` §9.6.

Two deliberate exceptions, both keyed off the codec's own decode path via
`KvQuant::feeds_bf16_k_at_decode()` / `feeds_bf16_v_at_decode()` — the same two
predicates `exit_prefill` gates the allocation on, so an unread mirror is never
materialised:

* **K-only family** (`IsoKOnly*`, `RotorKOnly*`) — keeps K quantized at decode
  and mirrors only V.
* **Fused rotor symmetric** (`rotor3_sym`, `rotor4_sym`) — mirrors **neither**:
  decode is `rotor_flash_decode_symv`, a flash kernel over both packed rotor
  rings. This is the only codec family whose advertised per-axis compression is
  its actual resident cost. It buys that with decode throughput — see the
  performance posture in the `rotor_flash_decode_symv` section.

---

## Layer-adaptive overrides

Two policies modify the per-layer codec assignment independently of the
request-level `KvQuant`:

**Tail layers** (`kv_quant_for_layer`, `LAYER_ADAPTIVE_TAIL_N = 8`): the last
8 layers are forced to `K8V8` regardless of the base mode. Last-layer KV
vectors carry the highest per-token information density; forcing 8-bit
recovers PPL quality lost to aggressive V quantization.

**Head layers** (at context ≥ 32K, `LAYER_ADAPTIVE_HEAD_N = 2`): the first 2
layers are forced to `K8V8`. First-layer K/V vectors carry large absolute
magnitudes (embedding residual is large before deep normalisation); q8_0
on the first 2 layers recovers 37–91% of turbo2 quality degradation at ≥32K
context.

When the base mode is already `K8V8`, both overrides are no-ops. The override
is by absolute layer index; it is arch-agnostic.

---

## Auto-by-context server policy

When neither `--kv-quant` nor `--cache-type-*` flags are passed, the server
selects the KV mode by prompt length (`kv_quant_for_ctx`):

| Prompt length (tokens) | Selected mode |
|---|---|
| ≤ 8 192 | `K8V4` |
| ≤ 16 384 | `None` (bf16) |
| ≤ 32 768 | `K8V8` |
| > 32 768 | `Planar` |

This policy applies after the arch-default resolver. Explicit flags always
take precedence and bypass `kv_quant_for_ctx`.

---

## CLI flags

### Preset interface

`--kv-quant <preset>` sets the K/V codec combo by name.

| Preset string | `KvQuant` variant |
|---|---|
| `none` / `bf16` / `f16` | `KvQuant::None` |
| `k8v8` | `KvQuant::K8V8` |
| `k8v4` | `KvQuant::K8V4` |
| `planar` | `KvQuant::Planar` |
| `k8vturbo3` | `KvQuant::K8VTurbo3` |
| `k8vturbo3tcq` | `KvQuant::K8VTurbo3Tcq` (Viterbi trellis 3-bit V; reuses turbo3 codebook) |
| `tsym4` | `KvQuant::TurboSym4` (symmetric WHT-4 K + tq4 V; rejected on Qwen MoE) |
| `k8vturbo2` | `KvQuant::K8VTurbo2` |
| `rot_k_tq4v` | `KvQuant::RotKTq4V` |
| `mixed_k<kb>g<kg>_v<vb>g<vg>` | `KvQuant::Mixed { .. }` |

Examples: `--kv-quant mixed_k8g64_v4g64`, `--kv-quant k8v4`.

### Named preset interface — `--kv-preset`

`--kv-preset <name>` is the high-level named preset flag. It resolves a short
human-readable name to a concrete `KvQuant` at clap parse time — no further
resolution needed at runtime.

**Conflict rule**: `--kv-preset` is mutually exclusive with `--kv-quant`,
`--cache-type-k`, `--cache-type-v`, and `--kv-bits`. Passing any combination
is a clap hard error (caught before the subcommand body runs).

#### Starter preset table

| Name | `KvQuant` | Notes |
|---|---|---|
| `fp16` | `KvQuant::None` | bf16 unquantized both sides (`KvQuant` variant named `None`, not `Option::None`) |
| `q8` | `KvQuant::K8V8` | symmetric 8-bit K+V |
| `speed` | `KvQuant::K8VTurbo3` | rMLX Lloyd-Max 3-bit V; matches mtq `turbo3` spirit |
| `quality` | `KvQuant::TurboSym4` | symmetric WHT-4 K + tq4 V, matches mtq `quality` byte-for-byte; rejected on Qwen MoE arch guard |
| `planar` | `KvQuant::Planar` | PlanarQuant V-side |

Future presets include: `balanced`, `max_compression`, `k_only_iso`,
`agents_8x16k`, `rot_k_quality`.

#### Preset semantics — divergence from mtq

rMLX `speed` maps to `TurboSym3` — symmetric WHT-3 K+V, matching mtq `speed`
preset definition exactly. Both K and V use the Lloyd-Max N(0,1) 8-centroid
3-bit codebook; K-side uses the GPU turbo3 MSL kernel. Arch guard: rejected on
Qwen MoE (K-side 3-bit is the PPL-disaster zone).

rMLX `quality` maps to `TurboSym4` (symmetric WHT-4 K + tq4 V),
matching mtq `quality` byte-for-byte. Both retain their historical CLI aliases —
no flag changes.

Examples:

```
rmlx serve --model <path> --kv-preset fp16
rmlx serve --model <path> --kv-preset q8
rmlx baseline --model <path> --kv-preset speed
rmlx info --model <path> --kv-preset planar
rmlx baseline --model <path> --kv-preset auto    # hardware-aware auto-selector
```

### Auto-selector — `--kv-preset auto`

`--kv-preset auto` runs the hardware-aware preset recommender at startup,
before any model is loaded. It avoids choosing a preset that requires more
VRAM than the system has.

#### Decision tree

```
model_bytes   = model_size_b × 2e9          (bf16 weights, SI bytes)
kv_bf16_bytes = model_size_b × ctx_tokens × 1e6   (rough KV-cache at bf16)
total_bf16    = model_bytes + kv_bf16_bytes
vram_budget   = available_vram_gb × 1e9 × 0.70  (70% safe utilisation cap)

if total_bf16        < budget → "fp16"           (unquantised)
if model + kv/2      < budget → "q8"             (8-bit K+V)
if model + kv/4      < budget → preferred_4bit() (quality → q8)
if model + kv/8      < budget → preferred_2bit() (max_compression → balanced → quality → q8)
else                           → preferred_2bit() (least-bad; warns in log)
```

The factor `1e6` in `kv_bf16_bytes` is a per-token KV footprint constant
calibrated for 7–70 B transformer layers at bf16.  The 70% utilisation cap
leaves headroom for activations, Metal command buffers, and the OS.

#### Fallback table

`preferred_4bit()` and `preferred_2bit()` walk the available preset table
at runtime so that as future presets land, the auto-selector automatically uses
them without code changes.

| Preference tier | Tries in order |
|---|---|
| 4-bit (preferred_4bit) | `quality`, `q8` |
| 2-bit (preferred_2bit) | `max_compression`, `balanced`, `quality`, `q8` |

Under the current starter preset set (no `max_compression`/`balanced` yet),
`preferred_2bit` falls through to `quality`.

#### Hardware query

Unified DRAM is queried via `sysctl hw.memsize` — the same value shown in
"About This Mac". On Apple Silicon, CPU and GPU share the same DRAM pool.
If the sysctl call fails (returns `None`), the auto-selector falls back to
a conservative 8 GB.

#### Model-size estimation

Model parameter count is estimated from `config.json` using the transformer
heuristic `hidden_size² × num_hidden_layers × 12 / 1e9 B`. Resolution order:
1. `text_config.hidden_size` + `text_config.num_hidden_layers` (Gemma4/multimodal).
2. Top-level `hidden_size` + `num_hidden_layers` (Qwen3/Bonsai flat layout).

If neither field is available, the selector falls back to 7.0 B.

#### Log pattern

Startup logs include:

```
INFO auto-selector chose preset model_size_b=7.2 context_tokens=4096 vram_gb=137.4 preset="fp16"
INFO --kv-preset resolved kv_quant=None
```

When the wanted preset is absent (future preset not yet landed), the selector
falls back and emits:

```
WARN auto-selector wanted max_compression, fell back to q8 (preset max_compression not yet landed)
```

#### Context tokens

The `context_tokens` input to the decision tree comes from (in order):
1. `--max-ctx` override (if provided).
2. `config.json` `max_position_embeddings` (model's native context window).
3. Default 4096 when neither is available.

Models with large native context windows (e.g. 131072) will appear to need
more VRAM for their KV cache; passing `--max-ctx 4096 --kv-preset auto`
caps the estimate to the actual deployment context.

### Per-side primitive interface

`--cache-type-k <tag>` / `--ctk <tag>` sets the K codec.
`--cache-type-v <tag>` / `--ctv <tag>` sets the V codec.

`--kv-quant` and `--cache-type-*` are mutually exclusive. Passing both is a
clap-time hard error.

Available K-side tags:

| Tag | Codec |
|---|---|
| `auto` | resolved by `KvCacheBuilder::resolve_default` |
| `bf16` / `f16` / `none` | unquantized bf16 |
| `q8_g128` | rMLX MSL q8_0, group=128 |
| `q8_g64` | MLX affine 8-bit, group=64 |
| `q8_g32` | MLX affine 8-bit, group=32 |
| `rot_k` | Hadamard-rotated affine 8-bit, group=64 |

Available V-side tags (includes all K-side affine tags plus):

| Tag | Codec |
|---|---|
| `q6_g64` | MLX affine 6-bit, group=64 |
| `q5_g64` | MLX affine 5-bit, group=64 |
| `q4_g128` | MLX affine 4-bit, group=128 |
| `q4_g64` | MLX affine 4-bit, group=64 |
| `q4_g32` | MLX affine 4-bit, group=32 |
| `q3_g64` | MLX affine 3-bit, group=64 (exploratory) |
| `q2_g64` | MLX affine 2-bit, group=64; V-side only |
| `tq4` / `turbo4` | TurboQuant 4-bit Lloyd-Max; head_dim ∈ {128, 256} |
| `planar4` | PlanarQuant 4-bit; head_dim % 32 == 0 |
| `planar3` / `planar_3` | PlanarQuant 3-bit; head_dim % 32 == 0 |

Notes:
- 2-bit K is not a supported combo. `combo_to_kv_quant` rejects K-side 2-bit
  because 2-bit K degrades attention scores into incoherent output.
- `rot_k` is the only K-side member of the rotation family. V-side rotation
  codecs (`tq4`, `planar4`, `planar3`) operate on the value tensor; `rot_k`
  operates on the key tensor via the pre-rotate-Q trick.
- SWA layers always use bf16 regardless of `--ctk` / `--ctv`. This matches
  mlx-lm semantics.
- `--paged-kv` is incompatible with `rot_k` / `rot_k_tq4v`.

### Canonical combo examples

```
rmlx serve --model <path> --kv-quant k8v8
rmlx serve --model <path> --kv-quant k8v4
rmlx serve --model <path> --kv-quant planar
rmlx serve --model <path> --ctk q8_g128 --ctv tq4      # equivalent to k8v4
rmlx serve --model <path> --ctk rot_k   --ctv tq4      # RotKTq4V hybrid
rmlx serve --model <path> --ctk rot_k   --ctv q4_g64   # RotK affine V
rmlx serve --model <path> --kv-quant mixed_k8g64_v4g64
rmlx serve --model <path> --paged-kv --kv-quant k8v4
```

---

### iso3 codec

**Algorithm — Quaternion SO(4) isoclinic rotation.**

iso3 applies a left-isoclinic SO(4) rotation to groups of 4 elements in the
V tensor before 3-bit quantization:

```
T(v) = q_L * v     (fast mode — 3 DOF, one quaternion per group)
```

where `*` is the **Hamilton product** and `v ∈ ℝ⁴` is treated as a quaternion
`v = (v₀, v₁, v₂, v₃)`. Inverse: `T⁻¹(r) = q̄_L * r` (conjugate multiply).

**Pipeline (per token):**

1. L2-normalise the full vector; store the scalar norm.
2. Reshape into `head_dim / group_size` quaternion groups.
3. Apply `r = q_L * v` via scalar Hamilton product.
4. Per-group scale: `max(|r_i|) / max_centroid`.
5. 3-bit Lloyd-Max nearest-centroid lookup.
6. Pack 10 codes per u32 (30 bits used, 2 wasted) — same Planar3 pack convention.

**Dequantize:** unpack → centroid lookup → rescale → inverse rotate → renorm.

**Memory truth.** iso stores, per 4-element group, a packed code word, an
f32 scale, and a 4×f32 quaternion, plus one f32 norm per token — ≈772 B per
token per kv_head at head_dim=128 versus 256 B for bf16. **iso3/iso4 are
net-NEGATIVE on memory at head_dim ≤ 256**; they are research codecs for
quality experiments, not size wins. The resolve-time net-negative warn
(`estimated_resident_bytes_per_layer` models the quaternion + norm sidebands
exactly) accounts for these sidebands.

**`head_dim % 4 == 0` constraint.** iso3 operates in groups of 4. Any
`head_dim` not divisible by 4 is rejected at encode/decode time with
`IsoQuantError::HeadDimNotMultipleOf4`.

**Fixed quaternion.** The current CPU implementation uses the golden-ratio
unit quaternion `q = (1, φ, φ−1, 1) / ‖(1, φ, φ−1, 1)‖` (where `φ = (1+√5)/2`)
applied uniformly to every group. This matches `multi_turboquant/methods/isoquant.py`
and provides good channel decorrelation without calibration. A follow-up will add
per-group optimised quaternions.

**Codebook divergence — rMLX Gaussian Lloyd vs Python Beta Lloyd.**

The Python references (`rotorquant/turboquant/lloyd_max.py`) derive a
Lloyd-Max codebook for the **Beta distribution** that arises after random
rotation of a unit vector. rMLX reuses `turboquant::lloyd_gaussian_codebook(3)`
(Lloyd-Max for N(0,1)) to stay consistent with TurboQuant and PlanarQuant
and avoid a new codebook solver.

For `head_dim ≥ 64`, Beta(d) → N(0, 1/d), and the per-group scale step
normalises to N(0,1) before the centroid lookup regardless of the source
distribution. The quality gap in practice is below measurement noise on LCG
fixtures. Published Python mtq cosine (0.9783, realistic KV vectors) is
a different measurement condition; rMLX LCG fixture measures mean ≈ 0.994
(group_size=4, 32 tokens × 128-dim).

**Wire-up status:**

| Component | Status |
|---|---|
| CPU encode/decode (`isoquant.rs`) | Done |
| `KvStorage::IsoV3` variant | Done |
| `KvQuant::Iso3` + `CacheType::Iso3` | Done |
| `KvCache::update_iso3` decode dispatch | Done |
| SDPA dispatch wiring | Done (dequant-then-SDPA legacy fallback; iso3 has no fused fast path, mirrors K8VTurbo3) |
| `KvBlockWriter`/`Reader` integration | Done (layout tag `iso_v_3`; K via `write_quant_k`; V via `write_quant_iso_v3` / `read_quant_iso_v3`) |
| SSD tier integration | Done |
| MSL kernel hook (`isoquant_msl.rs`) | Done |
| **MSL encode dispatch + on-demand `Array::from_bytes` dequant** | **Done — `update_iso3` / `update_iso3_sym` / `update_iso_k_only_3` route encode and dequant through `iso_quantize_v3_gpu` + `iso_dequantize_v3_gpu` when `device == Device::Gpu`; `QuantIsoV3::dequant_gpu` / `QuantIsoK3::dequant_gpu` rebuild GPU Arrays directly from CPU blocks via `Array::from_bytes` (no intermediate `Vec<f32>`)** |
| **GPU-resident `QuantIsoV3` mirror** | **Landed; hardcoded OFF (bench decision — bench showed no measurable benefit on the warm-TTFT path where the bf16 seed absorbs the dequant). `QuantIsoV3::append_gpu` retains the mirror infrastructure for future seedless workloads but the gate `gpu_resident_iso_enabled()` returns `false` unconditionally in production. CPU blocks are still populated for SSD spill (`.kvb` on-disk format unchanged). See `docs/PERF_BASELINE.md` for the bench rationale.** |
| `--kv-quant iso3` CLI flag | Done |

The MSL kernel ships as a future-reference hook. The GPU dispatch is on:
when `device == Device::Gpu`, `update_iso3` / `update_iso3_sym` /
`update_iso_k_only_3` route encode through `iso_quantize_v3_gpu` and
dequant through `QuantIsoV3::dequant_gpu` /
`QuantIsoK3::dequant_gpu`. The dequant methods concatenate per-block CPU
payload (codes / scales / quaternions / per-token-norm-expanded-to-per-group)
into single byte buffers, upload them to the GPU **once** via
`Array::from_bytes`, dispatch `iso_dequantize_v3_gpu`, then reshape the flat
f32 output to `[B, kv_h, S, D]`. No intermediate `Vec<f32>` is materialised
on the CPU side. CPU path remains intact and is the fallback for `Device::Cpu`.

**Warm-TTFT caveat:** the per-decode-step `update_iso3` codec is shadowed by
the warm-TTFT bf16 seed when `KvCache::decode_fp16_k.is_some()`, which is
the case for all current arch wirings (Bonsai 8B, Gemma4, Qwen3.6). The
GPU dispatch therefore fires once at `exit_prefill` and on cold cache
misses, not per step. Parity verified by
`iso_v3_dequant_gpu_matches_dequant_cpu` and
`iso_k3_dequant_gpu_matches_dequant_cpu` in
`crates/rmlx-kv-quant/src/isoquant_msl_tests.rs` (`#[ignore]`-gated).
Observed `max|cpu-gpu| ≤ 2.4e-7` on the LCG fixture (a few f32 ULPs from
different summation order between CPU `iso_decode_fast` and the MSL
kernel — not a real codec divergence). The parity test gates at 5e-3
(codebook tolerance) and additionally enforces a strict ≤ 1e-6 bound.

**Cosine quality (LCG fixture, group_size=4, head_dim=128, bits=3):**
mean = 0.994, min = 0.993. Test: `iso3_cosine_gate`
in `crates/rmlx-kv-quant/src/isoquant_tests.rs`. `QuantIsoV3` round-trip
matches `iso_decode_fast` reference within `max_abs_err < 1e-3`
(`quant_iso_v_roundtrip_dequant`).

**Smoke probes:** validated end-to-end on Bonsai-8B-2bit (head_dim=128),
Gemma4-e4b-mxfp8 (head_dim=512), and Qwen3.6-35B-A3B-8bit (head_dim=128).
No NaN/Inf, no infinite loops, 8-token generations complete. Decode TPS
reflects CPU-heavy V dequant on the initial version; GPU encode path reduces
overhead.

**Sequence-major buffer layout (whole Iso / Rotor family).** Every `Vec<Blocks>`
rotation-KV codec — `QuantIsoV3` / `QuantIsoV4`, `QuantIsoK3` / `QuantIsoK4`,
`QuantRotorV3` / `QuantRotorV4`, `QuantRotorK3` / `QuantRotorK4` — accumulates
one `*Blocks` entry per `append` and concatenates them on `dequant`. Because
the caller reshapes the concatenation head-major `[B, kv_h, S, D]`, a head-major
per-block store transposes per-head values across a multi-append GQA cache
(`kv_h > 1`, e.g. the post-SSD-hydrate decode-append path) — the same head
scramble fixed for `QuantK` / `QuantV`. Each `append` now reorders the
head-major chunk heads↔seq (`[B, new_seq, kv_h, D]`) before encoding and
`dequant` reorders back; single-chunk cold prefill is the identity. The codec is
per-token-row positional, so the sidebands stay correctly associated: Iso
per-(token, group) scale/norm and the constant `FIXED_QUAT` quaternion permute
with the rows; the Rotor static rotor table and QJL projection are
group/projection-keyed (untouched) while the per-token QJL `qjl_codes` /
`qjl_norms` permute with the rows. `QuantIsoV3` is the only GPU-resident member
and adds `Array::contiguous` after the heads↔seq transpose before its
raw-linear-index MSL encode kernel. The `.kvb` SSD format is byte-stable (only
the token-row order within a block changes). GPU round-trip verified on
`QuantIsoV3` (two-append GQA vs single-shot). See `docs/KV_CACHE.md` §5.7.3.

---

### iso4 codec

**Algorithm — Quaternion SO(4) isoclinic rotation, 4-bit codebook.**

iso4 is the natural 4-bit extension of [iso3](#iso3-codec).
Same rotation, same group geometry, same fixed quaternion — the only
differences are the codebook (16 centroids vs 8) and the pack density
(8 vals/u32 vs 10 vals/u32).

| Property | iso3 | iso4 |
|---|---|---|
| Code bits / element | 3 | 4 |
| Effective bits / element (incl. quaternion + norm sidebands) | ≈48.25 (≈772 B/token at head\_dim=128 — see Memory truth in iso3 section) | ≈52.25 (≈836 B/token at head\_dim=128: same quaternion + norm overhead, 4-bit codes) |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids) | `lloyd_gaussian_codebook(4)` (16 centroids) |
| Pack density (per u32) | 10 vals (30 bits used, 2 wasted) | 8 vals (32 bits used, 0 wasted) |
| Rotation | Golden-ratio fixed quaternion (`FIXED_QUAT`) | Same |
| Group size | 4 elements (one quaternion block) | Same |
| `head_dim` constraint | `% 4 == 0` | Same |
| MSL kernel | **Yes** — encode + dequant dispatch wired into `update_iso3` / `update_iso3_sym` / `update_iso_k_only_3`; `QuantIsoV3::dequant_gpu` / `QuantIsoK3::dequant_gpu` upload CPU blocks via `Array::from_bytes` (no intermediate `Vec<f32>`) | **Yes** — `iso_quantize_v4_gpu` / `iso_dequantize_v4_gpu` in `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs`; encode dispatch wired into `update_iso4` / `update_iso4_sym` / `update_iso_k_only_4` when `device == Device::Gpu` |

**Codebook divergence — same as iso3.** rMLX uses Gaussian Lloyd-Max
(N(0,1)) `lloyd_gaussian_codebook(4)`; Python references use Beta Lloyd.
The published multi-turboquant `iso4` cosine is 0.9951; rMLX LCG fixture
measures mean = 0.998638, min = 0.998092 (group_size=4, 32 tokens × 128-dim,
4-bit). Higher than the published number because rMLX's LCG fixture has
lower dynamic range than calibrated real KV (see iso3 note above).

**MSL kernel.** `crates/rmlx-kv-quant/src/isoquant_msl_v4.rs`
ships the sibling 4-bit kernel pair `iso_quantize_v4_gpu` /
`iso_dequantize_v4_gpu`, with the per-(token, group) thread layout, atomic-OR
pack via `(idx & 0xF) << shift`, and a dense 8-vals/u32 boundary table (15
mid-points derived from `lloyd_gaussian_codebook(4)`). The encode side is
wired into the three iso4 update paths (`update_iso4`,
`update_iso4_sym`, `update_iso_k_only_4`) under
`device == Device::Gpu`; the CPU codec remains the fallback.

**Warm-TTFT caveat.** The iso V hot path is shadowed by the bf16 exit-prefill
seed: the GPU encode fires **once at exit_prefill**, not per decode step. The
measured benefit lands on TTFT (large prefill chunk) rather than steady-state
decode TPS. The CPU dequant remains primary for the returned `v_full` Array —
full GPU end-to-end dequant is deferred (current CPU bookkeeping preserves SSD
spill / truncate semantics; switching to GPU-resident state is a follow-up).

**CPU ↔ GPU parity.** `iso_v4_msl_matches_cpu_within_eps` in
`crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs` asserts CPU
(`iso_encode_fast` + `iso_decode_fast`, `bits=4`) ↔ MSL bit-identity
within 5e-3 on a 32×128 LCG fixture (`#[ignore]`-gated; run via
`cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1`).

**Wire-up status:**

| Component | Status |
|---|---|
| CPU encode/decode (parameterized `iso_encode_fast` / `iso_decode_fast`) | Done (bits ∈ {3, 4}) |
| `KvStorage::IsoV4` variant + `QuantIsoV4` storage struct | Done |
| `KvQuant::Iso4` + `CacheType::Iso4` | Done |
| `KvCache::update_iso4` decode dispatch | Done |
| SDPA dispatch wiring | Done (dequant-then-SDPA legacy fallback, mirrors iso3) |
| `KvBlockWriter`/`Reader` integration | Done (layout tag `iso_v_4`; V via `write_quant_iso_v4` / `read_quant_iso_v4`) |
| SSD tier integration | Done |
| MSL kernel hook | Done (`isoquant_msl_v4.rs`, encode dispatch wired into `update_iso4` / `update_iso4_sym` / `update_iso_k_only_4`) |
| `--kv-quant iso4` / `--ctv iso4` CLI flags | Done |

**Cosine quality (LCG fixture, group_size=4, head_dim=128, bits=4):**
mean = 0.998638, min = 0.998092. Test: `iso4_cosine_gate`
in `crates/rmlx-kv-quant/src/isoquant_tests.rs`. SSD round-trip:
`roundtrip_iso4` in `crates/rmlx-kv-ssd/src/block_io_tests.rs` —
all four V buffers (codes_packed, scales, quaternions, norms)
bit-identical post-hydrate.

**Parameterize vs fork decision.**
The encode/decode CPU functions are parameterized over `bits ∈ {3, 4}`
(generic packer using `vals_per_word(bits) = 32 / bits`). The storage
struct is forked (`QuantIsoV3` + `QuantIsoV4`) because the bit-width is
fixed per storage variant and a generic rename would create large
cross-crate churn for no benefit. `IsoBlocks` is shared (codes:
`Vec<u32>` is bits-agnostic).

---

### rotor3 codec

**Algorithm — Cl(3,0) Clifford rotor sandwich, 3-bit codebook.**

rotor3 is the first member of the **Clifford rotation family** of KV codecs.
Each `head_dim`-element V-vector is embedded into Cl(3,0) (the 8-dimensional
multivector algebra of 3D Euclidean space) in groups of 3 grade-1 elements,
sandwiched by a per-(layer, head)-static rotor `R_g`, and 3-bit-quantised
against the Lloyd-Max N(0,1) codebook. The static rotor is stored once and
amortises across every token in the layer.

| Property | rotor3 |
|---|---|
| Effective bits / element | ~8 bpe pre-scale (8 codes × 3 bits per group of 3 real grade-1 elements + per-group scale + per-token norm). The 3.25 bpe target reported by the Python `rotorquant` reference is gated on the grade-aware codebook follow-up (deferred — see below). |
| Codebook | `lloyd_gaussian_codebook(3)` (8 centroids), shared across all 8 mv components (single-codebook simplification) |
| Pack density (per u32) | 10 vals (planar3 / iso3 convention; 8 codes ≤ 30 bits per group, 1 u32 per group) |
| Rotation | Static per-(layer, head) rotor table `[n_groups, 4]` in `[s, b12, b13, b23]` form, seeded from `ROTORQUANT_GLOBAL_SEED ^ (layer << 32) ^ (head << 16) + group` (see [`crate::clifford`]) |
| Group size | 3 elements (one Cl(3,0) grade-1 group; output multivector has all 8 components) |
| `head_dim` constraint | None — `head_dim % 3 != 0` is tail-padded with zeros at encode, masked off at decode |
| MSL kernel | **Yes** — `rotorquant_msl.rs`, V-side encode + K-side when QJL disabled |

**Single-codebook simplification.** The Python reference
(`rotorquant/turboquant/rotorquant.py`) ships a grade-aware codebook split
(separate `vector` and `trivector` codebooks at different bit budgets). rMLX
ships a **single 8-centroid codebook** for all 8 mv components; the
grade-aware variant is a deferred follow-up (cosine gate measured
empirically — see below).

**Per-layer rotor tables.** `KvCache` now carries a `layer_idx: usize` field,
set at construction by each arch builder via
`KvCache::with_quant_max_seq(…).with_layer_idx(i)`. The rotor3/rotor4 codec
constructors (`QuantRotorV3::new`, `QuantRotorV4::new`, `QuantRotorK3::new`,
`QuantRotorK4::new`) receive `self.layer_idx as u32` at every `exit_prefill`
and decode-time creation site. The `(layer << 32)` mixing term in
[`crate::clifford::rotor_seed`] is active, giving each layer a distinct
rotor table and restoring the cross-layer decorrelation that the algorithm
relies on.

**No QJL residual (V-only codec).** The Python reference includes an optional
1-bit QJL sign-quantization residual stage for unbiased inner-product recovery
on the K side. The base rotor3 codec is V-side only — QJL is not applied here
(see K-side rotor variants below).

**Sign-error correction.** The Python `clifford.py::geometric_product` and
the `rotor_fused.metal::gp_rotor_mv` kernel both have sign errors in the
grade-2 and grade-3 component formulas (e.g. `e23 * e1 = +e123` per the
Cl(3,0) multiplication table, but the Python formula yields `-e123`). The
Rust port uses a table-driven dense geometric product computed at compile
time from the algebra rules — these signs are correct by construction and
validated by the algebra tests in `clifford_tests.rs` (known-answer 90°
rotation, unit rotor identity, sandwich-of-grade-1-stays-grade-1). See
[CLAUDE.md hard rule 7][hr7] ("Document the truth, not the docstring").

[hr7]: ../CLAUDE.md

**MSL kernel.** `crates/rmlx-kv-quant/src/rotorquant_msl.rs`
ships GPU encode + decode kernels for both rotor3 and rotor4. The kernel
applies the Cl(3,0) sandwich as a closed-form 3×3 SO(3) rotation matrix
`M(R)` derived from `R * mv * R̃` (the grade-2 and grade-3 components
cancel identically for grade-1 input — verified algebraically). The
per-(layer, head, group) rotor table is passed as a buffer argument
(`rotors_in : f32 [n_groups, 4]`); kernels do not hardcode the table.
Dispatch is wired into `update_rotor3` / `update_rotor4` / sym variants /
`update_rotor_k_only_{3,4}` / `update_rotor_k_asym_{3,4}` and fires when
`device == Device::Gpu`. The CPU encoder remains the fallback. The V-side hot
path is shadowed by the warm-TTFT bf16 seed — the GPU encode fires once at
`exit_prefill` (large prefill slice), not per decode step; the speedup shows
up in TTFT, not decode TPS.

**K-side QJL caveat.** The K-side rotor codecs may carry a
1-bit QJL residual correction that needs the JL projection matrix `S` at
dequant time. The GPU dequant kernels in `rotorquant_msl.rs` do NOT
replicate QJL — when `crate::rotor_qjl::rotor_qjl_enabled()` is `true`
(the default), the K-side append/decode falls back to the CPU
`rotor3_k_encode` / `rotor3_k_decode` path. With `--rotor-qjl off`, the
GPU K-side kernel is engaged (TTFT drop measured on Bonsai 8B at ~10.8k
prompt tokens: 28.3 s → 11.5 s vs CPU K-encode).

**CPU↔GPU parity tests.** `crates/rmlx-kv-quant/src/rotorquant_msl_tests.rs`
asserts max-abs-error ≤ 5e-3 between CPU `rotor3_encode`/`rotor4_encode`
round-trip and the MSL round-trip (same per-codec tolerance policy as
iso3 / iso4). Tests are `#[ignore]`-gated:
`cargo test -p rmlx-kv-quant -- --ignored rotorquant_msl --test-threads=1`.

**Wire-up status:**

| Component | Status |
|---|---|
| Clifford module (`crate::clifford`) | Done (compile-time `MUL_TABLE`, sandwich, random rotor table) |
| CPU encode/decode (`crate::rotorquant`) | Done (single-codebook, planar3 / iso3 pack convention) |
| `KvStorage::RotorV3` variant + `QuantRotorV3` storage struct | Done (static rotors + per-token blocks; rotors counted once in `byte_size`) |
| `KvQuant::Rotor3` + `CacheType::Rotor3` | Done (with `rotor3` / `rotor_v_3` dual-spelling parse) |
| `KvCache::update_rotor3` decode dispatch | Done |
| SDPA dispatch wiring | Done (dequant-then-SDPA legacy fallback, mirrors iso3) |
| `KvBlockWriter`/`Reader` integration | Done (layout tag `rotor_v_3`; V via `write_quant_rotor_v3` / `read_quant_rotor_v3`; rotor table persisted on disk) |
| SSD tier integration | Done — round-trip parity in `roundtrip_rotor3` |
| MSL kernel | Done (`rotorquant_msl.rs`, V + K-no-QJL encode/decode; parity tests in `rotorquant_msl_tests.rs`) |
| `--kv-quant rotor3` / `--ctv rotor3` CLI flags | Done |

**Cosine quality (LCG fixture, head_dim=128, n_tokens=32, bits=3):**
mean = 0.995601, min = 0.994737 (post rotor-sandwich fix — original version
shipped a silent no-op sandwich — see [`crate::rotorquant`] history note).
Test: `rotor3_cosine_gate` in `crates/rmlx-kv-quant/src/rotorquant_tests.rs`.
The published Beta-codebook multi-turboquant `rotor3` number is 0.9780;
rMLX's Gaussian-codebook LCG measurement exceeds it (same effect documented
for iso3 / iso4 — Beta(d) converges to N(0, 1/d) for `head_dim ≥ 64`).

**SSD round-trip:** `roundtrip_rotor3` in
`crates/rmlx-kv-ssd/src/block_io_tests.rs` — all four V buffers
(`codes_packed`, `scales`, `norms`, `rotors`) bit-identical post-hydrate.
The rotor table is persisted alongside the per-token payload so
cross-restart identity is preserved independent of any change to
`ROTORQUANT_GLOBAL_SEED`.

**Paged-KV routing:** rotor3 does NOT route through the PagedAttention
block-table path. `PagedKStorage` is q8-only and `PagedPlanarVStorage` is
PlanarQuant-only; a paged rotor3 variant would need its own per-token
container plus a static rotor table inside the paged arena. Deferred per
the iso3 / iso4 precedent (opt-in codec, never an auto baseline).

**Smoke probes (16384 max_ctx, greedy decode):**

| Model | Decode TPS | Coherence |
|---|---|---|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 53.3 (10867 prompt tokens, 50 gen) / 134.4 (16 prompt) | yes — replies "4." to "What is 2+2?" |
| `mlx-community__gemma-4-e4b-it-mxfp8` | 67.3 (10808 prompt, 50 gen) | yes — replies "Paris." to "Capital of France?" |
| `mlx-community__Qwen3.6-35B-A3B-8bit` | 91.7 (11032 prompt, 50 gen) | yes — coherent reasoning chain in `reasoning_content` (thinking-model) |

---

### rotor4 codec

**Algorithm — Cl(3,0) Clifford rotor sandwich, 4-bit codebook.**

rotor4 is the 4-bit member of the Clifford rotation family. The algebra and
rotor-sandwich structure are identical to rotor3; the only difference is the
codebook and packing:

| Property | rotor4 |
|---|---|
| Effective bits / element | ~10.7 bpe pre-scale (8 codes × 4 bits = 32 bits per group of 3 real grade-1 elements = exactly 1 u32, plus per-group f32 scale + per-token norm). Same grade-aware split deferral as rotor3. |
| Codebook | `lloyd_gaussian_codebook(4)` (16 centroids), shared across all 8 mv components (single-codebook simplification, same as rotor3) |
| Pack density (per u32) | 8 vals / u32 (dense 4-bit packing: 8 components × 4 bits = 32 bits = 1 u32 per group; `ROTOR4_WORDS_PER_GROUP = 1`) |
| Rotation | Same static per-(layer, head) rotor table as rotor3 (`[n_groups, 4]`); seeded from the same `ROTORQUANT_GLOBAL_SEED` formula |
| Group size | 3 elements (same Cl(3,0) grade-1 group as rotor3) |
| `head_dim` constraint | None — same tail-padding as rotor3 |
| MSL kernel | **Yes** — `rotorquant_msl.rs`, shared with rotor3 via `rotor_quantize_v{3,4}_gpu` / `rotor_dequantize_v{3,4}_gpu` |

**Fork pattern.** `QuantRotorV4` is a fork of `QuantRotorV3` with `bits=4`
and `rotor4_encode`/`rotor4_decode` from `crate::rotorquant`. `RotorBlocks`
is bits-agnostic and shared. The encode/decode functions are the only
variant-specific code.

**Wire-up status:**

| Component | Status |
|---|---|
| CPU encode/decode (`crate::rotorquant`) | Done (`rotor4_encode` / `rotor4_decode` with 4-bit pack and 16-centroid codebook) |
| `KvStorage::RotorV4` variant + `QuantRotorV4` storage struct | Done (mirrors RotorV3; rotors counted once in `byte_size`) |
| `KvQuant::Rotor4` + `CacheType::Rotor4` | Done (with `rotor4` / `rotor_v_4` dual-spelling parse) |
| `KvCache::update_rotor4` decode dispatch | Done |
| SDPA dispatch wiring | Done (dequant-then-SDPA legacy fallback, mirrors rotor3) |
| `KvBlockWriter`/`Reader` integration | Done (layout tag `rotor_v_4`; V via `write_quant_rotor_v4` / `read_quant_rotor_v4`; rotor table persisted on disk) |
| SSD tier integration | Done — round-trip parity in `roundtrip_rotor4` |
| MSL kernel | Done (shared with rotor3; parity tests in `rotorquant_msl_tests.rs::rotor_v4_msl_matches_cpu_within_eps`) |
| `--kv-quant rotor4` / `--ctv rotor4` CLI flags | Done |

**Cosine quality (LCG fixture, head_dim=96, n_tokens=32, bits=4):**
mean = 0.998884, min = 0.998250.
Thresholds: mean ≥ 0.9978, min ≥ 0.9972.
Test: `rotor4_cosine_gate` in `crates/rmlx-kv-quant/src/rotorquant_tests.rs`.

**SSD round-trip:** `roundtrip_rotor4` in
`crates/rmlx-kv-ssd/src/block_io_tests.rs` — all four V buffers
(`codes_packed`, `scales`, `norms`, `rotors`) bit-identical post-hydrate.
The rotor table is persisted alongside the per-token payload so cross-restart
identity is preserved independent of any change to `ROTORQUANT_GLOBAL_SEED`.

**Smoke probes:** pending (requires live model run; not yet run).

**Paged-KV routing:** same deferral as rotor3 — RotorV4 does not route through
PagedAttention.

---

### iso K-side variants

Four variants mirror the V-side iso3 / iso4 codecs to the K axis. The
IsoQuant codec (`iso_encode_fast` / `iso_decode_fast`) is **axis-agnostic**
— the encoder consumes a flat `[B, kv_h, S, D]` row buffer and a per-row
`head_dim`; the K vs V distinction lives only in the role on
the SDPA path and the SSD writer/reader tensor names (`l{idx}.k.*` vs
`l{idx}.v.*`).

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD layout tag |
|---|---|---|---|---|
| `Iso3Sym` | iso3 (3-bit quaternion SO(4)) | iso3 (3-bit) | `(IsoK3, Iso3)` | `iso_sym_3` |
| `Iso4Sym` | iso4 (4-bit quaternion SO(4)) | iso4 (4-bit) | `(IsoK4, Iso4)` | `iso_sym_4` |
| `IsoKOnly3` | iso3 (3-bit) | **bf16** (parent `decode_fp16_v`) | `(IsoK3, Bf16)` | `iso_k_only_3` |
| `IsoKOnly4` | iso4 (4-bit) | **bf16** | `(IsoK4, Bf16)` | `iso_k_only_4` |

**A.y Qwen MoE arch guard (mandatory).** K-side ≤4-bit on Qwen MoE is the
PPL-disaster zone (218 → 8641 on Q4_K_M baseline; 7:1 GQA amplifies K-head
error through softmax). All four variants are flagged by `KvQuant::k_below_8bit()`
and `cache_type::validate_resolved` routes them through the dedicated
`ResolveError::QwenMoeIsoKRejected { variant }` error, which quotes the
offending variant by name. `KvCacheBuilder::resolve_default` never returns
any of the four variants for Qwen MoE (they are opt-in only — no auto path).

Smoke runs on `mlx-community__Qwen3.6-35B-A3B-8bit` are expected
to error with `exit 78` and the diagnostic
`"K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant <variant> …
rejected for Qwen3.5/3.6 MoE."` (positive guard test).

**IsoKOnly bf16-V layout.** The V buffer lives on the parent
`KvCache::decode_fp16_v` — same machinery as `KvStorage::None` and
`KvStorage::PlanarK` for V. The SSD writer emits only K-side tensors
(`l{idx}.k.codes_packed/scales/quaternions/norms`); the reader restores
the K side and the V side is rebuilt transparently from the live request's
bf16 buffer on first decode step.

**Status.** CPU-only on the hot path. The iso3 MSL kernel could in principle
be reused on the K axis (it is axis-agnostic), but on-disk and SDPA paths use
the CPU dequant fallback; an MSL re-wiring is a deferred follow-up.

**Decode-cost caveat.** `QuantIsoK3`/`QuantIsoK4` have no GPU-resident code
mirror on the live decode path: the CPU `dequant()` re-materializes every
accumulated block each step and the reconstructed K prefix is re-uploaded to
the GPU via `Array::from_bytes` — an O(kv_seq) per-step cost that grows
monotonically with context. (A `dequant_gpu` mirror exists but is gated behind
`gpu_resident_iso_enabled()`, hardcoded `false` in `rmlx-kv-quant/src/lib.rs`,
so it does not run.) The short-prompt anchors elsewhere in this doc (warm-TTFT
masks the cost after step 1) do not show it; long-prompt decode does (k_iso3
Bonsai ~59 TPS vs iso3_sym ~142). The GPU mirror is deferred until a bench arm
shows a win on a FusedQkShadow-incompatible path.

**Cosine empirical floors.** Measured on the LCG fixture at
`head_dim=128, n_rows=16, TEST_SEED` (see `quant_iso_k{,4}_tests.rs`):

| Variant | Measured cosine (min) | Gate |
|---|---|---|
| `iso_k_3` codec | ≥ 0.98 | 0.97 |
| `iso_k_4` codec | ≥ 0.99 | 0.99 |

The full symmetric / K-only KvQuant cosine is downstream of these K-side
floors plus the existing V-side iso{3,4} floors.

**SSD round-trip tests.** Four tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`:
`roundtrip_iso_sym_3`, `roundtrip_iso_sym_4`, `roundtrip_iso_k_only_3`,
`roundtrip_iso_k_only_4` — assert K-side codes bit-identical post-hydrate
and K dequant matches within 1e-3.

### rotor K-side variants

Four variants mirror the V-side rotor3 / rotor4 codecs to the K axis. They
add an **optional 1-bit QJL residual sideband** (Johnson–
Lindenstrauss sketch of the post-rotor MSE residual) when
`--rotor-qjl on` (the default). The storage format is controlled by a global
toggle ([`rotor_qjl_enabled()`] in `rmlx-kv-quant::rotor_qjl`).

| `KvQuant` | K codec | V codec | CacheType pair (`(K, V)`) | SSD tag (QJL off / on) |
|---|---|---|---|---|
| `Rotor3Sym` | rotor3 + QJL | rotor3 | `(RotorK3, Rotor3)` | `rotor_sym_3` / `rotor_sym_3_qjl` |
| `Rotor4Sym` | rotor4 + QJL | rotor4 | `(RotorK4, Rotor4)` | `rotor_sym_4` / `rotor_sym_4_qjl` |
| `RotorKOnly3` | rotor3 + QJL | **bf16** (parent `decode_fp16_v`) | `(RotorK3, Bf16)` | `rotor_k_only_3` / `rotor_k_only_3_qjl` |
| `RotorKOnly4` | rotor4 + QJL | **bf16** | `(RotorK4, Bf16)` | `rotor_k_only_4` / `rotor_k_only_4_qjl` |
| `RotorK3Asym { v_bits, v_group_size }` | rotor3 + QJL | **TurboQuant V** at `v_bits` (reuses K8V4 / K8VTurbo3 / K8VTurbo2 V codec; `v_group_size` is layout-tag-only — TurboQuant V uses GROUP_SIZE=32 regardless) | `(RotorK3, Q*G*)` for affine V tag | `rotor_k_asym_3_v{vb}_g{vg}` / `rotor_k_asym_3_qjl_v{vb}_g{vg}` |
| `RotorK4Asym { v_bits, v_group_size }` | rotor4 + QJL | **TurboQuant V** at `v_bits` (`v_group_size` is layout-tag-only — TurboQuant V uses GROUP_SIZE=32 regardless) | `(RotorK4, Q*G*)` | `rotor_k_asym_4_v{vb}_g{vg}` / `rotor_k_asym_4_qjl_v{vb}_g{vg}` |

**Asymmetric rotor-K variants.** The two
`RotorK{3,4}Asym` arms close the gap between `Rotor{3,4}Sym` (rotor V) and
`RotorKOnly{3,4}` (bf16 V) by carrying a TurboQuant V codec at `v_bits ∈
{2, 3, 4}`. The V slot routes through the same `QuantV` codec already used by
`K8V4` / `K8VTurbo3` / `K8VTurbo2` (Lloyd-Max N(0,1) codebook, fixed internal
group=32; the `v_group_size` field is carried through to the SSD layout key
for round-trip determinism, but the underlying codec keeps its 32-element
group). `(8, *)` tuples are rejected at compose / parse time because
TurboQuant has no 8-bit path — pair `--ctk k_rotor3` / `--ctk k_rotor4` with
`--ctv bf16` for the K-only path (`RotorKOnly{3,4}`) or with `--ctv
rotor_v_{3,4}` for the symmetric path (`Rotor{3,4}Sym`) instead.

Display form: `rotor_k_3_asym_v{v_bits}_g{v_group_size}` (similarly for `_4_`).
Compose forms:
- `--ctk k_rotor3 --ctv q4_g64` → `RotorK3Asym { v_bits: 4, v_group_size: 64 }`
- `--ctk k_rotor3 --ctv q4_g128` → `RotorK3Asym { v_bits: 4, v_group_size: 128 }`
- `--ctk k_rotor4 --ctv q3_g64` → `RotorK4Asym { v_bits: 3, v_group_size: 64 }`
- `--ctk k_rotor4 --ctv q2_g64` → `RotorK4Asym { v_bits: 2, v_group_size: 64 }`

Symmetric and K-only compose forms:
- `--ctk k_rotor3 --ctv rotor_v_3` → `Rotor3Sym`.
- `--ctk k_rotor3 --ctv bf16` → `RotorKOnly3`.

**Arch guard (Contract A.y)**: all `RotorK{3,4}Asym` variants are rejected on
Qwen MoE via the same `QwenMoeRotorKRejected` error as the sym / K-only
siblings (K-side ≤4-bit on Qwen MoE is the PPL-disaster path). The error's
`variant` field carries the full Display form (e.g.
`rotor_k_3_asym_v4_g64`) so the diagnostic is unambiguous.

**SDPA**: the K rotor codec dequants to bf16 (existing `RotorKOnly{3,4}` K
path); the affine V codec dequants to bf16 (existing `K8V4` V path); then
`scaled_dot_product_attention` runs.

**Decode-cost caveat.** Like the iso K-side codecs, the rotor K-side codecs
have no GPU-resident code mirror: with the default `--rotor-qjl on`, each
decode step re-decodes the full K prefix on the CPU (and re-encodes the newly
appended token), applies an O(head_dim²)-per-cached-token QJL score
correction, then re-uploads the K prefix — an O(kv_seq) per-step cost that the
short-prompt anchors mask but long-prompt decode exposes. The fused-QK fast path (which would avoid the
per-step marshaling) is reachable only with `--rotor-qjl off` and is
default-OFF; see the Fused-QK status below.

**Fused-QK status:** the 6 rotor variants (`Rotor3Sym`, `Rotor4Sym`,
`RotorKOnly3`, `RotorKOnly4`, `RotorK3Asym`, `RotorK4Asym`) are wired into
the fused-QK fast path via the shadow split (`FusedQkShadow` carries per-token
codes/scales/norms + a static `[n_groups * 4]` rotor table). Gated by
`--fused-qk on` AND `--rotor-qjl off` (the kernel does not consume the QJL
residual). Default-OFF (the auto/HOLD `--fused-qk` mode keeps the legacy bf16
SDPA path live). Bonsai bench (`--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl
off --fused-qk on`) regresses 63.5 → 12.3 decode TPS at 8k context because
the per-decode-step `concatenate([scales, norms, rotor_table])` marshaling cost
swamps the kernel's compute savings; the kernel is reachable and the A.y guard
is preserved, but the perf win remains a follow-up. See
`docs/PERF_BASELINE.md` for the full bench numbers and analysis.

**QJL residual — storage format.** When QJL is enabled at first `append`,
one extra 1-bit sign per `head_dim` element per token is stored alongside the
rotor codes. Wire format: packed `u8` row-major, shape
`[B, kv_h, max_seq, ceil(head_dim/8)]`. Bit order: LSB = element 0, MSB =
element 7 (matches Python `rotorquant/turboquant/rotorquant.py` reference).
The QJL projection matrix `S` (`[head_dim, head_dim]` f32, row-major) is
generated once per layer/head on first append and persisted to the SSD block
(`l{idx}.k.qjl_s`). The layout tag (`*_qjl`) distinguishes QJL-ON blocks
from QJL-OFF blocks so the reader can hydrate the projection matrix.

**QJL toggle.** CLI: `--rotor-qjl on|off` (default `on`). Env fallback:
`RMLX_ROTOR_QJL=0` disables. The toggle is read per-construction (not cached)
so env changes between tests still propagate.

**Score-time QJL correction.** The QJL correction is
applied **at decode time** inside `apply_qjl_correction` (called from
`rotor3_k_decode` / `rotor4_k_decode`) as a per-token K-side residual-add:

```
Δk[t, j] = ||r_t|| · sqrt(π/2)/m · sum_i ( S[i, j] · signs[t, i] )
K_corrected[t] = K_rotor[t] + Δk[t]
```

The downstream `Q · K_corrected` equals the Python reference's score-time
`term1 + term2` (`RotorQuantProd.inner_product` in
`rotorquant/turboquant/rotorquant.py:246-263`) because `term2` is linear in
`Q`. This lets the correction live entirely inside `rmlx-kv-quant`
(boundary contract preserved — no `rmlx-models`/`rmlx-runtime` reach-back)
and removes the need for any engine-side SDPA refactor: every existing
caller of `rotor3_k_decode`/`rotor4_k_decode` (CPU dequant path on
`Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4`, `RotorKAsym3/4`)
gets the correction for free.

Validation:
- **Math gate — bias-mean** (per-fixture, runs in `make ci`):
  `qjl_correction_score_estimator_unbiased` in
  `crates/rmlx-kv-quant/src/rotorquant_tests.rs` — reproduces the Python
  ref's `test_inner_product_unbiased` (n=1024 unit-normalized pairs,
  asserts `|bias| < 0.05` for both QJL on and off through the live
  `apply_qjl_correction` path).
- **Math gate — bit-equivalence linearity** (runs in `make ci`):
  `qjl_residual_add_matches_score_time_correction` in the same file —
  per-token, asserts `Q · K_on == Q · K_off + Python_term2` to within 1e-4
  absolute (f32 reorder noise; measured max_abs_err ≈ 6.7e-8, max_rel_err ≈
  4.1e-7 on head_dim=64 unit-normalized fixture). What this proves: the
  dequant-side residual-add is algebraically identical to the Python
  reference's score-time `term2` for every (Q, K, layer, head) given the same
  rotor MSE codes. Empirical real-model lift remains a deferred gate.
- **Per-K cosine** drops slightly on the LCG fixture (~0.002 at
  head_dim=128) — by design. Per-K cosine measures `cos(K_corrected, K_true)`
  and is not the relevant SDPA quality metric; the JL sketch trades a tiny
  per-element variance gain for an unbiased inner-product estimate, which is
  what attention scores actually consume.
- **Bonsai TPS regression bench** (see `docs/PERF_BASELINE.md`): decode TPS
  regression −0.06% (78.16 → 78.21 tok/s on Bonsai 4k prompt, 3 measured
  runs; well below the 15% ceiling).
- **Real-model output-logit lift** — deferred. The bit-equivalence linearity
  gate above supersedes the empirical 32-step cosine-lift gate as a stronger
  offline proof.

**GPU fused-QK kernels** (`rotor_fused_qk_msl.rs`) currently bypass QJL —
they only fire when `codec_has_gpu_encoder(codec) == true`
(q8/turbo3/turbo4 today; rotor is HOLD). When the rotor GPU
encoder lands, the kernel MUST either replicate the residual-add in MSL or
fall back to the CPU dequant path when `qjl_s_matrix.is_some()`.

**Storage round-trip** (carries the QJL sideband across SSD spill / hydrate)
is validated; the QJL wiring did not change the storage shape.

**A.y Qwen MoE arch guard (mandatory).** K-side ≤4-bit on Qwen MoE is the
PPL-disaster zone (218 → 8641 on Q4_K_M baseline; 7:1 GQA amplifies K-head
error through softmax). All four variants are flagged by `KvQuant::k_below_8bit()`
and `cache_type::validate_resolved` routes them through
`ResolveError::QwenMoeRotorKRejected { variant }`, which quotes the offending
variant by name. Error message verbatim:
`"K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant <variant> is rejected
for Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' (K stays 8-bit) or a V-only rotor
variant ('--kv-quant rotor3' / '--kv-quant rotor4')."` Smoke runs on Qwen MoE
rows for all four rotor K-side variants are expected to error with `exit 78`
(positive guard test only).

**MSL status.** CPU-only on the hot path. GPU axis-agnostic dispatch is a
deferred follow-up.

**SSD round-trip tests.** Eight tests in
`crates/rmlx-kv-ssd/src/block_io_tests.rs`:
`roundtrip_rotor_sym_3_{qjl,no_qjl}`,
`roundtrip_rotor_sym_4_{qjl,no_qjl}`,
`roundtrip_rotor_k_only_{3,4}_{qjl,no_qjl}` — each asserts K codes
bit-identical post-hydrate and the `use_qjl()` flag matches the tag. Tests
use `ROTOR_QJL_ENV_LOCK` (process-wide mutex) to prevent env-var races under
parallel `cargo test`.

---

## Per-arch default table

| Arch class | Condition | Default `KvQuant` |
|---|---|---|
| `Qwen3VLMoeForConditionalGeneration` | (any) | `None` (bf16) |
| `Qwen3_5MoeForConditionalGeneration` | (any) | `K8V8` |
| `Qwen3_5ForConditionalGeneration` | PARO checkpoint | `K8V4` |
| `Qwen3_5ForConditionalGeneration` | other | `K8V8` |
| `Qwen3ForCausalLM` | `weight_bits == 2` (ternary) | `Mixed{k8g64,v4g64}` |
| `Qwen3ForCausalLM` | other | `K8V8` |
| `Qwen2ForCausalLM` | (any) | `K8V8` |
| `LagunaForCausalLM` | (any) | `K8V8` |
| `Gemma3ForConditionalGeneration` | (any) | `Planar` |
| `Gemma4ForConditionalGeneration` | MoE (26B) | `K8V8` |
| `Gemma4ForConditionalGeneration` | hidden_size ≤ 2560, non-PARO | `K8V8` |
| `Gemma4ForConditionalGeneration` | hidden_size ≤ 2560, PARO | `K8V4` |
| `Gemma4ForConditionalGeneration` | hidden_size ≥ 5376 | `Planar` |
| (unknown) | — | `K8V8` |

Source: `KvCacheBuilder::resolve_default` in `kv_cache/mod.rs`.

---

## Memory and bit-rate summary

Approximate bytes per KV pair (`B=1, 1 layer, 1 head, D elements`):

| Mode | K bytes/tok | V bytes/tok | Total bytes/tok |
|---|---|---|---|
| `None` (bf16) | 2·D | 2·D | 4·D |
| `K8V8` | 1·D + D/128·4 | 1·D + D/128·4 | ~2.06·D |
| `K8V4` | 1·D + D/128·4 | 0.5·D + D/32·4 | ~1.65·D |
| `Planar` | 1·D + D/128·4 | 0.5·D + 0.5·D + D/16·4 | ~2.13·D |
| `Mixed{k8g64,v4g64}` | 1·D + 2·D/64·4 | 0.5·D + 2·D/64·4 | ~1.75·D |
| `K8VTurbo3` | 1·D + D/128·4 | 0.375·D + D/32·4 | ~1.51·D |
| `K8VTurbo2` | 1·D + D/128·4 | 0.25·D + D/32·4 | ~1.38·D |

PlanarQuant V carries extra rotation state (two u32 per group of 32), which
raises its byte cost above K8V4 despite the same 4-bit code width. The
quality improvement compensates on dense full-attention archs.

---

## TurboQuant calibration (`kv_calib.json`)

TurboQuant variants (K8V4-TQ, K8V8-TQ) require a `kv_calib.json` calibration
file that specifies per-head high-precision index sets. The file is generated
by `rmlx kv-calibrate` and consumed by the TurboQuant KV codec at runtime.

### Generation

```bash
rmlx kv-calibrate /path/to/model --recipe turbo3
# Writes /path/to/model/kv_calib.json
```

Internally, the command walks K/V projection weight tensors (dtype F32, BF16,
or F16), computes per-head L2 norms across the input dimension, and selects
the top-K highest-norm indices per head. These indices are stored as sorted
ascending `Vec<u32>` per head. The operation is CPU-only and acquires no
Metal claim.

### Recipe → outlier count

| Recipe | Internal | Ratio | head_dim=64 | head_dim=128 |
|---|---|---|---|---|
| `turbo2`, `turbo2_tcq` | `turboquant25` | 25% | 16 | 32 |
| `turbo3`, `turbo3_tcq`, `turbo4` | `turboquant35` | 50% | 32 | 64 |

Outlier count = `round(head_dim * ratio / 16) * 16` (GROUP_ALIGNMENT = 16,
round-half-away-from-zero). For standard head_dims (64/128/256) this matches
mtq exactly; rare divergence with Python's banker's rounding is possible only
at exact midpoints with non-standard head_dims.

### Schema compatibility

The `version` field is always `1`. rMLX extends the schema additively:

| Schema label | `version` | Extra fields |
|---|---|---|
| mtq v1 | `1` | *(baseline)* |
| rMLX v1.1 | `1` | `LayerCalib::codebook` (per-layer codebook override, optional) |

Key top-level fields:

| Field | Value |
|---|---|
| `version` | Always `1` |
| `recipe` | Internal recipe (`"turboquant25"` or `"turboquant35"`) |
| `head_size` | `head_dim` from `config.json` |
| `layers` | `BTreeMap<String, LayerCalib>` keyed by attention prefix |

The layer key is the attention module path up to and including the attention
block name, e.g. `"model.layers.0.self_attn"`.

**Backwards-compatibility**: v1 files (no `codebook` field) parse cleanly
into `codebook = None` via `#[serde(default)]`. Forward-compatibility:
v1.1 files with `codebook = Some(...)` are silently ignored by any reader
built against the plain v1 struct.

### Runtime lifecycle

At model-load time the server automatically discovers and wires the calibration
file. No CLI flag is needed. The lifecycle is:

1. **Discover** — `rmlx_loader::discover_kv_calibration(model_dir, expected_head_size)`
   probes `<model_dir>/kv_calib.json`. Returns `None` silently if the file is
   absent; emits `tracing::warn!` (and returns `None`) if the file is malformed,
   `version != 1`, or `head_size` mismatches the model's `config.json`.

2. **Validate** — Checked by `discover_kv_calibration`:
   - `version` must be `1`.
   - `head_size` must equal `ModelConfig::head_dim()` for the target model.
   Missing file or mismatch leaves the server fully functional with the default
   (uncalibrated) codec path — backwards-compatible.

3. **Attach** — `calibration: Option<KvCalibration>` is stored on
   `ModelLoadConfig` and forwarded through `KvCacheBuilder::with_calibration()`.
   The `KvCacheBuilder` makes the calibration available to per-arch construction.
   Per-arch wiring (calling `KvCacheBuilder::with_calibration` inside each arch's
   generator constructor) and codec-side consumption are deferred until calibrated
   codec paths are wired. No in-tree caller of `with_calibration` exists yet.

4. **Layer lookup** — `rmlx_models::kv_cache::lookup_layer_calibration(calib, layer_key)`
   resolves a layer's `LayerCalib` from the `BTreeMap`. Matching is:
   - **Fast path**: exact `BTreeMap` key lookup.
   - **Fuzzy path**: case-insensitive 3-component dotted-prefix match, e.g.
     `"model.layers.0.self_attn.k_proj"` matches key `"model.layers.0.self_attn"`.
     A 3-component query (e.g. `"model.layers.0"`) also matches via this path.
   Returns `None` if no entry matches. No in-tree caller exists yet.

5. **Consume** (deferred — per-layer `LayerCalib::value_high_precision_indices`
   will be passed to the TurboQuant codec to steer which V-projection dimensions
   receive high-precision treatment. `QuantV::high_precision_indices` stores the
   index sets; not read by any codec yet.

6. **Codebook consume** (wired on both CPU and GPU paths) —
   `LayerCalib::codebook.value` (if `Some`) is stored on `QuantV::value_codebook`.
   The CPU V-encode path passes it to `turbo_quantize_v_with_codebook`; the GPU
   V-encode path at `bits == 4` uploads it once into `value_codebook_gpu` and
   dispatches `turbo_quantize_v4_codebook_buf_gpu` (encode) and
   `turbo_dequantize_v4_codebook_buf_gpu` (decode). For `bits != 4` the GPU codec
   is not wired and the existing `KvStorage::K8VTurbo*` callers stay on the CPU
   path.

**Fallback**: if `kv_calib.json` is absent or fails validation, behaviour is
identical to uncalibrated operation. No error, no performance change.

### Rust API

```rust
use rmlx_loader::{
    discover_kv_calibration,
    read_kv_calibration, write_kv_calibration,
    KvCalibration, LayerCalib,
};

// Automatic discovery at load time:
let calib: Option<KvCalibration> =
    discover_kv_calibration(model_dir, head_dim as u32);

// Layer lookup inside per-arch construction:
use rmlx_models::kv_cache::lookup_layer_calibration;
if let Some(calib) = &builder.calibration {
    if let Some(layer) = lookup_layer_calibration(calib, "model.layers.0.self_attn") {
        // layer.value_high_precision_indices[head_idx] → sorted u32 indices
    }
}
```

`KvCalibration` and `LayerCalib` are `#[non_exhaustive]`; construct via the
writer or deserialize from JSON.

### Per-layer codebook override (rMLX v1.1)

`LayerCalib::codebook` is an optional `CodebookOverride` struct:

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
| `codebook.value` | Per-layer V-side codebook. `2^bits` centroids in **strictly ascending order**. |
| `codebook` absent | Omitted → `codebook = None` → built-in Lloyd-Max N(0,1) codebook used. |
| `codebook.value = []` | Empty vec deserializes cleanly but returns `Error::Quant` at first encode for that layer. |

**Semantics per layer:**
- `codebook = None` (absent from JSON or `null`) — use built-in Lloyd-Max. Zero behavior
  change; identical to uncalibrated behavior.
- `codebook.value = Some(cb)` — replace the 16-centroid Lloyd-Max with `cb` for V-side
  CPU encode on this layer. Length must equal `2^bits` (e.g. 16 for 4-bit). Centroids
  are per-layer and shared across all KV heads on that layer.

**GPU dispatch:**
The default MSL kernel (`rmlx_tq4_quantize`, `rmlx_tq4_dequantize`) has the
Lloyd-Max codebook hardwired in Metal source. The codebook-buffer variants
(`rmlx_tq4_quantize_codebook_buffer`, `rmlx_tq4_dequantize_codebook_buffer`)
take the 16 centroids as a kernel buffer argument and compute the 15 decision
midpoints `(cb[i]+cb[i+1])*0.5f` at runtime. `QuantV::append_inner` and
`QuantV::dequantize_choice` dispatch the codebook-buffer variants whenever
`value_codebook.is_some() && bits == 4`. The upload is cached on
`QuantV::value_codebook_gpu` (an `Array` of shape `[16]` f32, built once per
layer on the first GPU call). For `bits != 4` the per-layer override stays on
the CPU encode path because no GPU 2-bit / 3-bit codec is wired yet.

---

## Fused-QK kernels

The default decode path runs in two stages: K is **dequantized** from its
packed buffer back to bf16, then `scaled_dot_product_attention` runs the
full QKV/softmax/SV fused kernel against the bf16 K. That dequant is the
single largest decode-step bandwidth consumer on memory-bound models with
PlanarQuant-packed K.

The **fused-QK contract** lets a KV codec opt into a custom MSL kernel that
consumes the packed K (codes / scales / rotation indices) directly and emits
pre-softmax scores `[B, n_q_heads, 1, S_kv]` — no intermediate dequantized K
ever lives in HBM. Post-softmax, the legacy SV path (dequant V + matmul)
runs unchanged. Two follow-up work items complete the story:

* **Flash-decode kernel** — fuse the SV path too via a flash-decode kernel
  that keeps K, V, and online softmax all inside one threadgroup (mirrors
  mtq's `PLANAR_FLASH_DECODE_KERNEL` shape). Eliminates the V dequant +
  matmul ops and recovers the SDPA-internal fusion the split path gives up.
* **Codec generalisation** — generalise the fused-QK contract to other codecs
  (rotor, iso), so any codec that ships a packed K representation can ship an
  MSL kernel matching the same `(query, codes, scales, rot32 / sideband, dims)
  → scores` signature.

### PlanarK fused-QK scope

Implemented:
* `crates/rmlx-kv-quant/src/planar_fused_qk_msl.rs` — MSL kernel + Rust
  dispatcher. Reads PlanarQuant `(codes, scales, rot32)` triple, performs
  per-pair centroid lookup + inverse Givens rotation in registers, computes
  QK dot via per-thread multiply + threadgroup tree-reduction. Bit-exact
  with `planar_dequantize_v4_gpu` followed by reference matmul (tested in
  `planar_fused_qk_msl_tests.rs`, max abs error ≤ 1e-3 for both 4-bit and
  3-bit). One threadgroup per `(b, hq, s_kv)`; `head_dim` threads.
* `crates/rmlx-kv-quant/src/planar_fused_qk.rs` — CLI toggle (process-wide
  OnceLock, default `true`).
* `crates/rmlx-kv-quant/src/storage/quant_planar_k.rs::gpu_packed_view`
  — returns the sliced GPU codes/scales/rot32 for the accumulated `S`
  tokens, without dequantizing.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused`
  — appends K (packed), updates V (bf16), runs fused QK, adds the
  additive mask, precise softmax, GQA-broadcast matmul with V. Dispatch
  guard is **decode-step only** (`q_seq == 1`) — prefill chunks fall
  through to the legacy dequant+SDPA path so the cache state is not
  double-mutated.

### Storage applicability

| Variant | K codec | Eligible? | Notes |
|---|---|---|---|
| `KvStorage::PlanarK { k: QuantPlanarK, .. }` | PlanarQuant 4-bit | **YES** | The only path that goes through the fused-QK kernel today. K is `head_dim % 32 == 0`. Arch-guarded against Qwen MoE (PPL disaster — pre-existing). |
| `KvStorage::Planar { k: QuantK, v: QuantPlanarV, .. }` | q8_0 | NO | Planar is on the V axis. K is q8_0 (affine), not Planar-packed — the kernel does not apply. Future K-side ports (rotor, iso) would route via the same contract. |
| `KvStorage::Planar { bits: 3, .. }` (i.e. `KvQuant::Planar3`) | q8_0 | NO | Same as above — `Planar3` is the **V-side** 3-bit codec; K is still q8_0. |

### Performance posture (PlanarK fused-QK)

On Gemma4-e4b (only Planar-K-eligible test target — Bonsai's PlanarK
NIAH-retrieval gap was fixed by the warm-TTFT bf16-K shortcut; see
"Correctness gap" below — and Qwen3.6 MoE rejects PlanarK by arch guard)
decode TPS is within measurement noise of the legacy path (`+1%` over a 3-run
mean, 3-run stddev ≈ 1 TPS). The fused-QK path is approximately neutral
because the legacy `scaled_dot_product_attention` is already a single fused
flash kernel; the K-dequant cost saved is partly given back by the split SDPA
ops (softmax + matmul) the fused-QK requires. The real win lands with the
flash-decode kernel, which keeps QK + softmax + SV in one threadgroup and
restores the fused-kernel-vs-split trade-off — see anchors in
`docs/PERF_BASELINE.md`.

### CLI toggle

`--planar-fused-qk on|off` (default `on`). Process-wide OnceLock; the flag
is resolved at startup once. No env var fallback — keeps tests
env-lock-free, unlike `--rotor-qjl`'s `RMLX_ROTOR_QJL`. To bench the win
in isolation, run the same model with `on` and `off` (decode-step
sensitive — see `.rmlx/bench/perf_canary.csv` rows tagged
`planar-fused-qk-on` / `-off`).

---

## `rotor_flash_decode` — fused MSL flash-decode over rotor-quant K

Fused flash-decode for `KvStorage::RotorKOnly3` / `RotorKOnly4`: QK over the
packed rotor K store + online softmax + bf16-V SV, in two Metal dispatches per
decode step. The rotor codec's Cl(3,0) K-decode runs **inside** the attention
inner loop, so no bf16 / f32 K is materialised and nothing restages through the
host.

**What it replaced.** `update_rotor_k_only_{3,4}` called
`QuantRotorK{3,4}::dequant()` on every decode step — a full-prefix **CPU** rotor
decode into a `Vec<f32>` plus a re-upload. That is O(seq) host work per token
with the GPU idle, and it is what pinned the K-only rotor family in the
"Tier 3 — CPU-bound" bucket (0.05–8.8 TPS, see `docs/models/bonsai/27B/rMLX.md`).
The store is now GPU-resident (`storage::RotorGpuK`) and the kernel reads it
directly.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_msl.rs` — Rust dispatcher,
  header builder, dispatch counters.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_p1.metal` — pass-1 body
  (one body for **both** bit widths).
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — codec-agnostic
  pass-2 log-sum-exp merge, shared with `planar_flash_decode`.
* `crates/rmlx-kv-quant/src/storage/rotor_gpu_k.rs` — `RotorGpuK`, the
  GPU-resident packed ring (codes / per-group scales / per-token L2 norms) with
  paged growth and CPU-prefix seeding.
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

**QJL.** The optional 1-bit QJL residual (`--rotor-qjl on`, the default) is a
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
| `Rotor{3,4}Sym` | NO — uses the quant-V sibling | V side is also rotor-quantized, so it dispatches `rotor_flash_decode_symv` instead (same header, same `rf_decode_k_lane`). See below. |
| `RotorK{3,4}Asym` | NO | V side is affine-quantized; neither rotor flash kernel reads an affine V. |

### Ring eligibility is passed down, not inferred

`RotorGpuK` is only built for the codecs that can actually read it. The rotor K
GPU encode takes a `RingFeed` from its caller: `Maintain` from the two K-only
paths (prefill `update_rotor_k_only_*` and the fused decode entry), `Skip` from
the sym/asym mirrors. A ring for a non-eligible codec is not free —
`capacity * kv_h * n_groups * 8 + capacity * kv_h * 4` bytes per layer, growing
with context (order of a few hundred MB across a 36-layer model at 4k) — and
nothing would ever read it.

**Invariant: the ring either tracks `blocks` exactly, or it does not exist.** A
skipped feed *clears* rather than leaving the ring behind. A stale ring (blocks
grown, ring not) is the dangerous state: the next append takes `prev_seq` from
the longer `shape` and writes past the ring's filled region, leaving the gap
zeroed and attention silently wrong. Because a cleared ring re-seeds from
`blocks` on the next maintained append (`seed_from_cpu`), this is self-healing —
`reset()` / `truncate_to()` / a CPU `append()` all just drop it.

**`b > 1` skips.** The ring's per-step stride is `kv_h * n_groups` and does not
interleave batch, so a batched chunk cannot be laid into it (the encode carries
`b` × the span). That degrades to the CPU dequant path, which handles `b > 1`
correctly — it must not error, since a batched rotor cache worked before this
kernel existed. Both the append (`rotor{3,4}_sync_ring`) and the dispatcher
(`rotor_flash_shape_ok`) gate on it.

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

Against the *default* (`--rotor-qjl on`) baseline the same cells move
0.66 → 17.0 (Bonsai, 26×) and 2.35 → 51.8 (medgemma, 22×).

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

## `rotor_flash_decode_symv` — fused MSL flash-decode over rotor-quant K **and** V

The all-quant sibling of `rotor_flash_decode`, for `KvStorage::RotorSym3` /
`RotorSym4`: QK over the packed rotor K store + online softmax + SV over the
packed rotor **V** store, in two Metal dispatches per decode step. Neither axis
materialises bf16.

### Files

* `crates/rmlx-kv-quant/src/rotor_flash_decode_symv_msl.rs` — Rust dispatcher +
  dispatch counters. **Reuses `rotor_flash_decode_msl::build_rotor_flash_header`
  verbatim** — no second codebook, MUL table, or probe snapshot.
* `crates/rmlx-kv-quant/src/metal/rotor_flash_decode_symv_p1.metal` — pass-1
  body (one body for **both** bit widths).
* `crates/rmlx-kv-quant/src/metal/flash_decode_merge_p2.metal` — the shared
  codec-agnostic pass-2 LSE merge (third caller).
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_rotor_sym_fused`
  — dispatch site; `rotor_sym_flash_over_store` is the shared-KV consumer entry.

### Design: the V unpack is the K decode

The rotor codec is **axis-agnostic** — `rotor{3,4}_encode` (V) and
`rotor{3,4}_k_encode` (K) are the same function, the K fork only adding the
optional QJL sideband, and the dispatcher fires only with QJL off. Its per-lane
decode is also **self-contained**: lane `d` reads only its own group's code word,
that group's scale, the group's rotor, and the token's L2 norm.

So the SV loop needs no new decode math and no cross-lane exchange (unlike a
Hadamard-rotated codec such as PlanarQuant, whose fused-SV reference needs a
threadgroup butterfly). The body calls the header's `rf_decode_k_lane` **twice**
per token — once per axis — and multiplies the V result by the online-softmax
term already in registers. That function was emitted into the header by
`rotor_flash_decode` specifically so a quant-V kernel could call it unchanged;
this is that caller.

Each axis passes its **own** rotor table. They are seeded identically today
(`make_rotor_table(layer_idx, 0, n_groups)` on both stores), but reading V's
codes against K's table would be silently wrong the day the seeds diverge.

### V-side GPU ring

`QuantRotorV{3,4}` gained the same `storage::RotorGpuK` ring the K stores carry
(the type is named for the axis it landed with; its payload is the codec's own
and is axis-agnostic). `RingFeed::Maintain` now comes from the symmetric append
path on **both** axes; the V-only rotor variants still pass `Skip`.

The V GPU encode also now applies the same head-major → sequence-major reorder
the CPU `append` and the K GPU path already did. Without it a multi-token chunk
with `kv_h > 1` produced head-major blocks that `dequant()` un-transposed as if
sequence-major — and the seq-major ring would have read them scrambled.

### Memory: the mirror is gone

This is the point of the kernel. `Rotor{3,4}Sym` previously quantized both axes
at `exit_prefill` and then decoded from a full bf16 K+V mirror, which
`update_rotor{3,4}_sym` short-circuits to on its first line — the packed store
was written and never read. `feeds_bf16_k_at_decode()` / the new
`feeds_bf16_v_at_decode()` are both false for these variants, so `exit_prefill`
seeds neither mirror.

Measured (`serve`, `--rotor-qjl off`, 1838-token prompt, 64 gen, `kv_cache_bytes`
from the `generate: kv-cache bytes (N16)` line; "before" is the same commit's
parent binary, verified distinct by symbol):

| Model | Codec | KV before | KV after | Δ |
|---|---|---|---|---|
| Bonsai-8B | `rotor3_sym` / `rotor4_sym` | 590.0 MB | **390.8 MB** | −33.8% |
| gemma-4-e2b | `rotor3_sym` | 36.07 MB | **23.49 MB** | −34.9% |
| gemma-4-e2b | `rotor4_sym` | 36.07 MB | **23.58 MB** | −34.6% |

gemma-4-e2b reaches the kernel through the shared-KV producer path
(`try_dispatch_shared_store` → `SharedKv::Store` → `sdpa_shared`), so consumers
attend the quant store rather than forcing a bf16 materialisation. Verified at
`--log verbose`: 14 `rotor_flash_decode_symv_sdpa` dispatches, 6 shared-KV
producer dispatches, 0 CPU-dequant fallbacks.

### Performance posture — the mirror was **fast**

Same runs, decode TPS (median of 3 measured, warmup dropped; two independent
paired runs shown as a range where they differ):

| Model | Codec | bf16-mirror (before) | quant-V kernel (after) | Δ |
|---|---|---|---|---|
| Bonsai-8B | `rotor3_sym` | 145–148 | **14.1–15.8** | ≈ −89% |
| Bonsai-8B | `rotor4_sym` | 142–150 | **14.2–15.7** | ≈ −90% |
| gemma-4-e2b | `rotor3_sym` | 134–136 | **56.1–56.7** | ≈ −58% |
| gemma-4-e2b | `rotor4_sym` | 113–136 | **55.8–56.2** | ≈ −50…−59% |

**The regression is the kernel shell, not the V unpack.** On the same Bonsai
shape, `k_rotor3` — the bf16-V `rotor_flash_decode` from which this kernel is
copied — runs at **~23 TPS**. So the shared shell is already ~6.4× slower than
MLX's native bf16 flash attention (~145 TPS); adding the V unpack costs a
further ~1.6× (23 → 14–16). `rotor_flash_decode` shipped against a 1.34 TPS
**CPU** baseline, where 23 TPS is a 17× win; against a bf16 mirror it is a 6.4×
loss.

The shell's cost is structural, not a tuning miss: a `head_dim`-wide threadgroup
does a log2(head_dim)-round barrier tree reduction **per token** and then leaves
127 of 128 lanes idle while thread 0 runs the softmax — and the rotor sandwich
itself is ~64 FMAs per group, redone by each of a group's 3 lanes. Closing the
gap needs a different shell (simdgroup reductions instead of the barrier tree,
one decode per group instead of per lane), which is `rotor_flash_decode`'s
problem as much as this kernel's.

**So `rotor{3,4}_sym` is a memory/throughput trade, not a free win**: −34% KV for
−58…−90% decode. That was not the premise the work started from, and the
disposition of the trade is a product decision, not one this kernel can settle.

### Bit width, gates, ring eligibility

Identical to `rotor_flash_decode` above: `bits ∈ {3, 4}` via the header
(explicit; any other width is an `Err`), GPU device, QJL off (read from the
store's sticky decision, not the live toggle), `q_seq == 1`, `b == 1`, `head_dim`
a power of two and `<= ROTOR_FLASH_HEAD_DIM_MAX`. Any miss falls through to the
legacy CPU dequant path. Both axes' stores must report the same accumulated
sequence length or the dispatcher errors — a divergence would attend K over one
prefix and V over another.

## `planar_flash_decode` — single-pass MSL flash-decode for PlanarK

Single-pass MSL flash-decode for `KvStorage::PlanarK`: keeps QK + softmax
+ SV in one threadgroup over the decode-step (q_seq == 1) PlanarK K and
bf16 V buffers. Two-pass tile structure mirrors TurboFlash
(`turbo_flash_msl::TILE_SIZE = 64`).

### Files

* `crates/rmlx-kv-quant/src/planar_flash_decode_msl.rs` — MSL kernel,
  Rust dispatcher, `planar_flash_decode_enabled()` env gate, dispatch
  counter for NIAH.
* `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused`
  — dispatch site (when `planar_flash_decode_enabled()` is true, replaces
  the split fused-QK chain).
* `crates/rmlx-cli/src/commands/serve.rs::apply_planar_flash_decode_flags`
  — `--planar-flash-decode {on|off|auto}` CLI flag. Auto resolves OFF on
  every host (see below).

### Gate

`RMLX_PLANAR_FLASH_DECODE=1` enables the kernel.  CLI flag
`--planar-flash-decode {on|off|auto}` (in `rmlx serve`, default `auto`) is
the production switch and sets/removes the env var before the
`OnceLock` latches.  Default OFF on every host as of 2026-05-31 — see
"Auto-flip status" below.

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

The flash-decode kernel produces **byte-identical output** to the split chain
with **6× lower stddev** at the 4k canary shape, but does not beat the split
chain mean at this decode-token budget. The fused single-kernel save is
balanced by the loss of the upstream MLX flash kernel's tuning. See
`docs/PERF_BASELINE.md` for full data.

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

The q8 / TurboSym3 / TurboSym4 fused-QK MSL kernels are reachable from the
production decode path. These kernels were GPU-parity-correct but initially
unreachable at decode time because the per-codec K storage was either
CPU-only (`QuantKTurbo3`) or chunk-major (`QuantK` / `K8V*`), neither of
which matches the kernels' head-major flat-buffer contract. Iso
(`Iso3/4Sym`, `IsoKOnly3/4`) and rotor (`Rotor3/4Sym`, `RotorKOnly3/4`)
shims need a *segregated* combined buffer plus, for rotor, a per-(layer, head)
rotor table — neither expressible in a per-token shadow row. They are HOLD
pending a shadow split into `{per_token, sideband_table}`.

### Storage shape — q8 / turbo3 / turbo4

Added on `KvCache` as `fused_qk_shadow: Option<FusedQkShadow>`. Two
flat GPU arrays:

| Buffer | Shape | Per-token payload |
|---|---|---|
| `k_codes` | `u32 [B, kv_h, max_seq, codes_per_token]` | codec-specific packed codes |
| `k_combined_scales` | `f32 [B, kv_h, max_seq, combined_per_token]` | per-group f32 scales (no sidebands) |

The per-codec layout is computed by
`FusedQkLayout::for_codec(KvQuant, head_dim) -> Result<Option<Self>>` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_shadow.rs`. Wired codec
entries:

| `KvQuant` | `codes_per_token` (u32) | `combined_per_token` (f32) | sidebands |
|---|---|---|---|
| K8V4, K8V8 | `head_dim/4` | `head_dim/128` | — |
| TurboSym3 | `head_dim*3/32` | `head_dim/32` | — |
| TurboSym4 | `head_dim/8` | `head_dim/32` | — |
| Iso3Sym/IsoKOnly3, Iso4Sym/IsoKOnly4 | — | — | **HOLD** — `for_codec` returns `Ok(None)`; shadow split needed |
| Rotor3Sym/RotorKOnly3, Rotor4Sym/RotorKOnly4 | — | — | **HOLD** — same; plus rotor table cannot be per-token |

The q8 / turbo3 / turbo4 kernel shims read the codes / scales buffers
as flat 1-D inputs of length `tok_count * payload_per_token` where
`tok_count = B * kv_h * kv_seq`. The shadow is sliced
`[B, kv_h, max_seq, payload] → [B, kv_h, kv_seq, payload]` and
flattened on every dispatch — the dim-2 slice is non-contiguous, so
the flatten forces a per-step materialisation (see KV_CACHE.md §9.5
"Per-step cost framing").

### Dispatch wire-in

`KvCache::try_fused_qk_dispatch` in
`crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch.rs:113` is called
from `update_and_sdpa` (`crates/rmlx-kv-quant/src/kvcache/sdpa.rs:631`)
right after the K8V4-TurboFlash branch and before the legacy bf16 SDPA
fallback. Gates (in order):

1. `RMLX_FUSED_QK=1` (CLI flag `--fused-qk on|off|auto`,
   `rmlx_kv_quant::fused_qk_enabled()`).
2. `Device::Gpu`.
3. `q_seq == 1` (decode-only).
4. `head_dim ∈ {128, 256}` (kernel hard gate).
5. `kv_seq ≥ RMLX_FUSED_QK_MIN` (default 512; sub-threshold caches go to
   bf16 SDPA where the launch overhead is not amortised).
6. Codec is in the in-crate `lookup_fused_qk_kernel` table (mirrors the
   public `rmlx_models::kv_cache::attention_dispatch::FUSED_QK_TABLE`).
7. The codec has a GPU encoder wired in (`codec_has_gpu_encoder`).
8. `decode_fp16_k` is seeded (post-prefill).

The shadow is allocated lazily on the first dispatch (seeded by
quantising the prefill bf16 prefix in `decode_fp16_k`) then appended
head-major every subsequent decode step via 4-D `slice_update` at
`[:, :, prev_offset:prev_offset+new_seq, :]`. Bf16 `decode_fp16_k/v`
stay maintained as the fallback path.

### Codec coverage

| Codec family | GPU encoder | Status |
|---|---|---|
| q8 (K8V4, K8V8) | `q8_quantize_gpu` | **Wired** — cosine ≥ 0.999 vs bf16; dispatch delta proven |
| TurboSym3 | `turbo_quantize_v3_gpu` (axis-agnostic) | **Wired** — cosine ≥ 0.998 vs bf16 |
| TurboSym4 | `turbo_quantize_v4_gpu` (axis-agnostic) | **Wired** — cosine ≥ 0.999 vs bf16 |
| Iso3Sym / IsoKOnly3 / Iso4Sym / IsoKOnly4 | iso V-side GPU encoder exists; K-side blocked on shadow split | **HOLD** — `for_codec` returns `Ok(None)`; per-token shadow can't host segregated `[scales \| norms]` combined buffer |
| Rotor3Sym / RotorKOnly3 / Rotor4Sym / RotorKOnly4 | rotor K GPU encoder MISSING; also blocked on shadow split | **HOLD** — per-(layer,head) rotor table is not per-token |

### Dispatch counter

Aggregated across all 5 kernel families by
`rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count()`. NIAH / parity
tests use `delta = after - before > 0` to prove the kernel actually
fired through the production path.

### See also

* `crates/rmlx-kv-quant/tests/fused_qk_dispatch.rs` — GPU integration
  tests for q8 + TurboSym3 + TurboSym4.

---

## Sparse attention

Two-phase MSL kernel pair in `crates/rmlx-kv-quant/src/sparse_attn/`:

| Kernel | Role |
|---|---|
| `phase1_score_msl::phase1_score` | Per-(q, head) cheap-inner-product score against every KV slot; emits sorted partial top-K (`TOP_PER_TILE` slots per tile). |
| `phase2_sparse_attend_msl::phase2_sparse_attend` | Runs SDPA only on the phase-1 selected slots; per-tile partials are LSE-merged into the final attention output. |

Dispatcher: `rmlx_models::kv_cache::attention_dispatch::sparse_attn_dispatch_if_enabled`.
Gate: [`rmlx_kv_quant::sparse_attn_enabled`](../crates/rmlx-kv-quant/src/sparse_attn.rs)
reads `RMLX_SPARSE_ATTN=1` once into a `OnceLock`. CLI flag:
`--sparse-attn {auto|on|off}` (default `auto` → OFF on every host;
see `docs/CLI.md`).

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

Auto-policy: `apply_sparse_attn_flags::Auto` resolves OFF on every host
(same posture as `PlanarFlashDecodeMode::Auto`). The On override sets
`RMLX_SPARSE_ATTN=1` but does NOT cause the kernels to fire on a
warm-TTFT decode — that contract is structural, not gated.

Invariant tests:

* `crates/rmlx-models/tests/sparse_attn_dispatch.rs::sparse_attn_dormant_on_warm_ttft_update_and_sdpa`
  — warm PlanarK cache through `update_and_sdpa` with `RMLX_SPARSE_ATTN=1` keeps the counter flat.
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

## Per-arch defaults (composite-score audit)

### Composite score formula (3-term)

The NIAH term was deferred from the initial audit; a 3-term composite is used
with re-normalized weights (original 4-term: 0.4 TPS + 0.3 NIAH + 0.2 cosine
+ 0.1 1/mem; drop NIAH, re-normalize remaining by 0.7):

```
score = 0.571 × decode_tps_norm + 0.286 × cosine_norm + 0.143 × mem_norm

where:
  decode_tps_norm = decode_tps / max_decode_tps_per_model
  cosine_norm     = clamp((cosine_floor − 0.94) / 0.06, 0.0, 1.0)
  mem_norm        = (1 / mem_bits_per_token) / max(1 / mem_bits_per_token)
                    across all candidates for the same model
```

**Conservatism gate**: if the winner differs from the current default by
`< ±1% TPS` AND `< ±0.002 cosine`, keep the current default.

**Conservative tie-breaker**: `Δscore < 0.005` → prefer the
landed-earlier / more-tested codec.

NIAH will be added as a 4th term in a future audit cycle. Defaults will be
re-audited then.

---

### Data sources and exclusions

- **TPS**: `docs/PERF_BASELINE.md` § "Per-codec × per-model cells".
  Canary shape: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`,
  release-perf binary, M5 Max, 1 warmup + 3 measured runs, median.
- **Cosine floors**: per-codec empirical test floors from
  `crates/rmlx-kv-quant/src/*_tests.rs` (LCG fixture, pinned seed).
- **Mem per token**: K+V bits combined (e.g. K8V8=16, K8VTurbo3=11, Mixed{k8,v4}=12).

**Exclusions from direct comparison (shape mismatch)**:

The `iso3_sym`, `iso4_sym`, `k_iso3`, `k_iso4`, `rotor3_sym`, `rotor4_sym`,
`k_rotor3`, `k_rotor4`, and `tsym3` anchor runs used a 2-token short-prompt
shape which inflates decode TPS significantly vs the 4096-token canary
baseline used for the existing defaults. These cells appear in the
PERF_BASELINE.md table with measured values, but they cannot be directly
compared in the composite formula without shape-normalizing. They are excluded
from this audit cycle.

`planar_fused_qk (on)` for Bonsai (26.4 TPS) is also excluded — the
note marks it a short-ctx artifact, not representative of canary performance.

---

### Candidate evaluation per arch

#### Gemma4-e4b (hidden=2560, non-MoE, non-paroquant)

Candidates measured at 4096-token canary shape:

| Codec | TPS | cosine floor | mem bits | decode_norm | cosine_norm | mem_norm | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **K8V8** | 74.22 | 0.9990 | 16 | 1.000 | 0.983 | 0.625 | **0.942** |
| K8VTurbo3 | 73.16 | 0.9807 | 11 | 0.986 | 0.678 | 0.909 | 0.887 |
| turbo3_tcq | 73.54 | 0.9807 | 11 | 0.991 | 0.678 | 0.909 | 0.890 |
| turbo2_tcq | 73.97 | 0.9570 | 10 | 0.997 | 0.283 | 1.000 | 0.793 |

Max TPS = 74.22 (K8V8). Max 1/mem = 1/10 (turbo2_tcq).

**Winner: K8V8 (0.942)**. Prior default: K8VTurbo3.
Δscore = 0.055 (> 0.005 tie-breaker). +1.4% TPS, +0.0183 cosine — both
exceed conservatism gate (±1% / ±0.002). **FLIP: K8VTurbo3 → K8V8.**

K8VTurbo3 remains available via `--kv-quant k8vturbo3` for operators who
prefer the lower memory footprint (11 vs 16 bits/token) over the quality gain.

#### Qwen3.6-MoE (Qwen3_5MoeForConditionalGeneration, affine 8-bit)

A.y guard: K-side ≤4-bit rejected. All symmetric and K-side codecs skipped.

| Codec | TPS | cosine floor | mem bits | decode_norm | cosine_norm | mem_norm | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **K8V8** | 96.64 | 0.9990 | 16 | 0.991 | 0.983 | 0.625 | **0.937** |
| turbo3_tcq | 94.57 | 0.9807 | 11 | 0.970 | 0.678 | 0.909 | 0.878 |
| turbo2_tcq | 97.52 | 0.9570 | 10 | 1.000 | 0.283 | 1.000 | 0.795 |

Max TPS = 97.52 (turbo2_tcq). Max 1/mem = 1/10 (turbo2_tcq).

**Winner: K8V8 (0.937)**. No flip — default was already K8V8.

#### Bonsai / Qwen3ForCausalLM 2-bit (Mixed{k8g64,v4g64})

| Codec | TPS | cosine floor | mem bits | decode_norm | cosine_norm | mem_norm | score |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Mixed{k8,v4}** | 109.86 | 0.9937 | 12 | 1.000 | 0.895 | 0.833 | **0.946** |
| turbo3_tcq | 95.11 | 0.9807 | 11 | 0.866 | 0.678 | 0.909 | 0.819 |
| turbo2_tcq | 94.29 | 0.9570 | 10 | 0.858 | 0.283 | 1.000 | 0.714 |

Max TPS = 109.86 (Mixed). Max 1/mem = 1/10 (turbo2_tcq).
Mixed cosine floor: V-side turbo4 ≥ 0.9937 (the binding constraint).

**Winner: Mixed{k8g64,v4g64} (0.946)**. No flip — default was already Mixed.

#### Gemma4 dense (hidden ≥ 5376) and Gemma4 MoE

No cell data at canary shape. Defaults unchanged: Planar (dense),
K8V8 (MoE).

#### Qwen3ForCausalLM 8-bit (dense, non-Bonsai)

No cell data. Default unchanged: K8V8.

---

### A.y guard re-verification

`validate_resolved` (in `crates/rmlx-models/src/kv_cache/cache_type.rs`) was
inspected and confirmed to reject K-side ≤4-bit codecs on Qwen MoE arches:

- `TurboSym4` → `QwenMoeKBitsTooLow(4)`
- `PlanarK` → `PlanarKOnQwenMoe`
- `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4` → `IsoKOnQwenMoe`
- `Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4` → `RotorKOnQwenMoe`
- `TurboSym3` → `TurboSym3KOnQwenMoe`

None of these K-side ≤4-bit codecs are selected by `resolve_default` for
Qwen MoE — the `Qwen3_5MoeForConditionalGeneration` arm always returns
`K8V8`. The guard is intact; no weakening detected during the codec adds.

Empirical positive test: `validate_resolved_qwen_moe_low_k_bits_rejected_post_decompose`
in `crates/rmlx-models/src/kv_cache/cache_type_tests.rs` verifies the
runtime rejection path.

---

### Operator migration note

**If you pinned `--kv-quant k8vturbo3` explicitly**: your config is
unaffected. Explicit `--kv-quant` always overrides the auto-default.

**If you relied on the auto-default for Gemma4 small** (e2b / e4b, non-MoE,
non-paroquant): the default reverts from K8VTurbo3 back to K8V8 as of this
audit. Memory footprint increases from 11 → 16 bits/token on K+V combined.
To keep the lower-memory option, pass `--kv-quant k8vturbo3` explicitly.

**`for_arch_default` deprecated**: callers should migrate to
`KvCacheBuilder::resolve_default(arch_class, ResolverSignals::from_config(&cfg))`.
The deprecated function remains (returns K8V8 for all inputs) and will be
removed in a future cleanup.

---

## See also

- `docs/KV_CACHE.md` — flag surface, Qwen MoE PPL disaster, codec matrix.
- `docs/WEIGHT_QUANTS.md` — weight quantization families (separate from KV).
- `docs/SSD_TIER.md` — SSD spill / hydrate for long-context eviction.
