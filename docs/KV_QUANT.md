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
> Only the **policy / builder** items (`KvCacheBuilder`,
> `kv_quant_for_layer`, `DEFAULT_KV_QUANT`, `LAYER_ADAPTIVE_*`,
> `cache_type::*`) remain under `rmlx_models::kv_cache::*`.

Codec-level reference for every KV quantization variant in rMLX. Covers
storage layout, dispatch path, the auto default, and CLI flag surface.

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
| MSL wrappers: `q8_msl::*`, `turboquant_msl::*`, `planarquant_msl::*`, `turbo_flash_msl::*`, `rot_k_msl::*`, `k8vturbo3_append_msl::*` | `rmlx_kv_quant::*` |
| Rotation helpers: `rot_k::{hadamard_rotation, rotate_last_axis, …}` | `rmlx_kv_quant::rot_k` |
| SWA ring buffer: `rotating::*` | `rmlx_kv_quant::rotating` |

The policy / builder layer stays in `rmlx-models::kv_cache`:

* `kv_quant_for_layer`, `DEFAULT_KV_QUANT`
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

// Builder / policy stay on the rmlx-models side:
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

The residual net-negative is on the **global** layers, and only for the codecs
that keep a packed store *and* a bf16 mirror. A quantized global layer keeps a
warm-TTFT bf16 decode seed (`decode_fp16_k` / `decode_fp16_v`, gated by
[`KvQuant::feeds_bf16_k_at_decode`]); when the codec also builds a packed store,
the codes + scales are pure overhead on top of a buffer the same size as bf16
and the codec is strictly larger than `--kv-quant none`.

For the **bf16-mirror family** (`K8V4`, `K8V8`, `Planar*`, `PlanarK`,
`K8VTurbo*`, `TurboSym*`, `Iso3/4`, `Rotor3/4`, `RotorK*Asym`) that overhead is
gone: no decode path reads their store, so `exit_prefill` does not build one
(`KvQuant::materialises_packed_store()`, see `docs/KV_CACHE.md` §9.6 F3). Their
resident KV is exactly the two bf16 mirrors — the same bytes as
`--kv-quant none`, at every context and every geometry — and the warn never
fires for them. It still fires for `Mixed` / `RotK`, which read
their packed 3-tuples at decode and keep both mirrors as well, and for the
K-only re-quantise families' sideband-heavy K store.

Historical note: before that change, measured on Gemma4 e2b (7 global + 28
windowed layers, head_dim 256, 1 KV head) at a 4096-token prompt, `k8v4` ≈
125.0 MB vs `none` ≈ 113.2 MB (`kv_cache_bytes`) — `k8v4` was +11.7 MB on the
global layers, zero delta on the windowed layers. Those two numbers are equal
now.

rMLX emits one structured `warn!` at request build time when the resolved codec
is estimated to increase resident KV vs bf16 on the active layer mix:

```
WARN KV codec increases resident KV vs bf16 on this layer mix — the
per-global-layer warm-TTFT bf16 seed plus codec scales exceed the bytes saved
at this context; windowed layers already run bf16 and are unaffected. The byte
figure is an UPPER BOUND, not an estimate: the iso arm of the estimator sizes a
group from the CPU-block layout, which carries a per-group quaternion the GPU
ring the iso codecs actually decode from does not, so for those codecs it
overstates by roughly 3x. The sign is exact either way. Consider --kv-quant none
if memory is the goal.
  kv_quant=mixed_k8g64_v8g64 eff_seq=8192 n_global=7 n_windowed=28 est_extra_bytes_upper_bound=51380224
```

The field is named for what it is. Only the **sign** of that number is exact;
size nothing from its magnitude, and for `k_iso*` / `iso*_sym` expect it to run
about 3x high (`KvQuant::approx_code_bits`, iso arm).

The estimate is model-agnostic — keyed only on layer geometry (`head_dim`,
`kv_heads`, `window`) and codec attributes (`KvQuant::approx_code_bits`, the
per-group scale cadence, whether the codec retains a bf16 seed, and whether it
materialises a packed store at all). The decision lives in:

- `rmlx_kv_quant::KvQuant::estimated_resident_bytes_per_layer` /
  `estimated_net_saving_per_layer` (codec layer — the per-side byte model;
  windowed layers return saving 0).
- `rmlx_models::kv_cache::kv_codec_net_saving_total` /
  `warn_if_kv_codec_net_negative` (policy layer — sums the layer mix and emits
  the warn). Wired into the Gemma4, Qwen3 and Qwen3.5-MoE `generate` paths; any
  arch interleaving windowed + global attention can call it with its own
  `KvLayerShape` vector.

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

Fix — one float dtype for the whole model, adopted at load: every float model
parameter — norm weights, quant scales/biases, and embedding scales/biases —
takes the bf16 activation dtype. The
projection and norm outputs then stay bf16, so K and V store as bf16. The YARN
mscale scalar is also stored as bf16 at load (defense-in-depth; the scalar was
never the root cause, but prebuilding it as bf16 is cheaper than a per-step
cast and keeps the multiply unambiguous. Two unit-level dtype-lock tests pin
the `rms_norm` and `bf16_param` call paths.

Measured (Bonsai-8B-2bit, `--kv-quant none`): decode-time K/V resident dtype
flips f32→bf16 (4→2 B/elem), halving KV residency. Decode TPS gains widen with
context as KV bandwidth dominates: ~+34 % at 4 k, ~+73 % at 16 k, ~+100 % at
64 k — recovering the prior loss vs the mlx-lm champion on this model.

**The chosen dtype is bf16, and that is not what the reference does.** mlx-lm
applies the same one-dtype rule but takes it from the checkpoint: measured with
mlx-lm 0.31.2 on `prism-ml__Ternary-Bonsai-8B-mlx-2bit`, all 653 float params
load as **float16**, the forward returns float16 logits, and the KV cache is
float16. bf16 has 3 fewer mantissa bits than fp16, so rMLX decodes this
checkpoint coarser than both the weights on disk and the reference. It is a
deliberate trade for the numbers above — bf16 is what this engine's kernels and
KV codecs are built around — and it has a measurable price: it flips tokens at
near-tie logits. `bonsai_8b_mixed_k8g64_v4g64.golden.txt` predates this cast and
is **stale at index 18** because of exactly one such flip: the two candidates
tie exactly in rMLX (both `-0.74530375`) where the reference sees a 0.0859
margin. That fixture has **not** been regenerated — a mismatch at index 18 is
the expected consequence of this cast, not a new regression, and the bisect that
identified it is recorded in the issue history. Do not restate this cast as
matching mlx-lm.

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

**Byte accounting.** One method reports KV-cache size:

- `KvCache::resident_bytes()` — actual on-device allocation: reads the real
  `Array` shape × `dtype.itemsize()` for every GPU buffer, and `Vec.len()`
  for CPU codec blocks. Covers packed codes, scales, zero-points, optional
  rotation/residual buffers, the GPU rings behind the ring-backed K codecs,
  and the warm-TTFT bf16 mirrors. It backs `kv_cache_bytes` observations, the
  `kv_bytes` event, prompt-cache eviction, and `rmlx baseline`. **Cost is
  O(blocks)** — call it at request boundaries, not per-layer per-decode-step.

Each figure is delegated to the store that owns the buffers
(`KvStorage::resident_bytes` → per-codec `byte_size`), so it is derived from
the allocations themselves. There is deliberately **no** second, per-codec
bits-per-element formula: one existed (`approx_bytes`) and drifted, reporting
byte-identical totals for `k_iso3` and `k_rotor3` — two codecs with entirely
different storage — while missing their GPU rings outright. A nominal
bit-width is not a cache's memory.

**One sample point across every arch: post-decode.** `kv_cache_bytes` is
recorded at a single lifecycle position — after the decode loop, when every
resident KV allocation exists, including the decode-time GPU ring of a
ring-backed codec. It means the same thing in every row of the matrix,
regardless of arch or of whether the prompt cache hit. "One lifecycle point"
means **post-decode, and only when a decode actually ran**: a run that returns
before the decode loop — immediate-EOS, i.e. the first sampled token is EOS —
does **not** refresh `kv_cache_bytes` and leaves the prior value in place. That
is uniform across every arch now (the exact-hit paths and gemma4 / gemma3 /
qwen3.5-moe always behaved this way), and it loses no ring information: with
zero decode steps no ring is ever allocated, so the value that is *not* written
would equal the prefill snapshot.

A NaN prefill is **not** in that category: it aborts the whole request with an
error, so there is no run to attribute a byte count to at all. It used to return
`Ok` with one junk token, which is what made the stale-value case reachable from
a fault rather than only from an ordinary early stop.

The store takes a `PostDecode` witness minted only by a completed decode loop
(`pipelined_decode` and the per-arch `decode_loop` / `decode_from` helpers) and
required by `KvBytesCounter::store`. This **raises the bar and documents the
requirement**: the naive re-drift — co-locating the store back at the prefill
snapshot, reusing the loop's witness — is a compile error, because that witness
is not yet in scope there, and the witness parameter makes a wrong lifecycle
obvious in review. It is **not** an unforgeable compile-time guarantee: `seal()`
is `pub(crate)` (each per-arch decode helper has to mint its own), so a new arch
*could* mint a fresh witness at the prefill point and compile. The backstop for
that is review plus the `#[ignore]`d GPU re-drift test
(`kv_bytes_hit_equals_miss`), which goes red if a path samples pre-decode — it
runs on a manual GPU pass, not in `make ci`. Keyed off the decode lifecycle,
never an arch.

Earlier this differed per arch: gemma4 / qwen3.5-moe / gemma3 sampled after
decode, while the qwen3-miss / qwen2 / laguna / bitnet / qwen3-vl-moe paths
sampled at the prefill snapshot (before the ring existed), and qwen3 recorded
either figure depending on whether the prompt cache hit. Ring-backed cells of
those pre-decode arches were a lower bound; they now include the ring. The
prompt-cache snapshot is still cloned at the prefill point (it stores the
prompt's KV, not decode KV) — only the *metric* moved to post-decode.

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
    default (`--fused-qk`).
  * **K-only iso** (`k_iso3` / `k_iso4`) → **`None`** (Metal). No bf16
    early-return: `update_and_sdpa_iso_k_fused` GPU-encodes the step into the
    packed ring and `iso_flash_decode` reads that ring directly, so **both**
    sides are GPU-resident and nothing restages through the host. Before the
    flash-decode kernel this was a hybrid — the per-step dequant restaged the
    growing prefix host-side and re-uploaded it via `Array::from_bytes` — which
    is what held these codecs at single-digit TPS.
  * **K-only rotor** (`k_rotor3` / `k_rotor4`) → **QJL-dependent, default
    off**. No bf16 early-return; `update_rotor_k_only_{3,4}` gates the GPU K
    encode on the store's sticky `use_qjl()` flag — fixed at first append, the
    same source the sdpa fast path reads, so a later env toggle cannot
    reinterpret bytes already written. QJL **off** (default) →
    `rotor{3,4}_gpu_append_into_k_blocks` Metal MSL encode (`None`), and the
    **decode** side is fused too (see § `rotor_flash_decode` below), so the
    default path is fully GPU-resident. QJL **on** (opt-in `--rotor-qjl on`) →
    CPU (`Some`): the 1-bit residual has no MSL kernel, so it forces the K
    append + dequant onto the host every decode step (single-digit TPS). The
    residual bought no measured accuracy in a two-arch context sweep — identical
    temp=0 output and needle retrieval on vs off — so off is the default and on
    is the fidelity / ablation knob.

  The `Some` cases are the source of the 30–60× first-forward slowdown and the
  monotonic decode decay as KV grows.

### Per-codec verdict

| Codec family | Hot-path verdict | Notes |
|---|---|---|
| `none` | bf16, no kernel | nothing to compile |
| `k8v4` / `k8v8` / `planar` / `planar3` / `planar_k` | **Metal** | q8_0 K + tq4 / planar V GPU kernels |
| `mixed_*` / `rot_k_v*` | **Metal** | MLX-affine `mx.quantize` K + affine V (compiled Metal ops) |
| `k8vturbo3` / `k8vturbo2` / `*tcq` / `tsym3` / `tsym4` | **Metal K**, CPU V (bounded) | K=q8_0 GPU; V CPU-forced by the −1 %/−2 % TPS gate, cost small |
| `iso3` / `iso4` | **CPU** | bf16 decode seed shadows the GPU iso branch; prefill V-encode on host |
| `iso3_sym` / `iso4_sym` | **fully Metal** | both axes iso-quantized; decode is `iso_flash_decode_symv` over both packed rings (no bf16 mirror). Ring-as-sole-store. `cpu_hot_path_reason() == None` |
| `k_iso3` / `k_iso4` | **fully Metal** | iso K MSL encode into the packed ring + `iso_flash_decode` fused decode over that ring. No host restaging. `cpu_hot_path_reason() == None` |
| `rotor3` / `rotor4` / `rotor*_sym` / `rotor_k_*_asym_*` | **CPU** | bf16 decode seed shadows the GPU branch; GPU fused-QK encoder is `--fused-qk`-only |
| `k_rotor3` / `k_rotor4` | **QJL-dependent (default off)** | QJL off (default) → **fully Metal**: rotor K MSL encode + `rotor_flash_decode` fused decode; QJL on (`--rotor-qjl on`) → CPU. Gate reads the store's sticky `use_qjl()` |

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
MSL codec, plus the tq4 / planar V kernel for `k8v4` / `planar`.
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

**Arch defaults**: none. `auto` is bf16 on every arch, so this is what
`Qwen3VLMoeForConditionalGeneration` gets — which matters there beyond
uniformity, because quantized KV produces incoherent output on that
checkpoint.

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
`head_dim=128`, Gemma3 text KV `head_dim=256`, Gemma4 text KV `head_dim=256`
on SWA layers and `512` on full-attention layers), so the cold path is
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

**Arch defaults**: none. `auto` is bf16 on every arch; K8V8 is opt-in.

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
`DispatchPolicy::turbo_flash` (or `turbo_flash_lock`) on the cache's policy.

**TurboFlash default-OFF policy (HOLD)**: `--turbo-flash` accepts
`{on, off, auto}` with `auto` as the default, and `auto` resolves **OFF on
every host**. The kernel passes its crash and fidelity gates and fails its
throughput one: everywhere it fires it decodes several times slower than the
generic K8V4 path. It also changes the generated tokens — because it is the
only `k8v4` configuration in which the 4-bit V codec runs at decode at all, not
because the kernel is wrong; see "What the digest difference is, and is not"
below.

Measured with `rmlx bench` (n=3 + warmup, one process per cell, medians,
settle gate enforced) on a quiet host — same binary, temp=0, `--turbo-flash`
the only difference:

| cell | on vs off | token digest |
|---|---|---|
| Bonsai-8B `k8v4` @~1.7k (`RMLX_TURBO_FLASH_MIN=0`) | 1.93× slower | — |
| Bonsai-8B `k8v4` @8k | 2.74× slower | **differs** |
| Bonsai-8B `k8v4` @16k | 3.48× slower | identical |
| Bonsai-8B `k8v4` @32k (63.25 → 14.89 TPS) | 4.25× slower | **differs** |
| Bonsai-27B `k8v4` @16k | 1.98× slower | identical |

The loss scales with `kv_seq` rather than being a fixed per-request penalty.
Every cell above settled on both sides with zero settle-gate refusals (32k
ranges: 1.14% off, 3.42% on). Dispatch was proven by counter, not inferred:
1638 kernel dispatches in the ON arm against 0 in the OFF arm. Shrinking the
ring 4× recovers part of the gap but not the bulk, so this is not a
`--max-ctx` sizing artefact. The `on` arm also holds 722 468 864 B more
resident KV at 16k — the persistent head-major flash buffers sit *on top of*
the bf16 mirror and the packed store rather than replacing either.

**What the digest difference is, and is not.** An earlier revision of this
section claimed a byte-identical token digest in both arms; that held on one
cell and was generalised from it. It is not byte-identical — but the reason has
been wrong twice, in two different ways, and both are now measured.

First: *the comparison baseline is not a reference.* Turning the gate off does
not turn the codec off. With `--turbo-flash off`, `k8v4` decode shortcuts to the
bf16 mirror and the 4-bit V store is never read — `decode_reads_packed_store` is
`false` for `K8V4`, and a GPU capture of the OFF arm contains no
`custom_kernel_rmlx_*` at all, only `sdpa_vector_2pass_2_bfloat16_t_128`. The
ON arm is the **only** `k8v4` configuration in which the codec participates in
decode. So "the kernel is not bit-exact" was measured against a bf16 attention,
which **any** correct tq4-V kernel must also differ from, and the kernel's own
error had never been tested.

Second: *half the observed divergence was a dtype promotion.* When these cells
were measured, 8k and 32k both diverged while 16k and Bonsai-27B@16k matched.
The 32k divergence was the f32 promotion described below and is gone with the
dtype fix — see the digest table further down, where 32k now reproduces the
bf16 reference exactly. The 8k divergence survives.

**The kernel's own numerics, measured.** `turbo_flash_reference_sdpa`
(`turbo_flash_msl.rs`, `#[cfg(test)]`) is a dequantize-then-SDPA arm over the
*identical* `flash_k_codes` / `flash_k_scales` / `flash_v_codes` /
`flash_v_scales` buffers, unpacked with the same two codecs the kernel unpacks
inline and computed at the kernel's own f32 working precision. Every
quantization error is common to both arms and cancels; what is left is the
kernel's block tiling, its online softmax and its two-pass rescale. Measured at
the attention geometry of each architecture the kernel actually dispatches on,
plus a masked cell, on a ring whose stride is wider than its fill and whose
last block is partial:

| cell | `head_dim` | `kv_h` × heads/kv | cosine | worst per-row diff |
|---|---:|---:|---:|---:|
| Ternary-Bonsai-8B-2bit | 128 | 8 × 4 | 1.0 | 0.056 bf16 ULP |
| Qwen3.6-35B-A3B-8bit | 256 | 2 × 8 | 1.0 | **bit-identical** |
| Bonsai-8B + additive mask | 128 | 8 × 4 | 1.0 | **bit-identical** |

The gate is cosine ≥ 0.999999 and ≤ 0.5 bf16 ULP per row — the bound quoted
here is the gate's, not the measurement's, so a drift toward it cannot leave
this page true and CI green at the same time.

The kernel is therefore accurate **for its codec**, and the ≈0.997 SDPA cosine
against bf16 is the tq4-V codec's own floor — which at temp=0 flips greedy
argmax ties prompt-dependently. Guards:
`turbo_flash_matches_its_codec_reference_at_{bonsai_8b,qwen36_35b}_geometry`
and `..._with_an_additive_mask` (`#[ignore]`, GPU).
Mutation-checked: feeding the kernel `t_active` where `t_stride` belongs drops
the cosine to −0.04 / 0.65, and dropping a single tail KV token drops it to
0.995 (19–28 ULP) — the latter is *below* the 0.997 codec floor, which is why a
bf16-referenced gate at that floor would have passed that bug and a
codec-referenced one does not. The reference arm is also asserted not to move
the dispatch counter, so it cannot quietly become a second call into the thing
it is checking.

**Two things the reference had to be taught, both found by cells that did not
exist at first.** It validated `n_q_heads % n_kv_heads` when the *kernel* did
not — see "GQA divisibility" below — and it handed the kernel's f32 mask
straight to MLX SDPA, which refuses a mask that does not promote to the bf16
output. A reference is only a reference where it accepts and refuses exactly
what it references.

### GQA divisibility — a kernel-entry gap the reference exposed

`turbo_flash_sdpa` computed `n_repeats = n_q_heads / n_kv_heads` in integer
arithmetic with no divisibility check, and the MSL maps
`kv_head = q_head / n_repeats`. For `(n_q_heads, n_kv_heads) = (3, 2)` that
truncates to `n_repeats = 1`, so `q_head = 2` reads `kv_head = 2` against a
two-head store — past that batch's KV base, silently, with a plausible-looking
answer. `n_kv_heads == 0` divided by zero. The rule now lives in
`validate_flash_shapes`, which both arms call, and is covered by the GQA cells
in `reference_and_kernel_refuse_the_same_shapes_for_the_same_reason`. No
in-tree caller passes a non-multiple — `update_and_sdpa_k8v4_flash_inner`
derives both counts from the cache's own shapes — so this is entry-validation
hardening, not a live-path fix.

**Consequence for the HOLD.** The correctness half is discharged: it asked for
something no bf16 baseline could ever supply, and the reference arm supplies it.
What remains is throughput (see #340 for the P1 grid's per-query-head KV
re-read). The rows below are the pre-fix measurement, kept because the TPS and
residency figures still come from it. Reproduce it on Bonsai-8B at 8k with `rmlx bench --kv-quant k8v4
--max-ctx 16384 --prompt-tokens 8192 --max-tokens 64 --runs 2 --warmup 1`:

| arm | decode TPS | token digest | `kv_cache_bytes` |
|---|---:|---|---:|
| gate OFF | 110.80 | `0xb0273cf32cb9b715` | 1 668 005 888 |
| gate ON (`--turbo-flash on`) | 42.05 | `0x75a6992e38913e64` | 2 029 240 320 |

TurboFlash is therefore a decode loss that also applies a codec the generic
path skips — not pure cost, and not a kernel-accuracy problem.

**Every cell in the two tables above was measured while the kernel promoted the
decode graph to f32.** `turbo_flash_sdpa` declared f32 kernel outputs and
returned them without restoring the query dtype, so with the gate ON the
residual stream, the next layer's RMSNorm, its weight GEMV, its elementwise ops
and the sampler all re-instantiated at f32 — visible in a GPU capture as
`affine_qmv_fast_float_*` replacing `affine_qmv_fast_bfloat16_t_*`, plus
`rmsfloat32`, `vv_Addfloat32`, `vs_Multiplyfloat32` and `argmax_float32`. The
dispatcher now casts back (`turbo_flash_msl.rs`), and those f32 instantiations
are gone from the capture. Read the ratios above as an **upper bound on the
kernel's own cost**: part of what they measured was the promotion, not the
kernel. The gate posture is unchanged — the ON arm is still slower than the
generic path in the same direction on both cells re-run after the fix — and the
cells are due a re-measurement on a quiescent host before the numbers are
restated.

The digest picture changes too, and only at one of three contexts. Bonsai-8B
`k8v4`, temp=0, 32 generated tokens, `RMLX_TURBO_FLASH_MIN=0`, one process per
cell, digest over the emitted token ids:

| prompt | gate OFF | gate ON, before the dtype fix | gate ON, after |
|---|---|---|---|
| 4k | `587c5a59` | `587c5a59` | `587c5a59` |
| 8k | `a098059c` | `10323b3d` | `10323b3d` |
| 32k | `3466374f` | `163882de` | `3466374f` |

At 32k the ON arm now reproduces the bf16 reference exactly — that divergence
was the promoted graph, not the codec. At 8k it does not, and that one survives
the fix: it is the tq4-V codec floor the section above describes. Which is the
point of removing the confound — a digest difference is now attributable to the
codec, because both arms finally run at the same dtype. That 8k cell also pins
the extra resident KV to 361 234 432 B — exactly half the 16k figure, so the
flash buffers scale linearly with the ring.

Re-confirmed at the **production** threshold (no `RMLX_TURBO_FLASH_MIN`
override), temp=0, 32 tokens, `kv_bytes` as the dispatch witness:

| arch | prompt | `kv_bytes` OFF → ON | digest |
|---|---:|---|---|
| Ternary-Bonsai-8B-2bit | 8192 | 1 145 733 120 → 1 506 967 552 | **differs** |
| Ternary-Bonsai-8B-2bit | 32768 | 4 657 250 304 → 6 102 188 032 | identical |
| Qwen3.6-35B-A3B-8bit | 8192 | 227 840 000 → 283 414 528 | identical |
| gemma-4-e2b-mxfp8 | 8192 | 58 601 472 → 58 601 472 (0 B) | identical — did not fire |

Two things follow. **Qwen3.6-35B-A3B is a second *firing* architecture whose
digest does not move** (`head_dim` 256, MoE): a kernel that computed the wrong
thing would not be selective by prompt. And the 8k Bonsai ON digest is
**byte-identical** to the digest the `rot_k_tq4v` codec produced at the
same shape (measured before that codec was retired in this same change, so it
is not reproducible on this tree) — a completely different decode path (dequant-then-SDPA, and a
*different* K codec) whose only thing in common with this one is that it applies
TurboQuant-4 V at decode. Two independent implementations of the same V codec
landing on the same 32 token ids is what a codec floor looks like; it is not
what a kernel defect looks like.

**gemma-4-e2b is a null control, not a second architecture.** On that
shared-KV / windowed arch (`kv_h=1`, `head_dim=256`, SWA 512) the ±0.3% A/B at
`k8v4`@4k is inside the 1.3% noise floor — but the reason is that the kernel
never runs: `kv_cache_bytes` is bit-identical across both arms
(156 850 176 B), which proves the persistent flash buffers are never even
allocated and the `kv_seq > 4096` gate stops every dispatch. That is evidence
the gate holds, not evidence about where the kernel pays. The second *firing*
architecture is **Bonsai-27B** (`Qwen3_5ForConditionalGeneration`,
`head_dim=256`, `kv_h=4`), which loses 1.98× at 16k. Both supported head_dims
lose, so there is no arch where the kernel currently pays.

This supersedes the previous per-family default-ON policy. The validations that
policy rested on are unaffected and still stand — they were crash/fidelity
clearances, never throughput ones: 32k NIAH × 3 models on Apple ≤9
(commit fcb2e894ccc4, 100% needle retrieval at 5 depths × 3 ctx tiers), and the
Apple10 `head_dim = 256` hazard re-drive on M5 Max via
`crates/rmlx-kv-quant/tests/apple10_head_dim_256.rs` (no SIGSEGV, dispatch
fired, cosine min 0.997 vs bf16). Lifting the HOLD needs a decode measurement,
not another one of those.

`--turbo-flash on` is an explicit force-ON — the opt-in for ablation and for
the re-measurement that would lift the HOLD. `--turbo-flash off` is a hard
override that a stale shell `RMLX_TURBO_FLASH=1` does not survive; `auto`
honours that variable, so an operator who opted in keeps the kernel. That
last combination — flag resolving OFF while the kernel actually runs — logs at
`warn!` and names the cost, so a variable exported once in a shell or a CI job
cannot carry the regression silently.

`--turbo-flash` is a **global** flag: it resolves in `main` before subcommand
dispatch, so `rmlx bench` / `rmlx baseline` measure the same kernel
configuration `rmlx serve` runs. Until that was fixed the two disagreed on this
very gate, and the measurement commands were the ones reading OFF.

**head_dim coverage (TurboFlash kernel)**: `head_dim ∈ {128, 256}`. The
P1 kernel's register arrays (`q_vals`, `o_state`, `v_decoded`) are sized
for the larger dim (8 entries = 256/32). `head_dim = 256` is
hw-validated on Apple10 (M5 Max) — historical SIGSEGV
hazard at this size does not reproduce.

**Qwen MoE note**: K8V4 is safe for Qwen MoE because K stays 8-bit. Running
K below 8-bit on a 7:1 GQA model amplifies quantization error through
softmax and produces catastrophic PPL degradation (218 → 8641 observed).
The K-side codec is the safeguard — not the variant name.

**Arch defaults**: none. `auto` is bf16 on every arch and at every context;
K8V4 is opt-in. It was the default for `Qwen3_5ForConditionalGeneration` (PARO
checkpoints) and `Gemma4ForConditionalGeneration` (small + PARO), and the
per-context policy picked it at ≤8192 tokens, until both were retired.

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

**Memory truth — the planar V side is larger than bf16.** Those 352 B carry 128
values, so the store spends **22.00 bits per value against bf16's 16.0**, at
every `head_dim` and at both bit widths (planar3 and planar4 occupy
byte-identical storage). The split is codes 4.0 + **per-pair scales 16.0** +
rotation indices 2.0: one `f32` per 2 elements is a whole bf16 value's worth of
sideband before a single code bit is spent, so the code width is not what sets
the rate. Measured, not modelled — `kv_rate_tests.rs` reads the bytes
`planar_quantize` actually produced. Two consequences:

- The perf win above is real and is **not** a memory win on the V axis; it buys
  TPS and quality with resident bytes.
- The rate above is the **store's** rate, and on a seeded cache `Planar` no
  longer keeps a store: nothing reads it at decode, so `exit_prefill` does not
  build it (`docs/KV_CACHE.md` §9.6 F3) and the layer's resident V is the bf16
  mirror at 16.0 bits per value. `KvQuant::estimated_resident_bytes_per_layer`
  reports that directly and its old store sub-term — 5.0 bits per value against
  a measured 22.0, off by 4.4× and wrong in the codec's favour — no longer
  enters the figure for this codec. The 22.0-bit rate still governs a
  store-backed planar cache: a seedless one (hydrated, or never through a
  prefill bracket), and the resident bytes of any future decode path that reads
  the store.

**Arch defaults**: none. `auto` is bf16 on every arch and at every context;
Planar is opt-in. It was the default for `Gemma3ForConditionalGeneration` and
dense `Gemma4ForConditionalGeneration` (`hidden_size` ≥ 5376), and the
per-context policy picked it above 32K tokens, until both were retired.

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
`MixedKvState::update_and_fetch` applies a Hadamard rotation to K before
quantization, and `mixed_quantized_sdpa` applies the same rotation to Q
before the score matmul so the rotations cancel. See rot_k below.

**Perf characterization**: +24% decode TPS vs K8V4 on Bonsai (36/36
full-attention layers), because `quantized_matmul` eliminates the per-step
full dequantize that dominates the rMLX K8V4 hot path at long sequences.
On GQA-light MoE archs (25% FA layers) Mixed K8V4 regresses by 11–28% vs
K8V8 — the `quantize` + `quantized_matmul` overhead amortises poorly when
most layers are not full-attention.

**Arch defaults**: none. `auto` is bf16 on every arch; Mixed is opt-in. It
was the default for `Qwen3ForCausalLM` at `weight_bits=2` (Bonsai ternary)
until the per-arch table was retired.

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
quantization error in the rotated basis — **but only when the channels were
unequal to begin with**. Measured against the identical unrotated quantizer:
**+1.81 bits** on outlier-channel data and **−0.63 bits** on i.i.d. uniform
data, where the transform raises the peak-to-RMS ratio the group scale is set
by. See "Codec fidelity — measured" below.

K is never inverse-rotated — the rotation cancels algebraically. This
distinguishes rot_k from V-side rotation schemes (PlanarQuant, TurboQuant)
where the output must be un-rotated back to the value basis.

`R` is the normalized Walsh–Hadamard matrix `H_D / sqrt(D)`. It is orthogonal
and symmetric (`R = Rᵀ`), so the same matrix rotates both K and Q.
Construction requires a power-of-two head_dim (Sylvester recurrence).

**v1 path** (`rot_k.rs`): plain MLX `matmul` against a precomputed `[D, D]`
matrix. Correct and coherent; O(D²) arithmetic per step.

**Fused FWHT kernel** (`rot_k_msl.rs`, opt-in via `--rot-k-fused on` /
`RMLX_ROT_K_FUSED=1` → `DispatchPolicy::rot_k_fused`):
Fast Walsh-Hadamard Transform in Metal threadgroup shared memory, fused with
affine-8-bit quantize in a single kernel pass. O(D log₂ D) arithmetic and no
intermediate DRAM allocation for `K_rot`. For D=128 (Bonsai): 896 arithmetic
ops vs 16 384 for the matmul (~18×). Output *format* matches
`mx.quantize(mode="affine", bits=8, group_size=64)` — same shapes, same dtypes
— so it feeds directly into `mixed_quantized_sdpa` unchanged. It is not
bit-exact with it, and not for a width reason — MLX's `affine_quantize` also
loads into `float` and reduces in `float`, casting only at the store
(`mlx/backend/metal/kernels/quantized.h:2460-2489` in 0.31.2). The difference
is the affine parameterisation. MLX initialises `w_max = 0` (so an all-negative
group still spans up to zero), takes `scale = max((w_max - w_min)/n_bins, eps)`,
flips its sign to whichever end is larger in magnitude, then snaps the
zero-point: `q0 = round(edge/scale)`, `scale = edge/q0`, `bias = at_zero ? 0 :
edge`. `metal/rot_k_fwht_quantize_d128.metal:58-59` uses the plain unsigned
form, `scale = (gmax - gmin)/255` with `bias = gmin`. Same width, different
grid: a value can land one level apart between the two arms before the
FWHT-versus-matmul rotation difference enters at all.

That dtype match is load-bearing and was missing until 2026-08: the kernel
returned its scales and biases as the f32 it computed them in, while
`mx.quantize` returns them at K's dtype. `quantized_matmul` and `dequantize`
take their operand width from the scales, so with `--rot-k-fused on` a bf16
model decoded the whole layer stack in f32 — attention output, residual add,
next layer's norm and weight GEMV — for as long as the flag was set, and the
fused and non-fused arms of one codec silently ran at different widths.
Narrowing the scales moves the reconstruction by at most 0.0156 against the
`mx.quantize` reference — measured and gated at that value, not merely printed
(`fwht_quantize_types_scales_like_mx_quantize`) — and it does move greedy
output: on Bonsai-8B at 8k the fused arm's token digest changes from matching
the non-fused arm to differing from it.

Fidelity was measured rather than assumed: against the unquantized rotated K,
`mx.quantize` reconstructs at cosine 0.999969 and the fused arm at 0.999965 —
4e-6 apart, both gated in the same test. Narrowing the scales did not move the
codec's accuracy in either direction to any degree this shape can resolve.

What the digest A/B does **not** establish is why it moved. The pre-fix arm had f32 scales
*and* an f32 decode graph, and the post-fix arm has neither, so the two
variables moved together; "the wider graph was masking the quantizer's own
rounding" is a plausible reading of it and not a measured one. Isolating it
would need a third arm — bf16 scales with the promotion forced back on — which
nothing needs today.

A matching `rot_k_fwht_rotate_gpu` kernel applies the same FWHT to Q,
replacing the `rotate_last_axis` matmul when the fused path is active.

**Storage**: `KvStorage::Mixed`.
`MixedKvState` carries a `k_rotation: Option<Array>` field with the
precomputed `R` matrix.

**Requirements**: power-of-two `head_dim`. Pair with any affine V codec
(default V = `q4_g64`).

**CLI**: `--ctk rot_k [--ctv <affine-tag>]`.

**Cosine gate**: K-side cosine similarity ≥ 0.9970 (mean), ≥ 0.9990 (min)
on LCG fixture data (head_dim=64, 8-bit affine group_size=64).
Test: `rot_k_hadamard_8bit_cosine_gate` in `rot_k_tests.rs`. That gate
measures the quantizer, not the rotation — deleting the Hadamard leaves cosine
at 0.999881 against 0.999989, inside its 0.001 slack, so it passes.
The rotation is gated by `rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data`
and `hadamard_incoherence_ratio_beats_every_block_local_rotation` in
`rotation_fidelity_tests.rs`.

---

### Retired: `rot_k_tq4v` (rotated K + TurboQuant 4-bit V)

Withdrawn. `--kv-quant rot_k_tq4v` is rejected at parse and
`--ctk rot_k --ctv tq4` at resolve (`combo_to_kv_quant`); each error names
`rot_k_v4g64`, which is the same rotated affine 8-bit K with an MLX-affine
4-bit V. Recorded here so the design is not re-derived.

It was a dequant-then-SDPA path: every decode step appended to its packed store
and then rebuilt a **full bf16 K and a full bf16 V of the whole prefix** before
running an ordinary `scaled_dot_product_attention` over them. `mx.quantized_matmul`
cannot consume a Lloyd-Max codebook, so the affine-V pairing's fused route was
never available to it, and the one kernel in tree that reads TurboQuant-4 V at
decode (`turbo_flash`) is `auto`-OFF on every host and measured slower than the
generic path. There was no third option.

Measured against its affine-V sibling `rot_k_v4g64` at the same shape
(`kv_bytes` from the `kv cache bytes` debug event; digests over 32 greedy token
ids at temp=0; TPS sequential n=1 on a host that could not pass its quiescence
gate, so read the ratios as direction and the residency and digests as exact):

| arch | ctx | resident KV vs `rot_k_v4g64` | resident KV vs `none` | decode TPS vs `rot_k_v4g64` | digest |
|---|---:|---:|---:|---:|---|
| Ternary-Bonsai-8B-2bit | 4096 | +0.49% | +31.4% | ×0.75 | matches |
| Ternary-Bonsai-8B-2bit | 8192 | +0.83% | +31.1% | ×0.63 | **differs** |
| Ternary-Bonsai-8B-2bit | 32768 | +0.86% | +30.6% | ×0.42 | matches |
| gemma-4-e2b-mxfp8 | 4096 | +0.46% | +44.9% | ×0.97 | **differs** |
| gemma-4-e2b-mxfp8 | 8192 | +0.74% | +43.6% | ×0.97 | **differs** |
| gemma-4-e2b-mxfp8 | 32768 | +1.0% | +42.6% | ×0.94 | **differs** |

Two things in that table are worth keeping.

**The memory claim was true and misattributed.** The codec did hold 27–45% more
resident KV than `--kv-quant none`, exactly as reported — but so does its affine
sibling, to within one percent. That excess is the whole `Mixed` / `RotK`
family's: those codecs read their packed store at decode *and* `exit_prefill`
still materialises both bf16 mirrors for them. `tq4` V versus affine-4 V is the
0.5–1% column, not the 30–45% one. Retiring `rot_k_tq4v` therefore does not put
any codec below `none`; that is a separate, family-wide defect.

**The decode loss was real and was the tq4 V.** On the dense `kv_h > 1` arch the
loss grows monotonically with context (×0.75 → ×0.63 → ×0.42 against the affine
sibling at 4k/8k/32k) — the signature of a per-step cost proportional to prefix
length, which is the double materialisation. On gemma-4-e2b it is small because
28 of 35 layers are SWA and leave `update_and_sdpa` on the bf16 rotating ring
before any codec branch is reached.

Fidelity was worse too: at temp=0 it is the only codec of the four measured that
never reproduces its affine sibling's token ids on the shared-KV arch, and the
e2e manifest already carried it as DEGRADED on Bonsai-2bit.

A correctness reference for TurboQuant-4 V at decode still exists in tree —
`turbo_flash_reference_sdpa`, in the TurboFlash section above — so retiring the
codec does not retire the ability to check that codec's numerics.

---

### `KvStorage::K8VTurbo3` — q8_0 K, TurboQuant 3-bit V

**Status**: opt-in on every arch — `auto` is bf16 and never selects it. It was
briefly the auto default for Gemma4 small, then reverted to K8V8 by the
composite-score audit, and both of those tables are now retired (see "Retired:
the per-arch default table" below). Available via `--kv-quant k8vturbo3`.

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

**Measured claw-back: 0.000 dB**, at both shipped widths, on i.i.d. Gaussian
(codes byte-identical to plain turbo) *and* on a dim-axis sweep (codes differ
by tie-breaking, distortion identical to four decimals — the stronger
statement, and the one the non-strict cosine gate cannot make). Pinned by
`trellis_coded_quantization_claws_back_nothing` in `rate_distortion_tests.rs`,
as an equality, so giving the trellis a real constraint turns it red.

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

**MSL hook**: removed — the parked GPU Viterbi kernel had no production
dispatch path and rotted (kernel-load failure, never caught because its
tests were `#[ignore]`d). The hot path forces `Device::Cpu`; a GPU kernel
can be re-added later with a real dispatch caller from day one.

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

It is never the auto default; `auto` resolves to bf16 for every arch.

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

It is never the auto default; `auto` resolves to bf16 for every arch.

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
is not implemented) and for `RotK` (rotation codecs are not
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
quiescent — it runs only at `exit_prefill`. The K-only family (`IsoKOnly*`,
`RotorKOnly*`) is the deliberate exception: it keeps K quantized at decode and
mirrors only V. Full per-codec audit table + the keep-universal decision (with
real-model parity numbers) live in `docs/KV_CACHE.md` §9.6.

---

## Layer-adaptive overrides

Two policies modify the per-layer codec assignment independently of the
request-level `KvQuant`:

**Tail layers** (`kv_quant_for_layer`, `LAYER_ADAPTIVE_TAIL_N = 8`): the last
8 layers are forced to `K8V8` under every **quantizing** base mode. Last-layer
KV vectors carry the highest per-token information density; forcing 8-bit
recovers PPL quality lost to aggressive V quantization.

**Head layers** (`LAYER_ADAPTIVE_HEAD_N = 2`): the first 2 layers are forced to
`K8V8`. First-layer K/V vectors carry large absolute magnitudes (embedding
residual is large before deep normalisation). The reference sweep that set this
constant measured 37–91% of turbo2's quality degradation recovered at ≥32K
context — but that is the *evidence* for the constant, not a gate on it.
`kv_quant_for_layer` is never handed a context length and promotes the first 2
layers at every prompt size.

When the base mode is already `K8V8`, both overrides are no-ops.

### `--kv-quant none` is a bf16 control

`KvQuant::None` is **exempt from both overrides**. The promotion buys back
quantization loss; a base mode that quantizes neither side has none to buy
back, and promoting it would allocate a packed q8_0 K+V store *on top of* the
bf16 buffers the layer already holds. `kv_quant_for_layer` decides this from
the codec's own `approx_code_bits` — a side kept at model dtype reports 16 — so
it keys off a codec property, never a codec name or an arch. The K-only
families (`planar_k`, `k_iso3/4`, `k_rotor3/4`) are **not** exempt: their V is
already bf16, but their K is 3–4-bit and has loss the boundary layers want back.

Until this exemption landed the promotion fired under `None` too, which made
`--kv-quant none` a bf16/K8V8 mixture on every arch whose boundary layers hold
a real token-indexed cache. The promoted layer's packed store was written once
at `exit_prefill` and **never read on the RAM path** — decode attends the bf16
mirror on a `K8V8` layer (§"Warm-TTFT decode contract") — so within a process it
could not change an output bit and was pure resident cost.

**On an SSD hydrate it was not output-neutral**, which makes this a latent
correctness bug and not only a memory one. Only a `KvStorage::None` layer
persists its off-storage bf16 prefix (`none_bf16_payloads`,
`crates/rmlx-kv-ssd/src/block_io.rs`), and only a layer that persisted one gets
`with_decode_fp16_seed` back on read. A promoted `K8V8` layer therefore came
back from disk with **no** mirror, `update_k8v8`'s `decode_fp16_k.is_some()`
fast path did not fire, and decode dequantized from the packed q8_0 store — on
2 to 10 layers of a run the operator had asked to keep in bf16. That is the one
path where the promotion under `none` changed output, and it required the SSD
tier to be enabled (it is off by default), which is why it never showed up in a
RAM-only A/B.

Measured with `rmlx --metrics off --log debug baseline --prompt-tokens <N>
--max-tokens 32 --device gpu`, reading the per-generation `kv_bytes` event;
`--emit-token-ids` pins the generated ids across the pair. Every pair below is
**token-id identical**, which is the falsifier this change was accepted
against:

| model | promoted layers that owned a cache | ctx | `none` before | `none` after (= true bf16) | before ÷ after |
|---|---:|---|---:|---:|---:|
| Ternary-Bonsai-8B (`Qwen3ForCausalLM`) | 10 of 36 global | 4k | 641 581 056 | 560 480 256 | **1.145×** |
| Ternary-Bonsai-8B | 10 of 36 global | 32k | 5 327 683 584 | 4 657 250 304 | **1.144×** |
| gemma-4-26b-a4b (`Gemma4…`) | 2 of 5 global | 4k | 313 131 008 | 294 748 160 | **1.062×** |
| gemma-4-26b-a4b | 2 of 5 global | 32k | 1 060 003 840 | 914 022 400 | **1.160×** |
| Qwen3.6-35B-A3B (`Qwen3_5Moe…`) | 2 of 10 global | 4k | 152 604 672 | 143 953 920 | **1.060×** |
| gemma-4-e2b (`Gemma4…`) | **0** | 4k | 31 776 768 | 31 776 768 | 1.000× |
| gemma-4-e2b | **0** | 32k | 217 559 040 | 217 559 040 | 1.000× |

`--kv-quant k8v8`, `k8v4` and `mixed_k8g64_v4g64` are byte-identical and
token-identical across the same pair on all three architectures — the exemption
touches the `None` arm only.

The counts in column 2 are what makes the *effect* per-arch even though the
policy is not. Two independent things can make a promoted layer a no-op, and
both still apply to the quantizing base modes:

- **Windowed layers ignore the codec.** A cache built with a sliding window
  runs the bf16 rotating ring regardless of the flag
  (`KvCache::with_quant_max_seq_window` — mlx-lm's
  `RotatingKVCache.to_quantized` raises `NotImplementedError` and rMLX matches
  it). Promoting an SWA layer to `K8V8` changes nothing it stores.
- **Shared-KV layers own no cache.** Gemma4's `num_kv_shared_layers` points
  every layer from `n_layers - num_kv_shared_layers` onward back at an earlier
  layer's cache (`gemma4/loader.rs::build_previous_kvs`); those layers are
  handed `cache: None` and their own slot stays empty. Promoting them changes
  nothing either.

On gemma-4 e2b and e4b **both** filters apply and cancel the policy entirely:
the only promoted layers that own a cache are 0 and 1, and both are sliding.
That is why they read 1.000× above and are the null control for this change.
**It does not generalise to the larger gemma-4s** — 12B, 26b and 31b carry
`num_kv_shared_layers = 0`, so 2 of their global layers were promoted. 26b's
measured ratio climbs with context (1.062× at 4k, 1.160× at 32k) because its
fixed SWA-ring term dilutes the ratio at finite length; the asymptote derived
from its layer geometry is 1.21×, the largest in the release set.

The per-layer arithmetic behind the deltas, at cache offset
`S = prompt_len + max_tokens - 1`. The rates come from
`KvCache::resident_bytes`: a `KvStorage::None` layer is `filled_seq_bytes` over
the two bf16 mirrors, a q8_0 layer adds packed codes plus one `f32` scale per
128 values over a `KV_PAGE_SIZE = 256`-rounded capacity.

- **Bonsai-8B** (`S = 3801`, `kv_h = 8`, `head_dim = 128`, capacity 3840). A
  bf16 layer holds `2 × 8 × 128 × 2 B = 4096 B` per token, so the post-fix
  figure is the bf16 identity `36 × 4096 × 3801 = 560 480 256` exactly. The
  81 100 800 B that used to sit on top is `10 × 2112 × 3840`, i.e. the q8_0 K+V
  store on the 10 promoted layers. `k8v8` measures
  `36 × (4096 × 3801 + 2112 × 3840) = 852 443 136`, unchanged by the fix.
- **gemma-4-26b-a4b** (32k, capacity 34 560). The 145 981 440 B removed is
  `2 × 2112 × 34 560` — the same q8_0 rate on its 2 promoted global layers.
- **Qwen3.6-35B-A3B** (`S = 3885`, capacity 4096). The 8 650 752 B removed is
  `2 × 1056 × 4096`. Its raw `kv_bytes` also sums the fixed GDN recurrent state
  (64 389 120 B, codec-independent), which is why the whole-cache ratio reads
  1.060× where the attention-KV ratio is 1.109×.
- **gemma-4-e2b** (`S = 4148`). 12 SWA rings at `2 × 1 × 256 × 2 B = 1024 B`
  per token, capped at the 512 window, plus 3 global caches at
  `2 × 1 × 512 × 2 B = 2048 B` per token — `global_head_dim` is 512, not 256.
  `12 × 1024 × 512 + 3 × 2048 × 4148 = 31 776 768`, the measured figure on both
  sides of the change.

**Historical rows measured against the old `none`.** `runs.db` is append-only
and `docs/PERF_BASELINE.md` carries anchors taken while `none` was a mixture.
Those rows are not re-measured; they are restated by the per-arch factor in the
table above — a `1.04× none` Bonsai row is `1.19×` true bf16. Ratios *between*
two recorded codecs are unaffected, and so is every SEPARATED / INCONCLUSIVE
verdict; only "vs bf16" restatements move. New runs need no factor.

Decode TPS moved with the change: exempting `none` bought roughly +5% on the
affected architectures (Bonsai-8B 131.9 -> 138.6 TPS, Qwen3.6-35B-A3B
94.5 -> 100.1) and 0% on the exempt ones (gemma-4-e2b 128.2 -> 128.4).

**The per-layer mechanism recorded here at the time no longer reproduces, and
should not be quoted.** It read a `K8V8`-typed layer as costing ~0.041 ms/step
more than a `None` layer (Bonsai-8B 4k: all-`K8V8` 8.688 ms/step against
`none`'s 7.215 over 36 layers) even though both attend the same bf16 mirror.
Re-measured on the current tree, all-`K8V8` against all-`none` is
**INCONCLUSIVE at every cell**: ABBA, 8 slots, quiescent host, token ids
identical and `kv_cache_bytes` ratio exactly 1.0000 --- Bonsai-8B +0.45% at 4k
and +0.13% at 32k, Qwen3.6-35B-A3B +5.41% at 4k and +0.19% at 32k, gemma-4-e2b
+0.18% at 4k. Ranges overlap in all five, so none of those percentages is a
measured difference. The intervening change is the packed-store elision: the
figures above were taken while a `K8V8` layer still built and held a store that
decode never read, and with the store gone a `K8V8` layer is a `None` layer
under another name. The +5% the exemption bought is left as recorded --- it was
measured against that older arm and is not re-derivable now.

Recording the effective per-layer mixture alongside `kv_bytes` was considered
and not done: it is summed at 14 per-arch call sites, and with `none` meaning
none the requested codec is simply a true label for the row.

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
| `mixed_k<kb>g<kg>_v<vb>g<vg>` | `KvQuant::Mixed { .. }` |

Examples: `--kv-quant mixed_k8g64_v4g64`, `--kv-quant k8v4`.

### Named preset interface — `--kv-preset`

`--kv-preset <name>` is the high-level named preset flag. It resolves a short
human-readable name to a concrete `KvQuant` at clap parse time — no further
resolution needed at runtime.

**Conflict rule**: `--kv-preset` is mutually exclusive with `--kv-quant`,
`--cache-type-k`, `--cache-type-v`, and `--kv-bits`. Passing any combination
is a clap hard error (caught before the subcommand body runs).

#### Preset table

| Name | `KvQuant` | Notes |
|---|---|---|
| `fp16` | `KvQuant::None` | bf16 unquantized both sides (`KvQuant` variant named `None`, not `Option::None`) |
| `q8` | `KvQuant::K8V8` | symmetric 8-bit K+V |
| `speed` | `KvQuant::TurboSym3` | symmetric WHT-3 K+V, matches mtq `speed`; rejected on Qwen MoE |
| `quality` | `KvQuant::TurboSym4` | symmetric WHT-4 K + tq4 V, matches mtq `quality` byte-for-byte; rejected on Qwen MoE arch guard |
| `planar` | `KvQuant::Planar` | PlanarQuant V-side |
| `planar3` | `KvQuant::Planar3` | PlanarQuant 3-bit V-side |
| `k_only_planar` | `KvQuant::PlanarK` | PlanarQuant K-side, V bf16; rejected on Qwen MoE |

**None of the six non-`fp16` rows changes resident KV or output.** Each resolves
to a codec in the inert class (§"Codec disposition"): decode reads the bf16
mirror, so `exit_prefill` never builds the packed store, and the served request
holds the same bytes and emits the same token ids as `fp16`. A preset is a
codec name, not a memory setting. `no_preset_is_a_memory_lever` pins that claim
and fails the moment a preset's codec starts reading its own store.

No preset is planned. A new row is worth adding only once its codec's decode
reads its own packed store; before that it is another spelling of `fp16`.

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
rmlx baseline --model <path> --kv-preset auto    # == --kv-quant auto
```

### `--kv-preset auto`

`--kv-preset auto` resolves to `rmlx_models::kv_cache::DEFAULT_KV_QUANT` — the
same constant `--kv-quant auto` resolves to, read from the same place. It does
not consult the preset table and does not look at the hardware.

It used to. Until the disposition below was measured, `auto` ran a decision tree
over `sysctl hw.memsize` and an estimated parameter count, and returned a
"compressing" preset when the model plus its bf16 KV would not fit:

```
if total_bf16   < budget → "fp16"
if model + kv/2 < budget → "q8"
if model + kv/4 < budget → "quality"
...
```

Every branch of that tree returns a preset that holds **byte-identical** resident
KV to `fp16`. It answered a memory question with a codec that has no memory
effect, and it did so silently — the operator saw `auto-selector chose preset
q8` and had no way to learn that the choice changed nothing. Its own KV estimate
was, by its docstring, 10–30× off, so it could not have been repurposed into a
warning either. Both the tree and the two hardware queries that fed it
(`unified_memory_gb`, `estimate_params_billions`) are gone.

Two `auto` surfaces that resolve independently are two defaults that can
disagree. `preset_auto_is_the_same_default_as_kv_quant_auto` pins that they no
longer can.

### Per-side primitive interface

`--cache-type-k <tag>` / `--ctk <tag>` sets the K codec.
`--cache-type-v <tag>` / `--ctv <tag>` sets the V codec.

`--kv-quant` and `--cache-type-*` are mutually exclusive. Passing both is a
clap-time hard error.

Available K-side tags:

| Tag | Codec |
|---|---|
| `auto` | resolved to `DEFAULT_KV_QUANT` (bf16) |
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
- `--paged-kv` is incompatible with `rot_k`.

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

**Memory truth.** iso spends, per 4-element group, one whole `u32` code word
**and** one `f32` scale — 8 B for 4 values — plus one `f32` norm per token.
The nominal codebook width never reaches the store: iso3 uses 12 of its 32
code bits and iso4 uses 16, so **iso3 and iso4 occupy byte-identical
storage**. At head_dim=128 that is 260 B per token per kv_head against bf16's
256 B: **16.25 bits per value, 1.02× bf16**.

The rate itself does move with head dim; the *sign* does not. Reading it off
the allocation — `(D/4)·4 B` codes + `(D/4)·4 B` scales + `4 B` norm — gives

```
iso stored bits/value = 16 + 32/head_dim
```

so 16.25 at D=128, 16.125 at D=256, 16.0625 at D=512, approaching 16.0 **from
above** and never reaching it. Two planes at 8 B per 4 values are already
exactly bf16's density, and the per-token norm is what keeps the sum strictly
greater at every finite head dim. This is a derivation from
`QuantKGpuRing::alloc` (`Dtype::U32` codes, `Dtype::F32` scales and norms, one
element each per group per token per KV head), not a measurement.

The CPU `IsoBlocks` form adds a 4×f32 quaternion per group on top, taking the
same token to ≈772 B (≈48.25 bits per value, 3.0× bf16). That sideband is the
constant `FIXED_QUAT` replicated per group, not data, and the GPU ring the
K-only and symmetric codecs decode from does not carry it. **That figure is
hypothetical on a served request**: the V-only `iso3` / `iso4` codecs decode
from the bf16 mirror, so `exit_prefill` builds them no store and they measure
byte-identical to `none` (§"Codec disposition", Class 2). It is the rate they
would cost the day a decode kernel reads their store. The store-reading members
are `k_iso3/4` and `iso3_sym/4_sym`, and those measure 1.003–1.054× `none` on
the ring layout above (§"Codec disposition", Class 3 — whole-cache ratios
against a `none` that is a true bf16 control since the head/tail promotion
stopped applying to it). The per-group figures above are the store density and
are unaffected by that.

**No iso codec is a memory win, at any head_dim.** 8 B per 4 values is exactly
bf16's density before the per-token norm is added, so the packed side is
strictly larger than the bf16 side it replaces for every shape. These are
research codecs for quality experiments and kernel work, not size wins. The
sign is pinned by `iso_and_rotor_k_codecs_are_never_a_memory_win`
(`crates/rmlx-kv-quant/src/quant_tests.rs`) and surfaced to the operator by
the resolve-time net-negative warn, which the Gemma4, Qwen3 and Qwen3.5-MoE
generate paths call (the remaining arches do not call it yet).
`estimated_resident_bytes_per_layer` models the group layout directly (never
the codebook width) and counts the quaternion sideband, so its number is a
conservative upper bound for the ring-resident members.

**Crate-wide rate ceiling.** `crates/rmlx-kv-quant/src/kv_rate_tests.rs` reports
every store family's bits per value and fails any family above bf16's 16.0 that
does not carry a written exemption. An exemption is itself checked: a family
listed as exempt must actually measure above the floor, so a fixed codec turns
its own exemption red instead of silently keeping it.

Completeness is partial and stated as such. The `KvQuant` → family map is an
exhaustive `match`, so a new variant does not **compile** until someone writes
down where its bytes go — that much is mechanical. The list of measured
representatives is hand-maintained and nothing forces it to grow, so a variant
that declares its families and never gets a representative is unmeasured and the
gate stays green; catching that is review's job. Closing it mechanically needs
enum iteration (a `strum`-style derive), which is a dependency decision.
Table at `head_dim = 128`:

| Family | Stored bits / value | Provenance | Verdict |
|---|---|---|---|
| turbo2 / tcq2 | 3.00 | measured | under |
| turbo3 / tcq3 | 4.00 | measured | under |
| turbo4 | 5.00 | measured | under |
| q8 (group 128) | 8.25 | measured | under |
| affine (`CacheType::Q8G32`) | 10.00 | layout formula | under |
| bf16 | 16.00 | by definition | the floor |
| **rotor3 / rotor4** | **21.75** | measured | exempt |
| **planar3 / planar4** | **22.00** | measured | exempt |
| **iso3 / iso4** (`IsoBlocks`) | **48.25** | measured | exempt |

"Measured" means the summed byte length of the buffers that family's own CPU
encoder produced over a shared fixture. The two non-measured rows have no CPU
encoder in this crate: bf16 is two bytes per value by definition, and MLX affine
is `bits + 64/group` — code bits plus one `f32` scale and one `f32` bias per
group.

**The affine row is not a bound over every parseable config.** It covers the
widest *enumerable* `CacheType` (`q8_g32`). The `mixed_k<b>g<g>_v<b>g<g>`
grammar reads the group size as a bare `u16` with no whitelist and no floor, so
`mixed_k8g4_v8g4` parses and stores `8 + 64/4 = 24` bits per value — above the
floor and invisible to an enum-driven gate, because the rate is a property of a
runtime field with an unbounded domain rather than of the variant. Pinned by
`mixed_grammar_admits_affine_rates_above_the_floor` rather than fixed: adding a
parser floor rejects configs that parse today, which is a CLI-surface decision.

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
head-major chunk heads↔seq (`[B, new_seq, kv_h, D]`) before encoding, and
`dequant` reorders **each block back at its own sequence offset**
(`seq_layout::transpose_chunked_seq_heads`) — reordering the whole concatenation
in one pass is only equivalent at `B == 1`; see the `b > 1` note under the
truncation planner. Single-chunk cold prefill is the identity. The codec is
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
| Delivered bits / element, ring-resident (`k_iso*`, `*_sym`) | **16.25** (260 B/token at head\_dim=128 — see Memory truth in iso3 section) | **16.25** — byte-identical to iso3; the codebook width never reaches the store |
| Delivered bits / element, CPU-blocks form (`iso3` / `iso4`) — **hypothetical, not resident** | ≈48.25 (≈772 B/token at head\_dim=128, incl. the constant quaternion sideband). These two codecs decode from the bf16 mirror, so `exit_prefill` builds them no store and they measure byte-identical to `none` (§"Codec disposition", Class 2); this is the rate they would cost once a kernel reads one | ≈48.25 — same sideband, same code word, same hypothetical |
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
dispatches the sibling 4-bit kernel pair `iso_quantize_v4_gpu` /
`iso_dequantize_v4_gpu`, with the per-(token, group) thread layout, atomic-OR
pack via `(idx & 0xF) << shift`, and a dense 8-vals/u32 boundary table (15
mid-points derived from `lloyd_gaussian_codebook(4)`). The bodies live in
`src/metal/isoquant_quantize_iso4.metal` and
`src/metal/isoquant_dequantize_iso4.metal`, gated by `make check-metal-compiles`
against the captured header snapshot `src/metal/probes/isoquant_iso4.hdr.metal`.
The encode side is wired into the three iso4 update paths (`update_iso4`,
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
| Delivered bits / element | **21.75** at head\_dim=128 (348 B/token/kv\_head vs bf16's 256 B, **1.36× bf16**), split **10.75 codes + 10.75 scales + 0.25 norm**. 43 groups cover a 128-element row, and each spends one whole `u32` code word **and** one `f32` scale for 3 real grade-1 elements. rotor3 and rotor4 therefore occupy byte-identical storage. Read off the same allocation, `rotor stored bits/value = (64·⌈D/3⌉ + 32) / D` — 21.75 at D=128, 21.625 at D=256, with a floor of 64/3 = **21.33** as D grows. A derivation from `QuantKGpuRing::alloc`, not a measurement. The 3.25 bpe target reported by the Python `rotorquant` reference is gated on the grade-aware codebook follow-up (deferred — see below). **No rotor codec is a memory win at any head\_dim** — see the iso3 "Memory truth" note, `iso_and_rotor_k_codecs_are_never_a_memory_win`, and the crate-wide rate ceiling in `kv_rate_tests.rs`. |
| Dead code budget | **5 of the 8 codes per group carry no information.** A rotor sandwich is grade-preserving, so embedding 3 values as the grade-1 part leaves the scalar, three bivector and pseudoscalar slots algebraically zero on encode; on decode the inverse sandwich keeps every non-grade-1 part out of the reconstructed vector, so quantising those slots injects no error either. 15 of the 24 code bits per group are pure waste. Pinned by `clifford_tests::sandwich_of_grade1_in_3d_stays_grade1` (encode side) and `clifford_tests::inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1` (decode side). Removing them is a format + MSL-kernel change and is not scheduled; the 10.75 bits/value of `f32` scale would still dominate afterwards. |
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
(opt-in `--rotor-qjl on`), the K-side append/decode falls back to the CPU
`rotor3_k_encode` / `rotor3_k_decode` path. With QJL off (**the default**), the
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
| Delivered bits / element | **21.75** at head\_dim=128 — byte-identical to rotor3 (8 codes × 4 bits = exactly 1 `u32` per group of 3 real grade-1 elements, plus the same per-group `f32` scale and per-token norm; rotor3 spends 24 of the same 32 code bits). Codes alone are ~10.7 bpe. Same grade-aware split deferral as rotor3, and the same "never a memory win" conclusion. |
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
offending variant by name. `auto` never returns any of the four variants on
any arch (they are opt-in only — no auto path).

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

**Status.** GPU-resident on the hot path. `QuantIsoK3` / `QuantIsoK4` each embed
a `QuantKGpuRing`; the K encode writes the packed ring on-GPU and
`iso_flash_decode` reads that ring directly — see § `iso_flash_decode` below.

**Decode-cost caveat (historical — fixed).** These stores used to have no
GPU-resident mirror on the live decode path: the CPU `dequant()` re-materialised
every accumulated block each step and re-uploaded the reconstructed K prefix via
`Array::from_bytes` — an O(kv_seq) per-step cost that grew monotonically with
context. Short-prompt anchors (warm-TTFT masks the cost after step 1) hid it;
long-prompt decode showed it. Measured on Bonsai-8B `k_iso3`, that cost was
~48.5 µs per KV token, taking decode to 0.96 TPS at 16k and 0.59 at 32k. The
flash-decode kernel removes it (16k: 0.96 → 10.6; 32k: 0.59 → 6.6). Kept here
because the shape of the bug — an O(seq) host restage hidden behind a warm bf16
seed — recurs across codecs.

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
Lindenstrauss sketch of the post-rotor MSE residual) under
`--rotor-qjl on`, which is **not** the default — QJL has no MSL kernel, so
enabling it moves the rotor K encode + dequant onto the CPU. The storage format
is controlled by a global toggle ([`rotor_qjl_enabled()`] in
`rmlx-kv-quant::rotor_qjl`), whose default is `false`.

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

**Decode-cost caveat — opt-in only.** With `--rotor-qjl on` the rotor K-side
codecs have no GPU-resident code mirror: each decode step re-decodes the full K
prefix on the CPU (and re-encodes the newly appended token), applies an
O(head_dim²)-per-cached-token QJL score correction, then re-uploads the K
prefix — an O(kv_seq) per-step cost that the short-prompt anchors mask but
long-prompt decode exposes. That is why QJL is **off** by default: the default
path runs the rotor K encode and `rotor_flash_decode` on Metal. The fused-QK
fast path (which would avoid the per-step marshaling) also needs
`--rotor-qjl off` and is separately default-OFF; see the Fused-QK status
below.

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

**QJL toggle.** CLI: `--rotor-qjl on|off` (default `off`). Env fallback:
`RMLX_ROTOR_QJL=1` enables. The toggle is read per-construction (not cached)
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

## The auto default

`--kv-quant auto` — the value when no codec flag is given — resolves to
**unquantised bf16** (`KvQuant::None`), for every architecture, every
checkpoint and every prompt length. One constant,
`rmlx_models::kv_cache::DEFAULT_KV_QUANT`, is the only producer; the CLI, the
server load path, the image branch, the arch dispatcher and all six
speculative drafter stacks read it and nothing else. There is no per-arch table
and no per-context re-selection behind it.

Two things this replaced, both removed rather than retuned:

* a **per-arch table** that returned `K8V8`, `K8V4`, `Planar` or
  `Mixed{k8g64,v4g64}` depending on arch class, `hidden_size`, the MoE flag,
  the PARO flag and `quantization.bits`; and
* a **per-prompt-length server policy** that re-picked a codec per request,
  overriding whatever the table had chosen at load. Three of its four bands
  quantised, including the longest one — where a wrong codec costs most.

  Its reach was narrower than the description suggests, and the qualifier
  belongs with the claim: on `rmlx serve --model` it never fired, because the
  CLI resolves `auto` before `run_serve` and passes a concrete codec down, which
  the server treats as operator-supplied. It was live under `--registry`, where
  nothing pre-resolves. Measured on the pre-change binary in registry mode, one
  gemma-4-e2b served `K8V4` / `K8V4` / `None` / `K8V8` across four requests at
  110 / 3 010 / 9 010 / 30 010 prompt tokens.

### Why bf16

Neither half of the codec axis pays, and both halves were re-measured on the
current tree rather than inherited:

* **The bf16-mirror family costs bytes it does not save.** `K8V8`, `K8V4`,
  `Planar*`, `PlanarK`, the `K8VTurbo*` / `TurboSym*` families and the
  `Iso3/4` / `Rotor3/4` / `RotorK*Asym` asymmetric families all decode off the
  bf16 mirror and never read their packed store, so no store is built
  (§"Per-layer net-benefit decision" above). Their resident KV equals bf16's
  **byte for byte**, and so does their output at temp=0.
* **The one store-reading codec a default ever picked loses on memory, and
  does not win on speed.** `Mixed` really does read its packed 3-tuples at
  decode — and holds them *beside* a bf16 mirror, so it is bf16 plus a store.
  Measured against `none` on Ternary-Bonsai-8B it is 1.29x / 1.29x / 1.29x the
  resident KV at 4k / 8k / 32k and lossy, while `none` is faster at 4k and 8k
  (SEPARATED) and indistinguishable at 32k. Note the *decode* gap narrows with
  context in that series rather than widening — the growing-with-context loss
  recorded for this trade is a different cell, Qwen3.8-27B at 130 848 tokens
  (§"Fused flash-decode over a quant store"), and is not what the Bonsai rows
  below show.

So on every cell measured here — the bf16-mirror family and `Mixed`, three
architectures, 4k to 32k — nothing is smaller than bf16 and nothing is faster.
A default that picked one of those was charging for a label.

That is a claim about the codecs a default could plausibly have picked, **not**
about the whole codec axis. The fully symmetric families (`Iso3Sym`, `Iso4Sym`,
`Rotor3Sym`, `Rotor4Sym`) return `false` from both `feeds_bf16_k_at_decode` and
`feeds_bf16_v_at_decode`, so they keep no mirror on either axis and their
resident KV is structurally *below* bf16. None of them was ever an auto default,
none is measured in the table below, and each carries its own arch guards and a
CPU-bound V path — so they are out of scope for this decision, not evidence
against it. Making one of them the default is a different question with a
different burden of proof.

### Measured

Every row below is one `scripts/perf_ab.sh` run: ABBA-interleaved, 8 slots
(4 per arm), both arms the same binary and differing only in `--kv-quant`,
`release-perf`, M5 Max. Host quiescent for every row -- none carries the
harness's TAINTED verdict, and the busiest foreign process during each
comparison was 7.2-7.4% of one core. `INCONCLUSIVE` means the two arms' per-slot
ranges overlap: the percentage beside it is the gap between two point estimates
and is **not** a measured difference. Only `SEPARATED` rows license a direction.

Arm A is the codec `auto` used to pick for that model; arm B is `none`.

The `bin` column is the sha256 prefix `perf_ab.sh` printed for that run's
binary, because "one binary" is a claim and not a given: a build finishing in
the background has already replaced `target/release-perf/rmlx` mid-campaign
here. `f2f889b9` is the pre-change tree, `cae129bc` the branch; the arms within
any single row share one binary, which is what the comparison needs. `run`
distinguishes the two gemma-4-e2b 4 096 rows, which are the same cell measured
in two separate sessions and are a deliberate repeat, not a transcription slip.

| model | ctx | arm A | bin | run | resident KV B/A | token ids | decode B/A | verdict |
|---|---:|---|---|---|---:|---|---:|---|
| gemma-4-e2b | 4 096 | `k8v8` | `f2f889b9` | 1 | **1.0000** | identical | 0.9985 | INCONCLUSIVE |
| gemma-4-e2b | 4 096 | `k8v8` | `cae129bc` | 2 | **1.0000** | identical | 1.0018 | INCONCLUSIVE |
| gemma-4-e2b | 8 192 | `k8v8` | `f2f889b9` | 1 | **1.0000** | identical | 1.0013 | INCONCLUSIVE |
| gemma-4-e2b | 32 768 | `k8v8` | `f2f889b9` | 1 | **1.0000** | identical | 0.9912 | INCONCLUSIVE |
| Ternary-Bonsai-8B | 4 096 | `k8v8` | `cae129bc` | 2 | **1.0000** | identical | 1.0045 | INCONCLUSIVE |
| Ternary-Bonsai-8B | 32 768 | `k8v8` | `cae129bc` | 2 | **1.0000** | identical | 1.0013 | INCONCLUSIVE |
| Qwen3.6-35B-A3B | 4 096 | `k8v8` | `cae129bc` | 2 | **1.0000** | identical | 1.0541 | INCONCLUSIVE |
| Qwen3.6-35B-A3B | 32 768 | `k8v8` | `cae129bc` | 2 | **1.0000** | identical | 1.0019 | INCONCLUSIVE |
| Ternary-Bonsai-8B | 4 096 | `mixed_k8g64_v4g64` | `cae129bc` | 2 | **0.7771** | diverge at id 57 | 1.0300 | SEPARATED |
| Ternary-Bonsai-8B | 8 192 | `mixed_k8g64_v4g64` | `cae129bc` | 2 | **0.7751** | diverge at id 56 | 1.0258 | SEPARATED |
| Ternary-Bonsai-8B | 32 768 | `mixed_k8g64_v4g64` | `cae129bc` | 2 | **0.7736** | diverge at id 35 | 0.9994 | INCONCLUSIVE |

The two binaries are not a confound: a codec-vs-codec comparison never crosses
a row, and the repeated gemma-4-e2b 4 096 cell -- the one point measured on
both -- returns the same verdict and the same 1.0000 residency on each.

Two things to read off it. The eight `k8v8` rows are a *null* result by
construction -- same bytes, same bits, no measurable time -- across three
architectures and two KV shapes (`kv_h = 1` shared-KV/SWA at `head_dim` 256/512,
and `kv_h = 8` dense at 128). The three `mixed` rows are the only place the
default's behaviour actually changes, and every axis moves toward `none`.

The same comparison run once per branch of the retired per-arch table that the
release set reaches (4 096-token prompt, 32 tokens, temp=0, resident KV and token-id digest
only) closes the branch table: `gemma-4-e2b` (`K8V8`), `gemma-4-12B` unified
(`K8V8`), `gemma-4-26b-a4b` MoE (`K8V8`), `gemma-4-31b` (`Planar`), `medgemma`
(`Gemma3`, `Planar`), `Qwen3.6-35B-A3B` (`K8V8`), `Qwen3.8-27B` (`K8V8`),
`Qwen3.6-27B-PARO` (`K8V4`) -- all eight byte-identical in `kv_cache_bytes` and
identical in token ids against `none`. `Ternary-Bonsai-8B` (`Mixed`) is the
ninth and the only one that moves.

### What stays

Every codec remains selectable by name — `--kv-quant`, `--cache-type-k` /
`--cache-type-v`, `--kv-bits`, `--kv-preset`. Nothing is deprecated or removed
by this; only what `auto` resolves to is fixed. `--kv-preset auto` resolves to
the same constant (§"`--kv-preset auto`"); the hardware-aware selector it used
to run has since been removed, because every preset it could return holds
resident KV byte-identical to bf16.

`DEFAULT_KV_QUANT` is where a future answer changes. When fused decode over a
quantised store becomes profitable, that constant moves — on a fresh
measurement, not by restoring a table.

---

## Memory and bit-rate summary

Approximate bytes per KV pair (`B=1, 1 layer, 1 head, D elements`). These are
**packed-store** rates — what a codec's codes and scales occupy. They are not
resident KV for the bf16-mirror family, which builds no store: those codecs sit
at the `None` row, 4·D, measured byte-identical (§"Per-layer net-benefit
decision"). Which rows are live and which are hypothetical is stated under the
table.

| Mode | K bytes/tok | V bytes/tok | Total bytes/tok |
|---|---|---|---|
| `None` (bf16) | 2·D | 2·D | 4·D |
| `K8V8` | 1·D + D/128·4 | 1·D + D/128·4 | ~2.06·D |
| `K8V4` | 1·D + D/128·4 | 0.5·D + D/32·4 | ~1.65·D |
| `Planar` | 1·D + D/128·4 | 2.75·D (measured) | ~3.78·D |
| `Mixed{k8g64,v4g64}` | 1·D + 2·D/64·4 | 0.5·D + 2·D/64·4 | ~1.75·D |
| `K8VTurbo3` | 1·D + D/128·4 | 0.375·D + D/32·4 | ~1.51·D |
| `K8VTurbo2` | 1·D + D/128·4 | 0.25·D + D/32·4 | ~1.38·D |
| `k_iso3` / `k_iso4` | 2·D + 4 | 2·D (bf16) | ~4·D + 4 |
| `iso3_sym` / `iso4_sym` | 2·D + 4 | 2·D + 4 | ~4·D + 8 |
| `k_rotor3` / `k_rotor4` | 8·⌈D/3⌉ + 4 | 2·D (bf16) | ~4.67·D + 4 |
| `rotor3_sym` / `rotor4_sym` | 8·⌈D/3⌉ + 4 | 8·⌈D/3⌉ + 4 | ~5.33·D + 8 |

PlanarQuant's V row is **measured**, not a layout formula, and it is the one
row an earlier revision of this table got wrong (`~2.13·D`, from a per-group
sideband cadence the codec does not use). Its scale is per **pair** — one `f32`
per 2 elements — which is 16 bits per value before a single code bit, so the
store is **22.00 bits per value at every head_dim and at both bit widths**
(`planar3` and `planar4` are byte-identical). That is the widest rate in the
crate's rate gate, above rotor's 21.75, and it is why `planar3` / `planar4` /
`planar_k4` carry a written exemption in
`crates/rmlx-kv-quant/src/kv_rate_tests.rs` rather than a fix: a scale-cadence
change is a format change. The quality improvement is what it buys on dense
full-attention archs.

Two groups of rows here have a decode that reads the store they describe, so
their rates are live rather than hypothetical: the `Mixed{k8g64,v4g64}` row and
the four ring rows (§"Codec disposition", Class 3). Every other row is a rate
its codec would cost the day a kernel reads its store. Of all of them, only the
four ring rows are **above** the `None` row *on rate*; `Planar` is the closest
of the rest at 3.78·D against 4·D, i.e. a 5% saving for a 4-bit name. `Mixed` is
below the `None` row here and still measures larger resident, because it keeps
both bf16 seeds beside its store — a residency fact this table does not carry.

Each ring side spends one `u32` code word and one `f32` scale per group
whatever the codebook width, so the nominal
3-/4-bit label never reaches the store: iso is `16 + 32/head_dim` bits per
value and rotor floors at `64/3` = 21.33, both strictly above bf16's 16.0 at
every finite head dim (§ iso3 "Memory truth", § rotor3). The rate is a property
of the ring layout, not of the algorithms; a layout that packs its scale plane
separately is unbuilt and is the open question § "What this disposition does not
decide" names.

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

## Fused flash-decode over a quant store — the break-even condition

The four sections that follow — plus TurboFlash, whose posture and measured
cells live on `rmlx_cli::commands::serve::TurboFlashMode` — each describe a
hand-written MSL kernel that reads a packed KV store directly at decode instead
of a bf16 mirror. They all exist for one reason: a smaller store means fewer
bytes moved per decode step. This section states the condition under which that
trade actually pays, and records what it measures on this tree. Read it before
concluding that a new fused kernel — or a denser store — will make some codec
win.

### The condition

Two quantities, both measurable with `rmlx bench`:

* **ρ** — bytes the fused kernel reads, over bytes MLX `sdpa_vector` reads from
  the bf16 mirror for the same cell. Measured as the codec's resident KV in
  excess of `none`'s.
* **ε** — the fraction of MLX `sdpa_vector`'s per-byte throughput the
  hand-written kernel achieves: `ε = ρ / (marginal-slope ratio vs none)`, where
  the slope is `b` in `ms/step = a + b·(KV tokens/1000)`.

A fused decode beats bf16 iff **ρ < ε**. Note that ε is a property of the
*kernel shell*, not of the codec, and ρ is a property of the *store*, not of the
kernel — so a codec and a kernel can each be reasonable and the pair still lose.

### Measured (`d8a6169`, release-perf, M5 Max, n=3, contended host)

| kernel | arch | `heads_per_kv` | ρ | slope ratio | ε |
|---|---|---|---|---|---|
| TurboFlash (`k8v4`) | Bonsai-8B | 4 | 0.262 | 6.37× | **0.041** |
| `iso_flash_decode_symv` (`iso3_sym`) | Bonsai-8B | 4 | 1.013 | 7.51× | **0.135** |
| `iso_flash_decode_symv` (`iso3_sym`) | gemma-4-e2b | 8 | ~1.01 | 19.2× | **0.052** |

The TurboFlash row is the sharpest statement available: at 32k on Bonsai-8B the
fused path reads **3.8× fewer bytes** and costs **6.4× more time per KV token**.
Decode through these kernels is **not bandwidth-bound**, so a smaller store does
not buy time.

Dispatch was witnessed, not inferred. `--turbo-flash on` at a 4k prompt (below
the kernel's own `kv_seq > 4096` gate) is bit-identical to `off` in resident
bytes and within noise on TPS; dropping the gate at the *same* prompt moves
resident KV by +180 MB and decode 129.8 → 65.9 TPS. The 16k/32k byte deltas
(+722 MB / +1445 MB, exactly linear in `kv_seq`) are the kernel engaging.

### Cross-backend calibration — ε is a platform number, not an rMLX number (TAINTED host)

The three rows above are all rMLX kernels, so on their own they cannot separate
"fused decode over a quant store is hard on this GPU" from "our kernels are
bad". A second, independent implementation now answers that.

`llama-cpp-turboquant` implements the same idea in a different framework with
hand-written Metal: TurboQuant K/V blocks read directly inside
`flash_attn_ext_vec_kturbo*_vturbo*_dk{128,256}_dv{128,256}`, with the WHT
applied to `q` once per query as a graph op. It was measured against **its own
upstream merge-base** (`7fc1c4ef7`) at f16 KV — same GGUF file, same graph, same
build flags, so within one cell the only variable is the codec and its kernels.
Both models are `n_q_heads/n_kv_heads = 32/8` → `heads_per_kv = 4`, the same
ratio as Bonsai-8B.

**Measurement conditions.** Every cell ran `--allow-busy-host` and came back
**TAINTED** — entry gate 25.6–55.0% of one core, measured windows 29–56%
(`WindowServer`; `syspolicyd` steady at 32.2–32.5%), and two slots of the 7 722
run with a `node` process at 139.9%/159.8%. ABBA cancels a *steady* load and the
arm ranges below are disjoint by far more than the interference, so the
**separation verdicts survive**; the **absolute TPS does not** and is not quoted
across runs. Per-slot windows are in each result JSON.

| model | prompt tok | n/arm | fork turbo3 / upstream f16 decode | ms/step ratio | KV MiB f16 → turbo3 | peak RSS B/A |
|---|---:|---:|---:|---:|---|---:|
| Qwen3-8B-Q8_0 | 3 753 | 4 | 0.733x | 1.364 | 648.00 → 126.69 | 0.945x |
| Qwen3-8B-Q8_0 | 7 722 | 4 | 0.705x | 1.419 | 2 304.00 → 450.13 | 0.830x |
| Qwen3-8B-Q8_0 | 31 536 | 4 | 0.690x | 1.449 | 5 760.00 → 1 125.13 | 0.677x |
| Llama-3.1-8B-Instruct-Q8_0 | 61 709 | **2** | 0.653x | 1.532 | 8 192.00 → 1 600.13 | 0.607x |

Arm ranges are disjoint in all four cells. **The 61 709 row is n=2/arm**
(`--pairs 2`), not 4 — it is simultaneously the most extreme ratio, the only
second-architecture point, and the least-replicated cell. Nothing below is
derived from it alone. Its stored result JSON records the verdict as
`SEPARATED`; the harness now labels a disjoint pair at n=2 `SEPARATED-WEAK`, and
refuses `--pairs 1` outright, because two single-point "ranges" cannot overlap.

`ρ` is exact here and needs no inference: llama.cpp keeps no bf16 mirror, so the
reported KV buffer *is* what the kernel reads. Across all four cells and both
models **ρ = 0.1953 ± 0.0002** (0.195509 / 0.195369 / 0.195335 / 0.195328) — a
1.8e-4 spread, which is the codec's block layout reproducing itself exactly.

**ε is derived from the Qwen3-8B series alone.** The `ms/step = a + b·x`
monotonicity argument is only valid within one `(a, b)` pair, i.e. within one
model; extending it across the model boundary to the Llama-3.1 row would not be
a bound at all. Two derivations, both same-model:

* **Drift-free upper bound.** The within-run ms/step ratios rise 1.364 → 1.419 →
  1.449 and are still climbing, so the marginal slope ratio is at least 1.449 and
  **ε ≤ 0.135**. This uses only within-run ratios, so host drift cancels.
* **Two-endpoint slope regression.** Fitting `b` per arm from the 3 753 and
  31 536 cells gives `b = 0.330`, `b' = 0.536`, slope ratio **1.63** and
  **ε ≈ 0.120**. This one reads absolute ms across two runs, so it carries the
  drift caveat.

So ε ∈ **[0.120, 0.135]** for the fork's kernel, and the upper bound is *exactly*
rMLX's own best measured shell:

| kernel | `heads_per_kv` | geometric ceiling | ε | % of ceiling |
|---|---|---|---|---|
| rMLX `iso_flash_decode_symv`, Bonsai-8B | 4 | 0.250 | 0.135 | 54% |
| `llama-cpp-turboquant` vec-turbo FA, Qwen3-8B | 4 | 0.250 | 0.120–0.135 | 48–54% |

**Two independently written Metal kernels, two frameworks, land on the same ε at
the same `heads_per_kv`.** And for the same structural reason: the fork's vec
dispatch is `dispatch_threadgroups(…, (ne01 + nqptg - 1)/nqptg, ne02, ne03, …)`
with `ne02` the **query**-head count (`ggml-metal-ops.cpp`), so its
`heads_per_kv` threadgroups re-read the identical KV bytes exactly as ours do.
The ceiling in the next section is not an rMLX artifact.

Consequences, and they are the load-bearing part of this whole section:

* **ρ = 0.1953 > ε ≤ 0.135, so the fork loses too** — by 1.45x on ms/step at
  31 536, which is the 0.690x decode ratio measured. rMLX's `turbo_flash` being
  slower than its generic path is a *degree* of the same result, not a different
  kind of result. Its ε = 0.041 says our TurboFlash *shell* is ~3x off what is
  achievable; even a perfect fix of that shell reaches ε ≈ 0.135 and still loses
  at ρ = 0.1953.
* **The trend runs the wrong way**, and this is established on the three
  **same-model** Qwen3-8B points alone: 0.733 → 0.705 → 0.690 as context grows
  8.4x. A codec whose store is 5x smaller should gain as KV's share of
  bytes/step rises; it does not, because the per-step dequant work scales with
  the same `t_seq`. "Measure it at longer context and it will pay" is falsified
  on the reference implementation, not only on ours. The Llama-3.1 point at
  0.653x is *consistent* with the trend but cannot extend it — it changes model
  and context simultaneously, and there is no short-context Llama-3.1 cell to
  separate "ratio falls with context" from "Llama-3.1 is worse for this codec".
* **What the fork does deliver is the memory axis**, and it is monotone with
  context rather than flat: peak process memory **0.945x / 0.830x / 0.677x /
  0.607x** at 3 753 / 7 722 / 31 536 / 61 709, against a flat 0.1953x KV buffer.
  The campaign declared a `≤0.85x` peak-memory criterion in advance; **at 3 753
  tokens the fork does not clear it** (0.945x), because the KV saving is small
  beside 8.3 GB of weights until context grows. The trade is real from ~8k up and
  absent below it.
* **Coherence was observed only where it was captured, and no quality axis was
  measured.** Output capture was added to the harness after the 7 722 and 31 536
  runs, so every slot of those two carries `output_first_64: null`. Where
  captured (3 753, 61 709, and the confound control) both arms produce fluent,
  on-topic English. At 61 709 the two arms visibly disagree on extracted facts,
  which is expected of a lossy KV codec and is *not* evidence either way about
  quality — no perplexity or task score was run.
* Its own deepest fusion (`TurboFlash`, a two-pass fused kernel) is **disabled by
  default because it emits garbage on Apple10**, reproduced here on this host.
  Two independent projects disabling their deepest Metal fusion on this GPU
  family is a platform signal.

**Scope of the confound control.** "The only variable is the codec" is licensed
by a `fork @ f16` vs `merge-base @ f16` cell that returned INCONCLUSIVE with
fully overlapping ranges (1.013x, n=4/arm) — but that control ran at **one
context (7 722) on one model (Qwen3-8B)**. The 61 709 Llama-3.1 cell has no
control of its own, so a long-context-specific or Llama-specific regression among
the fork's 211 non-upstream commits is unmeasured there.

Method, raw slots and the fork-side detail: `scripts/bench_llama_ab.sh`,
`scripts/ingest/llama_ab_ingest.py`, results under `$RMLX_HOME/bench/llama_ab/`.

### Why ε is small — grid geometry

Every P1 kernel here indexes its grid by **query** head (`n_bh = b · n_q_heads`)
and addresses KV with `kv_h_idx = hq / heads_per_kv`. So `heads_per_kv`
threadgroups each stream the identical KV bytes. That caps the shell at
**ε ≤ 1/heads_per_kv** before any kernel-body cost:

| arch | `heads_per_kv` | geometric ceiling | measured ε |
|---|---|---|---|
| Bonsai-8B | 4 | 0.250 | 0.135 (54% of ceiling) |
| gemma-4-e2b | 8 | 0.125 | 0.052 (42% of ceiling) |

Measured ε differs between the two archs by 2.6×; `heads_per_kv` alone predicts
2.0×. The geometry is the dominant term. The residual ≈2× is the f32
`partial_o` P1→P2 DRAM round trip plus the thread-0-serial online-softmax
section, where all but one lane idle twice per KV token.

**Corollary for `kv_h == 1` architectures.** e2b's geometric ceiling (0.125) is
*below* the densest store in the tree (`tsym3`, ρ = 0.158). On such an arch no
fused decode over any store that exists or has been proposed can beat bf16 **even
with a perfect kernel body**, until the grid stops re-reading KV per query head.

### What ρ would have to be

| kernel efficiency | break-even store (bf16 = 32 bits per K+V pair) |
|---|---|
| ε = 0.135 (best measured) | 4.3 bits/pair = **2.2 bits per value per axis** |
| ε = 0.052 (e2b) | 1.7 bits/pair = **0.83 bits per value per axis** |
| ε = 0.041 (TurboFlash) | 1.3 bits/pair = **0.66 bits per value per axis** |

Against what exists or has been specced:

| store | bits/value | ρ | clears ε = 0.135? |
|---|---|---|---|
| `tsym3` — densest store in the tree | 2.5 | 0.158 | no, 1.2× over |
| the repacked iso/rotor store specced in "Memory truth" | 3.75 | 0.234 | no, 1.7× over |
| rotor with its structurally-zero components dropped | 14.0 | 0.876 | no, 6.5× over |
| iso / rotor as stored today | 16.25 / 21.75 | 1.02 / 1.36 | no |

**The binding constraint is ε, not ρ.** Repacking a store is necessary for some
of these codecs to stop costing memory, but it is not sufficient to make a fused
decode win, and on `kv_h == 1` it is not even close. Order kernel-shell work
first; judge a store repack on its memory merits, not on an expected decode win.

### Codec disposition — what every codec in the tree is for

The KV enum spells **28 codecs**. This section gives each one an explicit
disposition and the evidence behind it. "Nobody selects it" is not a
disposition; the classes below are.

The classification is not prose. `KvQuant::materialises_packed_store` is the
predicate the runtime dispatches on, `quant_tests.rs::DISPOSITIONS` names all
28 by hand, and `every_codec_carries_a_disposition` fails when the two
disagree. A variant added to the enum cannot reach `ALL_KV_QUANTS` without
being classified.

#### The measurement

`scripts/bench/codec_inertness_probe.sh`, one `rmlx baseline` per codec at
temperature 0, `--max-tokens 100`: 27 codec spellings × 2 architectures × 2
contexts, 108 runs, all exit 0. Two architectures on purpose — gemma-4-e2b is
`kv_h == 1` with shared-KV and sliding-window layers, Ternary-Bonsai-8B is
`kv_h == 8` dense — because a KV result at one shape is not a result at the
other.

27 spellings, not 28 variants: the four parameterised families are driven at
one representative parameter set each, and `rotor_k_3_asym_v*_g*` is left to
`disposition_is_a_property_of_the_family_not_its_parameters`, which pins that
the classification cannot move with the parameters. Every unit-variant codec
was served.

Both reported quantities are load-independent: `kv_cache_bytes` is a byte
count and the digest is over token **ids**, not text. No throughput claim is
made here; the probe's `decode_tps` column is single unpaired runs on a shared
host and is not comparable row to row (see `docs/PERF_BASELINE.md` for the
conditions on this machine).

| | gemma-4-e2b 4k | gemma-4-e2b 32k | Bonsai-8B 4k | Bonsai-8B 32k |
|---|---:|---:|---:|---:|
| `none` resident KV (B) | 32 194 560 | 217 976 832 | 570 507 264 | 4 667 277 312 |
| codec spellings byte-identical **and** id-identical to `none` (incl. `none`) | 17 | 17 | 17 | 17 |
| codec spellings larger than `none` | 10 | 10 | 10 | 10 |
| codec spellings **smaller** than `none` | **0** | **0** | **0** | **0** |

**No codec in the tree reduces resident KV, on either architecture, at either
context.** That is the finding the dispositions follow from.

#### Class 1 — the baseline (1 codec)

`none`. bf16 both sides, the resolved `auto` default, the smallest resident KV
measured, and the reference every other row is compared against.

**Disposition: keep.** Nothing else is in this class, and
`exactly_one_codec_is_the_baseline` keeps it that way.

#### Class 2 — inert, mirror-fed (17 codecs)

`k8v4`, `k8v8`, `planar`, `planar3`, `planar_k`, `k8vturbo3`, `k8vturbo3tcq`,
`k8vturbo2`, `k8vturbo2tcq`, `tsym3`, `tsym4`, `iso3`, `iso4`, `rotor3`,
`rotor4`, `rotor_k_3_asym_v*_g*`, `rotor_k_4_asym_v*_g*`.

Decode reads the bf16 mirror on both axes, so `exit_prefill` skips the packed
store and prefill never encodes one either — the codec math does not execute at
all on a prefill-bracketed flow. That is a property of
`KvQuant::decode_reads_packed_store`, checked by
`exit_prefill_builds_a_store_exactly_when_the_predicate_says_so`. In all four
cells every one of these reports the identical `kv_cache_bytes` and the
identical 100-token id digest as `none`.

The probe's `store_skipped` column corroborates and does **not** classify: it
is set when *any* layer-cache in the run logged the skip, and the layer-adaptive
head/tail promotion types some layers `K8V8`, so a store-*reading* codec sets it
too — `mixed_k8g64_v4g64` does, on Ternary-Bonsai-8B, where 10 of 36 layers are
promoted. Bytes and the id digest are the deciding columns.

**These are equivalent to `none`, and unselected — not dominated.** The first
draft of this section said "dominated, strictly better on one axis (it does not
carry a quantised layer type through dispatch)". That axis does not survive
measurement: the ~0.041 ms/layer/step figure it rests on was re-measured after
the packed-store elision and is **INCONCLUSIVE at all five ABBA cells** — see
"`--kv-quant none` is a bf16 control", which states in the same words that with
the store gone "a `K8V8` layer is a `None` layer under another name". Nothing in
this class costs anything a measurement here can see, and nothing in it buys
anything either.

So the honest reading, axis by axis: resident KV identical (4 cells, exact
bytes); output token ids identical (4 cells); decode throughput INCONCLUSIVE
(5 ABBA cells); TTFT INCONCLUSIVE (the mirror family's own spread at Bonsai-8B
32 768 is 18 320–20 582 ms, wider than any gap inside it). An operator who
passes `--kv-quant iso3` gets bf16 KV under another name — no better, no worse,
just not what the name says — which is why `validate_resolved` says so at
`warn!` and why no `--kv-preset` row is described as a memory setting any more.

**What this does to the dominated-vs-unused split.** The word belongs to
Class 3, not here. On the axes anything is measured on, `none` is strictly
smaller than every Class 3 codec (4 cells, 1.003×–1.541×) and not slower, so
Class 3 *is* dominated by the baseline; Class 2 merely ties it. The two classes
are still kept for different reasons — see each disposition — but the reason is
not that one is beaten and the other is not.

**Disposition: keep parseable and selectable; stop advertising.** Not deleted,
for three reasons, in descending order of force:

1. **The store is the re-enable path.** `exit_prefill` keeps a bulk-encode arm
   for each of them behind the predicate. A codec that grows a decode kernel
   over its own store flips one arm in `decode_reads_packed_store` and the arm
   fills the buffer that kernel reads. Deleting the codec deletes the landing
   site, and the algorithm is not what failed — see "the tension", below.
2. **Recorded rows must stay readable.** `observations` is append-only and
   metrics labels are free-form; a name that has been recorded has to keep
   parsing after it stops being recommended.
3. **The widest-matrix goal.** CLAUDE.md names the rotation KV families as a
   differentiator. Removing them would narrow the matrix without making
   anything true.

Two pairs inside this class are *exact duplicates* of each other rather than
merely equivalent to `none`: the decoder of `k8vturbo3tcq` is bit-for-bit the
plain `k8vturbo3` decoder and only the encode-time assignment differs, so while
the encode never runs the two names are one behaviour; likewise `k8vturbo2tcq`
and `k8vturbo2`. They are fold candidates the day a turbo decode kernel lands,
and indistinguishable today.

#### Class 3 — reads its own packed store (10 codecs)

`mixed_k<kb>g<kg>_v<vb>g<vg>`, `rot_k_v<vb>g<vg>`, `iso3_sym`, `iso4_sym`,
`k_iso3`, `k_iso4`, `rotor3_sym`, `rotor4_sym`, `k_rotor3`, `k_rotor4`.

These are the only codecs whose quantization a served request touches. Every
one of them is **larger** than bf16, in all four cells:

| codec | e2b 4k | e2b 32k | Bonsai 4k | Bonsai 32k |
|---|---:|---:|---:|---:|
| `k_iso3` / `k_iso4` | 1.015× | 1.003× | 1.027× | 1.007× |
| `iso3_sym` / `iso4_sym` | 1.029× | 1.007× | 1.054× | 1.013× |
| `k_rotor3` / `k_rotor4` | 1.154× | 1.167× | 1.159× | 1.131× |
| `rotor3_sym` / `rotor4_sym` | 1.309× | 1.334× | 1.317× | 1.262× |
| `mixed_k8g64_v4g64` | 1.339× | 1.396× | 1.287× | 1.293× |
| `rot_k_v8g64` | 1.541× | 1.533× | 1.384× | 1.384× |

The iso / rotor rows are the ring-layout result restated on real serving: a
per-group `f32` scale beside each packed word puts 16.25–21.75 bits on the
store per value against bf16's 16.0, so the family cannot win on bytes at any
finite head dim. The `mixed` / `rot_k` rows still carry both bf16 seeds beside
their store, which is a separate, unlanded elision; their decode result does
not depend on it (returning an unread seed frees memory, it does not speed a
quantized-matmul decode) and is measured at 0.763× of `none` at 130 848 tokens
with the arms' ranges disjoint.

**Disposition: keep, unchanged, and do not treat as memory levers.** These
*are* dominated by `none` on the measured axes — strictly larger resident KV in
all four cells, and for `mixed` also slower (0.763× at 130 848 tokens, ranges
disjoint). They are kept anyway, and the reason is not that they are competitive:
they are the only codecs in the tree that decode over a packed store at all, so
they are the substrate any fused-decode work has to stand on, and a dominated
codec with a function is not the same object as a beaten one with none.

Being dominated today is a statement about the ring layout and about ε (the
byte-to-time conversion efficiency, ≈0.04–0.135 on every path measured, and
shared with a non-MLX runtime on the same hardware), not about the codecs.

Four **within-family** dominations are measured here, and they are exact:
`iso3_sym` and `iso4_sym` are byte-identical in all four cells, as are
`k_iso3`/`k_iso4`, `rotor3_sym`/`rotor4_sym` and `k_rotor3`/`k_rotor4`. The
3-bit member of each pair therefore buys nothing over the 4-bit member and
carries measurably worse distortion. They are the tree's only strictly
dominated-by-a-sibling codecs and the first fold candidates, tracked
separately; the pairing is recorded here because the probe shows it at four
cells rather than at a fixture.

#### The tension with the widest-matrix goal

CLAUDE.md sets out to ship "the widest weight × KV quantization matrix MLX can
express, including rotation-based KV families no other MLX server ships". A
disposition that retired codecs would be in direct tension with that, and the
tension is not resolved by preferring one side.

What the evidence indicts is **the ring layout and the byte-to-time
conversion**, not the algorithms. The scale-beside-the-word layout is what puts
iso/rotor above 16 bits per value; a layout that packs scales separately is
unbuilt and open. ε is a platform property — an independent runtime's fused
decode over a quantised KV store loses ~30% on the same hardware, and its
efficiency is statistically the same as this tree's best kernel — so "our codec
implementation is bad" is not what the numbers say. Deleting a codec would
remove a differentiator on evidence that convicts something else.

So the matrix stays whole, and what changes is the **claim**: rMLX ships 28 KV
codecs, 17 of which are presently inert and 10 of which are presently larger
than bf16, and it says so in `--help`, in a resolve-time `warn!`, and here.
That is the honest form of the same capability. A codec matrix nobody can be
misled by is worth more than one name fewer.

#### What this disposition does not decide

* **Whether any codec should eventually be deleted.** That turns on whether the
  re-enable path can be made to pay. It is decided by a fused decode kernel over
  a packed store that beats bf16 at a context this tree can serve — not by
  another residency measurement, which is now saturated at "no codec is
  smaller".
* **Whether a better ring layout clears 16 bits/value for iso/rotor.** Nothing
  measured here touches it; the layout, not the algorithm, is what fails.
* **Which member of a byte-identical 3-bit/4-bit pair survives a fold.** The
  bytes are settled (identical); the distortion comparison that would pick a
  survivor is a fidelity question, not a residency one.

### `kv_frac` bounds a codec claim — and is not a statement about context

`kv_frac` is the KV share of a decode step's byte stream,
`kv_bytes_step / (weight_bytes_step + kv_bytes_step)`. It is the ceiling on how
much of decode any KV-codec change can possibly touch, and
`scripts/perf_ceiling.py` prints it in the last column of every row.

**It is a property of the (model, context) pair, not of the context.** At a
4096-token prompt it spans 22× across the release set. This table is a **static
prediction** — `perf_ceiling.py` over `config.json` plus the safetensors index,
no model launched — unlike the measured cells below it:

| model | 4 096 | 8 192 | 32 768 | 131 072 |
|---|---:|---:|---:|---:|
| Qwen3.8-27B-mxfp8 (26.4 GB/step of weights) | 0.010 | 0.020 | 0.075 | 0.245 |
| Qwen3.6-35B-A3B-8bit (MoE, 3.1 GB/step) | 0.026 | 0.051 | 0.177 | 0.462 |
| gemma-4-e2b-mxfp8 (SWA on most layers) | 0.030 | 0.053 | 0.170 | 0.445 |
| Ternary-Bonsai-8B-2bit (2.1 GB/step) | **0.221** | **0.362** | **0.694** | 0.901 |

So "measured at 4k" and "measured where the codec axis is near-zero" are not
the same qualifier, and a claim scoped by the first is not scoped by the
second. A 2-bit 8B model at a 4k prompt already puts 22% of its decode bytes on
the codec axis; a dense 27B at 131k puts 25%.

**A large `kv_frac` is necessary, not sufficient.** Measured on this tree,
`none` vs `mixed_k8g64_v4g64` ABBA-paired (`scripts/perf_ab.sh`, n=4/arm; the
two Qwen3.8 cells untainted at 7.0–7.3 % foreign CPU, the two Bonsai cells
tainted only by a monitoring process at their entry gate — see
docs/PERF_BASELINE.md for per-cell conditions and spreads):

| model | prompt tok | `kv_frac` | predicted ceiling B/A | measured decode B/A | measured resident B/A |
|---|---:|---:|---:|---|---:|
| Ternary-Bonsai-8B | 3 770 | 0.211 | 1.099 | 0.975 ranges disjoint | 1.287 |
| Ternary-Bonsai-8B | 31 553 | **0.687** | **1.419** | 1.002 INCONCLUSIVE | 1.293 |
| Qwen3.8-27B | 3 892 | 0.010 | 1.004 | 0.977 SEPARATED | 1.219 |
| Qwen3.8-27B | 130 848 | 0.245 | 1.149 | **0.763 SEPARATED** | 1.349 |

Two rows carry the argument. The Bonsai **31 553** row is the high-`kv_frac`
end: at 0.687 — the largest any release-set model reaches *at a context this
tree can serve*; the static table above projects 0.901 for the same model at
131 072, which no measured cell reaches — a codec that cuts the decode KV
stream to 0.571× and is predicted +42% moves decode by +0.3%, inside the noise,
while costing +29% resident KV — a null with the power to have seen a 1%
effect, since the arm spreads there are 0.04% and 0.15%. The Qwen3.8
**130 848** row is the long-context
end: `kv_frac` 0.245, predicted +14.9%, measured **−23.7% with the arms'
per-slot ranges disjoint in the losing direction**, and +35% resident
whole-cache — which on that hybrid arch understates the attention-KV ratio,
1.355 excluding its codec-independent GDN state.

**Scope on arm B.** It is `mixed_k8g64_v4g64` as it stands here: still
materialising a packed store *and* retaining both bf16 seeds, because the change
that stopped building a store for a mirror-fed codec excluded the Mixed / RotK
family (whose decode does read its store). Its **resident** figures are
therefore pre-seed-elision and say nothing about a seed-elided variant. Its
**decode** figures need no such variant — returning an unread seed frees memory,
it does not speed a quantized-matmul decode — so the throughput result stands
for the family either way.

The byte model is not what is wrong. At those offsets `perf_ceiling.py` puts
arm A's resident KV at 4 667.3 MB against 4 667.3 MB measured on Bonsai, and on
Qwen3.8 it is exact once the codec-independent GDN recurrent state that arch
carries (a flat ~152–154 MB, identical on both arms at both contexts) is added
back. What fails is the conversion of bytes into time — the ε of the section
above, ≈0.04–0.135 on every path measured — and at the Qwen3.8 long cell the
packed path does not merely fail to convert: its non-bandwidth per-step cost
grows 12.0 → 44.2 ms between 3 892 and 130 848 tokens while `none`'s stays flat
at 10.5 → 14.7, so halving the byte stream still loses.

Read together with the ε table: **`kv_frac` bounds the prize, ε decides how much
of it is collectable, and ε is the binding term.** State `kv_frac` next to a
codec cell so a reader can see the bound; do not infer an effect from it.

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
  (`update_rotor_k_only_*`) and the non-fused decode fallback, which `dequant()`
  the whole prefix on the same step and so need the block immediately.
- **`MaintainRingOnly`** — feed the ring **without** pushing a CPU block: a
  **ring-only tail**. Used by the fused decode entry
  (`rotor{3,4}_k_only_gpu_append`). The flash kernel reads the ring, never the
  block, so skipping the per-step host download (`rotor_gpu_outputs_to_cpu`) is
  the win; `shape[2]` still advances so the ring and the attention length stay
  in lockstep.
- **`Skip`** — clear the ring. Used by the sym/asym mirrors, and as the `b > 1`
  fallback of a ring-only feed (which reverts to the block path).

A ring for a non-eligible codec is not free —
`capacity * kv_h * n_groups * 8 + capacity * kv_h * 4` bytes per layer, growing
with context (order of a few hundred MB across a 36-layer model at 4k) — and
nothing would ever read it.

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
GPU-encode append (`QuantIsoV3::append_gpu`, the V side of the legacy
`update_iso3` / `update_iso3_sym` entries) pushed a CPU block without touching a
live ring, leaving the ring stale *and* the blocks short — it now calls
`QuantIsoV3::reconcile_ring(device, RingDisposition::Drop)`, which takes the
ring's prefix back and then drops the ring, matching what the CPU `append` does
by clearing. That is one body, shared with `materialize_iso_v3_ring_tail`, which
passes `RingDisposition::Keep` because its caller's `sync_ring` decides the
ring's fate immediately after — the disposition is a parameter precisely because
it is the only thing the two callers disagree on. The iso4 V
side had the same hole in a separate ring-unaware helper; both of its callers now
go through the ring-aware `iso4_gpu_append_into_v_blocks` with a `Skip` feed, and
the helper is gone (it also stored its block head-major, unlike every other iso
append). On the read side `QuantIsoK3::dequant_gpu` and
`QuantIsoV3::dequant_gpu` counted `self.blocks` directly while their CPU siblings
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
`quant_rotor_v3_tests::quant_rotor_v3_truncate_at_b_gt_1_stays_loud`.

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
`quant_rotor_k3_tests.rs` (`RotorKBlocks`, QJL sideband on),
`quant_rotor_v3_tests.rs` (`RotorBlocks`) and `quant_iso_v_tests.rs`
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
  1. `SpeculativeGenerator::from_snapshots_with_id` takes a fourth,
     drafter-agnostic branch — `SpeculativeDispatcher::load_speculative` — when
     `draft_kind` is `None`, and `rmlx-cli`'s serve gate keys on
     `draft_path.is_some()` alone. `draft_model` has a `projects.toml` profile
     key while `draft_kind` does not, so `draft_model = Some, draft_kind = None`
     is reachable and never trips clap's `requires = "draft_kind"` (that only
     binds CLI-supplied flags). That branch calls plain `load_model` on both
     sides and `spec_generate_greedy_cached` builds caches from
     `num_hidden_layers()` with **no arch check**, rolling back through
     `KvCache::truncate_to`.
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
* `crates/rmlx-kv-quant/src/storage/quant_rotor_v{3,4}.rs` — the V stores gain
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
not a memory win because the store is not smaller than bf16 to begin with (see
"Memory truth" above — 21.75 bits/value for rotor, 16.25 for iso), so there is
no bandwidth prize to collect and no context at which a crossover exists. It is
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

**What it replaced.** `update_iso_k_only_{3,4}` called `QuantIsoK{3,4}::dequant()`
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
* `crates/rmlx-kv-quant/src/storage/quant_iso_k.rs` / `quant_iso_k4.rs` — the iso
  K stores, each embedding a `QuantKGpuRing`.
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

Empirical positive test: `validate_resolved_qwen_moe_low_k_bits_rejected_post_decompose`
in `crates/rmlx-models/src/kv_cache/cache_type_tests.rs` verifies the
runtime rejection path. The declared-vs-resolved bypass is covered by
`crates/rmlx-models/tests/resolved_arch_class.rs`, which builds a snapshot
that declares dense while shipping MoE tensors and asserts the guard fires.

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
| iso | 3 | 19.292 dB | 14.616 dB | −0.777 | 48.25 † |
| iso | 4 | 25.395 dB | 20.224 dB | −0.859 | 48.25 † |
| rotor | 3 | 20.432 dB | 14.616 dB | −0.966 | 21.75 |
| rotor | 4 | 26.555 dB | 20.224 dB | −1.052 | 21.75 |

No shipped cell is short of its anchor by more than 0.35 bits.

† **The iso rate is path-specific, and neither path is resident today.** 48.25
is the CPU `IsoBlocks` figure, which carries a per-group quaternion sideband. It
is the rate the V-only `iso3` / `iso4` stores *would* cost — those codecs decode
from the bf16 mirror, so `exit_prefill` builds them no store at all and they
measure byte-identical to `none` (§"Codec disposition", Class 2). `k_iso3/4` and
`iso3_sym/4_sym` do build a store: a GPU ring that does not carry the quaternion
— it is the constant `FIXED_QUAT` replicated per group, not data — and they sit
at **16.25** bits/value. See § iso3 "Memory truth". Read against rotor's 21.75 without that distinction the table inverts
the comparison: on the ring path iso is the cheaper of the two. Distortion is
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
