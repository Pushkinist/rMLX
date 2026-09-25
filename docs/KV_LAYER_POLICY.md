# KV codec per layer

This doc tells which KV codec each layer gets: the per-layer net-benefit
decision, bf16 at `--kv-quant none`, byte accounting, hot-swap, the dispatch
axis, the layer-adaptive overrides and the Qwen MoE rejection of low-bit K
codecs.

The other KV quantization docs: [`KV_QUANT.md`](KV_QUANT.md) (the contract:
API, CLI flags, the auto default, bit rates, codec disposition);
[`KV_CODECS.md`](KV_CODECS.md) (storage per `KvStorage` variant, TurboQuant
calibration); [`KV_ROTATION_CODECS.md`](KV_ROTATION_CODECS.md) (the iso and
rotor codecs); [`KV_FUSED_KERNELS.md`](KV_FUSED_KERNELS.md) (fused-QK, fused
flash-decode, sparse attention);
[`KV_STORE_TRUNCATION.md`](KV_STORE_TRUNCATION.md) (`truncate_to` per store);
[`KV_CODEC_FIDELITY.md`](KV_CODEC_FIDELITY.md) (measured codec fidelity).

---

## Per-layer net-benefit decision + net-negative warn

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
  store, so `exit_prefill` does not build it. The resident KV is the two bf16
  mirrors. This is the same byte count as `--kv-quant none`.
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

The estimate runs **low** for `k_iso*` and `iso*_sym`. It sizes an iso side from
the GPU ring. But `exit_prefill` encodes into CPU `IsoBlocks`. For iso3 at
`head_dim = 128` they are 6.07× the ring: 43.25 against 7.125 bits per value.
The first fused decode step frees the blocks (`drop_blocks_when_ring_live_iso_*`) on
a layer the fused path serves. The fused path's shape gate rejects batch > 1 and
a `head_dim` that is not a power of two at most 512 (`head_dim = 80` is one
example). On such a layer, `update_and_sdpa_iso_k_fused` returns before any
mutation, the ring is not allocated, and the layer keeps the blocks for the
whole request.

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

## Gemma4 global KV is bf16 at `--kv-quant none`

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

## Qwen3 dense KV is bf16 at `--kv-quant none`

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

## Qwen3.6 MoE KV is bf16 at `--kv-quant none`

The Qwen3.5-MoE arch (`Qwen3_5MoeForConditionalGeneration`) uses the same
load-time cast. The `qwen3_5_moe` loader calls `load_util::bf16_param` on every
float param: FullAttention (q/k-norm weights, quant scales and biases,
embedding scales and biases) and the GDN recurrent layers (`conv1d_weight`,
`norm_weight`). Thus an fp16 repack also stays bf16 in compute. Two CPU tests
pin this: `moe_stream_stays_bf16_with_bf16_params` and
`bf16_param_casts_fp16_to_bf16` (both in `qwen3_5_moe/moe_tests.rs`).

## KV byte accounting

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

## Per-request hot-swap

The `KvQuant` of a request is not tied to the model load. A running
`rmlx serve` accepts a per-request `kv_quant` field (OpenAI route). The field
selects the codec for that request. The weights stay resident; only the KV
cache is rebuilt. If the field is absent, the launch `--kv-quant` applies.

The prompt and prefix cache is **partitioned by codec**, so a switch cannot
serve mismatched cached K/V. `KvQuant::cache_key_salt()` is XOR'd into the
block-hash seed with the SSD `layout_key`. See `docs/PROMPT_CACHE.md`
§ "Codec namespacing" and `docs/SERVER.md` § "Per-request KV-config hot-swap".

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

## Qwen MoE rejects low-bit K codecs

`validate_resolved` (`crates/rmlx-models/src/kv_cache/cache_type.rs`) rejects
these codecs when the arch class is `Qwen3_5MoeForConditionalGeneration` or
`Qwen3VLMoeForConditionalGeneration`:

| Codec | Error |
|---|---|
| `Mixed` with `k_bits < 8` | `QwenMoeKBitsTooLow(k_bits)` |
| `PlanarK` | `QwenMoePlanarKRejected` |
| `Iso3Sym`, `Iso4Sym`, `IsoKOnly3`, `IsoKOnly4` | `QwenMoeIsoKRejected` |
| `Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4`, `RotorK3Asym`, `RotorK4Asym` | `QwenMoeRotorKRejected` |
| `TurboSym3` | `QwenMoeTurboKRejected` |
| `TurboSym4` | `QwenMoeKBitsTooLow(4)` |

Every other codec stores K at 8 bits or wider and passes. On any other arch
class `validate_resolved` rejects nothing.
`crates/rmlx-models/src/kv_cache/cache_type_tests.rs` tests each row of the
table.

### What the guard keys off

`validate_resolved` takes an architecture *string*. Which string it is handed
decides whether the guard can fire.

Both Qwen3.5 arch strings (`Qwen3_5MoeForConditionalGeneration` and the dense
`Qwen3_5ForConditionalGeneration`) load into one `Architecture` variant. The
loader selects dense or sparse MoE per layer from the tensor witness
`mlp.switch_mlp.gate_proj.weight`, not from the declaration. So
`architectures[0]` and the built model can disagree.

The enforcing check is therefore keyed on the **resolved** architecture:

- `Architecture::arch_class()` reports what the loader built. For the Qwen3.5
  variant it asks `has_sparse_moe_layers()`. Qwen3-VL has no registered dense
  arch string. So an all-dense Qwen3-VL checkpoint reports the MoE class and
  is refused these codecs too.
- `Architecture::validate_kv_quant()` runs `validate_resolved` against that
  class, before any KV cache exists.
- `load_model` emits a `warn!` naming `declared_arch` and `resolved_arch` when
  they differ. A deliberate alias (`registry::is_declared_arch_alias`, only
  Gemma4-unified) logs at `debug!` instead.

The check covers the cache of the model that serves the request, or of the
verifier on a speculative path. There are two production families, and they do
not share a call graph:

| Path | Cache built by | Enforced at |
|---|---|---|
| Non-speculative | per-arch `generate_greedy` (`gemma4`, `gemma3`, `qwen2`, `qwen3`, `qwen3_5_moe`, `qwen3_vl_moe`, `laguna`, `bitnet`), reached on production paths from `Architecture::generate_greedy` / `generate_image` | those two methods, and `ArchGenerator::from_snapshot_with_id` at startup |
| Speculative | `speculative::round_common::cache_stack` builds the verifier's caches; `Architecture::generate_greedy` is never called | `SpeculativeGenerator::from_snapshots_with_id` at startup, and its per-request `generate` |

The MTP and Eagle3 drafter caches use `KvQuant::None`, which every arch
accepts. The two-model paths build the draft model's stack with the verifier's
codec, and only the verifier is checked.

The CLI resolver (`rmlx-cli` `resolve_kv_quant`) reads `architectures[0]`,
because it runs before the model loads. It exits 78 on a rejected
`--kv-quant`. The server's `resolve_kv_quant_for_load` validates nothing. The
server's startup check is `ArchGenerator::from_snapshot_with_id` (or its
speculative counterpart), on the loaded model. The per-request checks cover a
request's `kv_quant` field, which arrives after startup.

`crates/rmlx-models/tests/resolved_arch_class.rs` relabels a MoE snapshot as
dense and asserts that the guard still fires. It is `#[ignore]` and needs the
`mlx-community__Qwen3.6-35B-A3B-8bit` snapshot (or `RMLX_TEST_MODEL_QWEN36`).
