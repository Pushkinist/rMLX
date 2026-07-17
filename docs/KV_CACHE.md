# KV Cache Quantization

> The codec implementation lives in **`rmlx-kv-quant`**
> (storage enums, MSL kernels, per-layer `KvCache`, paged-KV, mixed/rot-K
> codecs). The policy / builder layer (`KvQuant` resolution, `KvCacheBuilder`,
> `kv_quant_for_layer`, the SSD spill/hydrate plumbing) and the per-arch
> entry points live in **`rmlx-models::kv_cache`**. See `docs/KV_QUANT.md
> § Public API` for the canonical import paths.

User-facing reference for rMLX KV cache codec configuration: `--kv-quant`
presets and the `--cache-type-k` / `--cache-type-v` (aliases `--ctk` / `--ctv`)
primitives. Applies to `rmlx serve`, `rmlx chat`, `rmlx info`, and
`rmlx baseline`.

---

## 1. TL;DR

- **Presets** — `--kv-quant <k8v8|k8v4|planar|none|mixed_…>` picks a named
  combo wired into the codebase. Default: `auto`, resolved per-arch by
  `KvCacheBuilder::resolve_default` (see §6).
- **Primitives** — `--cache-type-k <tag>` / `--cache-type-v <tag>` (long
  aliases `--ctk` / `--ctv`) set the K and V codec independently from §4's
  namespace. `auto` on either side delegates that side to the resolver.
- **Default recommendation**: leave both unset — the auto resolver picks the
  best-known combo per architecture (see §6 table). Use the primitives only
  when sweeping novel combos or migrating from llama.cpp.

**Note for llama.cpp users**: rMLX flags use double-dash (`--ctk`), not
single-dash (`-ctk`).

`--kv-quant` and `--cache-type-{k,v}` are mutually exclusive on the same
command. Passing both is a clap-time hard error (§9).

---

## 2. Why KV cache matters

Every decoded token re-reads every prior token's K and V tensors through
SDPA. The cache footprint scales linearly with sequence length:

```
kv_bytes = 2 · n_layers · n_kv_heads · head_dim · seq_len · bytes_per_value
```

The `2` is K plus V. `n_kv_heads` (not `n_q_heads`) reflects GQA: a model
with 32 query heads sharing 8 KV heads stores 8× less than full MHA. On
modern long-context models the cache dominates RAM:

| Knob               | Effect on `kv_bytes`                              |
|--------------------|---------------------------------------------------|
| Doubling `seq_len` | 2× (the dominant axis at 32K+)                    |
| 8-bit vs bf16      | ~0.5× (8 bits vs 16 bits)                         |
| 4-bit V            | ~0.25× on V (mixed with 8-bit K → ~0.375× total)  |
| GQA (8 KV heads)   | 4× less than MHA at the same n_q_heads            |

At 128k context many 8B–35B models spend **more bytes on KV than on
weights**. Picking the right codec is therefore a memory decision first and
a quality / TPS decision second.

---

## 3. Flag surface

| Flag                                  | Type      | Purpose                                            |
|---------------------------------------|-----------|----------------------------------------------------|
| `--kv-quant <preset>`                 | preset    | Named combination (e.g. `k8v8`, `k8v4`, `planar`). |
| `--cache-type-k <tag>` / `--ctk <tag>`| primitive | K-side codec only. See §4 for tags.                |
| `--cache-type-v <tag>` / `--ctv <tag>`| primitive | V-side codec only.                                 |

When to use which:

- **Presets** — daily driver. Stable, regression-tested, documented
  per-arch champions.
- **Primitives** — sweeping codec combos in a bench loop, exploring a
  new asymmetric pair, or matching a llama.cpp invocation (§7).

Passing `--ctk` or `--ctv` alone fills the other side with `auto`; the
resolver then overrides only the user-specified side.

---

## 4. Supported types

The source of truth is `rmlx info --list-cache-types`:

```text
tag        codec                        bits  group  sides
---        -----                        ----  -----  -----
auto       auto (engine picks)          —     —      K, V
bf16       bf16 (unquantized)           —     —      K, V
q8_g128    rMLX MSL affine              8     128    K, V
q8_g64     MLX affine                   8     64     K, V
q8_g32     MLX affine                   8     32     K, V
q6_g64     MLX affine                   6     64     K, V
q5_g64     MLX affine                   5     64     K, V
q4_g128    MLX affine                   4     128    K, V
q4_g64     MLX affine                   4     64     K, V
q4_g32     MLX affine                   4     32     K, V
q3_g64     MLX affine                   3     64     K, V
q2_g64     MLX affine (V-only)          2     64     V
tq4        TurboQuant rotation          —     —      V
planar4    PlanarQuant rotation         —     —      V
```

Aliases: `f16` and `none` are accepted as `bf16`; `turbo4` is accepted as
`tq4`.

**Approximate bytes per value** (excludes per-group scale/bias overhead, which
adds a few percent at group=128, more at group=32; reduction ratios are
regression-tested in `crates/rmlx-models/src/kv_cache/tests.rs` —
`kv_reduction_ratios_match_table`):

| Tag       | Bits | Bytes/value (codes only) | Codec family    |
|-----------|-----:|-------------------------:|-----------------|
| `bf16`    | 16   | 2.0                      | unquantized      |
| `q8_*`    | 8    | 1.0                      | affine 8-bit     |
| `q6_g64`  | 6    | 0.75                     | affine 6-bit     |
| `q5_g64`  | 5    | 0.625                    | affine 5-bit     |
| `q4_*`    | 4    | 0.5                      | affine 4-bit     |
| `q3_g64`  | 3    | 0.375                    | affine 3-bit     |
| `q2_g64`  | 2    | 0.25                     | affine 2-bit (V only) |
| `tq4`     | 4    | 0.5                      | TurboQuant Lloyd-Max (V only) |
| `planar4` | 4    | 0.5                      | PlanarQuant + Givens rotation (V only) |

Accuracy class (rough, vs bf16 baseline at 4k context):

| Tag                        | Class    | Notes                                       |
|----------------------------|----------|---------------------------------------------|
| `bf16`                     | reference | No quantization error.                     |
| `q8_g128` / `q8_g64`/`g32` | very high | rMLX K8V8 + K8V4 K-side use `q8_g128`.     |
| `q6_g64` / `q5_g64`        | high     | Niche; mostly for sweeps.                  |
| `q4_g128` / `q4_g64`       | moderate | Best V-side affine option below 5 bits.    |
| `tq4`                      | moderate | Lloyd-Max codebook; +TPS on supported head_dim. |
| `planar4`                  | moderate | Per-pair rotation; rMLX-first on Apple Silicon. |
| `q4_g32`                   | moderate | Fine-grained group; matches llama.cpp `q4_0` block size. |
| `q3_g64`                   | low      | Exploratory — quality risk on most archs.  |
| `q2_g64`                   | low      | V-side only (§5.10). ~8× codes-only / ~6.4× effective (g=64) vs bf16. Coherent on Bonsai V-side; pure 2-bit K gated (§5.10). |

Two **8-bit affine** codecs exist and are NOT interchangeable:

- `q8_g128` — rMLX MSL symmetric (no bias). The K-side codec used by
  `K8V8`, `K8V4`, and `Planar`.
- `q8_g64` / `q8_g32` — MLX affine (scale + bias). Used by `Mixed` and the
  primitives path.

Rotation codecs are V-side only (§5.3). `q3_g64` is V-side only in
practice; the resolver still accepts it on K but the resulting combos are
rejected as `UnsupportedCombo`. `q2_g64` is V-side only and pure 2-bit K is
**actively rejected** at resolve time (§5.9) — 2-bit is too lossy for K.

---

## 4.5 Dynamic prefill grow + hard cap

Prefill chunks accumulate into a per-layer `[B, kv_h, max_seq, head_dim]`
raw buffer (`KvCache::prefill_raw_k/v`) before quantisation in
`exit_prefill`. The **initial** `max_seq` is always the small lazy default
`KV_MAX_SEQ_DEFAULT = 4096` (capped at the ceiling — see §4.6), regardless of
`--max-ctx`; the buffer grows from there to fit the prompt. The grow path
(below) takes it up to the resolved `--max-ctx` ceiling.

Before this fix the buffer was sized once at cache construction and never
grew. If a prompt exceeded `max_seq`, the chunked-prefill loop wrote at
`[prev_offset, new_offset]` inside a `[..., max_seq, ...]` buffer that
had already filled — `slice_update` clamped the target to zero-width and
the upstream broadcast failed with shapes like
`(1, 2, 512, 512) vs (1, 2, 0, 512)`.

### Grow contract

`update_prefill_raw` now calls `ensure_prefill_capacity(needed_seq, …)`
at the top of every call. When `needed_seq > storage.max_seq`:

1. Allocate a fresh `[B, kv_h, new_max_seq, head_dim]` buffer where
   `new_max_seq = next_pow2_seq(needed_seq)` (power-of-two ≥ needed,
   clamped to `2^30`). When a virtual ceiling is set (§4.6), the doubled
   size is additionally clamped down to the ceiling, so the ring never
   allocates past `--max-ctx`.
2. Copy the filled prefix `[..prev_offset]` from the old buffer into
   the new one with `slice_update`.
3. Bump `max_seq` on the active `KvStorage` variant so the downstream
   `exit_prefill` quant buffers and per-axis `QuantK::append` /
   `QuantV::append` capacity caps honour the new size.

Quantised payload buffers (codes / scales / biases) are not yet
allocated when grow triggers — they materialise in `exit_prefill` from
the (now-larger) raw prefill buffer, so there is no quantised state
to migrate. Already-finalised storage variants (post-`exit_prefill`,
on the decode hot path) are unaffected.

### Hard cap

`RMLX_KV_MAX_SEQ_HARD_CAP` (env var, optional) is an opt-in upper bound
on the prefill sequence length. When set to a positive integer:

* `needed_seq > cap` returns `Error::KvHardCapExceeded { requested, cap }`
  before any allocation. The HTTP layer maps this to a fatal class via
  `crates/rmlx-server/src/retry.rs`.
* When unset (default), there is **no cap**. The buffer grows
  unbounded by the power-of-two policy above, bounded only by
  available unified memory.

The env var is resolved once at process start (`OnceLock`); changes
during the process lifetime are ignored.

## 4.6 `--max-ctx` is a virtual ceiling (lazy grow)

`--max-ctx N` is a **virtual ceiling**, not an eager allocation. Both
`rmlx serve` and `rmlx baseline` start the KV ring at the lazy default
(`KV_MAX_SEQ_DEFAULT = 4096`, capped at the ceiling) and grow it by the
power-of-two policy above **up to** the ceiling as the prompt fills. A short
request on a server started with a large `--max-ctx` therefore pays only for
what it fills — it does **not** carry a multi-GB-to-tens-of-GB resident KV
cache sized to the ceiling.

> **Why this matters (issue #25).** Decode speed tracks the *allocated* ring
> capacity, not the *filled* length: an oversized resident KV sitting next to
> the weights starves decode bandwidth/locality. Eagerly sizing the ring to
> `--max-ctx` cost short requests ~20-25% decode TPS (e.g. gemma-4-e2b 4k:
> ~95 TPS oversized vs ~117 TPS right-sized). Lazy grow removes the tax; the
> only cost is an occasional realloc+copy when a request crosses a doubling
> boundary (amortised O(1), rare).

### Mechanism

* The resolved ceiling is `min(--max-ctx, max_position_embeddings)` when an
  override is given, else `min(max_position_embeddings, KV_MAX_SEQ_DEFAULT)`.
  Computed once per request by `rmlx_models::kv_cache::kv_max_seq_and_ceiling`,
  which returns `(initial_max_seq, ceiling)`. This is the same chain the
  server's `effective_max_ctx` uses for the per-request prompt-length guard,
  so cache ceiling and request guard agree.
* The ceiling is recorded on each `KvCache` via
  `KvCache::with_max_seq_ceiling(ceiling)` (field `max_seq_ceiling`). It is
  preserved across branch clones (`try_deep_clone`).
* `ensure_prefill_capacity` rejects a prefill that needs more than the
  ceiling with `Error::KvCeilingExceeded { requested, ceiling }` **before**
  any allocation, and clamps grown allocations to the ceiling. The server
  route additionally rejects over-long prompts up front
  (`effective_max_ctx_for`), so this engine-level guard is defense-in-depth
  (and the sole guard for the CLI `baseline`/`chat` paths).
* `ceiling <= 0` means "no ceiling" — unbounded lazy grow (bounded only by
  `RMLX_KV_MAX_SEQ_HARD_CAP` / unified memory).

SWA / rotating layers are unaffected: their bf16 ring is sized to the
`sliding_window`, never to `max_seq`, so they carry no long-context tax.

### Windowed-layer ring sizing (issue #35)

Issue #35 alleged that windowed / sliding-window-attention (SWA) layers
allocate KV to the full context length rather than the attention window. This
is **not the case** — it was falsified by direct `resident_bytes()` measurement
and is documented here so the question does not recur.

SWA layers use the `rotating` ring (`rmlx-kv-quant::rotating`, a byte-for-byte
port of mlx-lm `RotatingKVCache`). The ring's *physical* buffer is bounded to
`sliding_window + prefill_chunk` rows **regardless of context** — never the
filled context length. Mechanism: `RotatingState::update_in_place` only grows
while `buf_len < max_size`, and `update_concat` trims to `max_size - 1` before
concatenating the next prefill chunk, so the buffer peaks at
`max_size - 1 + chunk` (the `+ chunk` is the in-flight prefill block). The
logical `offset` field tracks total tokens seen (mirroring mlx-lm) but is **not**
the physical fill. `KvCache::resident_bytes()` (issue #33) counts the ring
buffers, so the measurement is exact.

Measured windowed (rotating) vs global (full-attention) bf16 `resident_bytes`
for a single Gemma4-shaped layer (`B=1, kv_h=1, head_dim=256, window=512,
chunk=512`), chunked prefill at growing context:

| ctx  | windowed (bytes) | global (bytes) | windowed scales with ctx? |
|------|------------------|----------------|---------------------------|
| 4k   | 1,047,552        | 4,194,304      | no                        |
| 16k  | 1,047,552        | 16,777,216     | no                        |
| 64k  | 1,047,552        | 67,108,864     | no (64× smaller)          |

The windowed ring is **flat** across context; only the global-attention layers
grow with `max_seq`. This is exactly what #35 asked for — already implemented by
the rotating ring. Pinned by
`crates/rmlx-kv-quant/src/kvcache/windowed_ring_sizing_tests.rs`
(`windowed_ring_stays_bounded_while_global_grows` for the byte bound;
`windowed_ring_retains_full_swa_window` for the correctness guard that the
retained set always contains the full most-recent `window` — no still-attended
key is ever evicted).

Scope note: the rotating ring is the bf16 SWA path used by Gemma3/Gemma4 (the
only archs with interleaved windowed + global attention). Quantized SWA codecs
are not currently routed through the ring (mlx-lm's reference keeps SWA bf16
too — `RotatingKVCache.to_quantized` raises), but the SWA layers are bf16 by
default (§5.7), so the bound applies on every shipping SWA configuration.

### `resident_bytes` counts the live-inference KV (filled prefix, not ceiling)

`KvCache::resident_bytes()` reports the bytes of the K/V that **actually serves
decode** — the *filled* prefix of each buffer, not its pre-allocated capacity.
This matters because the per-position decode mirrors (`decode_fp16_k/v`, and the
sole bf16/f32 storage on the `KvQuant::None` path) are allocated to the
`--max-ctx` ceiling and only compacted to the filled length *after*
`exit_prefill` / decode-time reclaim. Naively summing the whole allocation made
the same `(model, prompt, KV quant)` cell report different totals depending on
when the metric was read (ceiling vs compacted), and made `kv_cache_bytes`
depend on the chosen ceiling rather than the prompt — so a high `--max-ctx`
inflated the reported KV even for a short prompt.

To keep the figure consistent and comparable for bytes-per-KV-token /
tokens-per-KV-GB, the seq-scaled buffers are counted by their filled length
(`offset`, clamped to the buffer capacity) at each buffer's real per-position
size (shape × dtype, so per-layer head_dim differences — e.g. the windowed
layers' `head_dim` vs the full-attention layers' larger `head_dim` — and the
real decode dtype are both picked up). Quantized storage is already compacted at
`exit_prefill`, so it is counted as-is. The windowed ring is already bounded to
the window (above), so clamping is a no-op once the ring has filled.

The prompt-cache snapshot (a deep clone of every layer cache, held per slot in
the arch prompt cache) is **never** part of this sum — the store sites iterate
only the active decode caches. So `kv_cache_bytes` means live-inference KV on
every arch, with or without a prompt-cache snapshot resident.

### Per-request `max_ctx` override (issue #26)

Because the ceiling is resolved **per request** (`kv_max_seq_and_ceiling`), it
can be overridden per request without reloading weights. The OpenAI route
accepts an optional `max_ctx` field that supplies the override for that one
request (else the launch `--max-ctx`); the server's `context_length_exceeded`
prompt-length guard uses the per-request ceiling when present. This pairs with
the per-request KV-codec hot-swap (also issue #26) so a resident model can sweep
`(codec × ctx)` cells with no reload. See `docs/SERVER.md` § "Per-request
KV-config hot-swap".

## 5. Hard invariants

These are enforced by the resolver. Violation exits with code 78
(`EX_CONFIG`) and a self-describing message.

### 5.1 `head_dim % group_size == 0` (affine codecs)

The MLX affine quantizer groups consecutive elements into blocks of
`group_size` and shares one scale (and bias) per block. The head_dim must
be a multiple of group_size or the kernel cannot tile evenly.

Trigger: `--ctv q4_g64` on a model with `head_dim = 80`.

### 5.2 MLX bit-packing rule (`head_dim % (32 / bits) == 0`)

MLX packs `bits ∈ {2..8}` into 32-bit words. The element count per word is
`32 / bits`; head_dim must be a multiple of that.

Trigger: `--ctv q3_g64` on `head_dim = 64` (64 % ⌊32/3⌋=10 ≠ 0).

### 5.3 K-side rotation codecs

The **V-side** rotation codecs `tq4` and `planar4` remain V-side only — their
kernels assume V-style distributional properties and apply an *inverse*
rotation in the V dequant. There is no `tq4`/`planar4` K-side implementation.

Trigger (still rejected): `--ctk tq4` or `--ctk planar4` → exit 78
(`KSideRotationCodec`).

**`rot_k` — the one K-side rotation codec.** K is affine-quantized at
8-bit/group=64 in a **Hadamard-rotated basis**, and Q is **pre-rotated** by the
same matrix `R` before the score matmul, so the rotations cancel:

```text
  (Q Rᵀ) · (K Rᵀ)ᵀ = Q (Rᵀ R) Kᵀ = Q Kᵀ      (R orthogonal, Rᵀ R = I)
```

Because the K rotation is cancelled by the pre-rotated Q, **K is never
inverse-rotated** — it stays quantized in the rotated basis for the whole cache
lifetime (this is the "pre-rotate-Q trick" that makes the K side cheaper than a
naive rotate→quantize→inverse-rotate path; cf. the V side, which *must*
inverse-rotate). The rotation decorrelates K channels and equalizes their
dynamic range, lowering affine-K quant PPL.

- Tag: `--ctk rot_k`. **K-side only** — `--ctv rot_k` is rejected (`RotKVSide`).
- K is fixed at 8-bit/group=64 (rotation is a PPL win *over* affine K, not a
  way to drop K below 8-bit — hard rule 6). Pair with any **affine** V codec;
  default is `--ctv q4_g64`. (Non-affine V — `tq4`/`planar4`/`bf16` — is
  rejected, since `rot_k` reuses the Mixed `mx.quantize` 3-tuple path.)
- Requires a **power-of-two head_dim** (Walsh–Hadamard `R`); else
  `RotKHeadDimNotPow2`. Bonsai head_dim=128 qualifies.
- **Opt-in only** — never an `auto`/per-arch default; existing KV quants and
  goldens are byte-identical (plain Mixed keeps `rotate_k=false` and quantizes
  the unrotated K).

**v1 implementation (document-the-truth):** the rotation is applied as a
plain MLX `matmul` against a precomputed `[D,D]` `R` — correct + coherent,
but O(D²) arithmetic and materialises an intermediate `K_rot` tensor.

**Fused FWHT kernel:** `crates/rmlx-models/src/kv_cache/rot_k_msl.rs`
implements a fused Metal kernel using the Fast Walsh-Hadamard Transform (FWHT),
which is O(D log₂D) instead of O(D²). For D=128 (Bonsai): 896 ops vs 16 384 —
~18× fewer arithmetic ops plus elimination of the intermediate `K_rot` DRAM
round-trip. Both K-encode and Q-pre-rotation benefit.

Infrastructure note: `planarquant_msl.rs` served as the template for the MSL
kernel infrastructure pattern (`MetalKernel` singleton,
`MetalKernelInvoke` dispatch) was borrowed — NOT the Givens pair-rotation math.
The FWHT is a butterfly network operating on the full D-element row simultaneously,
which is fundamentally different from the Givens 2×2 micro-rotation.

Default-OFF. Enable via `RMLX_ROT_K_FUSED=1`. Supported D: {32, 64, 128, 256, 512}.
Falls back to v1 matmul on unsupported D or kernel error.

Reference math: `crates/rmlx-models/src/kv_cache/rot_k.rs`; rotorquant README
(`../rotorquant/`).

Example: `--ctk rot_k --ctv q4_g64` on Bonsai → coherent.

### 5.4 Qwen MoE family requires K-bits ≥ 8

Symmetric ≤4-bit K on Qwen2.5/Qwen3 MoE causes catastrophic perplexity
collapse (218 → 8641 in the historic test). The 7:1 GQA ratio amplifies
K-head quantization error through softmax. Re-checked **after** auto-
decompose so resolver-table changes can't bypass it.

Trigger: `--ctk q4_g64` on `Qwen3_5MoeForConditionalGeneration`.

### 5.5 `tq4` requires `head_dim ∈ {128, 256}`

The TurboQuant 4-bit kernel hardcodes its group layout for these two
head_dim values. Other sizes are rejected.

Trigger: `--ctv tq4` on `head_dim = 64`.

**Gemma4 (all variants) is rejected here.** The resolver checks the
**full-attention** head_dim — `ModelConfig::head_dim()` returns
`text_config.global_head_dim`, which is `512` for e2b, e4b, and 26b-a4b
alike (their `text_config.head_dim = 256` belongs only to the SWA layers,
which stay bf16 unconditionally — §5.7 — and are never tq4-quantized). Since
the only layers tq4 would touch (full-attention) are head_dim=512,
`--ctv tq4` on any Gemma4 model fails fast at startup with
`Tq4UnsupportedHeadDim(512)` (exit 78). This is the correct general outcome:
tq4 is genuinely unsupported at that head_dim, not a missing feature. tq4
remains available wherever the FA head_dim ∈ {128, 256} (e.g. Qwen3 @ 128,
Qwen3.5-MoE @ 256) via the `sdpa_dispatch` TurboFlash path.

### 5.6 `planar4` requires `head_dim % 32 == 0`

PlanarQuant's group size is 32. Same divisibility rule as §5.1, just
codec-specific.

Trigger: `--ctv planar4` on `head_dim = 80`.

### 5.7 SWA layers always bf16

Gemma3 / Gemma4 sliding-window-attention layers bypass KV quantization
unconditionally — the rotating buffer codec is bf16. Quantization flags
apply to full-attention layers only. A one-shot `info!` log discloses this
at startup when a quantized type is requested on a SWA model.

The bf16 rotating ring is **not** serialised to the SSD KV tier (the ring
layout is not expressed in the `.kvb` format), so SWA layers hydrate as empty
`KvStorage::None`. A hydrated prompt-cache entry whose length is not
block-aligned therefore cannot be reused as a prefix (the empty SWA prefix
would corrupt attention) — the consume path detects the payload-less SWA layer
and degrades the entry to a full re-prefill. See `docs/SSD_TIER.md` §"SWA layers
are not spilled". Serialising the SWA ring is a future enhancement.

### 5.7.1 `QuantK` GPU buffer is sequence-major

The `QuantK` storage (q8_0 K — the K side of K8V8 / K8V4 / Planar, the V side
of K8V8, and the K side of the V-only Iso / Rotor codecs) lays its flat
codes / scales buffer out **sequence-major**: the logical `[B, kv_h, S, D]`
cache is stored as `[B, S, kv_h, D]`, so per token all heads are contiguous and
chunk `n` occupies a single contiguous run at `prev_seq * words_per_seq`.

This is the only ordering that lets the active prefix be read back as one
contiguous `slice` after **any** number of appends. `QuantK::append` transposes
the incoming head-major chunk before quantizing; `QuantK::dequantize_choice`
transposes back to the logical `[B, kv_h, S, D]`. A single-chunk cold prefill is
the identity *at the logical-mapping level* (the transposes cancel), so the
common path stays correct. It is **byte-identical** to the pre-fix head-major
grouping only when `head_dim % 128 == 0` — then every q8 group of 128 stays
inside one head. That holds for every current QuantK-routed target arch
(Qwen3.5-MoE linear `head_dim=128`, Gemma3 and Gemma4 text KV `head_dim=256`),
so the cold path is byte-identical in practice. When `head_dim` is not a
multiple of 128 (no current target arch, but exercised by the `d=64` cross-head
round-trip test) a q8 group spans a (head,token) boundary, so its per-group
`abs_max` scale differs from the old head-major grouping: the cold path is
logically correct and within q8 noise but **not** bit-identical to the base
commit's decode output.

This matters for the **post-SSD-hydrate decode path**. After a hydrate, the bf16
decode mirror (`decode_fp16_k`) is absent, so K is read from the quantized
buffer; the first decode step appends a second chunk. Under the old head-major
chunk store + `[B, kv_h, S, D]` reshape read, that second chunk landed after all
heads' prefixes while the reshape mapped one head's new-token slot onto another
head's prefix — a head transposition that silently corrupted K for every GQA
model (`kv_heads > 1`). Steady-state decode (bf16 mirror live) was never
affected. The spill / hydrate / paged-grow paths copy the contiguous active
prefix `[0 .. filled]` and are layout-agnostic, so the on-disk `.kvb` payload is
unchanged.

### 5.7.2 The same sequence-major rule covers `QuantV`, `QuantKTurbo3/4`, and the paged handoff

The `prev_seq * words_per_seq` write vs head-major reshape is a buffer-shape
property, not a K-vs-V one, so the sibling flat-buffer storage structs carry the
**same** latent head-transposition and use the **same** sequence-major fix:

- **`QuantV`** (TurboQuant V — K8V4 / Planar V side, bits=4 GPU + CPU
  `Vec<TurboBlocks>`): `append` reorders the chunk heads↔seq before quantizing
  on both backends; `dequantize_choice` reorders back. Byte-identical cold
  prefill at `head_dim % 32 == 0` (TurboQuant group of 32). The CPU
  `Vec<TurboBlocks>` path concatenates per-append blocks, so it carries the same
  cross-block scramble and the same fix.
- **`QuantKTurbo3` / `QuantKTurbo4`** (symmetric K side of `TurboSym3` /
  `TurboSym4`): identical TurboQuant codec reuse; the "axis-agnostic" header
  rationale was the root cause (a *positional* codec means physical buffer order
  decides correctness across appends).
- **Paged KV** (`update_paged` → `PagedKStorage` / `PagedVStorage` /
  `PagedPlanarVStorage`): the page slabs are physically token-major
  (`words_per_token` per token slot), so `update_paged` reorders the head-major
  `new_k` / `new_v` to sequence-major **before** quantizing and transposes the
  dequant output back. `--paged-kv` is default-off.

In every case the GPU side adds `Array::contiguous` after the heads↔seq
transpose because the custom quant / dequant MSL kernels read their buffers by
raw linear index and ignore MLX lazy-transpose strides — a CPU-only test cannot
catch that; the layout fixes are GPU round-trip verified.

**Now covered (PlanarQuant K/V — closes the class):** the PlanarQuant structs
(`QuantPlanarK` / `QuantPlanarV`) carried the same multi-append head scramble
(reproduced on CPU and GPU: a two-chunk GQA append with `kv_h > 1` and a
multi-token second chunk drove per-row cosine far negative against the
head-major reference). `QuantPlanarK` was the hard case because it also exposes
its packed codes buffer to the `planar_fused_qk` / `planar_flash_decode` MSL
kernels (and the sparse-attn phase-1/2 kernels) via `gpu_packed_view` on the
post-hydrate decode path (gated on `decode_fp16_k.is_none()`). Those kernels
read the packed K buffer **per token**, so the fix is a coordinated change:

- **Storage goes sequence-major** like the rest of the family. `append` reorders
  the incoming head-major chunk heads↔seq before quantizing (GPU: `transpose`
  then `Array::contiguous`, because the raw-linear-index MSL kernel ignores
  lazy-transpose strides; CPU: `transpose_heads_seq` on the flat data with the
  reordered chunk shape), and `dequantize_choice` reshapes the prefix
  `[B, S, kv_h, D]` and transposes back to the logical `[B, kv_h, S, D]`.
- **The packed-K kernels switch to a sequence-major token base.** `planar_fused_qk`,
  `planar_flash_decode` (P1), and the sparse-attn phase-1/2 score kernels now
  index K as `kv_tok = (b * kv_seq + s) * kv_h + kv_h_idx` instead of the old
  head-major `(b * kv_h + kv_h_idx) * kv_seq + s`. The V offset stays head-major
  in the flash / sparse kernels because V is the separate bf16 decode mirror
  (`decode_fp16_v`), not the planar-packed buffer.

**Rotor K joins the same convention.** `QuantRotorK{3,4}`'s CPU blocks were
already sequence-major (`append` calls `transpose_heads_seq` before encoding),
but the GPU encode path (`rotor_gpu_encode_block`, QJL-off) fed the kernel the
head-major chunk unreordered — consistent only because the decode step it ran on
is `new_seq == 1`, where the transpose is the identity. The GPU-resident ring
(`storage::RotorGpuK`) that backs `rotor_flash_decode` is sequence-major like the
rest of the family, so the K-side GPU encode now reorders heads↔seq
(`transpose` + `Array::contiguous`, skipped on the `new_seq == 1` hot path) to
match its CPU sibling. The `rotor_flash_decode` P1 kernel indexes K as
`kv_tok = (b * kv_seq + t) * kv_h + kv_h_idx`; V stays head-major (bf16 mirror).

For a single decode token (`new_seq == 1`) the heads↔seq transpose is the
identity, so the decode hot path is byte-unchanged; for a single cold-prefill
chunk (`prev_seq == 0`) the append and dequant transposes cancel. PlanarQuant is
layout-agnostic (it processes the flat element stream group-by-group and
`head_dim % GROUP_SIZE == 0`, so no group spans a (head, token) boundary), so
the reorder is bit-exact, not just within-noise — planar3 and planar4 packing
are untouched. The `.kvb` SSD format is byte-stable (only the token-row order
within the buffer changes; spill and dequant agree on sequence-major). With this
change **the entire multi-append head-scramble class is closed** — every
flat-buffer quantized KV storage (`QuantK`, `QuantV`, `QuantKTurbo3/4`, paged,
all eight Iso/Rotor codecs, and now `QuantPlanarK` / `QuantPlanarV`) is
canonically sequence-major. CPU + GPU round-trip verified (two-append GQA vs
single-shot, plus the fused-QK / flash-decode / sparse-attn parity suites all
green on real Metal).

### 5.7.3 The Iso / Rotor `Vec<Blocks>` codecs use the same sequence-major rule

The rotation-KV `Vec<Blocks>` codecs — `QuantIsoV3` / `QuantIsoV4`,
`QuantIsoK3` / `QuantIsoK4`, `QuantRotorV3` / `QuantRotorV4`,
`QuantRotorK3` / `QuantRotorK4` — accumulate one `*Blocks` entry per `append`
and concatenate them on `dequant`, with the caller reshaping head-major
`[B, kv_h, S, D]`. That is the **same** cross-block head transposition as
`QuantV` (a multi-append GQA cache with `kv_h > 1` scrambles per-head values).
Each `append` now reorders the head-major chunk heads↔seq before encoding and
`dequant` reorders back to the logical `[B, kv_h, S, D]`; for a single chunk
(cold prefill) the two reorders cancel.

These codecs are **per-token-row positional** (the codec sees `n_tokens` rows
of `head_dim`; it does not interpret the B / kv_h / S axes), so the sideband
parameters stay correctly associated after the value reorder:

- **Quaternions / per-(token, group) scale + norm** (Iso): keyed by row
  position, so they permute together with the value rows. The iso fast-mode
  quaternion is the constant `FIXED_QUAT` for every group regardless.
- **Static rotor table / QJL projection matrix** (Rotor): keyed by
  group-position-within-`head_dim` (rotor) or by the JL projection (QJL), not
  by token — the reorder leaves them untouched. The **per-token** QJL sideband
  (`qjl_codes` / `qjl_norms`) permutes with the value rows.

`QuantIsoV3` is the one GPU-resident member: its `append_gpu` adds
`Array::contiguous` after the heads↔seq transpose before the iso3 encode kernel
(raw-linear-index MSL kernel; lazy-transpose strides are ignored), and both
`dequant_gpu` paths (mirror fast-path and CPU-staged `from_bytes`) reshape the
flat decode to `[B, S, kv_h, D]` then transpose back. The remaining seven are
CPU-only (`QuantIsoK3` also drives the shared iso3 dequant kernel via the
CPU-staged path; iso4 / rotor have no MSL kernel). The `.kvb` SSD format is
byte-stable — only the token-row order **within** a block changes, and spill
and dequant agree on sequence-major. GPU round-trip verified on `QuantIsoV3`
(two-append GQA vs single-shot, `kv_h=1` control).

### 5.7.4 The unquantised store boundary floors K/V to bf16

The `KvQuant::None` / warm-TTFT decode mirror (`decode_fp16_k/v`) is bf16 by
contract — the dtype every sibling MLX backend stores for unquantised KV. But the
incoming K/V inherit whatever dtype the model's attention stream happened to
produce, so a single f32 scalar leaking upstream (Gemma4 mxfp8 stream, Qwen3
fp16 norm/scale params — both fixed per-arch) silently promoted the cache to f32
and doubled resident KV, found months later in a bench.

The store boundary therefore **casts incoming K/V to bf16 independent of the
inbound dtype**, at the single model-agnostic choke point every arch funnels
through (`update_prefill_raw` for the seed, `update_decode_fp16` for the decode
append, both in `rmlx-kv-quant`). The cast is idempotent (a `dtype == Bf16`
check, no `astype` on the already-bf16 hot path) so it is pure insurance. It is
**defense-in-depth, not a substitute for the per-arch source fix**: it caps the
*memory* damage but leaves any upstream f32 *compute* (RoPE / SDPA) still f32.
The bytes-per-element detector (an f32 input must store as 2 B/elem) lives in
`resident_bytes_tests.rs` and runs under `make model-check` (now `-p
rmlx-kv-quant`), so a future arch leak trips CI. Full reference:
`docs/KV_QUANT.md` §`KvStorage::None`.

### 5.7.5 `exit_prefill` runs on a worker thread — MLX stream affinity

`exit_prefill` (the bulk prefill→quant step) executes on the request's tokio
`spawn_blocking` worker, the same thread the prefill forward built its graph on.
That co-location is load-bearing because of an MLX threading contract:

- Since MLX ≥0.31 the default CPU/GPU streams are **thread-local**
  (`mlx/stream.cpp`: `static thread_local … default_streams`) and the CPU
  backend resolves a stream's `CommandEncoder` through a **thread-local** map
  first (`mlx/backend/cpu/encoder.cpp::get_command_encoder`). An `Array::eval()`
  of ops that were **built on a different thread** throws
  `There is no Stream(cpu, N) in current thread.` (surfaced by
  `mlx-c … array.cpp`). Streams from `mx.new_stream` are documented as usable
  only on their thread of creation.
- A graph that is **built and evaluated on the same thread always evals cleanly**
  — for CPU and GPU alike, whether or not any stream guard ran. Only a
  *cross-thread* eval of an unscheduled (lazy) array faults; an
  already-materialised array is safe to read from any thread.
- The K8V8 `exit_prefill` q8 quantize + eval is therefore safe on its own (all
  its ops are worker-built). The observed rare, self-healing
  `no Stream(cpu, 0)` crash on dense `Qwen3_5ForConditionalGeneration` (issue
  #206 — 48× per request, one per layer → zero tokens) is a *cross-thread* eval:
  a stream-bound lazy op that crossed the load→generation thread boundary. It
  self-heals once that shared array materialises.

**Guard (worker-thread stream hygiene).** The generation entry points
(`arch::generate_greedy`) call `rmlx_mlx::ensure_cpu_default_stream()` — the CPU
analog of the pre-existing `ensure_gpu_default_stream()` — before building any
graph, so the worker registers its **own** default CPU stream up front (GPU stays
guarded by device, as before). This removes first-touch nondeterminism and keeps
every worker-built graph (the whole per-request prefill/decode path,
`exit_prefill` included) eval-clean. The same one-line CPU guard is mirrored at
the other worker-thread eval entry points (embeddings, image, audio, transcribe,
speculative).

**Limitation (document-the-truth).** The guard registers the *worker's own*
stream; it does **not** let a worker evaluate an array whose ops were built on
another thread — that needs an MLX `thread_unsafe` (process-global) stream,
which **mlx-c 0.6.0 does not expose**, or materialising the shared array on its
building thread before hand-off. The mechanism and this bound are pinned by
`cross_thread_eval_faults_documents_mlx_limit` (rmlx-mlx) and the worker-thread
`k8v8_q8_quantize_eval_on_worker_thread` regression (rmlx-kv-quant).

### 5.8 TurboQuant requires Flash Attention

The `tq4` V-side path is only valid through the Flash Attention dispatch.
Enforced upstream by `sdpa_dispatch`; users do not flip this.

### 5.9 `head_dim` must be declarable

The resolver refuses to operate without a `head_dim`. If
`ModelConfig::head_dim()` returns `None` and no safe fallback derives one,
exit 78 with `HeadDimUnknown`. The fix is either to use `--kv-quant`
(preset path bypasses the affine validators) or to report the model so its
loader can populate `head_dim`.

### 5.10 Pure 2-bit K is gated

`q2_g64` is **V-side only**. 2-bit on the K side collapses attention
score precision and produces incoherent / repetitive output (CLAUDE.md hard
rule 6 — smoke-probed on Bonsai). `combo_to_kv_quant` rejects any K-side
`q2_g64` with `UnsupportedCombo` before it can reach the kernel — there is
no garbage-output path. Use the asymmetric form instead:

- `--ctk q8_g128 --ctv q2_g64` (8-bit K, 2-bit V), or
- `--kv-bits 2` (K stays 8-bit, V drops to 2-bit; group from `--kv-group-size`).

2-bit V flows through the existing `Mixed` `mx.quantize` path unchanged —
MLX's affine quantizer packs 16 two-bit values per `u32`, so no new rMLX
kernel was required. Coherent on Bonsai (`Mixed{k=8, v=2, g=64}`); effective
V-side compression is ~6.4× vs bf16 at group=64 (8× on codes alone, less the
per-group scale+bias overhead which is bf16).

Trigger: `--ctk q2_g64` (any V).

### 5.11 `k8vturbo3` — research probe only

`--kv-quant k8vturbo3` wires K = affine q8_0 (group_size=128) and V = TurboQuant
**3-bit Lloyd-Max** (group=32, N(0,1) codebook). This variant exists as a bench
probe and is **not recommended for production use**.

**First pass (CPU V-dequant path, Gemma4 e4b + 26b-a4b, ctx≈17k):**

| Model | k8v8 TPS | mixed3 TPS | turbo3 TPS | turbo3 vs mixed3 | KV bytes (turbo3 vs mixed3) |
|---|---|---|---|---|---|
| Gemma4-e4b | 65.12 | 66.79 | 64.70 | **−3.1%** | same |
| Gemma4-26b | 65.81 | 65.29 | 65.50 | +0.3% (noise) | same |

- **TPS gate** (≥ −2%): FAIL on e4b, pass on 26b (within noise).
- **PPL gate** (turbo3 ≤ mixed3 − 0.3): NOT measured. No native rMLX PPL harness.
- **Memory**: identical KV bytes to mixed3 at same bit-width (no advantage).

**Second pass (Metal 3-bit kernel, same setup):**

| Model | k8v8 TPS | mixed3 TPS | turbo3-GPU TPS | turbo3-GPU vs mixed3 |
|---|---|---|---|---|
| Gemma4-e4b | 63.88 | 66.01 | 63.70 | **−3.5%** |
| Gemma4-26b | 62.66 | 64.91 | 60.45 | **−6.9%** |

- **TPS gate**: FAIL on both models — GPU kernel does not unlock a TPS win
  over the CPU path; the GPU pipeline overhead per decode step ends up
  similar to the K8V8 GPU path (note `turbo3-GPU ≈ k8v8`).
- **PPL gate**: still N/A — `rmlx eval ppl` is Qwen3 arch only;
  `K8VTurbo3` wires to Gemma4 arch only, so no overlap exists.
- **Smoke probe**: coherent prose on both Gemma4-e4b and Gemma4-26b with the
  Metal V3 kernel enabled (validated 2026-05-26 prior to revert).

**Decision**: GPU dispatch wiring **reverted** at `update_k8vturbo3` (V-side
stays on CPU as in the first pass).  The Metal kernel source is kept under
`crates/rmlx-models/src/kv_cache/k8vturbo3_append_msl.rs` as a
future-reference hook with full bit-equivalence unit-test coverage (CPU
vs GPU max abs diff < 1e-3 on a fixed-seed input).  Re-wiring it later is
a one-line dispatch-site change once Gemma4-arch PPL coverage exists.


### 5.12 Paged-KV append model — no slot-mapping / negative-skip semantic

rMLX paged-KV uses **sequential append**, not scatter-by-slot-index. This
section documents the contract so callers do not expect vLLM-style
`slot_mapping` semantics.

**Architecture** (`crates/rmlx-kv-quant/src/paged/ops.rs`):

The three paged storage types (`PagedKStorage`, `PagedVStorage`,
`PagedPlanarVStorage`) expose `append(new_shape, codes, scales, device)`.
Each call fills the next contiguous token slots in the current page, allocating
a new page when the current one is full. The `block_table: Vec<usize>` grows
monotonically — logical page index = `block_table.len() - 1`, physical page id =
`block_table[logical]`. There is no scatter parameter, no sparse skip, and no
negative-index convention.

**Call chain** (`crates/rmlx-kv-quant/src/kvcache/update.rs:1898`):

```
KvCache::update → KvStorage::Paged dispatch → update_paged
  → q8_quantize_gpu  → PagedKStorage::append
  → turbo/planar_quantize_gpu → PagedVStorage/PagedPlanarVStorage::append
```

**No masked positions upstream**: chunked prefill partitions `prompt_ids`
into clean contiguous chunks (`prompt_ids.chunks(prefill_chunk)`). No padding
tokens, no negative position ids, no masked slots enter the paged path at any
call site.

**Contrast with multi-turboquant**: the Python reference (`multi-turboquant
tests/test_integration.py::TestPagedCacheOps::test_slot_mapping_negative_skips`)
uses a `slot_mapping` tensor where `slot == -1` means "skip this token" — a
vLLM batch-scatter interface. rMLX does not have this interface, and negative
skip is not load-bearing for any current rMLX feature (no batch scatter, no
eviction-driven slot reuse, no padding-token masking in the paged path).

**Contract**: negative-skip / slot-mapping is **absent by design** at the
current single-request sequential-append architecture. If continuous-batching
ever adds a scatter interface, that work introduces the contract and must add
the corresponding negative-skip test at that time.

**Codecs routed through Paged**: `K8V4`, `K8V8`, `Planar`, `Planar3`.

**Codecs NOT routed through Paged** (auto-fall-through to their contiguous
storage variants regardless of `--paged-kv`):

- `K8VTurbo3`, `K8VTurbo2`, `TurboSym4`, `PlanarK`, `RotKTq4V` — each owns a
  CPU- or hybrid-path V codec that does not fit either `PagedVStorage`
  (q8 / tq4-only) or `PagedPlanarVStorage` (PlanarQuant-only).
- `Iso3`, `Iso4` — quaternion-quantized V; per-token quaternion + scale +
  norm payload would need its own paged container.
- `Rotor3` — Cl(3,0) Clifford rotor sandwich V codec. A paged rotor3 variant
  would need both a per-token container (codes / scales / norms) AND a
  layer-static rotor table living inside the paged arena — doubling the
  paged-storage surface area for an opt-in codec. Deferred as a follow-up;
  the iso3 / iso4 precedent is the operative pattern.
- `Rotor4` — same deferral as Rotor3 (4-bit codebook, denser pack, identical
  structural argument; paged variant deferred).

---

## 6. Default policy

When neither `--kv-quant` nor the primitives are passed, the per-arch
default comes from `KvCacheBuilder::resolve_default` in
`crates/rmlx-models/src/kv_cache/mod.rs`. The mapping (read from the
current source):

| Arch class                              | Signal                          | Default `KvQuant` |
|-----------------------------------------|---------------------------------|-------------------|
| `Qwen3_5MoeForConditionalGeneration`    | (any)                           | `K8V8`            |
| `Qwen3_5ForConditionalGeneration`       | `is_paroquant = true`           | `K8V4`            |
| `Qwen3_5ForConditionalGeneration`       | otherwise                       | `K8V8`            |
| `Qwen3ForCausalLM`                      | `weight_bits = 2` (Bonsai)      | `Mixed{8,4,g=64}` |
| `Qwen3ForCausalLM`                      | otherwise                       | `K8V8`            |
| `Qwen2ForCausalLM`, `LagunaForCausalLM` | (any)                           | `K8V8`            |
| `Gemma3ForConditionalGeneration`        | (any)                           | `Planar`          |
| `Gemma4ForConditionalGeneration`        | `has_moe`                       | `K8V8`            |
| `Gemma4ForConditionalGeneration`        | `hidden_size ≤ 2560` (e2b/e4b)  | `K8V8` (`K8V4` if `is_paroquant`) |
| `Gemma4ForConditionalGeneration`        | `hidden_size ≥ 5376` (31b)      | `Planar`          |
| `Gemma4ForConditionalGeneration`        | hidden_size in (2560, 5376)     | `K8V8` (safe fallback) |
| unknown                                 | —                               | `K8V8`            |

One-line rationale per row:

- Qwen3.5 MoE — Mixed K8V8 regressed on the hybrid GDN+FA arch;
  K8V8 fused dequant + fast SDPA wins on FA-light layouts.
- Qwen3.5 dense PARO — K8V4 bit-exact vs paroquant reference, wins memory
  and TPS.
- Qwen3 dense (Bonsai 2-bit) — Mixed routes directly into
  `mx.quantized_matmul` and skips the per-step full dequant.
- Qwen2 / Laguna — safe K8V8 default (11/11 coherent at 4k bench).
- Gemma3 (medgemma) — Planar wins TPS; chat-template divergence is not a
  kernel issue.
- Gemma4 MoE (26b-a4b) — K8V8 ties Planar on TPS; prefer the universally
  validated path.
- Gemma4 small (e2b/e4b) — K8V8 baseline; PARO variant gets K8V4.
- Gemma4 dense 31b — Planar wins TPS at acceptable PPL with low GQA.
- Unknown arch — K8V8 is the cross-arch safe default.

The auto-by-context server policy (`kv_quant_for_ctx`) is a separate path
documented in `crates/rmlx-models/src/kv_cache/mod.rs`; it is invoked only
when no explicit `--kv-quant` flag is given and only by `serve`/`chat`.

---

## 7. Migration from llama.cpp

| llama.cpp                       | rMLX equivalent                                  | Notes |
|---------------------------------|--------------------------------------------------|-------|
| `-ctk q8_0`                     | `--cache-type-k q8_g32`                          | Block=32 closest match. `q8_g128` is rMLX's faster default. |
| `-ctk q4_0`                     | `--cache-type-k q4_g32`                          | Block=32 closest match (V-side only in practice).            |
| `-ctk q4_1`                     | `--cache-type-k q4_g32`                          | Closest block size; rMLX has no separate min/max codec.      |
| `-ctk q5_0`                     | `--cache-type-k q5_g64`                          | No g32 5-bit codec exists; granularity differs.              |
| `-ctk iq4_nl`                   | `--cache-type-k q4_g64`                          | rMLX has no non-linear 4-bit codec.                          |
| `-ctk f16 -ctv f16`             | `--cache-type-k bf16 --cache-type-v bf16` (or just leave default `auto`) | rMLX stores bf16 (Apple Silicon native). |
| llama.cpp asymmetric K8V4       | `--kv-quant k8v4` **or** `--ctk q8_g128 --ctv tq4` | Preset and primitives produce the same `KvQuant::K8V4`.   |

The `q8_0` / `q4_0` / `q4_1` / `q5_0` / `q5_1` / `iq4_nl` tags return a
`NotImplemented` parse error pointing at the closest rMLX equivalent.
Reason: those are llama.cpp legacy block-32 codecs with **fp16 per-block
scale** packed inside the block layout — a different on-disk layout from
either rMLX's MSL `q8_0` (group=128, no fp16-inline scale) or MLX's affine
3-tuple `(codes_u32, scales, biases)` at arbitrary group sizes.
Implementing them would require a separate codec; for now, pick the
closest rMLX codec from the table above.

---

## 8. Examples per model class

Each example uses `rmlx baseline` for a single bench cell; swap to
`rmlx serve` / `rmlx chat` to run the same configuration as a server or
REPL.

**Bonsai dense (Qwen3, 2-bit weights)** — `Mixed{k=8,v=4,group=64}` is the
auto default; the preset spelling is equivalent:

```bash
rmlx baseline \
    --model /path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit \
    --kv-quant k8v4
```

**Gemma4 dense (e4b)** — auto resolves to `K8V8`; explicit form:

```bash
rmlx baseline \
    --model /path/to/mlx-community__gemma-4-e4b-it-mxfp8 \
    --kv-quant k8v8
```

**Gemma4 MoE (26b-a4b)** — auto resolves to `K8V8`; explicit form:

```bash
rmlx baseline \
    --model /path/to/mlx-community__gemma-4-26b-a4b-it-mxfp8 \
    --kv-quant k8v8
```

**Qwen3.6 MoE (35B-A3B)** — K-side **must be ≥ 8 bits** (§5.4). Auto
resolves to `K8V8`; do NOT lower K below 8 on this family:

```bash
rmlx baseline \
    --model /path/to/mlx-community__Qwen3.6-35B-A3B-8bit \
    --kv-quant k8v8
```

To explore the V-side rotation codec via primitives on the same model:

```bash
rmlx baseline \
    --model /path/to/mlx-community__Qwen3.6-35B-A3B-8bit \
    --ctk q8_g128 --ctv planar4
```

**Qwen3 dense (non-Bonsai, e.g. affine 8-bit)** — auto resolves to `K8V8`:

```bash
rmlx baseline \
    --model /path/to/qwen3-dense-snapshot \
    --kv-quant k8v8
```

---

## 9. Failure modes

| Exit | Source             | Meaning                                            | Example trigger                                          | Try instead                                  |
|------|--------------------|----------------------------------------------------|----------------------------------------------------------|----------------------------------------------|
| 2    | clap               | Flag collision: preset and primitive both set.     | `--kv-quant k8v4 --cache-type-k q8_g128`                 | Pick one path (preset OR primitives).        |
| 1    | anyhow             | Unknown cache type tag.                            | `--ctv garbage`                                          | Use a tag from `rmlx info --list-cache-types`. |
| 1    | anyhow             | Reserved llama.cpp legacy tag.                     | `--ctv q8_0`                                             | Use the suggested rMLX substitute (§7).      |
| 78   | resolver (EX_CONFIG) | `HeadDimUnknown` — model config has no head_dim. | `--ctv q4_g64` on a model without `head_dim` declared.   | Use `--kv-quant` (preset bypasses validator). |
| 78   | resolver (EX_CONFIG) | `KSideRotationCodec`.                              | `--ctk tq4`                                              | Move the codec to V-side (`--ctv tq4`).      |
| 78   | resolver (EX_CONFIG) | `QwenMoeKBitsTooLow`.                              | `--ctk q4_g64` on Qwen MoE.                              | `--ctk q8_g128` or `--kv-quant k8v8`.         |
| 78   | resolver (EX_CONFIG) | `Tq4UnsupportedHeadDim`.                           | `--ctv tq4` on `head_dim = 64`.                          | `--ctv q4_g64` or `--ctv planar4`.            |
| 78   | resolver (EX_CONFIG) | `GroupSizeNotDivisible`.                           | `--ctv q4_g64` on `head_dim = 80`.                       | Pick a codec whose group divides head_dim.   |
| 78   | resolver (EX_CONFIG) | `MlxBitPackingViolation`.                          | `--ctv q3_g64` on `head_dim = 64`.                       | Use a higher-bit codec (`q4_g*` / `q8_g*`).  |
| 78   | resolver (EX_CONFIG) | `UnsupportedCombo`.                                | `--ctk q8_g64 --ctv tq4` (K8V4 needs K=`q8_g128`).       | Set `--ctk q8_g128` or pick affine V codec.  |

Every exit-78 message is self-describing: it names the offending input
and points at an actionable alternative.

---

## 9.5 Head-major persistent K shadow

A head-major K shadow (`KvCache::fused_qk_shadow`) sits on top of the
per-codec storage, sized to `[B, kv_h, max_seq, codes_per_token]`
(u32 codes) + `[B, kv_h, max_seq, combined_per_token]` (f32 scales).
The shadow is the input contract the fused-QK MSL kernels expect, and
is allocated **lazily on the first fused-QK decode dispatch when
`RMLX_FUSED_QK=1`**.

**Scope** — q8 (`K8V4`, `K8V8`), `TurboSym3`, `TurboSym4`. The iso
(`Iso3Sym`, `IsoKOnly3`, `Iso4Sym`, `IsoKOnly4`) and rotor (`Rotor3Sym`,
`RotorKOnly3`, `Rotor4Sym`, `RotorKOnly4`) kernel shims read a
*segregated* `[scales_all_tokens | norms_all_tokens | (rotors)]`
combined buffer plus, for rotor, a per-(layer, head) rotor table. That
shape is not expressible as a per-token shadow row, so `for_codec`
returns `Ok(None)` for iso/rotor and dispatch falls back to the legacy
bf16 SDPA path. A follow-up will split the shadow into
`{per_token, sideband_table}` and unblock iso/rotor.

State machine:

1. **Allocate** on first dispatch — zero-init `[B, kv_h, max_seq, *]`.
2. **Seed** by quantising the bf16 prefix `[B, kv_h, prev_offset, D]`
   from `decode_fp16_k` head-major into `[:, :, 0:prev_offset, :]`.
3. **Grow** every decode token by quantising the new bf16 chunk and
   `slice_update`ing at `[:, :, prev_offset:prev_offset+1, :]`.

The bf16 mirror in `decode_fp16_k/v` is **always maintained alongside**
the shadow so the legacy SDPA fallback remains the safety net. Reset on
`KvCache::reset`; truncated in-place on `KvCache::truncate_to` (no
buffer reallocation — only the `filled` cursor moves; rotating caches
never allocate a shadow, gated by `storage_max_seq_for_fused_qk`).
Per-codec shape details and the dispatch wire-in: see
`docs/KV_QUANT.md` § "Fused-QK head-major K storage".

**Per-step cost framing.** The shadow stores data head-major, but the
kernel input is built by slicing `[B, kv_h, max_seq, payload]` down to
`[B, kv_h, kv_seq, payload]` along the non-contiguous `kv_seq` axis and
then flattening. The reshape forces a per-step materialisation of
`B * kv_h * kv_seq * payload` bytes; the "head-major no-copy" framing
in earlier notes describes the *storage* layout, not the per-step
kernel-input cost. Reducing this copy (kernel reads `[B, kv_h, max_seq,
payload]` directly with an explicit `max_seq` row stride and bounds
iteration to `kv_seq`) requires touching every fused-QK MSL kernel and
is tracked as a separate optimisation.

## 9.6 Warm-TTFT bf16-K decode contract

**This is an architectural contract, not an accident.** Every quantized KV
mode in rMLX serves K **and** V from a bf16 decode mirror
(`decode_fp16_k` / `decode_fp16_v`) during normal autoregressive decode. The
per-codec quantizer runs **once at `exit_prefill`** (bulk-encode the prompt
prefix) and is then **quiescent for the whole decode window** — it is not
re-invoked per generated token. Decode-phase K and V are bf16.

### Origin

The mirror was introduced by `0806148`
(`exp(qwen3_5_moe): fused decode fp16 buffer — closes long-ctx decode TPS
gap`, 2026-05-08). Genesis numbers (Qwen3.5-MoE, K8V8): decode TPS
27.30 → 48.06 (+76%) at 128k; +53% at 64k; +37% at 16k; +23% at 4k.

> *"Per-step per-layer KV dequant was the dominant decode bottleneck at
> long context: O(N) Metal kernel launches × 64 layers × 8 KV heads.
> Replaced with persistent fp16 decode buffer pre-expanded to max_seq at
> exit_prefill, plus O(1) slice_update per step."*

The decode-time win is **not** memory-bandwidth on the cache read (a
2-bit cache is smaller to read); it is the elimination of per-token,
per-layer **encode + dequant Metal dispatches**. Quantizing one new K/V
column per step launches a kernel per layer per head; the bf16 mirror
collapses that to one `slice_update`. At long context this dominates.

### Mechanism

* `exit_prefill` quantizes the prefill K/V into the codec storage buffer
  **and** unconditionally stores a compact bf16 seed on
  `decode_fp16_k`/`decode_fp16_v` (the generic seed tail after the
  per-`KvQuant` match, `update.rs:2200`).
* Each `update_<codec>` begins with
  `if self.decode_fp16_k.is_some() { return self.update_decode_fp16(...); }`.
  Once the seed is live (always, post-prefill), the codec is bypassed and
  the decode step appends bf16 K **and** V via one `slice_update`.
* Consequence: tokens **generated during decode** are never quantized in
  the codec buffer for the shortcut family — they live only in the bf16
  mirror. The codec buffer is frozen at the prefill length. This is
  correct: the codec buffer is not consulted at decode-read time.

### Per-codec audit table

`decode-K` / `decode-V` = the dtype actually fed to SDPA per decode step.
`codec-at-decode` = does the per-codec quantizer run on each generated
token? "frozen" = no (bf16 shortcut); "K-only" = K codec runs, V bf16.

| Codec (`KvQuant`)        | shortcut? | decode-K | decode-V | codec-at-decode | intentional |
|--------------------------|-----------|----------|----------|-----------------|-------------|
| `None` (bf16)            | n/a       | bf16     | bf16     | none            | yes         |
| `K8V8`                   | yes       | bf16     | bf16     | frozen          | yes         |
| `K8V4`                   | yes       | bf16     | bf16     | frozen          | yes         |
| `Mixed{k,v}`             | yes¹      | bf16     | bf16     | frozen          | yes         |
| `Planar` / `Planar3`     | yes       | bf16     | bf16     | frozen          | yes         |
| `PlanarK`                | yes²      | bf16     | bf16     | frozen          | yes         |
| `K8VTurbo2/3` (+`Tcq`)   | yes       | bf16     | bf16     | frozen          | yes         |
| `TurboSym3` / `TurboSym4`| yes       | bf16     | bf16     | frozen          | yes         |
| `Iso3` / `Iso4`          | yes       | bf16     | bf16     | frozen          | yes         |
| `Iso3Sym` / `Iso4Sym`    | yes       | bf16     | bf16     | frozen          | yes         |
| `Rotor3` / `Rotor4`      | yes       | bf16     | bf16     | frozen          | yes         |
| `Rotor3Sym` / `Rotor4Sym`| yes       | bf16     | bf16     | frozen          | yes         |
| `RotorK{3,4}Asym`        | yes       | bf16     | bf16     | frozen          | yes         |
| `IsoKOnly3` / `IsoKOnly4`| **no**    | **quant**| bf16     | **K-only**      | yes³        |
| `RotorKOnly3`/`KOnly4`   | **no**    | **quant**| bf16     | **K-only**      | yes³        |

1. `Mixed`/`RotKTq4V` are driven through `update_and_sdpa` (the direct
   `update()` arm errors); the bf16 mirror is surfaced to cross-layer-KV
   consumers via `update_and_sdpa_shared_source` (see §10.2 Mixed note).
2. PlanarK was the **sole** codec missing the shortcut; it re-encoded K
   through lossy Lloyd-Max + Givens every decode step, which broke
   `niah_pflash_bonsai_8k_d50` retrieval. The fix (`b1d9dca`) restored the
   shortcut to match its siblings. Regression-locked by
   `warm_ttft_tests.rs`.
3. The K-only family deliberately keeps K quantized at decode (the K codec
   has no bf16 shortcut in its body) so that the K-side reduction it exists
   to provide is actually realized. V rides the bf16 mirror via
   `update_decode_fp16_v_only` (which must NOT touch `decode_fp16_k`, or it
   would silently re-arm the shortcut and drop K back to bf16). Because the
   K-only decode body never reads `decode_fp16_k`, `exit_prefill` no longer
   materialises the bf16 K seed for these variants (gated on
   `feeds_bf16_k_at_decode()`); only the bf16 V seed is kept. See the F2
   reclaim note below.

**F2 (RAM-only reclaim).** Previously, `exit_prefill` populated `decode_fp16_k`
for **every** quant arm, including the K-only codecs whose `update_<arch>` never
reads it. For `IsoKOnly*` / `RotorKOnly*` the bf16 K seed was allocated at
prefill-end and then held, unused, for the entire decode window — wasted RAM
equal to one bf16 K buffer per K-only layer.

**Fix.** `exit_prefill` now gates the K-seed materialisation (clone + eval **and**
store) on the predicate [`KvQuant::feeds_bf16_k_at_decode()`], which is `false`
for the K-only family (`IsoKOnly3/4`, `RotorKOnly3/4`) and `true` for every
shortcut codec. The `KvStorage::None` bf16-fallback path always forces the K seed
(it reads bf16 K at decode regardless of `self.quant`). The bf16 **V** seed stays
unconditional — the K-only decode path consumes it via
`update_decode_fp16_v_only`. Pure RAM reclaim: decode behaviour and output are
**byte-unchanged** (verified on Bonsai-8B-2bit `k_iso3`, base vs branch decoded
string identical, 2026-06-03). `resident_bytes` counts the K and V seeds
independently and so reflects the dropped buffer. Residency reclaimed for Bonsai
(`num_kv_heads=8`, `head_dim=128`, 36 layers): 288 MiB @ 4k ctx, 576 MiB @ 8k,
1152 MiB @ 16k (= `seq × kv_h × head_dim × 2 B × n_layers`). Pinned by
`warm_ttft_cross_codec_tests::iso_k_only3_quant_at_decode`, which asserts the
seed is **absent** and that the reported total is the K store plus the surviving
V seed and nothing else. `resident_bytes` is the **only** byte diagnostic: it
reads actual buffer sizes rather than re-deriving from shape fields.

### Decision: keep warm-TTFT universal

Verified, not assumed. Real-model proof (Bonsai-8B-2bit, `longctx_4k`
prompt, `ctx_max=8192`, `max_tokens=64`, 1 warmup + 3 measured, GPU,
2026-06-03, `release-perf`):

| KV mode | decode_tps (3 runs)        | median |
|---------|----------------------------|--------|
| `none`  | 95.98 / 94.70 / 96.39      | 95.98  |
| `k8v4`  | 94.01 / 95.48 / 95.39      | 95.39  |
| `planar`| 95.56 / 95.65 / 95.53      | 95.55  |

All three are within ~1% — exactly the warm-TTFT prediction: every mode
reads bf16 K+V at decode, so the codec adds **zero** per-step decode cost.
The quant differs only at prefill (encode) and in RAM footprint.
Coherence parity confirmed end-to-end: `/v1/chat/completions` "capital of
France" returns "Paris" identically under `k8v4` and `none`.

Forcing **quant-at-decode** universally would (a) re-introduce the per-token
per-layer dispatch storm the mirror was built to remove (the +76%@128k
regression, in reverse) and (b) for lossy K codecs, compound per-position K
drift across the softmax tail — the exact failure caught on PlanarK
(NIAH retrieval break). The memory-bandwidth win of reading a smaller cache
does not pay for the per-step encode/dequant dispatch cost except possibly
in a 32k+ kv_seq regime, and even there the correctness risk on lossy K
makes it a per-codec opt-in, not a default. **Warm-TTFT stays universal.**

### Where the codec kernels DO fire (seedless paths)

The per-codec decode kernels (and the fused-QK / flash-decode
dispatchers) are **not** dead — they fire on **seedless** caches, i.e. any
decode-firing path where `decode_fp16_k` is `None`:

* The K-only family (`IsoKOnly*`, `RotorKOnly*`) — K codec runs every
  decode step by design (it ignores the bf16 K seed).
* Speculative / draft-model decode and PPL-eval fixtures that drive
  `update()` without an `exit_prefill` seed.
* Prompt-cache hydration paths that restore codec state without re-seeding
  the bf16 mirror.

This is why `--sparse-attn` and `--fused-qk` resolve OFF ("warm-TTFT dormant")
under Auto: in the normal generate flow the seed is always live, so those
kernels would never dispatch. They are gated for the seedless workloads above,
not normal generation. The fused-QK / flash-decode perf gates must be read as
**seedless-path** gates, not normal-decode gates.

### GPU-resident V mirror analysis (iso3 V)

Settled by code + `warm_ttft_cross_codec_tests`: `update_iso3` and
`update_iso3_sym` (the codecs where the GPU-resident `QuantIsoV3` mirror's
`append_gpu`/`dequant_gpu` live) both begin with the
`decode_fp16_k.is_some()` shortcut. In normal generate the seed is live, so
their per-step `append_gpu`/`dequant_gpu` **never runs** — the iso3 V encode
fires only once at `exit_prefill` (a single bulk slice) or on a seedless
cache. A GPU-resident V mirror therefore **cannot** yield a ≥10% per-token
TTFT win on iso3 V in the warm-TTFT decode loop; its only beneficiaries are
the one-shot exit_prefill encode and seedless decode. **Result: NO gain**
for the normal decode path.

## 10. Bench reference

Decode TPS (warm) for every cache-type combo, measured on the
`longctx_4k` prompt with `ctx_max=8192`, `max_tokens=32`, 2 warmup + 3
measured runs per cell, median over the measured set. Sample stddev shown
in parentheses. `ttft_ms` is the warm-prefill TTFT for the median run.

Combo column lists the (`--ctk` / `--ctv`) primitive pair; the `kv_quant`
column shows the canonical preset string the resolver picks for that pair
(see §4.4 and §6). Preset-form rows (e.g. `k8v8`) test the `--kv-quant`
spelling — they share `kv_quant` with the equivalent primitive-form row.

Status legend:
- **ok** — measured, decode TPS reported.
- **skip** — resolver rejected the combo at startup (exit 78). See §5
  for the rule, §9 for the error mapping.
- **runtime_fail** — combo loaded but crashed at decode time. Bug, not
  a config error. (The Gemma4+Mixed rows were previously **skip** under the
  old startup guard; they are now **ok** — see the Mixed notes in §10.2/§10.3.)

Rows ordered by decode_tps_warm descending. The `auto` row is pinned to
the top for reference — it is the resolver default for that family.

### 10.1 Bonsai-8B-2bit (`Qwen3ForCausalLM`, weights = 2bit)

| Combo                | kv_quant              | decode_tps_warm (stddev) | ttft_ms | Status |
|----------------------|-----------------------|--------------------------|---------|--------|
| `auto`               | `k8v8`                | 15.88 (±0.235)           | 65      | ok     |
| `q4_g64`  / `q4_g64` | `mixed_k4g64_v4g64`   | 15.82 (±0.167)           | 64      | ok     |
| `q8_g128` / `q8_g32` | `mixed_k8g128_v8g32`  | 15.78 (±0.088)           | 64      | ok     |
| `bf16`    / `bf16`   | `none`                | 15.41 (±0.012)           | 65      | ok     |
| `q8_g128` / `tq4`    | `k8v4`                | 15.34 (±0.085)           | 65      | ok     |
| `q8_g128` / `q4_g128`| `mixed_k8g128_v4g128` | 15.28 (±0.039)           | 65      | ok     |
| `q8_g128` / `planar4`| `planar`              | 15.19 (±0.271)           | 68      | ok     |
| `q8_g128` / `q8_g64` | `mixed_k8g128_v8g64`  | 14.79 (±0.139)           | 68      | ok     |
| `k8v4` preset        | `k8v4`                | 14.71 (±0.208)           | 68      | ok     |
| `q8_g128` / `q4_g64` | `mixed_k8g128_v4g64`  | 14.61 (±0.077)           | 69      | ok     |
| `k8v8` preset        | `k8v8`                | 14.52 (±0.262)           | 70      | ok     |
| `planar` preset      | `planar`              | 14.18 (±0.263)           | 72      | ok     |
| `q8_g128` / `q6_g64` | —                     | —                        | —       | skip   |

The `q6_g64` V-codec is rejected by `MlxBitPackingViolation` on
`head_dim = 64` (Bonsai). Decode-TPS spread across ok rows is ~12% top
to bottom; `auto` ties the top cluster within stddev.

### 10.2 Gemma4-e4b-mxfp8 (`Gemma4ForConditionalGeneration`, dense, hidden_size = 2560)

| Combo                | kv_quant              | decode_tps_warm (stddev) | ttft_ms | Status       |
|----------------------|-----------------------|--------------------------|---------|--------------|
| `auto`               | `k8v8`                | 27.50 (±0.124)           | 36      | ok           |
| `q8_g128` / `planar4`| `planar`              | 27.57 (±0.066)           | 36      | ok           |
| `bf16`    / `bf16`   | `none`                | 27.54 (±0.086)           | 36      | ok           |
| `k8v4` preset        | `k8v4`                | 27.50 (±0.174)           | 37      | ok           |
| `k8v8` preset        | `k8v8`                | 27.48 (±0.229)           | 36      | ok           |
| `planar` preset      | `planar`              | 27.44 (±0.153)           | 36      | ok           |
| `q8_g128` / `q4_g64` | `mixed_k8g128_v4g64`  | ~75 (short prompt)       | 66      | ok           |
| `q8_g128` / `q8_g64` | `mixed_k8g128_v8g64`  | (supported)              | —       | ok           |
| `q8_g128` / `q4_g128`| `mixed_k8g128_v4g128` | (supported)              | —       | ok           |
| `q8_g128` / `q8_g32` | `mixed_k8g128_v8g32`  | (supported)              | —       | ok           |
| `q4_g64`  / `q4_g64` | `mixed_k4g64_v4g64`   | (supported)              | —       | ok           |
| `q8_g128` / `q6_g64` | —                     | —                        | —       | skip         |

**Mixed is now supported on Gemma4 via dequant-before-share.** The former startup
rejection is gone. `update_and_sdpa_shared_source` now surfaces the accumulated
**bf16** K/V (prefill-raw buffer during prefill, the maintained `decode_fp16`
accumulator during decode — the same tensors the fused quantized SDPA was computed
from) to the cross-layer-KV consumer layers. Verified coherent: e4b
`--kv-quant mixed` (`mixed_k8g64_v4g64`) generates correct text at ~75 decode TPS
(short prompt, on par with `k8v8` ~74). The `q6_g64` V-codec is still rejected by
`MlxBitPackingViolation` on `head_dim = 64`. (Note: the default 10.8K-token
benchmark prompt currently fails chunked prefill on e4b for ALL KV quants incl.
`k8v8` — a pre-existing long-context SWA mask bug.) `auto = k8v8` remains the
conservative default pick.

### 10.3 Gemma4-26b-a4b-mxfp8 (`Gemma4ForConditionalGeneration`, MoE, has_moe = true)

| Combo                | kv_quant              | decode_tps_warm (stddev) | ttft_ms | Status       |
|----------------------|-----------------------|--------------------------|---------|--------------|
| `auto`               | `k8v8`                | 3.47 (±0.026)            | 288     | ok           |
| `q8_g128` / `planar4`| `planar`              | 3.46 (±0.034)            | 290     | ok           |
| `k8v8` preset        | `k8v8`                | 3.28 (±0.008)            | 304     | ok           |
| `planar` preset      | `planar`              | 3.28 (±0.013)            | 306     | ok           |
| `k8v4` preset        | `k8v4`                | 3.24 (±0.009)            | 309     | ok           |
| `bf16`    / `bf16`   | `none`                | 3.11 (±0.015)            | 322     | ok           |
| `q8_g128` / `q4_g64` | `mixed_k8g128_v4g64`  | ~80 (short prompt)       | 384     | ok           |
| `q8_g128` / `q8_g64` | `mixed_k8g128_v8g64`  | (supported)              | —       | ok           |
| `q8_g128` / `q4_g128`| `mixed_k8g128_v4g128` | (supported)              | —       | ok           |
| `q8_g128` / `q8_g32` | `mixed_k8g128_v8g32`  | (supported)              | —       | ok           |
| `q4_g64`  / `q4_g64` | `mixed_k4g64_v4g64`   | (supported)              | —       | ok           |
| `q8_g128` / `q6_g64` | —                     | —                        | —       | skip         |

**Mixed is now supported on Gemma4-26b too** (same dequant-before-share
mechanism as e4b). Verified coherent: `--kv-quant mixed` (`mixed_k8g64_v4g64`)
generates correct text at ~80 decode TPS on a short prompt. `auto = k8v8`
remains the conservative default; the asymmetric primitive form
(`q8_g128`/`planar4`) ties within stddev. `bf16` baseline costs ~10% TPS and
12% TTFT against `k8v8`. (Long-context numbers omitted — the §10.2 long-context
chunked-prefill caveat applies to all KV quants on the Gemma4 family.)

### 10.4 Qwen3.6-35B-A3B-8bit (`Qwen3_5MoeForConditionalGeneration`, MoE)

| Combo                | kv_quant              | decode_tps_warm (stddev) | ttft_ms | Status |
|----------------------|-----------------------|--------------------------|---------|--------|
| `auto`               | `k8v8`                | 4.46 (±0.017)            | 224     | ok     |
| `q8_g128` / `q8_g64` | `mixed_k8g128_v8g64`  | 4.45 (±0.015)            | 225     | ok     |
| `q8_g128` / `q4_g128`| `mixed_k8g128_v4g128` | 4.45 (±0.242)            | 225     | ok     |
| `bf16`    / `bf16`   | `none`                | 4.44 (±0.018)            | 224     | ok     |
| `q8_g128` / `q4_g64` | `mixed_k8g128_v4g64`  | 4.40 (±0.008)            | 227     | ok     |
| `k8v8` preset        | `k8v8`                | 4.37 (±0.020)            | 230     | ok     |
| `planar` preset      | `planar`              | 4.35 (±0.034)            | 231     | ok     |
| `q8_g128` / `planar4`| `planar`              | 4.28 (±0.020)            | 234     | ok     |
| `q8_g128` / `tq4`    | `k8v4`                | 4.22 (±0.032)            | 234     | ok     |
| `q8_g128` / `q8_g32` | `mixed_k8g128_v8g32`  | 4.11 (±0.220)            | 244     | ok     |
| `k8v4` preset        | `k8v4`                | 4.10 (±0.038)            | 246     | ok     |
| `q4_g64`  / `q4_g64` | —                     | —                        | —       | skip   |
| `q8_g128` / `q6_g64` | —                     | —                        | —       | skip   |

K-side < 8 bits is rejected by `QwenMoeKBitsTooLow` (§5.4) — that is why
the symmetric `q4_g64`/`q4_g64` row shows as skip on this family but
runs on Bonsai. Top cluster (auto, the two asymmetric mixed pairs, and
bf16) is within 0.5%; pick `auto = k8v8` unless memory is the gate.

---

### 10.5 Why are the §10 decode_tps numbers so low? (prefill-contamination note)

The `decode_tps_warm` cells in §10.1–§10.4 were measured with the **old
combined-TPS metric**: `rmlx baseline` (before the decode-only timing fix)
started its clock before `generate_greedy`, so the denominator was
`prefill + decode`. For the MoE models (Gemma4-26b, Qwen3.6-35B) whose
4k-token prefill takes 8–10 s against ~1.4 s of decode at max_tokens=32,
that buries the true decode rate — the §10.3/§10.4 cells (3.47 and 4.46
TPS) are dominated by prefill latency, **not** decode speed.

The corrected **decode-only** numbers and their derivation are in
[`docs/PERF_BASELINE.md`](PERF_BASELINE.md). Short summary: once prefill is
excluded, all four models
run at **1.8–2.7× the 614 GB/s bandwidth ceiling**, which is the normal
batch-1 band that llama.cpp and mlx-lm hit on dense models — there is no
inference defect. The "headline" decode-only figures are:

| Model | §10 cell (combined) | decode-only (PERF_BASELINE.md) | ratio vs ceiling |
|---|---:|---:|---|
| Bonsai 8B 2bit | 15.88 TPS | ~110 TPS | ~2.7x |
| Gemma4-e4b mxfp8 | 27.50 TPS | ~74 TPS | ~2.1x |
| Gemma4-26b MoE | 3.47 TPS | ~72 TPS | ~2.4x |
| Qwen3.6-35B MoE | 4.46 TPS | ~96 TPS | ~1.8x |

For steady-state decode reasoning — picking a KV quant, estimating
throughput, comparing combos — use the **decode-only** TPS from
`docs/PERF_BASELINE.md`, not the §10 cells. The §10 matrix is still useful
for **relative** comparison across combos (the ratio between cells is
unaffected by the fixed prefill constant being added to each denominator),
and for TTFT comparisons (unaffected by the TPS metric).

The `scripts/perf_canary.sh` canary anchors are the forward-looking baseline:
Bonsai ~110, Gemma4-e4b ~74, Qwen3.6 ~97 decode-only TPS under `release-perf`.
Future measurements against those anchors are decode-only by construction.

---

### Methodology + reproduction

All cells in §10.1–§10.4 share the partition key
`(prompt_id = longctx_4k, ctx_max = 8192, warmup = 2, measured = 3,
median)`. This is distinct from the historic `(longctx_8k, ctx_max =
16384, 2-run cold+warm)` partition that populates older rows of
`BENCHMARK_CHAMPIONS.md` — the SQLite `bests` view keys on the full 8-tuple
so both partitions coexist without collision.

Per-cell raw data lives in `metrics/runs.db` under the
`ctype-<combo>` label naming convention; query via:

```bash
sqlite3 .rmlx/metrics/runs.db \
  "SELECT model, kv_quant, value, decode_stddev, notes
     FROM observations
    WHERE metric='decode_tps_warm'
      AND notes LIKE '%ctype-%';"
```

Companion KV-bytes growth measurements for Qwen3.6 (a separate
multi-turn experiment, not the §10 cache-type matrix) live in
`.rmlx/bench/qwen36_multiturn_kvbytes.csv` (local-only; gitignored.
Regenerate via `.rmlx/bench/qwen36_multiturn.sh`).

---

## 11. How to bench your own combo

Two paths.

**Quick — single cell via `rmlx baseline`**:

```bash
rmlx baseline \
    --model /path/to/snapshot \
    --cache-type-k q8_g128 \
    --cache-type-v q4_g64 \
    --prompt-tokens 4096 \
    --gen-tokens 32 \
    --record \
    --label my-test
```

`--record` writes a single observation row to `metrics/runs.db`; the
`--label` value is preserved on the row for filtering. The bench runs
2 warmup + 3 measured per cell (D7 methodology).

**Matrix — sweep multiple combos via the bench cell script**:

```bash
CTK=q8_g128 \
CTV=q4_g64 \
PROMPT_ID=longctx_4k \
MAX_TOKENS=32 \
CTX_MAX=8192 \
WARMUP_RUNS=2 \
MEASURED_RUNS=3 \
./scripts/bench_cell.sh /path/to/snapshot
```

Loop over `CTK` / `CTV` values in a shell `for` to fill a matrix.
`scripts/bench_cell.sh` writes one buffer file per cell into
`metrics/buffer/pending/` and invokes `rmlx metrics record --file <path>`
on success.

After a sweep, regenerate the records table:

```bash
rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md
```

See `docs/METRICS_DB.md` for the full DB operating rules (cell identity,
prompt registry, `bests` view, query API).

---

## 12. Further reading

- llama.cpp 4-bit KV cache discussion: <https://github.com/ggml-org/llama.cpp/pull/5932>
- MLX `mx.quantize` API reference: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html>
- ParoQuant (z-lab): <https://github.com/z-lab/paroquant>
- IsoQuant (ParaMind2025): <https://github.com/ParaMind2025/isoquant>
- Metrics database operating rules: `docs/METRICS_DB.md`
- Profiling runbook (samply, Instruments, dhat): `docs/PROFILING.md`
