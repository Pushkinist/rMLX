# Adding a Model

This guide lists the integration points a new **text** architecture touches.
It covers the generative path only. Vision and audio towers and the
`/v1/embeddings` encoder path are not covered; `docs/MODELS.md` lists what
each architecture supports.

## Integration points

Per-arch source lives under `crates/rmlx-models/src/<arch>/`. `qwen3.rs` is
one file instead of a directory. Shared seams live at the crate root and
under `layers/`.

1. **Config parse**, `<arch>/config.rs`. Deserialize the `config.json`
   block: quant overrides, RoPE and sliding-window parameters.
2. **Model and layers**, `<arch>/model.rs` and `<arch>/layers.rs` (or a
   `layers/` directory). The forward math is per-arch.
   - Attention goes through `KvCache::update_and_sdpa` (`rmlx-kv-quant`).
   - Masks come from `layers/mask.rs`: `build_chunked_prefill_mask`,
     `build_swa_prefill_mask`, `pick_attn_mask_mode`.
   - `layers/` also holds shared `Linear`, `Embedding`, `RmsNorm`, `Mlp` and
     `MoeBlock` types. An arch whose layer differs keeps its own type.
   - The model struct carries `kv_bytes: KvBytesCounter` and
     `model_sig: u64`, per instance, never static. The loader sets both.
     The generate path folds `model_sig` into `cache_seed`. That keeps the
     prompt cache and the SSD tier from serving one model's K/V to another.
3. **Loader**, `<arch>/loader.rs`. Fetch tensors through `Weights`
   (`load_util.rs`) and pick each tensor's quant with `resolve_quant`
   (`layers/quant.rs`). Add per-arch wiring (MoE expert stacking, shared K/V
   heads, ParoQuant rotations) only where the checkpoint has it.
4. **Generate**, `<arch>/generate.rs` (or a `generate/` directory). Choose
   the prompt-cache policy, build the caches, and hand a `forward_step`
   closure to the shared decode loop.
   - Build the per-layer codec vector with
     `kv_cache::kv_layer_quants(n_layers, kv_quant, shares_kv)`. The SSD
     `layout_key` and `cache_seed` describe that vector.
   - `make check-kv-layer-quants` fails on a `kv_quant_for_layer` call
     outside `kv_cache/`. A deliberately uniform stack carries a
     `// kv-layer-quants: uniform — <reason>` marker instead.
   - Every prefill path calls `reject_nan_prefill` on its logit row before
     it picks a token (see below).
5. **Prompt cache** (optional), `<arch>/prompt_cache.rs`. Write an `Entry`
   struct and its `PromptCacheEntry` accessors; `kv_bytes`, `truncate_kv_to`
   and the SSD spill come with the trait. For SSD hydrate, implement
   `HydratedEntry` (`rmlx-kv-ssd`): one `const SHARES_KV` and one
   `from_hydrated`.
6. **Enum variant**, `arch/mod.rs`. Add one `Architecture` variant and its
   arms in the methods that match on it (`forward_seq`, `Debug`,
   `config_summary` and the rest).
7. **Registry**, `arch/registry.rs` and `arch/loader.rs`. Add
   `architectures[0]` to `KNOWN_ARCHS` and a `load_model` arm. When one
   arch string covers several checkpoint shapes, branch on checkpoint facts
   such as `cfg.is_paroquant()`, not on the string.
8. **SSD attach** (optional), `ssd_tier.rs`. Add one arm to `attach_at_load`
   for the arch's prompt cache.
9. **Prefill chunk**, `prefill_chunk.rs`. Add a row to `arch_default`, or
   take the 64-token fallback.
10. **Server features** (optional), in `rmlx-server`: `tool_parser.rs` for
    tool calls, `engine/think.rs` for thinking tags,
    `engine/arch_generator.rs` for image prompts.

`bitnet`, `laguna`, `qwen2` and `qwen3_vl_moe` keep their own decode loop.
`qwen3_vl_moe` uses the shared `chunked_prefill`.

## Shared seams

- `crates/rmlx-models/src/decode_loop.rs`:
  - `pipelined_decode(ctx, first_id, steps, forward_step)`, with
    `forward_step: impl FnMut(&Array) -> Result<Array>`.
  - `chunked_prefill(caches, ids, resolved_chunk, device, arch,
    forward_chunk)` returns the last-position logits.
  - `choose_token(ctx, logits_flat, mask_active)` applies sampling,
    penalties and the constraint mask.
  - `reject_nan_prefill(arch, dtype, nan_count, max_abs_logit,
    prompt_len)` refuses a prefill row with NaN, and a row whose dtype the
    NaN scan cannot read. Call it after counting NaN and before
    `choose_token`. Pass the row's real dtype.
- Speculative drivers call
  `speculative::guard_verifier_prefill_logits(verifier, logits, prompt_len)`
  at the same point.
- `crates/rmlx-models/src/load_util.rs`: `Weights::new(shards, idx)` and
  `Weights::scan_only(shards)`; `.array`, `.has`, `.raw`, `.linear`; and
  `.resolve_prefix(candidates, witness)`, which picks the tensor-name prefix
  a checkpoint uses.
- `crates/rmlx-models/src/layers/quant.rs`:
  `resolve_quant(tensor_name, has_biases, defaults, overrides)`.
- `SsdHydrator::lookup_seeded` (`crates/rmlx-kv-ssd/src/hydrate.rs`) holds
  the SSD seed formula.
- Spill: `impl<E: PromptCacheEntry> SpillSink<E> for SsdSpiller` in
  `crates/rmlx-models/src/prompt_cache.rs`.
- Hydrate: `impl<E: HydratedEntry> SsdHydrate<E> for SsdHydrator` in
  `crates/rmlx-kv-ssd/src/traits.rs`.

## Verification

1. **Smoke-probe the snapshot** (`CLAUDE.md` hard rule 6; flags in
   `docs/CLI.md`). `arch::load_model` refuses affine `bits` outside
   `rmlx_quant::affine::SUPPORTED_BITS` before any weight loads, for every
   arch. A new quant mode may need its own check; see `docs/WEIGHT_QUANTS.md`
   §4.4.
2. **Add a golden-token test** under `crates/rmlx-models/tests/` and add it to
   the `make model-check-full` list. Run `make model-check-full
   MODEL=<snapshot>`.
3. **Update the NaN gate.** `make check-no-decode-swallow` RULE 4 pins the
   number of `count_nan_in_bytes` and guard call sites. Each must propagate
   before `step_fn`. A new arch moves both counts; update them in
   `scripts/check_no_decode_swallow.sh`.
4. **Pass the f32-leak gate**, `make check-no-scalar-f32-leak`. It scans
   `rmlx-kv-quant`, `rmlx-models` and `rmlx-mlx` source, minus `laguna/`,
   for `scalar_f32(` with no guard. An f32 scalar combined with a bf16
   activation promotes the result to f32. Guard it with a non-f32 cast on
   the same chain:
   ```rust
   scalar_f32(x).astype(operand.dtype(), device)?
   ```
   A genuinely f32-only scalar carries `// f32-ok: <reason>` on its line or
   in the comment block directly above.
5. **Document it** in `docs/MODELS.md`. The KV flags are in
   `docs/KV_QUANT.md`, the prompt cache in `docs/PROMPT_CACHE.md`, the SSD
   tier in `docs/SSD_TIER.md`, bench rows in `docs/METRICS_DB.md`.
