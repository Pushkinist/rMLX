# Adding a Model

This guide lists every integration point a new **text** architecture touches
after the model-agnostic refactor. Arch-specific forward math stays per-arch by
design (see the standing guardrails at the bottom); the shared seams below
remove the copy-paste that used to dominate a new arch.

It covers the generative path only. Vision / audio towers and the
`/v1/embeddings` encoder path are additive and out of scope here — see
`docs/MODELS.md` for per-arch modality coverage.

---

## Integration points

Per-arch source lives under `crates/rmlx-models/src/<arch>/` (some single-file
archs — e.g. `qwen3.rs` — keep everything in one module instead of a directory).
Shared seams live at the crate root (`crates/rmlx-models/src/`) and under
`layers/`. The ten points a new generative arch touches:

| # | Integration point | File | What you write |
|---|---|---|---|
| 1 | Config parse | `<arch>/config.rs` | Arch-specific (genuine) — deserialize the `config.json` block, quant overrides, RoPE/SWA params. |
| 2 | Model + layers | `<arch>/model.rs`, `<arch>/layers.rs` (or a `<arch>/layers/` dir) | Arch-specific forward math. Attention goes through `KvCache::update_and_sdpa` (`rmlx-kv-quant`); build masks with `layers/mask.rs` (`build_chunked_prefill_mask`, `build_swa_prefill_mask`, `pick_attn_mask_mode`). The model struct also carries two shared fields: `kv_bytes: KvBytesCounter` and `model_sig: u64`. Both are set by the loader row, written/read by the generate row and the enum-variant row. Per instance, never a static — the KV-byte counter cross-attributes two resident models of the same arch, and `model_sig` is what keeps the shared per-arch prompt cache from serving one model's K/V to the other. |
| 3 | Loader | `<arch>/loader.rs` | Thin: `Weights` (`load_util.rs`) for tensor fetch + `resolve_quant` (`layers/quant.rs`) for the per-tensor quant rule + per-arch wiring (MoE expert stacking, `k == v` head sharing, PARO rotation parts) only where present. |
| 4 | Generate | `<arch>/generate.rs` (or a `<arch>/generate/` dir) | Prompt-cache policy + cache construction + a `forward_step` closure handed to the shared `decode_loop` (`pipelined_decode` / `chunked_prefill` / `choose_token`). No decode-loop copy. |
| 5 | Prompt cache (optional) | `<arch>/prompt_cache.rs` | `Entry` struct + accessor one-liners impl'ing `PromptCacheEntry`; `kv_bytes` / `truncate_kv_to` and the SSD spill path are inherited. Hydrate = `SsdHydrator::lookup_seeded` + a struct literal. |
| 6 | Enum variant | `arch/mod.rs` | One `Architecture` variant + the match arms (`forward_seq`, `Debug`, config summary, …). |
| 7 | Registry string | `arch/registry.rs`, `arch/loader.rs` | ~2 lines: add `architectures[0]` to `KNOWN_ARCHS` and a `load_model` match arm. When one arch string covers several checkpoint shapes (quant codec, dense vs MoE), branch the match arm on **checkpoint facts** — e.g. `cfg.is_paroquant()`, tensor presence — not on the arch string alone. The Qwen3.5 arm routes PARO vs standard this way. |
| 8 | SSD attach (optional) | `ssd_tier.rs` | 1 match arm wiring the arch's `PROMPT_CACHE` into `attach_at_load`. |
| 9 | Prefill chunk default | `prefill_chunk.rs` | 1 row in `arch_default` (omit to take the 64-token fallback). |
| 10 | Tool / think / vision flags (optional) | `rmlx-server`: `tool_parser.rs`, `engine/think.rs`, `engine/arch_generator.rs` | Only if supported. Tool-call extraction, thinking-tag splitting, image-prompt assembly. |

### Verified shared-seam signatures

These are the seams a new arch leans on instead of re-typing. Symbols current
as of this writing; confirm against the tree before relying on an exact arg
list.

- `decode_loop.rs` (`crates/rmlx-models/src/decode_loop.rs`):
  - `pipelined_decode(ctx, first_id, steps, forward_step)` — `forward_step: impl FnMut(&Array) -> Result<Array>`. The decode loop; you supply only the per-step forward.
  - `chunked_prefill(...)` — chunked prompt prefill returning last-position logits.
  - `choose_token(ctx, logits_flat, mask_active)` — sampling / penalties / constraint-mask token pick.
- `load_util.rs` (`crates/rmlx-models/src/load_util.rs`):
  - `Weights::new(shards, idx)` / `Weights::scan_only(shards)` — tensor-fetch handle.
  - `.array(name)`, `.has(name)`, `.raw(name)`, `.linear(...)` — fetch / probe / dequant helpers (replaces hand-rolled `load_array` / `embed` / `linear`).
  - `.resolve_prefix(candidates, witness) -> Result<String>` — pick the tensor-name prefix the checkpoint actually uses (e.g. `language_model.model` vs `model.language_model`) by probing shard headers for `<prefix>.<witness>`. Use this instead of hardcoding a prefix per loader when an arch ships under more than one layout.
- `layers/quant.rs` (`crates/rmlx-models/src/layers/quant.rs`):
  - `resolve_quant(tensor_name, has_biases, defaults, overrides) -> Result<QuantParams>` — the shared `.biases`-sibling / affine rule. `QuantParams`, `QuantMode` live here too.
- `lookup_seeded` lives in **`rmlx-kv-ssd`** (`crates/rmlx-kv-ssd/src/hydrate.rs`, `SsdHydrator::lookup_seeded`) — single-sources the FNV seed so no arch re-types the seed formula.
- The blanket `impl<E: PromptCacheEntry> SpillSink<E> for SsdSpiller` lives in `crates/rmlx-models/src/prompt_cache.rs` — one spill impl covers every arch's `Entry`.

---

## What the refactor removed

Versus the pre-refactor era, a new arch no longer hand-copies:

- A **~200-LOC decode loop**, previously re-pasted across each prompt-cache
  path (exact-hit / partial-hit / cold-miss) per arch. Now one `forward_step`
  closure into `pipelined_decode` / `chunked_prefill`.
- **~40 LOC of spill + ~25 LOC of hydrate + truncate** impls per arch. Spill is
  the blanket `SpillSink<E>`; truncate / `kv_bytes` are defaulted on
  `PromptCacheEntry`; hydrate collapses to `lookup_seeded` + a struct literal.
- **~130 LOC of `load_array` / `embed` / `linear` boilerplate** per loader,
  now `Weights` accessors.
- The **FNV seed formula**, previously re-typed from a sibling arch and a
  silent drift risk. Now single-sourced by `lookup_seeded`.

Numbers are approximate (`~`) — they index the scale of the copy-paste, not an
exact line budget.

---

## Verification ritual

Run these in order before declaring a new arch done.

1. **Smoke-probe the snapshot.** Short generation, reject incoherent output —
   CLAUDE.md hard rule 6. Every new snapshot / quant gets a smoke probe before
   it enters the registry. Surface and flags: `docs/CLI.md`. Before the smoke
   probe even loads weights, `arch::load_model` (point 7) pre-flights the
   declared affine `bits` against `rmlx_quant::affine::SUPPORTED_BITS` — a new
   arch needs no extra code for this; it fires for every arch string
   uniformly. Only add to it if a new *quant mode* (not arch) introduces its
   own variable-bit-width kernel matrix — see `docs/WEIGHT_QUANTS.md` §4.4.
2. **Add a golden-token test.** Temperature-0, byte-identical output against a
   recorded reference. Gate it with:
   ```
   make model-check-full MODEL=<snapshot>
   ```
   Each golden reads `config.json` and skips gracefully when the arch does not
   match, so the target stays green for any single test-target model.
3. **Add the registry row.** `KNOWN_ARCHS` + the `load_model` match arm (point
   7 above), so the server can validate the arch before paying weight-I/O cost.
   Per-arch support details: `docs/MODELS.md`. KV-quant defaults and the
   flag surface: `docs/KV_QUANT.md`. Prompt-cache reuse policy:
   `docs/PROMPT_CACHE.md`. SSD-tier attach: `docs/SSD_TIER.md`.
4. **Record a `BENCHMARK_CHAMPIONS.md` cell.** Land a bench row through the
   metrics DB and regenerate the markdown:
   ```
   rmlx metrics export --markdown
   ```
   Never hand-edit `BENCHMARK_CHAMPIONS.md`. Ingest path, schema, and operating
   rules: `docs/METRICS_DB.md`.
5. **Pass the f32-leak gate.** The CI gate `make check-no-scalar-f32-leak` scans
   every file under `crates/rmlx-models/src/` for unguarded `scalar_f32(` calls.
   A `scalar_f32(x)` combined with a BF16 activation silently upcasts the
   residual stream, Q/K/V tensors, and the KV cache to F32 — this class of bug
   has shipped multiple times and is invisible at review.

   **Canonical form for scalars that enter the activation stream:**
   ```rust
   scalar_f32(x).astype(operand.dtype(), device)?
   ```
   The `.astype(` call may appear on the same line as `scalar_f32(` or anywhere
   in the immediately following continuation lines of the same method chain
   (lines that do not yet end the statement). Note: `.astype(Dtype::F32, …)` is
   NOT a guard — it preserves the F32 scalar and the gate will still flag it.

   **Allowlisting a genuine f32-only scalar** (e.g. inside a vision tower or
   audio encoder that runs entirely in f32, or a scalar passed to an f32-only
   API): add `// f32-ok: <reason>` on the same line or in the comment block
   directly above the `scalar_f32(` call:
   ```rust
   // f32-ok: SigLIP tower runs entirely in f32 (weights upcast at load)
   let inv_k = scalar_f32(1.0 / k as f32);
   ```
   The reason must be specific enough to explain why the f32 promotion is safe
   (e.g. "tower is f32", "output is Vec<f32>", "terminal pre-sampling logits").

   Run `make check-no-scalar-f32-leak` before pushing any new arch or new layer.

---

## Standing guardrails

The per-arch seams that stayed per-arch are deliberate. Do **not**, without a
benchmarked perf case:

- **Unify per-arch `Linear` / `Embedding` / `RmsNorm` types** into a single
  shared tower. The small per-arch differences (quant mode, bias presence,
  per-head norm) are cheap to keep separate and expensive to abstract wrongly.
- **Fold the speculative / own-loop archs** (`laguna`, `qwen3_vl_moe`,
  `bitnet`) into the shared `decode_loop`. These keep their own
  `generate_greedy` on purpose — they diverge from the pipelined path (hybrid
  recurrent state, vision token interleave, ternary-weight matmul) and were
  measured, not assumed, to be better left standalone.

When in doubt, three similar lines beat a wrong abstraction (CLAUDE.md
simplicity rules). A guardrail comes down only behind a bench number, not a
tidiness argument.
