# Speculative Decoding

Speculative decoding reduces wall-clock decode latency by parallelising the
verifier's sequential token budget across multiple positions per forward pass.
A small, fast draft model proposes K tokens; the verifier evaluates all K+1
positions in a single cached forward; accepted tokens are emitted for free.

## Overview

The fundamental contract: the output distribution produced by the speculative
loop is **identical** to what the verifier would produce alone (Leviathan et
al., 2023, Theorem 1). No quality is traded. The speedup comes from the
verifier's cost-per-position being sub-linear when processing multiple positions
at once on Metal, while the draft model is fast enough that its cost per token
is negligible relative to the verifier.

Key constraints enforced at construction time:

- The verifier and drafter vocabularies must match (`vocab_size` assertion in
  `SpeculativeDispatcher::new`). Draft token IDs only make sense as verifier
  logit indices if the vocabularies align. For EAGLE-3, which uses a reduced
  draft vocabulary, a `d2t` offset table maps draft IDs to target IDs.
- Both models load under the single Apple Silicon Metal context. Loading is
  sequential; both models share the same `Device` at construction.
- The speculative path does not support `ConstraintEngine` (structured output /
  `response_format`). Requests mixing speculative decoding with structured
  output receive `HTTP 400`. The single-arch path handles that case correctly.

Two acceptance rules are available:

- **Greedy** (`temperature == 0`): longest prefix of draft tokens that match
  the verifier's argmax at each position. Deterministic.
- **Stochastic** (`temperature > 0`): Leviathan acceptance — each draft token
  is accepted with probability `min(1, p(x)/q(x))` vs a uniform draw; on first
  rejection a correction is sampled from `normalize((p - q)+)`. The `p` and `q`
  distributions are built with the same post-temperature / top-p / top-k /
  min-p pipeline, ensuring the output is the verifier's distribution exactly.

## Round-loop

The production path (`spec_generate_greedy_cached` /
`spec_generate_stochastic_cached`): persistent per-layer verifier and draft KV
caches with `KvCache::truncate_to`-based rollback on partial acceptance. This
cuts per-round verifier cost from O(prompt\_len) to O(K).

For hybrid architectures (Qwen3.5/3.6-MoE) that carry a GatedDeltaNet (GDN)
recurrent state in addition to the standard KV cache, rollback requires
snapshot/restore rather than truncation, because the GDN state has no sequence
axis.

```text
# Initialisation
prefill verifier + draft on prompt[..-1]
y  = prompt[-1]   # verifier carry token
ds = prompt[-1]   # draft seed

# Per-round
loop until n_tokens emitted:
  if hybrid (GDN): snapshot_lin(verifier_lin), snapshot_lin(draft_lin)

  # Phase A — draft
  draft_tokens = draft.decode_n(ds, K, draft_caches, draft_lin)

  # Phase B — verify
  v_input = [y] + draft_tokens           # length K+1
  v_logits = verifier.forward(v_input, K+1, verifier_caches, verifier_lin)
    # → [1, K+1, vocab]; computes K+1 logits in one Metal dispatch

  # Phase C — accept
  greedy: accept = longest prefix where v_argmax[i] == draft_tokens[i]
  stochastic: accept = accept via min(1, p/q) per-position Leviathan rule

  emit v_tokens[0..=accept]   # accept + 1 tokens this round
  y = v_tokens[accept]        # next round's carry

  # Phase D — cache rollback
  v_drop = K - accept          # positions to discard from verifier cache
  d_drop = max(K - accept - 1, 0)
  verifier_caches.truncate_to(offset - v_drop)
  draft_caches.truncate_to(offset - d_drop)

  if GDN partial accept:
    restore_and_replay_lin(verifier_lin, verifier_lin_snap, v_input[..kept])
    restore_and_replay_lin(draft_lin, draft_lin_snap, d_fed[..kept])

  if accept == K:
    ds = [last_draft, y]   # draft cache lagged one position — prepend dK
  else:
    ds = [y]
```

On full acceptance (`accept == K`) the verifier has consumed all K draft
tokens, but the draft cache stopped one step short. The next draft seed
prepends the last draft token so the draft cache re-aligns before the new
proposals begin.

### Partial accept always cuts mid-block

Phase D truncates a KV store to a position that is **never** an append boundary:
the verifier writes its whole `K + 1`-token chunk as one append, and
`truncate_to(offset - v_drop)` lands inside it by construction.

It is not the only such caller. `PromptCacheEntry::truncate_kv_to_block`
delegates to `truncate_kv_to(block_count * BLOCK_TOKENS)` with
`BLOCK_TOKENS = 256`, while gemma4 prefills in 1024-token chunks and
qwen3.5-MoE in 2048 — so a `ReusePolicy::Partial` trim also lands inside an
append block on those arches. Phase D is the caller that hits it on *every*
partial accept; the prompt-cache trim hits it whenever the block granularity and
the prefill chunk disagree.

That matters for every store whose CPU payload accumulates independently of
`shape[2]` — the rotor and iso K/V stores plus `QuantV` (`Vec<TurboBlocks>`),
`QuantKTurbo3/4`, `QuantPlanarK` / `QuantPlanarV` (`Vec<PlanarBlocks>`), and
`QuantK` (a flat append-only `codes`/`scales` pair). Lowering `shape[2]` alone
rolls back none of them.

Dropping the whole trailing block would discard the accepted tokens along with
the rejected ones and leave the payload covering fewer rows than `shape[2]`; the
store then aborts the request rather than fabricate a zeroed gap.
`storage::truncate_plan` therefore **splits** the trailing block, cutting every
per-row buffer — codes, per-group or per-pair scales, per-group quaternions,
rotation indices, per-token norms, and the rotor QJL sideband — to the accepted
row count. `QuantK` has no blocks and cuts its flat buffer to the leading `n`
positions directly. See `crates/rmlx-kv-quant/src/storage/mod.rs`,
`storage/truncate_plan_tests.rs` and `storage/cpu_block_truncate_tests.rs`.

A live GPU ring can rebuild the gap, which is why this was survivable on the
fused decode path. It is not survivable wherever the ring is absent: the CPU
append path (a QJL-carrying rotor K store, or a `Device::Cpu` run), and the
legacy `update_rotor{3,4}_sym` / asym appends, which clear the ring by design.
Those are exactly the paths a `q_seq > 1` verifier forward takes, since the
fused decode entry is gated on `q_seq == 1`. The turbo / planar / affine stores
have no ring at all, so their CPU payload is the only copy — but they also have a
flat GPU mirror that carries the whole store under `Device::Gpu`, which bounds
where their CPU payload is load-bearing: the V side of `K8VTurbo2/3`, both TCQ
variants and `TurboSym3` (whose encode is forced to CPU even on GPU), plus any
`Device::Cpu` run and any post-SSD-hydrate store. `Planar` / `PlanarK` do **not**
accumulate CPU blocks on a normal GPU serve despite being the Gemma3 /
Gemma4-dense arch default.

**Scope.** `KvStorage::truncate_to` no longer contains a bare `shape[2] = n` in
any arm — `K8V4`, `K8V8`, `Planar`, `PlanarK`, `TurboSym3/4`, `K8VTurbo2/3`, both
TCQ variants, `IsoV3/4`, `RotorV3/4` and **both** axes of `RotorKAsym3/4`
delegate to a store-level `truncate_to`, so no codec truncates its two axes with
different semantics any more.

The split itself is still `b == 1` only, and that bound is unchanged: at `b > 1`
the block concatenation is not readable by the decode path, so a mid-block cut
stays a loud error rather than becoming a scrambled store. What changed is that
"loud" is now true for the turbo / planar / affine stores too — every CPU dequant
path checks that its payload decodes to exactly `prod(shape)` and errors with
`"refusing to zero-pad / truncate"` otherwise. Previously the turbo stores
zero-padded (`out.resize(total, 0.0)`) and the planar stores panicked on an
out-of-range index. See `docs/KV_QUANT.md` § "Scope — every CPU-side store now
cuts, and every one of them is loud".

## Per-drafter Deep Dive

### MTP — Multi-Token Prediction (Qwen3.5 sidecar)

**Source**: `speculative/mtp.rs`

An MTP drafter is a sidecar head that conditions on the verifier's penultimate
hidden state (the decoder trunk output before the final RMSNorm and LM head),
rather than a second independent model. For each draft step it:

1. Embeds the input token through the verifier's `embed_tokens` table (reuses
   `Architecture::embed_tokens_raw`, the same seam DFlash uses; `embed_scale = 1.0`).
2. Applies two pre-FC RMSNorms to the embedding and the conditioning hidden
   state independently.
3. Concatenates them along the feature axis (`[embed_norm; hidden_norm]`,
   width `2H`) and projects through a `fc` linear (`2H -> H`).
4. Runs one small Qwen3.5-MoE decoder layer (full-attention GQA, per-head q/k
   RMSNorm, partial RoPE, sparse MoE FFN) over the drafter's own KV cache.
5. Applies a final RMSNorm, then re-uses the verifier's LM head
   (`Architecture::logits_from_hidden`) to pick the next draft token greedily.

The single decoder layer is the **reused** Qwen3.5-MoE `DecoderLayer`
(`crate::qwen3_5_moe::MtpLayer` — `FullAttention` + `SparseMoeBlock`), not a
second hand-ported attention/MoE implementation: the sidecar's `layers.0` has
identical tensor names to the verifier (`full_attention_interval = 1` so the
single layer is full-attention + 256-expert MoE). The conditioning hidden is
the verifier's last-decoder-layer residual stream (pre-final-norm), captured in
the same combined `Architecture::forward_verify_capture` pass that yields the
verify logits (`capture_layer_ids = [num_hidden_layers - 1]`).

Weight layout (Qwen3.5 `mtp.*` prefix, stripped by `qwen3_5_mtp/split.py`):

| Tensor | Shape | Role |
|--------|-------|------|
| `fc.weight` (+`scales`/`biases`) | `[H, 2H]` (quant-packed) | Concat projection (quantized in the 5-bit sidecar) |
| `pre_fc_norm_embedding.weight` | `[H]` | Embedding branch norm |
| `pre_fc_norm_hidden.weight` | `[H]` | Hidden branch norm |
| `norm.weight` | `[H]` | Post-layer RMSNorm |
| `layers.{i}.self_attn.{q,k,v,o}_proj` | — | Gated GQA (`q_proj` out = `n_heads*head_dim*2`) |
| `layers.{i}.self_attn.{q,k}_norm` | `[head_dim]` | Per-head RMSNorm |
| `layers.{i}.{input,post_attention}_layernorm` | `[H]` | Pre/post norms |
| `layers.{i}.mlp.gate` / `switch_mlp.{gate,up,down}_proj` | — | 256-expert top-8 MoE |
| `layers.{i}.mlp.shared_expert.{gate,up,down}_proj` / `shared_expert_gate` | — | Shared expert |

Status: **fully wired + live-validated** against
`mlx-community/Qwen3.6-35B-A3B-MTP-5bit` (sidecar) +
`mlx-community/Qwen3.6-35B-A3B-8bit` (verifier). The round-loop
(`mtp_generate_greedy`) mirrors the DFlash loop structurally: verifier prefill →
round-0 penultimate-hidden + first-bonus capture → per-round autoregressive
`draft_n` (RoPE offset = sidecar `_next_position` = verifier prefix length +
appended count) → one combined verify forward → `walk_deferred_greedy` accept →
emit → GDN snapshot/restore verifier-KV rollback + sidecar-KV `truncate_to` on
partial acceptance. Note the MTP sidecar config carries `model_type` but no
`architectures` array; `ModelConfig::architectures` is now `#[serde(default)]`
so the standalone drafter config loads cleanly.

Norm-weight contract: `qwen3_5_mtp.py::sanitize` adds `+1.0` to each 1-D norm
weight ONLY when the source is not already mlx-format. The `mlx-community`
snapshots are mlx-format, so the split stores weights verbatim — rMLX loads them
verbatim and applies a plain `rms_norm` (matching the verifier's own RmsNorm),
adding no centring shift.

Acceptance walk (`walk_deferred_greedy`): for each position from 0 to
`n_draft` (inclusive), the verifier's hidden at that position is projected
through the LM head lazily (only until the first reject). On a match, the
token is accepted. On a mismatch or at position `n_draft`, the verifier's
prediction at that position is emitted as the correction/bonus. Budget-capped.

### DFlash — Draft-Flash Attention (Qwen3.6-MoE target)

**Source**: `speculative/dflash.rs`

DFlash (z-lab "Block Diffusion for Flash Speculative Decoding") drafts a whole
block of `block_size` tokens in **one non-autoregressive pass**. Three
properties distinguish it from MTP and EAGLE-3:

**1. Multi-layer conditioning.** The drafter conditions on the verifier's
residual stream at multiple layers (`target_layer_ids`, e.g. five layers for
the Qwen3.6-MoE target), concatenated along the feature axis and projected
to the drafter width `H` through a `fc` linear (`len(ids)*H -> H`, no bias)
followed by a `hidden_norm` RMSNorm. This conditioning vector `h_ctx` is
shared across all 8 drafter decoder layers.

**2. Non-autoregressive block decode.** The drafter receives a masked input
block: `[seed_token, mask, mask, ..., mask]` of length `block_size`. The 8
Qwen3-style decoder layers (GQA, per-head q/k RMSNorm, YARN RoPE) denoise all
masked positions simultaneously. Queries come from the proposal positions;
keys and values are sourced from both the conditioning context prefix and the
proposal tokens (context/proposal split attention). Block self-attention is
intentionally non-causal (`mask = None`) — the whole block is denoised at once.
The verifier LM head (with optional `final_logit_softcapping`) picks greedy
tokens at positions `1..block_size`.

**3. Adaptive block size.** `dflash_next_block_size` adjusts the block
ceiling each round based on the last 8 rounds of `(accepted, drafted)` history:

- `accept_rate < 0.30` or `mean_accept < 2.0`: halve (if current >= 8) or
  subtract 2, floored at `min(block_size, 4)`.
- `0.30 <= accept_rate < 0.50`: subtract 2, floored.
- `0.50 <= accept_rate < 0.85`: hold at current.
- `accept_rate >= 0.85` and `full_hit_rate >= 0.75`: add 2, capped at ceiling.

**YARN RoPE.** The Qwen3.6-35B DFlash drafter was trained with YARN RoPE
(factor 64, `original_max_position_embeddings` 4096). Applying plain RoPE
collapses the accept rate to near zero. `compute_yarn_freqs` precomputes the
inverse-frequency table and `mscale` at load time; the numeric alignment is
pinned by a test that verifies `mscale = 1.4158...` and specific frequency
values against the mlx-lm reference output.

**GDN-aware rollback.** The Qwen3.6-MoE verifier carries GDN (GatedDeltaNet)
recurrent layers in addition to its KV cache. On partial acceptance,
`DFlashRoundState::snapshot` captures all GDN layer states before the verify
forward; `DFlashRoundState::restore` + a kept-prefix replay re-aligns the GDN
state with the truncated KV cache.

**Accumulated conditioning context.** The drafter conditions on the
accumulated verifier hidden across all rounds (equivalent to the Python
reference's persistent draft KV cache). After each round the committed slice of
the verifier hidden `v_hidden[:, :n_committed, :]` is concatenated onto
`h_ctx_raw`; this grows monotonically and is projected freshly each round.

Weight layout:

| Tensor | Shape | Role |
|--------|-------|------|
| `fc.weight` | `[H, len(ids)*H]` | Multi-layer hidden projection |
| `hidden_norm.weight` | `[H]` | Post-projection norm |
| `norm.weight` | `[H]` | Post-stack RMSNorm |
| `layers.{i}.self_attn.{q,k,v,o}_proj.weight` | — | GQA projections |
| `layers.{i}.self_attn.{q,k}_norm.weight` | `[hd]` | Per-head RMSNorms |
| `layers.{i}.{input,post_attention}_layernorm.weight` | `[H]` | Pre/post norms |
| `layers.{i}.mlp.{gate,up,down}_proj.weight` | — | SwiGLU MLP |

Status: fully wired and live-validated against `z-lab/Qwen3.6-35B-A3B-DFlash`
plus `mlx-community/Qwen3.6-35B-A3B-8bit` verifier. The numeric alignment
(round-0 first proposal and full-run accept rate of ~0.515 on the test prompt)
matches the mlx-vlm `_dflash_rounds` reference.

### EAGLE-3 (Qwen3.6-MoE target)

**Source**: `speculative/eagle3.rs`

EAGLE-3 (Li et al., 2025, arXiv:2503.01840) drafts tokens **autoregressively**
with a single transformer decoder layer conditioned on the verifier's
multi-layer fused hidden state. Three structural properties define it:

**1. Multi-layer feature fusion.** The drafter reads the verifier residual
stream at three auxiliary layers (`eagle_aux_hidden_state_layer_ids`, e.g.
`[3, 19, 35]` for the Qwen3.6-35B target; captured as `[2, 18, 34]` — one
position earlier per the mlx-vlm convention). The three slices are concatenated
along the feature axis (`3*H = 6144`) and projected to `H = 2048` through `fc`
(`[2048, 6144]`, no bias). A per-aux RMSNorm variant (`fcs.{0,1,2}`) applies
an individual norm to each aux slice before re-concatenating; this is
auto-detected from tensor presence (`RMLX_EAGLE3_NO_FCS=1` forces raw concat).
Applying `fcs` more than doubled the measured greedy accept rate on the Dogacel
checkpoint (0.09 → 0.21).

**2. Eagle3FirstLayer embed/hidden fusion.** The single decoder layer
(`Eagle3FirstLayer`) takes `concat(input_layernorm(embed), hidden_norm(h_proj))`
as the `2*H = 4096` attention input — both the token embedding and the projected
verifier hidden — so the `q/k/v_proj` weights are `[*, 4096]`. The residual
is the un-normed projected hidden (`norm_before_residual=False`), not the
concatenated input. The autoregressive draft loop advances the drafter's own
KV cache one position per step.

**3. Reduced draft vocabulary with d2t remap.** The Dogacel drafter's `lm_head`
predicts over a 32,000-token draft vocabulary, a subset of the verifier's
248,320-token vocabulary. `d2t[draft_id]` holds an offset such that
`target_id = draft_id + d2t[draft_id]`. The `hot_ids_host` vector precomputes
these mapped IDs at load time. The EOS token IDs are appended to the hot IDs
list so EOS can be selected at intermediate positions.

**Restricted-vocab hot path.** When the drafter has a reduced vocabulary
(`d2t` non-empty), the verifier's logit materialisation is reduced by ~7.8x:

1. Run the verifier forward, capturing both the 3-aux hidden and the
   final-normed hidden at all K positions via `forward_verify_capture_hot`.
2. Compute restricted-vocab logits for all K positions by gathering only the
   `draft_vocab_size` rows from the LM head weight via `hot_logits_from_final_hidden`.
3. Map the argmax of each restricted-logit position through `hot_ids_host` to
   get the target-vocab token IDs.
4. Find `full_pos`: the first position where the draft token differs from the
   restricted-logit prediction (or the bonus position if all match).
5. Compute full-vocabulary logits for only the single `full_pos` position and
   replace the token at that position with the full-vocab correction.

This avoids materialising a `[1, K, 248320]` logit tensor for the accepted
positions.

**Verifier prefill chunking.** For prompts longer than 1024 tokens, the
verifier prefill uses `forward_verify_capture_chunked`: non-final chunks run
`forward_hidden_states_multi` (no logit materialisation); only the final chunk
runs `forward_verify_capture` to obtain the last-position logits. The drafter
prefill uses 512-token windows (`DRAFTER_PREFILL_CHUNK`), driven by the Metal
watchdog limit on the drafter's single-layer quadratic attention kernel.

**Per-round accept-and-reseed.** After each acceptance walk,
`Eagle3Drafter::accept_and_reseed` rolls the drafter KV cache back to the
pre-round offset, re-runs the drafter forward on the accepted prefix plus the
correction token conditioned on the verifier's 3-aux hiddens at those positions,
and samples the drafter's next-token prediction from the correction position
hidden. This precomputed token (`d_seed_tok`) is passed as `precomputed_first_tok`
to the next `draft_block`, preventing the correction position from being
processed twice (a structural bug fixed during the reference-alignment pass).

**Block-size schedule.** `eagle3_next_block_size` is non-adaptive for the
Dogacel checkpoint (it uses the configured block ceiling capped to the
remaining budget). The DFlash adaptive schedule fires only when
`adaptive_max_block_size` is present in the config.

**GDN rollback.** Identical to the DFlash path: `DFlashRoundState::snapshot`
before each verify forward, `restore` + kept-prefix replay on partial accept.

Per-step trace is available via `RUST_LOG=rmlx_models::speculative::eagle3=trace`.

### Gemma4 Assistant Drafter

**Source**: `speculative/gemma4_assistant.rs`

The Gemma4 "assistant" drafter is a small, standalone Gemma4 decoder stack (4
layers for the E2B sidecar) that reads the **verifier's own K/V cache** rather
than computing its own. This shared-K/V design differs from all other drafters:

- Each drafter sliding-attention layer attends over the verifier's last
  sliding-layer K/V.
- The single drafter full-attention layer attends over the verifier's last
  full-layer K/V.
- The drafter has no `k_proj` or `v_proj` weights; only `q_proj` and `q_norm`
  per layer.

Per draft step:

1. Embed the previous token through the verifier's `embed_tokens` table
   (embed scale 1.0 for Gemma4).
2. Concatenate `[embed(tok); last_hidden]` (`2 * backbone_hidden`).
3. Project through `pre_projection` (`2*B -> D`, where `B` is the backbone/
   verifier hidden width and `D` is the draft hidden width).
4. Run the 4-layer decoder stack using the shared K/V from the verifier's most
   recent forward pass. Sliding layers receive an additive bidirectional-window
   bias mask when the KV length exceeds the sliding window. That bias is passed
   to SDPA via mlx-c's `"array"` mask mode (masked cells use a large finite
   penalty `-1e30`, bias cast to the bf16 Q/K/V dtype) — the same convention as
   the verifier SWA path (`gemma4::layers::build_attn_mask`). mlx-c does **not**
   accept an `"additive"` mode string; using it aborts every decode step
   (fixed, issue #24).
5. Apply `norm` (final RMSNorm), then `post_projection` (`D -> B`) to produce
   the `last_hidden` for the next step.
6. Pick the next token via the **centroid-routed sparse LM head**
   (`MaskedEmbedder`): score centroids, take top-K centroids, gather the
   `vocab_per_centroid` candidate token embeddings for each, compute logits by
   inner product, and argmax.

The round-loop (`mtp_assistant_generate_greedy`) seeds the drafter using the
verifier's **normed trunk hidden** (`apply_final_norm(hidden_raw)`) at the
accepted position, obtained via `forward_hidden_states_shared_kv`. On partial
accept, the verifier KV caches are truncated to the valid prefix length;
the shared K/V is sliced accordingly before the next draft step.

**Verify-step mask invariant (issue #32).** The verifier's multi-token verify
forward (`forward_hidden_states_shared_kv` over `[b, draft…]`, `query_len > 1`)
builds its SWA / chunked-prefill array mask in `gemma4::layers::Attention::
forward`. The mask's key dim **must equal the post-update K seq dim** the SDPA
attends, or mlx-c rejects the broadcast (`mask (1,1,5,kv+1)` vs scores
`(1,8,5,kv)`). For a cache-holding producer layer the post-update K length is
`producer_offset + seq` (non-rotating) or the ring-capped
`min(window-1, producer_offset) + seq` (rotating), where `producer_offset` is
the **cache-holding layer's own `KvCache::offset()`** — NOT the model-wide
`cache_base_offset` (read from the first full-attention cache). Those two can
desync by one position across a partial-accept verify-block rollback (the
rotating sliding cache that drives `v_target` rolls back with no-op semantics
once it wraps), so sizing the mask from `cache_base_offset` produced a mask one
key too long only at non-trivial prompt lengths (window no longer covers the
whole KV). RoPE still uses the model-wide absolute `offset`; only the mask's
key dim is bound to the producer's own K. A guard in the producer branch fails
loudly if `mask.shape()[3] != k_full.shape()[2]`. This is downstream of the
issue-#24 `additive`→`array` mode fix, not a regression. The shared-KV consumer
layers and the `draft_n` (`query_len == 1`) path already size their mask from
the actual K (`k.shape()[2]`) and are unaffected.

Weight layout (E2B sidecar, `model.*` prefix):

| Tensor | Shape | Role |
|--------|-------|------|
| `model.embed_tokens.weight` | `[vocab, D]` | Tied LM-head weight |
| `pre_projection.weight` | `[D, 2B]` | Input concat projection |
| `post_projection.weight` | `[B, D]` | Output projection |
| `model.norm.weight` | `[D]` | Final RMSNorm |
| `model.layers.{i}.self_attn.q_proj.weight` | — | Q-only projection |
| `model.layers.{i}.self_attn.q_norm.weight` | `[hd]` | Per-head q norm |
| `model.layers.{i}.self_attn.o_proj.weight` | — | Output projection |
| `model.layers.{i}.mlp.*` | — | GeGLU-tanh MLP |
| `masked_embedding.centroids.weight` | `[D, num_centroids]` | Centroid scorer |
| `masked_embedding.token_ordering` | `[vocab]` i64 | Centroid→token map |

Full-attention layers use `global_head_dim` (512); sliding layers use
`head_dim` (256). The proportional RoPE frequencies for the full-attention
layer are precomputed at load time.

## SpeculativeDispatcher

`SpeculativeDispatcher` is the top-level (verifier, draft) container for the
two-independent-model speculative path (Gemma4 31B + Gemma4 E2B). The
drafter-conditioned paths (MTP, DFlash, EAGLE-3) each have their own
round-loop function and are dispatched from the serve layer based on the
`draft_kind` parameter.

Construction (`SpeculativeDispatcher::new`) asserts:

- `verifier.vocab_size() == draft.vocab_size()` — mismatched vocabularies
  make speculation meaningless.

`spec_generate_greedy` is the public entry point:

- `sampler_cfg.sampling_active()` routes to `spec_generate_stochastic_cached`
  (Leviathan stochastic acceptance, temperature > 0).
- Otherwise routes to `spec_generate_greedy_cached` (deterministic argmax,
  temperature == 0).

Both paths share the same KV cache structure and rollback logic. The
`spec_generate_greedy_no_cache` method (full re-prefill every round, no
persistent KV cache) is retained as a fallback reference.

## Accept-rate Gates and Verifier-prefill Chunking

The verifier and draft KV caches are allocated at round-loop entry with
`KvCache::with_quant_max_seq_window`, one cache per layer. Sliding-window
layers receive their layer-specific `window` value; full-attention layers
receive `max_seq`. The `max_seq` bound is derived from the verifier's
`max_position_embeddings` (clamped to `KV_MAX_SEQ_DEFAULT`).

Verifier prefill (all paths) uses `prefill_chunked`: per-arch chunk sizes
(Gemma4 and Qwen3.5-MoE have different Metal-optimal chunk sizes) gate how
much sequence length is dispatched per Metal command buffer. Non-final chunks
forward with `forward_seq_last_k_with_cache` discarding the returned logits;
between chunks the KV cache state is flushed via `eval_prefill_state`. The
`enter_prefill` / `exit_prefill` bracket optimises cache memory layout.

GDN recurrent state is snapshotted before every draft round using
`snapshot_lin`. On partial acceptance (`accept < K`), `restore_and_replay_lin`
restores the pre-round GDN state and re-runs the forward over the kept token
prefix through a throwaway scratch KV cache, leaving the GDN state byte-
consistent with the truncated production KV.

## CLI

Speculative decoding is activated on the `rmlx serve` and `rmlx chat`
subcommands by supplying `--draft-model` together with `--draft-kind`.

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--draft-model <PATH>` | directory | (none) | Path to the drafter snapshot directory. Requires `--draft-kind`. |
| `--draft-kind <KIND>` | `mtp`, `dflash`, `eagle3` | (none) | Drafter architecture family. Requires `--draft-model`. |
| `--draft-block-size <N>` | integer ≥ 1 | 4 | Tokens the drafter proposes per round. Upper-bounded by the drafter's trained `block_size`. |

Environment variable fallbacks: `MLX_VLM_DRAFT_KIND` and
`MLX_VLM_DRAFT_BLOCK_SIZE` for `--draft-kind` and `--draft-block-size`
respectively.

`--draft-kind none` is not a valid value; to disable speculative decoding omit
`--draft-model` and `--draft-kind` entirely.

### `--draft-kind mtp` dispatch (arch-family routing)

`--draft-kind mtp` fronts **two structurally different drafter loaders**, and
the serve layer (`engine::speculative::classify_mtp_draft`) routes by the draft
model's **detected architecture family** (`architectures[0]` and `model_type`),
never a substring guess:

| Draft family (`model_type` / `architectures[0]`) | Drafter loaded | Notes |
|---|---|---|
| `qwen3_5_mtp` | `MtpDrafter` (Qwen3.5/3.6-MoE sidecar head) | The MTP head reuses the verifier's embedding, LM head, and one Qwen3.5-MoE decoder layer. |
| `gemma4_assistant` / `Gemma4Assistant*` | `Gemma4AssistantDrafter` | The dedicated `*-it-assistant-bf16` snapshot — a small Gemma4 decoder stack that reads the verifier's own K/V cache. |
| anything else (incl. plain `Gemma4ForConditionalGeneration`) | **rejected at load** | Typed `Error::SpeculativePairing`; for a plain Gemma4 draft the message points at the assistant snapshot. |

A **plain Gemma4 model** (`Gemma4ForConditionalGeneration`, e.g.
`gemma-4-e2b-it-mxfp8`) is **not** a valid `--draft-kind mtp` draft: it has no
MTP sidecar head and is not the assistant drafter. Passing one is rejected at
load with an actionable error rather than falling through to the Qwen3.5
sidecar loader (which previously leaked a confusing `text_config missing
num_experts` error — issue #23). Use the `*-it-assistant-bf16` assistant
snapshot for Gemma4 speculative decoding.

### Example invocations

```text
# Gemma4 assistant speculative (verifier + dedicated assistant drafter):
# the draft MUST be the *-assistant-bf16 snapshot, NOT a plain Gemma4 model.
rmlx serve \
  --model   /path/to/gemma-4-e2b-it-mxfp8 \
  --draft-model /path/to/gemma-4-E2B-it-assistant-bf16 \
  --draft-kind  mtp \
  --draft-block-size 6

# Qwen3.6-MoE + DFlash drafter:
rmlx serve \
  --model   /path/to/Qwen3.6-35B-A3B-8bit \
  --draft-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --draft-kind  dflash \
  --draft-block-size 16

# Qwen3.6-MoE + EAGLE-3 drafter:
rmlx serve \
  --model   /path/to/Qwen3.6-35B-A3B-8bit \
  --draft-model /path/to/Qwen3.6-35B-A3B-Eagle3 \
  --draft-kind  eagle3 \
  --draft-block-size 5
```

## Reference Accept Rates

Accept rates depend on the prompt distribution and temperature. The values
below are measured at temperature 0 (greedy) on a representative coding prompt
against the local Open Models snapshots. Stochastic acceptance (temperature > 0)
tends to reduce the rate by 5-15 percentage points for equivalent block sizes.

| Verifier | Drafter | Draft kind | Block size | Accept rate |
|----------|---------|------------|-----------|-------------|
| Gemma4-31B-mxfp8 | Gemma4-E2B-mxfp8 | MTP (assistant) | 6 | ~0.70 |
| Qwen3.6-35B-A3B-8bit | Qwen3.6-35B-A3B-MTP-5bit | MTP (sidecar) | 3 | ~0.86 |
| Qwen3.6-35B-A3B-8bit | Qwen3.6-35B-A3B-DFlash | DFlash | 16 | ~0.515 |
| Qwen3.6-35B-A3B-8bit | Qwen3.6-35B-A3B-Eagle3 | EAGLE-3 | 5 | ~0.21 (restricted) |

Notes:
- Two distinct MTP paths: the Gemma4 assistant shared-K/V path
  (`mtp_assistant_generate_greedy`) and the Qwen3.5-MoE sidecar path
  (`mtp_generate_greedy` / `MtpDrafter`). The sidecar ~0.86 is measured at
  temp=0 on the "capital of France" probe (block_size 3 = 2 drafts/round,
  190/220 accepted, ~95 decode-TPS).
- The DFlash rate of ~0.515 matches the mlx-vlm `_dflash_rounds` reference
  exactly on the test prompt.
- The EAGLE-3 rate of ~0.21 is with the `fcs` per-aux norms active
  (Dogacel speculators-format checkpoint). Without `fcs` the rate drops to
  ~0.09. The restricted-vocab hot path does not change the accepted token IDs;
  it only avoids materialising the full-vocab logit tensor for accepted positions.

## See also

- `docs/KV_CACHE.md` — KvCache asymmetric K/V quant, `truncate_to`, sliding
  window cache, and `LinearAttnCache` GDN rollback.
- `docs/MODELS.md` — architecture loading, `Architecture` trait surface, the
  verifier-side seams (`forward_verify_capture`, `forward_hidden_states_multi`,
  `embed_tokens_raw`, `hot_logits_from_final_hidden`).
