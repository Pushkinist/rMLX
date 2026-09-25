# Speculative Decoding

A drafter proposes a block of tokens. The verifier scores the whole block in one
cached forward and commits the prefix it agrees with, plus one token of its own.
The output is the verifier's: the drafter changes speed, not the answer's
distribution (Leviathan et al., 2023, Theorem 1).

Speculative decoding runs on `rmlx serve` only, when `--draft-model` is set.

## Overview

Every drafter runs one round loop, `run_rounds` in
`crates/rmlx-models/src/speculative/round_loop.rs`. Each drafter kind has an
entry function that refuses what its loop cannot run, resolves the request's
block and hands its rounds to that loop. The interface a drafter implements
(`RoundDrafter`) and what the loop requires of it are in
`docs/SPEC_ROUND_SKELETON.md`.

| Kind | Drafter | Entry | Verifier |
|---|---|---|---|
| `mtp` | Qwen3.5-family MTP sidecar (`MtpDrafter`) | `mtp_generate` | Qwen3.5-family hybrid (recurrent layers) |
| `mtp` | Gemma4 assistant (`Gemma4AssistantDrafter`) | `mtp_assistant_generate` | Gemma4 |
| `dflash` | DFlash 1 block drafter | `dflash_generate` | Qwen3.5-family hybrid |
| `dflash2` | DFlash 2 block drafter | `dflash2_generate` | Qwen3.5-family hybrid |
| `eagle3` | EAGLE-3 | `eagle3_generate` | Qwen3.5-family hybrid |
| `two_model` | a smaller full model | `spec_generate_greedy_cached`, `spec_generate_stochastic_cached` | any registered architecture |

The first five are sidecar heads: small drafters that read the verifier's
hidden states or K/V. The two-model kind loads a second complete
`Architecture` with its own caches. `spec_generate_greedy` is the two-model
entry guard: it routes a request to the stochastic entry when the sampler is
active and to the greedy entry otherwise.

Each entry refuses a prompt under two tokens. The MTP sidecar, DFlash 1,
DFlash 2 and EAGLE-3 entries refuse a verifier with no recurrent state. The
assistant entry refuses a verifier that is not Gemma4.

### Acceptance rules

Every kind honours the request's temperature. The rule it runs depends on what
its drafter can supply.

- **Greedy** (`temperature == 0`, every kind): accept the longest prefix of
  proposals that match the verifier's own token at each position, then commit
  the verifier's token at the first mismatch or after the last proposal. This
  is `speculative::accept_prefix`.
- **Two-model, sampled** (`temperature > 0`): the draft model samples, so each
  proposal `x` is accepted with probability `min(1, p(x)/q(x))`. On the first
  rejection a correction is drawn from `normalize((p - q)+)`. `p` and `q` pass
  through the same temperature, top-p, top-k and min-p pipeline.
- **Sidecar, sampled** (`temperature > 0`): a sidecar proposes an argmax, which
  is a point mass. With `q = δ_x` the rule above accepts with probability
  `p(x)`, and the residual is `p` with `x` removed. A verifier that draws its
  own token at each position and accepts on agreement does exactly that. The
  walk reaches position `i` only when every earlier proposal agreed, so each
  committed token is a draw from the verifier at exactly that prefix.

`a_point_mass_proposal_emits_the_target_and_matches_sample_and_match` in
`crates/rmlx-models/src/sampler_tests.rs` pins the point-mass reduction. It runs
200k trials of both forms and holds them to one emitted distribution and one
acceptance rate.

`VerifierDraw` in `crates/rmlx-models/src/speculative/mod.rs` is the one place a
round reads the verifier's token: the argmax at temperature 0, a draw above it.
It is also the request's one RNG stream. The two-model sampled rule draws its
proposals, its acceptance coins and its corrections from it, so a seeded
request reproduces one sequence.

**What a sampled sidecar request costs.** An argmax is one device reduction over
the verified block. A draw is a host round trip and a full-vocabulary softmax at
every position, rejected positions included. This cost has not been measured.

**EAGLE-3's restricted read-back is sound at temperature 0 and only there.** An
argmax over the drafter's reduced vocabulary equals the full argmax whenever the
full argmax is in that vocabulary. A distribution needs the whole row's
normalising constant, and top-p and top-k over a subset cut a different set. A
sampled EAGLE-3 request therefore scores every position over the full
vocabulary (`restricted_read_back` is false).

### What the speculative path does not apply

- **Penalties and `logit_bias`.** No kind applies `repetition_penalty`,
  `frequency_penalty`, `presence_penalty` or `logit_bias`. A request that sets
  one runs without it, and the server logs a `warn` naming the fields
  (`dropped_sampling_fields` in `crates/rmlx-server/src/engine/speculative.rs`).
- **Constrained decoding.** A request that carries a constraint engine is
  refused: `response_format` with `json_object` or `json_schema`, and
  `tool_choice` set to required or to a named tool. The error names
  `speculative_decode_with_response_format_unsupported`.

## Drafters

### Which drafter a snapshot is

`rmlx_models::Declared::from_snapshot` reads the draft's `architectures[0]`
and `model_type`. It reads both because export tools set one or the other.
The first row that matches decides:

| Declaration | Kind |
|---|---|
| `architectures[0]` contains `Eagle3` (`LlamaForCausalLMEagle3`, over `model_type = llama`) | `eagle3` |
| `architectures[0]` contains `DFlash2` (`DFlash2DraftModel`, over `model_type = qwen3`) | `dflash2` |
| `architectures[0]` contains `DFlash` (`DFlashDraftModel`, over `model_type = qwen3`) | `dflash` |
| `model_type = gemma4_assistant`, `architectures[0]` contains `Gemma4Assistant`, or either field contains `qwen3_5_mtp` | `mtp` |
| a registered generative architecture (`Gemma4ForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, …) | `two_model` |
| anything else, a registered encoder included | none: `--draft-kind` is required |

The architecture is read before `model_type` because DFlash and EAGLE-3 declare
a plain family `model_type`. `DFlash2DraftModel` contains `DFlash`, so DFlash 2
is read first.

`--draft-kind` names the kind for a snapshot that declares none. The server
(`decide_draft_kind` in `crates/rmlx-server/src/engine/speculative.rs`)
refuses it in two cases, naming both sides:

- It contradicts a sidecar marker. No loader can build a snapshot as a kind it
  is not.
- It is `eagle3`, `dflash` or `dflash2` on a registered full model, which
  carries no such head.

The `two_model` row is an inference from the model registry, not a marker in
the snapshot. Any other flag outranks it: `--draft-kind mtp` on a full model
goes to the `mtp` router below, which refuses a full model before any weight is
read.

### `mtp` dispatch (arch-family routing)

`mtp` fronts two loaders. The server (`classify_mtp_draft`) routes by the
draft's `architectures[0]` and `model_type`:

| Draft family | Drafter loaded |
|---|---|
| `model_type = gemma4_assistant`, or `architectures[0]` contains `Gemma4Assistant` | `Gemma4AssistantDrafter` |
| either field contains `qwen3_5_mtp` | `MtpDrafter` (MoE or dense sidecar) |
| both fields empty | `MtpDrafter`, which loads by tensor name |
| anything else | refused at load with `Error::SpeculativePairing`, naming the family |

A plain Gemma4 model (`Gemma4ForConditionalGeneration`) has no MTP head. Given
bare, it drafts as `two_model`. Given with `--draft-kind mtp`, it is refused,
and the message points at the `*-it-assistant-bf16` snapshot.

### MTP sidecar (Qwen3.5 family)

Source: `speculative/mtp.rs`, `qwen3_5_moe/mtp_layer.rs`.

The sidecar head conditions on the verifier's last-decoder-layer residual
stream, before the final norm. The verify forward captures it
(`Architecture::forward_verify_capture`). Each draft step:

1. Embeds the input token through the verifier's embedding
   (`embed_tokens_raw`, scale 1.0).
2. Normalises the embedding and the conditioning hidden with separate RMSNorms
   (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`).
3. Concatenates them (`2H`) and projects through `fc` to `H`.
4. Runs one Qwen3.5 full-attention decoder layer over the sidecar's own KV
   cache. The layer is the verifier's own `DecoderLayer` type (`MtpLayer`).
5. Applies the final norm and picks the next token through the verifier's LM
   head.

Steps past the first condition on the sidecar's own output hidden. The chain
can therefore run past the `block_size` the sidecar declares, which is the
depth it was trained at. Acceptance falls with depth.

**The FFN is read off the tensors.** `MtpLayer::load` builds a sparse-MoE FFN
when `layers.0.mlp.switch_mlp.gate_proj.weight` is present and a dense SwiGLU
otherwise. A dense sidecar omits the expert keys. `num_experts` defaults to the
`0` "no experts" sentinel. `num_experts_per_tok` has no sentinel, so the MoE
branch refuses a checkpoint that omits it.

| Sidecar | FFN |
|---|---|
| `Qwen3.6-35B-A3B-MTP-5bit` | MoE |
| `Qwen3.8-27B-MTP-mxfp8` | dense |

The `mlx-community` snapshots store norm weights in MLX form, so they load
verbatim and apply a plain `rms_norm`. The sidecar `config.json` carries
`model_type` and no `architectures`; `ModelConfig::architectures` defaults to
empty for it.

`draft_n` runs one extra `forward_token` after the last proposal, so the last
drafted token gets a KV slot. `MtpDrafter::truncate_to` warns when a layer is
already shorter than the target.

`crates/rmlx-models/tests/qwen3_5_mtp_drafter_alignment.rs` checks the FFN
probe and the greedy-tracking property on the MoE and the dense pair.

### Gemma4 assistant

Source: `speculative/gemma4_assistant.rs`.

The assistant is a small Gemma4 decoder stack that reads the verifier's K/V
instead of computing its own. Its layers carry `q_proj` and `q_norm` and no
`k_proj` or `v_proj`. Each sliding layer attends over the verifier's last
sliding-layer K/V. The full-attention layer attends over the verifier's last
full-layer K/V. The verifier exposes both through
`forward_hidden_states_shared_kv`.

Each draft step embeds the previous token through the verifier's embedding,
concatenates it with the carried hidden (`2B`), projects to the drafter width
(`pre_projection`), runs the stack, applies `norm`, and projects back to the
verifier width (`post_projection`). That output is the next step's hidden.

Two LM heads ship under `model_type = gemma4_assistant`, told apart by tensor
presence:

- **Sparse head:** `masked_embedding.centroids.weight` and
  `masked_embedding.token_ordering` are present. The next token is an argmax
  over a centroid-selected shortlist (`masked_argmax`).
- **Plain tied head:** no `masked_embedding.*` tensors. The next token is a
  full-vocabulary argmax over the tied embedding (`plain_argmax`).

Sliding layers get a bidirectional-window bias once the verifier's KV is longer
than the window. It goes to SDPA through mlx-c's `"array"` mask mode; mlx-c has
no `"additive"` mode. In the verifier's multi-token verify forward, a
cache-holding layer sizes its mask from its own `KvCache::offset()`, and a
guard fails loudly if the mask's key dimension differs from the K length.

The assistant is full attention, so its rollback is a K/V truncation and no
refold.

### DFlash 1

Source: `speculative/dflash/`.

DFlash drafts a whole block in one non-autoregressive pass. The input block is
the seed token followed by `block_size - 1` mask tokens. A stack of Qwen3-style
layers denoises it with bidirectional block attention (`mask = None`). The
verifier's LM head picks the tokens at positions `1..block_size`.

**Conditioning.** The drafter reads the verifier's residual stream at
`target_layer_ids`, concatenated and projected through `fc` and `hidden_norm`.
The conditioning accumulates over the whole request. Each round projects only
the rows it committed and appends them to the carried projection. `fc` and
`hidden_norm` are row-wise, so this equals re-projecting the whole history in
exact arithmetic. At the checkpoint's dtype the matmul height differs, so rows
can differ by a last-place rounding. `report_conditioning_residual` logs that
gap per request. `crates/rmlx-models/tests/spec_conditioning_residual.rs`
measures it on the shipped DFlash 1 and DFlash 2 pairs.

The buffer is not bounded. The checkpoint declares no sliding window, and its
block attention reads every carried row.

**YaRN RoPE.** The drafter was trained with YaRN RoPE. `compute_yarn_freqs`
builds the table at load, and a test pins `mscale` and frequency values against
the mlx-lm reference.

**Adaptive block.** `dflash_next_block_size` starts from the round's block
ceiling and moves the block from the last eight rounds that drafted:

- accept rate below 0.30, or mean accepted below 2.0: halve (when the block is
  at least 8) or subtract 2;
- accept rate below 0.50: subtract 2;
- accept rate at least 0.85 and full-accept rate at least 0.75: add 2;
- otherwise: hold.

The block never drops below `min(ceiling, 4)` and never exceeds the ceiling.
DFlash 1 is the one entry in `rmlx_metrics::cell::ADAPTIVE_DRAFTERS`, so its
`decode_config` names the adaptive policy.

### DFlash 2

Source: `speculative/dflash2/`.

DFlash 2 keeps DFlash 1's shape and adds two weight families. A two-tap dynamic
depthwise convolution wraps each sublayer. A candidate-path selector turns the
block's per-position candidates into one chain. It is its own kind because the
two checkpoints differ by 23 of 81 tensors, and a row records only the kind and
the block.

**Selector.** `DFlash2Drafter::select_chain` keeps the `selector_top_k`
highest-scoring tokens at each block position. It scores adjacent pairs as
`S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>` against two vocabulary codebooks
and traces the chain from the seed. `U_t` is the verifier's LM head over the
drafter's hidden states; the drafter has no head of its own. The chain is built
from lazy device ops and read back once per round.

The candidate set depends on how MLX's partition breaks ties at the k-th place.
On the CPU kernel the tie goes to the higher token id; production drafts on
Metal. A different tie-break moves the accept rate and no committed token.
`a_tie_at_the_candidate_boundary_breaks_toward_the_higher_token_id` pins the
CPU behaviour.

**Checked against the reference.** The forward and the selector are checked
against the z-lab MLX reference on a synthetic scale model
(`crates/rmlx-models/tests/fixtures/dflash2_scale`) and on the published
weights (`crates/rmlx-models/tests/dflash2_loader.rs`).

**Conditioning.** The prefill captures the prompt through
`forward_verify_capture_chunked` and keeps only the rows the drafter's
`sliding_window` reaches. Rounds carry the projected rows, as DFlash 1 does,
bounded by that window. The per-layer K/V over the window is rebuilt on every
call, with RoPE from position zero.

**The block** is `--draft-block-size` capped by the checkpoint's `block_size`.
It is not adaptive; only the token budget shortens a round.

**Loader refusals.** The loader reads `block_size` and the other drafter keys
from `dflash_config` and the RoPE base from `rope_parameters`. It defaults
nothing: a missing key is a refusal naming the key. It also refuses:

- `is_causal: true` (the forward's block mask is bidirectional);
- `input_embedding_scale` or `output_multiplier` other than 1, or a positive
  `final_logit_softcapping` (the port applies none of them);
- `mask_token_id` outside the vocabulary;
- `selector_top_k` below 2 or above the vocabulary;
- `block_size` below 2 or above `MAX_BLOCK_SIZE` (1024);
- `sliding_window` below 2 or wider than an `i32`.

**Acceptance.** The drafter is greedy. Above temperature 0 the sidecar sampled
rule applies. The reference's candidate-restricted rejection sampling is not
implemented, so a published acceptance figure taken under that rule is a
different quantity from this port's.

### EAGLE-3

Source: `speculative/eagle3/`.

EAGLE-3 drafts autoregressively with one decoder layer
(`Eagle3FirstLayer`) conditioned on a fused multi-layer hidden.

- **Feature fusion.** The drafter reads three layers named by
  `eagle_aux_hidden_state_layer_ids`. The capture takes each one position
  earlier (`id - 1`), following mlx-vlm. The slices are concatenated and
  projected through `fc`. A checkpoint that ships `fcs.{0,1,2}` gets a
  per-slice RMSNorm before the concatenation. `RMLX_EAGLE3_NO_FCS=1` forces the
  raw concatenation.
- **Embed and hidden fusion.** The layer attends over
  `concat(input_layernorm(embed), hidden_norm(h))`. The residual is the
  projected hidden.
- **Reduced draft vocabulary.** The drafter's `lm_head` predicts over a draft
  vocabulary; `target_id = draft_id + d2t[draft_id]`. The EOS ids are appended
  to the mapped ids.

**Restricted read-back.** At temperature 0, with a reduced vocabulary, the
verify pass computes logits only over the mapped ids
(`hot_logits_from_final_hidden`). It takes full-vocabulary logits at the first
position where the proposal and the restricted argmax differ, or at the bonus
position. An accepted position therefore carries the drafter's argmax. The
round reports its restricted prefix (`Verdict::restricted`), and the loop
records a `DecidedBy` per committed token. `docs/SPEC_ANSWER_EQUIVALENCE.md`
reads it.

**Prefill.** The verifier prefill runs `forward_verify_capture_chunked` in
1024-token chunks and keeps every row, because the drafter prefill conditions on
every prompt position. The drafter prefill runs in 512-token windows
(`DRAFTER_PREFILL_CHUNK`).

**Reseed.** After each acceptance walk `Eagle3Drafter::accept_and_reseed` rolls
the drafter's cache back, re-runs the drafter over the accepted prefix and the
correction, and keeps the next proposal as the next round's first token.

**Block.** `--draft-block-size` capped by the checkpoint's block. There is no
adaptive schedule.

Per-step trace: `RUST_LOG=rmlx_models::speculative::eagle3=trace`.

### Two-model drafter — a separate full model

Source: `speculative/mod.rs` (`SpeculativeDispatcher`),
`speculative/two_model.rs`.

A smaller full model of the verifier's family proposes one token per forward.
It has its own KV cache and, on a hybrid, its own recurrent state.
`TwoModelRound` is the drafter. `Acceptance::Prefix` and
`Acceptance::Stochastic` are its two rules, so the greedy and the stochastic
entry are one drafter under two rules.

After a round that accepted every proposal, the draft cache is one token
behind. The next drafting pass feeds that token ahead of the correction.

**Load checks.** `SpeculativeDispatcher::load_speculative` runs these before any
weight is read:

- The verifier and the draft must be different directories.
- The two `tokenizer.json` files must agree id by id over every id both carry.
  The refusal names the first id whose piece differs. `vocab_size` cannot see
  a mismatch: Gemma 3 and Gemma 4 both declare 262144.
- A tail of ids only one side carries is admitted up to 128
  (`VOCAB_TAIL_TOLERANCE`). One family can ship a different tail of special
  tokens.

`SpeculativeDispatcher::new` then requires equal `vocab_size`, because the
stochastic rule indexes `p` and `q` by one id.

The draft model's cache stack is built at the verifier's codec and context
ceiling (`round_common::cache_stack`). Only the verifier's codec goes through
the server's KV codec validation.

`crates/rmlx-models/tests/two_model_stochastic.rs` checks that the stochastic
rule runs: one seed reproduces one sequence, and a second seed and temperature 0
do not. `crates/rmlx-models/tests/spec_sampled_distribution.rs` checks what the
sampled rule draws from, on a sidecar pair.

rMLX has no early stop on draft confidence. The only depth policy is DFlash 1's
adaptive block.

## The round loop

Per round, `run_rounds` narrows the block against the tokens the request may
still emit, lets the drafter propose, verifies the carry token and the
proposals in one forward, walks the acceptance, commits, and rolls back. The
verifier's cache stack is built by `round_common::verifier_cache_stack`:

- The codec is the request's `--kv-quant`, or the default.
- The ceiling comes from `context::resolve_context` over the verifier's limits.
  A `--max-ctx` above the verifier's positional capacity is refused there, as on
  the plain path (`docs/CLI.md` § "Context ceiling").
- A layer with a sliding window gets a bf16 ring at that window, whatever the
  codec.

**Prefill.** The MTP sidecar, the assistant, DFlash 1 and the two-model kind
prefill through `prefill_chunked`. It uses the verifier's per-architecture
chunk (`prefill_chunk::resolve`) and logs the size and its source at `debug`.
DFlash 2 and EAGLE-3 prefill through `forward_verify_capture_chunked` in
1024-token chunks, because they need the captured hidden states.

**Seed.** The five sidecar entries emit one token drawn from the prefill
forward before the first round. The two-model entries emit nothing before a
round; their first round carries the prompt's last token.

### One rollback, and the tape it rebuilds from

On a partial accept the verifier's caches return to the accepted prefix. There
is one implementation, `round_common::rollback_round`. It is called by
`run_rounds` for the verifier and by the two-model drafter for the draft model.

- **Attention layers** are truncated to the target (`KvCache::truncate_to`). The
  target always lands inside the verify block, never on an append boundary.
  Every quantized store cuts its own per-row payload; see
  `docs/KV_STORE_TRUNCATION.md` § "Every store cuts its own payload". A sliding
  ring gives back the block tail it just wrote; see `docs/KV_CACHE.md`
  § "Rolling the SWA ring back".
- **Recurrent (GDN) layers** have no sequence axis. Before the verify forward
  the loop arms a round tape (`arm_lin_tapes`). While it is armed, each GDN
  layer records its recurrence inputs and the state the round started from.
  Those inputs are causal, so on a partial accept the recurrence kernel refolds
  the accepted prefix from the tape. The refold reads no weights and runs no
  second forward. On a full accept the tape is dropped. A two-model drafter
  takes one forward per token, so its tape holds several segments (`GdnTape` in
  `crates/rmlx-kv-quant/src/linear_attn.rs`).

`a_round_tape_refolds_to_what_the_replay_produced` in
`crates/rmlx-models/src/speculative/round_common_tests.rs` checks the refold
against a replay forward. On a dense hybrid the two agree bit for bit. On a
mixture-of-experts hybrid they differ by that stack's own disagreement between a
batched and a stepped forward; the test measures that disagreement in the same
process and holds the refold to it.

The retained prefix is always at least one token, because the carry token is
kept.

### Which hidden the LM head is fed

A verify forward captures the verifier's residual stream before the final norm.
`Architecture::logits_from_hidden` applies the final norm and then the head.
`logits_from_final_hidden` is the head alone, for a caller that holds an
already-normed hidden.

## Reading a run

### The request record

Every entry closes a request with one `done` line at `info`, including a request
whose seed token was a stop token. The event is `<entry>: done`, for example
`mtp_generate: done` or `spec_generate_greedy_cached: done`.
`crates/rmlx-models/src/speculative/round_stats.rs` holds the counters, the
derivation and the log site.

| Field | Meaning |
|---|---|
| `rounds`, `total_draft`, `total_accept` | rounds run, tokens proposed, proposals accepted |
| `emitted` | tokens the request committed, seed included |
| `seed_emitted` | tokens committed before the first round |
| `emitted_in_rounds` | tokens the rounds committed |
| `accept_rate` | `total_accept / total_draft` |
| `accepted_per_step` | `total_accept / rounds` |
| `tokens_per_round` | `emitted_in_rounds / rounds` |
| `decode_tps` | first emitted token to last, prefill excluded; `Some(x)`, or `None` under two tokens |
| `elapsed_ms`, `prefill_ms` | wall clock of the whole request and of the prefill |
| `round_ms`, `draft_ms`, `verifier_ms` | the round loop, and the drafter and verify spans inside it |
| `draft_ms_per_round`, `verify_ms_per_round`, `loop_ms_per_round` | per round; the last is `round_ms` less the other two |
| `conditioned_rows` | rows a projecting drafter's rounds have projected |
| `block_size` | the block the entry resolved |
| `charged` | whether phases were forced (below) |
| `decode_config` | the cell the request's rows are filed under, for example `dflash2/block=8` |

`decode_tps` renders through `Debug`, so the field reads `Some(20.98)` or
`None`. `emitted / elapsed_ms` is a different, lower figure, because
`elapsed_ms` includes the prefill.

**`tokens_per_round` is the figure a speculative result is read with.** It
counts what the rounds produced, not the seed. It equals
`1 + accept_rate × (block − 1)` only while every round drafts the full block;
DFlash 1 resizes its block, so it is recorded, not derived.

The engine logs an `error!` when the counts do not add up:

- `seed_emitted + emitted_in_rounds` differs from `emitted`;
- the rounds committed more than `total_accept + rounds`;
- `draft_ms + verifier_ms` exceeds `round_ms`;
- `conditioned_rows` is further than one block from `emitted_in_rounds`.

### The round event

Every round logs one `speculative round` event, target `rmlx::spec::phase`, at
`debug`, through `round_stats::log_round`:

```
loop_kind round accept num_draft n_committed emitted_total
condition_rows projected_rows
v_offset_before v_target d_offset_before d_target refolded charged
round_ms draft_ms verify_ms walk_ms rollback_ms other_ms
```

- A field a drafter does not have is absent, not zero. `condition_rows` and
  `projected_rows` belong to drafters that carry conditioning;
  `d_offset_before` and `d_target` to drafters that keep a cache.
- `n_committed` is what the round committed, after the request's budget.
- `emitted_total` is every committed token so far, the seed included.
  `scripts/lib/spec_round_log.py` prints an `emitted_total` that sums
  `emitted_in_rounds` and excludes the seed. On a sidecar the two differ by
  `seed_emitted`.
- `v_offset_before` is the verifier offset before or after the verify forward,
  as the drafter declares in `VERIFIER_OFFSET_BASIS`. Both name the position
  the rollback is computed from.
- `refolded` is true only when the verifier's rollback refolded a recurrent
  state. `rollback_round` returns it.
- `rollback_ms` covers the verifier rollback, the drafter's own rollback and the
  next round's conditioning.
- `other_ms` is the time no phase claimed: emission, detokenization and
  bookkeeping. When the phases claim more than the round, the event carries no
  `other_ms` and an `error!` names the phases.

### Charged and uncharged phases

The engine evaluates lazily. A span is charged for the work it issued only when
something inside it blocks, so uncharged phase times drift. A round's rollback
is typically paid inside the next round's `draft_ms`.

At `trace` on `rmlx::spec::phase` (`--log verbose`, or
`RUST_LOG=info,rmlx::spec::phase=trace`), each phase forces its own work before
its span closes. That run is slower than the run it describes, because the
forced evaluations drain the pipeline.

- `phases_charged()` is read once, in the entry, and travels in `RoundCfg`. It
  reaches `rollback_round` as an argument and the record as `charged`.
- Three entries can charge: the MTP sidecar, the assistant and DFlash 2. The
  other four pass `false`, so their phase times are always uncharged.
- On a charged round, `log_round` checks the arrays the round hands to the next
  one. Any that are still unevaluated are named in an `error!` ("these arrays
  are still unevaluated"). After a change to a drafter's forcing, a charged run
  with no such line is the check.

`scripts/spec_bench.sh` refuses to file a charged row, because `observations`
is append-only. `RUST_LOG` takes precedence over `--log`, so unset it before a
bench.

### Benching an arm

`scripts/spec_bench.sh` measures a speculative arm and a no-drafter arm on one
prompt:

- The speculative arm's decode rate is the `decode_tps` of the `done` line,
  read by `scripts/lib/spec_round_log.py`. That reader re-derives every derived
  field from the event's own counters and refuses a mismatch.
- The no-drafter arm's rate is `1000 / step_mean_ms` from `GET /metrics/cache`,
  read by `scripts/lib/server_decode_tps.py`.
- Both use the window from the first emitted token to the last.
  `scripts/lib/sse_decode_window.py` times the same window client-side. A
  disagreement past `CROSS_CHECK_BAND_PCT` (10) stops the run.
- Each measured completion is digested. Without a sampler, a speculative run
  whose answer differs from the plain arm's stops the run before any median.
  Whether a sampler ran is read from the engine's own event
  (`scripts/lib/server_sampling.py`). Under a sampler both rows record
  `answer_check=sampled`.

The bench's answer check decides whether to file a row. It does not judge
correctness; that is the next section.

## Judging a draft-side change

**Byte equality is not proof.** Greedy verification commits the verifier's own
token at every position, whatever the drafter proposed. A byte-identical stream
after a drafter change shows only that the run met no near-tie that moved.

A different stream is not proof of a defect either. A different proposal
changes the accept split, which changes the height of the next verify block.
The verifier's later logits are then computed under a different dispatch, and a
near-tie the verifier sits on can resolve the other way.

Judge a draft-side change with one of three checks. Each is blind to something
different:

- **An identity-row CPU test** on the drafter's own math. Blind to the round
  loop and the verifier.
- **The accept stream** (`accept_rate`, `tokens_per_round`). It moves on
  near-ties, so it is a signal, not an oracle.
- **An equivalence pair** judged by the divergence-confidence oracle in
  `docs/SPEC_ANSWER_EQUIVALENCE.md`. It asks where the first divergence sits in
  the plain arm's own margin distribution.
  `crates/rmlx-models/tests/spec_greedy_equivalence.rs` holds the pairs. The
  two-model stochastic path runs only above temperature 0 and has no pair.

The per-round event stream of `the_assistant_round_loop_reproduces_plain_greedy`
is pinned against
`crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256`.

## CLI

| Flag | Values | Default | Description |
|---|---|---|---|
| `--draft-model <PATH>` | directory | none | The drafter snapshot: a sidecar head or a smaller full model. Its kind is read from its `config.json`. |
| `--draft-kind <KIND>` | `mtp`, `dflash`, `dflash2`, `eagle3`, `two_model` | from the snapshot | Names the kind for a snapshot that declares none. Requires `--draft-model`. Env: `MLX_VLM_DRAFT_KIND`. |
| `--draft-block-size <N>` | 2 to 1024 | 5, capped by the drafter's declared depth | Tokens the verifier scores per round, its own included; the drafter proposes one fewer. Env: `MLX_VLM_DRAFT_BLOCK_SIZE`. |

- The block has one meaning for every kind. Absent, it is 5 capped by the
  depth the drafter's checkpoint declares, and 5 for a drafter that declares
  none (the assistant, a full model).
- An explicit block is capped by the checkpoint's `block_size` on DFlash 1,
  DFlash 2 and EAGLE-3. The MTP sidecar is not capped, because it chains on its
  own hidden.
- Every block is bounded by `MAX_BLOCK_SIZE` (1024): one verify forward scores
  the whole block in a single un-chunked pass.
- A `[profile.<name>]` in `<RMLX_HOME>/profiles.toml` can set `draft_model`;
  `--profile` is in `docs/CLI.md` § `serve`.
- There is no `--draft-kind none`. Omit `--draft-model` to serve without a
  drafter.

```text
# Two full models; the kind is read off the draft's config.json.
rmlx serve --model /path/to/gemma-4-e4b-it-mxfp8 \
  --draft-model /path/to/gemma-4-e2b-it-mxfp8

# Gemma4 assistant: the *-assistant-bf16 snapshot declares itself mtp.
rmlx serve --model /path/to/gemma-4-e2b-it-mxfp8 \
  --draft-model /path/to/gemma-4-E2B-it-assistant-bf16 --draft-block-size 6

# Qwen3.6-MoE with an MTP sidecar.
rmlx serve --model /path/to/Qwen3.6-35B-A3B-8bit \
  --draft-model /path/to/Qwen3.6-35B-A3B-MTP-5bit
```

## See also

- `docs/SPEC_ROUND_SKELETON.md` — the `RoundDrafter` interface and what the
  shared loop requires of a drafter.
- `docs/SPEC_ANSWER_EQUIVALENCE.md` — the equivalence oracle and its coverage.
- `docs/KV_CACHE.md` — `KvCache`, `truncate_to`, the sliding-window ring and
  `LinearAttnCache`.
- `docs/MODELS.md` — the verifier-side seams (`forward_verify_capture`,
  `forward_hidden_states_multi`, `embed_tokens_raw`,
  `hot_logits_from_final_hidden`).
