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

- The verifier and drafter vocabularies must match. Draft token ids only make
  sense as verifier logit indices if the two tokenizers read them the same way,
  so a full draft model is checked piece by piece against the verifier's
  `tokenizer.json` before either model is loaded (§ "Two-model drafter"), and
  the two logit rows must be the same width (`SpeculativeDispatcher::new`). For
  EAGLE-3, which uses a reduced draft vocabulary, a `d2t` offset table maps
  draft ids to target ids.
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

A round loop's rollback is not only about the store it can slice. A sliding-window
layer's ring must give back the block tail it just wrote, and it can — see
`docs/KV_CACHE.md` § "Rolling the SWA ring back" — but only in the state a block
write leaves it in. It used to refuse silently, keeping the rejected drafts while
every full-attention layer dropped theirs, which is one of the two defects the
answer-equivalence gate found. `truncate_to` now returns that refusal, and the
gemma4-assistant loop reads its rollback target before the verify forward rather
than off a cache afterwards.

**Which hidden the LM head is fed.** A verify forward's capture is the
verifier's **pre-final-norm** residual stream, so a caller that projects one goes
through `Architecture::logits_from_hidden`, which applies the final norm.
`logits_from_final_hidden` is the head-only counterpart, for callers that hold an
already-normed hidden: the Qwen capture forwards, the EAGLE-3 full-vocab
correction, and the two drafters whose own final norm ends their stack. One name
meant both contracts once, and the arch that did not norm quietly reweighted the
vocabulary by the norm's weight vector for every caller handing it a raw capture.

**Answer equivalence.** Neither of those defects changed an accept rate enough to
notice, and neither was visible to the 48-token prefix checks the alignment
suites run. `crates/rmlx-models/tests/spec_greedy_equivalence.rs` is the gate that
found them, over 256 generated tokens; `docs/SPEC_ANSWER_EQUIVALENCE.md` is its
reference.

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

  # Phase D — cache rollback (one helper, both sides)
  v_drop = K - accept          # positions to discard from verifier cache
  d_drop = max(K - accept - 1, 0)
  rollback_round_caches(verifier, verifier_caches, verifier_lin, verifier_lin_snap,
                        v_input, v_pre_round_offset, offset - v_drop)
  rollback_round_caches(draft, draft_caches, draft_lin, draft_lin_snap,
                        d_fed, d_pre_round_offset, offset - d_drop)

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
have no ring at all, so their CPU payload is the only copy — but for them the
binding constraint is not the ring, it is the **bf16 decode seed**.
`exit_prefill` materialises `decode_fp16_{k,v}` for every quant whose
`feeds_bf16_k_at_decode()` is true (all of them), and each quantized
`update_<codec>` then early-returns into `update_decode_fp16` on its first line —
so the codec store is not read at decode time at all, and because it is not,
`exit_prefill` does not build it either (`KvQuant::materialises_packed_store()`;
see `docs/KV_CACHE.md` §9.6 F3). A plain GPU serve therefore cannot observe
whether the cut happened, on any of these codecs — there is no store to cut —
including the ones whose encode is forced to `Device::Cpu` (`K8VTurbo2/3`, both
TCQ variants, `TurboSym3`) — that forced-CPU `append` sits *below* the same early
return.

Where it is observable: a **hydrated** cache (`KvCache::from_storage` leaves
`decode_fp16_k: None`, so the codec arm runs every decode step and the blocks are
the cache), a `Device::Cpu` run, and any cache that never bracketed a prefill.
An uncut store is also still *written out* wherever one exists — the SSD spill
serialises `blocks` against `shape[2]`, and the prompt-cache snapshot clones
them — which is how the defect travels into the hydrated cache that later reads
it. That route now applies only to the codecs that keep a store (the K-only and
fused-symmetric families, `Mixed` / `RotK`) and to already-hydrated
store-backed caches; a seeded bf16-mirror cache spills its mirror instead. See
`rmlx-kv-ssd/src/hydrate_tests.rs` for the round-trip that pins this.

**Scope.** `KvStorage::truncate_to` no longer contains a bare `shape[2] = n` in
any arm — `K8V4`, `K8V8`, `Planar`, `PlanarK`, `TurboSym3/4`, `K8VTurbo2/3`, both
TCQ variants, `IsoV3/4`, `RotorV3/4` and **both** axes of `RotorKAsym3/4`
delegate to a store-level `truncate_to`, so no codec truncates its two axes with
different semantics any more. `Mixed` no longer drops its store either: it rolls
the fill marker back instead, which is what makes a partial-accept rollback
under `--kv-quant mixed_*` keep the prefix it was told to keep. See
`docs/KV_QUANT.md` § "The `Mixed` arm truncates, it no longer resets". `KvStorage::reset` had the identical defect and is
rewired the same way. Truncation is also clamped to be monotone-decreasing on
these six stores: a rollback into the decode window arrives with a target past
the frozen store's fill, and raising `shape[2]` to meet it would invent coverage
no payload backs.

The split itself is still `b == 1` only, and that bound is unchanged: a block's
rows run `[B, S_block, kv_h, D]`, so keeping a row prefix at `b > 1` would cut
one batch element and leave the other whole. A mid-block cut therefore stays a
loud error rather than becoming a half-cut store. What changed is that
"loud" is now true for the turbo / planar / affine stores too — every CPU dequant
path checks that its payload decodes to exactly `prod(shape)` and errors with
`"refusing to zero-pad / truncate"` otherwise. Previously the turbo stores
zero-padded (`out.resize(total, 0.0)`) and the planar stores panicked on an
out-of-range index. See `docs/KV_QUANT.md` § "Scope — every CPU-side store now
cuts, and every one of them is loud".

### Where a round's time goes

The request-level `done` line reports `draft_ms`, `verifier_ms` and a residual
`loop_ms_per_round`. Those are wall-clock spans around call sites, and this
engine evaluates lazily, so a span is charged the work it *issued* only when
something inside it blocks. Two consequences that have to be read with the
numbers:

- **A span with no blocking evaluation in it measures graph construction, not
  work.** Both MTP loops now read the verifier's argmax back inside the verify
  span, so `verifier_ms` is the verify forward in both. It was not always: the
  sidecar loop used to close its span on the unevaluated forward and re-derive
  the LM head position by position afterwards, which put most of the verify
  forward into the residual and made the two loops' residuals incomparable.
- **The rollback replay is still lazy by default.** Its output is discarded and
  the state it writes is not read until the next round's verify forward, so the
  second weight read a partial-accept round pays is billed to the *next* round's
  `verifier_ms`.

Per-round attribution comes from the `speculative round` event, target
`rmlx::spec::phase`, one per round at `debug`, emitted by one shared
`RoundPhases::log` so the two loops cannot drift into two record shapes:

```
loop_kind round accept num_draft replayed charged
round_ms draft_ms verify_ms walk_ms rollback_ms other_ms
```

`other_ms` is what no phase claimed: emission, tokenizer decode, slicing and
host bookkeeping. The four phases are disjoint sub-spans of the round, so
claiming more than the round has means a timer started outside it — that is an
`error!` naming the phases rather than an `other_ms` near zero that reads like
rounding. `replayed` says whether that round took the GDN replay arm of
`rollback_round_caches`.

`charged` is the field that says how to read the rest. At `debug` the phases are
timed but not forced, so the lazy tails above still move between them. At
`trace` on the same target — `--log verbose`, or
`RUST_LOG=info,rmlx::spec::phase=trace` for the split without per-token traces —
each phase forces its own work before its span closes and the split becomes
attributable. That run is also slower than the run it describes: the forced
evaluations drain a pipeline the loop would otherwise keep full, and shedding
exactly those drains is part of what the loop is being measured for. Compare a
charged run's `round_ms` against an uncharged one's before trusting either.

**The decision is the loop's, made once per request, and it travels on the
record.** `phases_charged()` is read at the loop head and passed down — to
`rollback_round_caches` as an argument, and onto `RoundStats::charged`, which
every loop's `done` line carries. Two things depend on that.
`rollback_round_caches` is shared by six loops and only two of them time their
phases; a switch it read on its own behalf would change how the other four
schedule work, with nothing on their records saying so — they pass `false` and
report `charged=false`.

And a charged request's `verifier_ms`, `loop_ms_per_round` and `decode_tps`
describe a differently scheduled engine. `scripts/lib/spec_round_log.py` reads
the flag back off the `done` line, refusing a value that is not a boolean rather
than coercing one — `bool("false")` is `True`, and the row it would decide is
permanent. `scripts/spec_bench.sh` puts it in the row's `notes` and **refuses to
file a charged row at all**: `observations` is append-only and `bests` is a view
over it, so such a row could not be taken back out and would compete in its cell
on a reading nobody wanted. Passing `--log info`, which that script does, is not
enough on its own — `RUST_LOG` takes precedence over the `--log` preset
(`crates/rmlx-cli/src/startup.rs`), so an ambient
`RUST_LOG=info,rmlx::spec::phase=trace` reaches a bench run that never asked for
it. Unset it before benching.

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
4. Runs one small Qwen3.5 decoder layer (full-attention GQA, per-head q/k
   RMSNorm, partial RoPE, and whichever FFN the sidecar carries — sparse MoE or
   dense SwiGLU) over the drafter's own KV cache.
5. Applies a final RMSNorm, then re-uses the verifier's LM head
   (`Architecture::logits_from_hidden`) to pick the next draft token greedily.

The single decoder layer is the **reused** Qwen3.5 `DecoderLayer`
(`crate::qwen3_5_moe::MtpLayer` — `FullAttention` + `SparseMoeBlock` or
`DenseMlp`), not a second hand-ported attention/FFN implementation: the
sidecar's `layers.0` has identical tensor names to the verifier
(`full_attention_interval = 1`, so the single layer is full-attention).

Its FFN follows the verifier family, and **which one is a tensor fact, not a
config flag**. `MtpLayer::load` probes `layers.0.mlp.switch_mlp.gate_proj.weight`
— the same witness `qwen3_5_moe::loader::build_mlp` probes per layer — and
builds `MlpBlock::Moe` when it is present, `MlpBlock::Dense` when it is not.
`DecoderLayer::forward` already handles both. A dense sidecar also omits
`num_experts` / `num_experts_per_tok` from its `text_config` entirely, so those
are read against the same `num_experts == 0` "dense, no experts" sentinel
`Qwen3_5MoeConfig` uses; a checkpoint that carries MoE tensors while reporting
no experts is refused by name rather than mis-built. Concretely:

| Sidecar | `layers.0.mlp` tensors | `text_config` expert keys | Loaded FFN |
|---|---|---|---|
| `Qwen3.6-35B-A3B-MTP-5bit` | `gate`, `switch_mlp.*`, `shared_expert*` | present (`num_experts` 256) | `MlpBlock::Moe` |
| `Qwen3.8-27B-MTP-mxfp8` | `{gate,up,down}_proj` only | absent | `MlpBlock::Dense` |

The conditioning hidden is
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
| `layers.{i}.mlp.gate` / `switch_mlp.{gate,up,down}_proj` | — | MoE sidecar only — router + expert stack |
| `layers.{i}.mlp.shared_expert.{gate,up,down}_proj` / `shared_expert_gate` | — | MoE sidecar only — shared expert |
| `layers.{i}.mlp.{gate,up,down}_proj` | — | Dense sidecar only — plain SwiGLU, no router |

Status: **fully wired + live-validated** against two pairs — the MoE sidecar
`mlx-community/Qwen3.6-35B-A3B-MTP-5bit` + `mlx-community/Qwen3.6-35B-A3B-8bit`,
and the dense sidecar `mlx-community/Qwen3.8-27B-MTP-mxfp8` +
`mlx-community/Qwen3.8-27B-mxfp8`. `crates/rmlx-models/tests/qwen3_5_mtp_drafter_alignment.rs`
gates both the FFN-shape probe and the greedy-tracking property. The round-loop
(`mtp_generate_greedy`) mirrors the DFlash loop structurally: verifier prefill →
round-0 penultimate-hidden + first-bonus capture → per-round autoregressive
`draft_n` (RoPE offset = sidecar `_next_position` = verifier prefix length +
appended count) → one combined verify forward → `accept_prefix` walk over the
verifier's own argmax → emit → GDN snapshot/restore verifier-KV rollback +
sidecar-KV `truncate_to` on partial acceptance.

`draft_n` proposes `block_size - 1` tokens but only ever feeds back `block_size - 2`
of them, so the last one used to get no KV slot. A full-accept round then commits
`block_size` verifier positions against `block_size - 1` sidecar slots, and
`MtpDrafter::truncate_to` — which skips a layer already shorter than the target —
absorbed the difference in silence. The gap grew one slot per full-accept round
and the sidecar acquired a permanent context hole. It cannot corrupt an emitted
token (every one of those is the verifier's own argmax over its own verify
forward, which never consults the drafter), so the only symptom was
accept-rate decay. `draft_n` now runs one more `forward_token` at the end and
discards the hidden, purely so the last drafted token gets its slot, and the skip
in `truncate_to` warns instead of passing quietly.

A dense sidecar omits `num_experts` / `num_experts_per_tok` entirely, and the two
keys are not read the same way for that reason. `num_experts` has a sentinel — `0`
means "dense, no experts", the same one `Qwen3_5MoeConfig` uses — so it defaults.
`num_experts_per_tok` has none: every value it can take is a legal routing width,
so a default would load a top-8 checkpoint at top-1 and show up only as a quietly
worse accept rate. It is carried as an `Option` and refused by name in the MoE
branch of `MtpLayer::load`, which is the only branch that reads it.

The GDN half of that rollback is not a truncation: the recurrent state has no
sequence axis, so it is restored from a pre-round snapshot and **replayed** over
the kept prefix. That replay runs the whole layer stack, and on this hybrid the
full-attention layers sit between GDN layers — layer 3's output is the residual
layers 4-6 consume. It therefore replays through the **real** KV caches, rolled
back to the pre-round offset first, so the FA layers attend their true prefix at
their true positions and land back on `v_target`. Replaying through a fresh
scratch KV stack instead makes those FA layers attend a `v_kept`-token prefix at
positions `0..v_kept`, and every downstream GDN layer advances on a wrong
hidden.

Every round loop that can partially accept goes through **one** implementation of
this — `speculative::rollback_round_caches`. It owns the whole rollback: the
full-attention `truncate_to` loop, the GDN snapshot restore, and the replay.
A full-attention arch (`lin` absent or empty) takes its short arm and truncates
straight to the target; a GDN hybrid takes the replay arm. Its seven callers are
`mtp_generate_greedy`, `dflash_generate_greedy`, `eagle3_generate_greedy`, and
the classic two-model loop's four (verifier + drafter, greedy + stochastic).
There is deliberately no second copy: the defect below lived in four independent
implementations at once, and a rollback inlined per loop is how it got there.

Measured, greedy, temp=0, one plain-greedy arm per pair (`common prefix` from the
alignment suites in `crates/rmlx-models/tests/`), scratch-stack replay vs
real-cache replay:

| Round loop | Pair | Scratch stack | Real caches |
|---|---|---|---|
| MTP sidecar | Qwen3.8-27B + its MTP sidecar | 4 / 31 | 31 / 31 |
| EAGLE-3 | Qwen3.6-35B-A3B + specdrift eagle3 | 13 / 96 | 93 / 96 |
| Two-model | Qwen3.8-27B + ornith-1.0-9b | 17 / 96 | 96 / 96 |

The corrupted arm also degenerated into a repetition loop on longer prompts. The
remaining flips in the correct arms are ordinary near-ties — the verify pass
scores a whole block in one forward, which is a different reduction order from a
one-token-at-a-time decode — which is why those gates assert a threshold and not
bit-identity.

The classic two-model loop needed one more thing to reach the rollback at all: it
read the pre-round offset from `caches[0]`, and on a GDN hybrid layer 0 is a
recurrent layer whose `KvCache::offset` never leaves 0. That drove the truncation
target negative on every round. It now takes the max across the stack, the way
the other three loops already did.

Note the MTP sidecar config carries `model_type` but no
`architectures` array; `ModelConfig::architectures` is now `#[serde(default)]`
so the standalone drafter config loads cleanly.

Norm-weight contract: `qwen3_5_mtp.py::sanitize` adds `+1.0` to each 1-D norm
weight ONLY when the source is not already mlx-format. The `mlx-community`
snapshots are mlx-format, so the split stores weights verbatim — rMLX loads them
verbatim and applies a plain `rms_norm` (matching the verifier's own RmsNorm),
adding no centring shift.

Acceptance walk (`speculative::accept_prefix`, shared with the gemma4-assistant
and two-model-greedy loops; the stochastic two-model loop applies a different,
Leviathan acceptance rule and does not use it): the verify forward already
projects all `block_size` positions through the LM head, so the loop reads that
argmax back once and walks it on the host. For
each position from 0 to `n_draft` (inclusive), the verifier's own token is
compared with the draft; on a match the draft is accepted, and on a mismatch or
at position `n_draft` the verifier's token is emitted as the correction/bonus.
Budget-capped.

The walk used to re-derive the head one position at a time and stop at the first
reject. That saves head FLOPs and costs a full read of the head tensor — a
separate quantised matrix, not tied — plus a pipeline drain, per position walked.
On a bandwidth-bound verifier that is the wrong way round: one batched read of
the head serves every position, and the block's logits were already in the verify
forward's graph, dropped unevaluated.

**The trade is conditional, not a pure gain.** The head cost is now constant in
`k` where it was proportional to `accept + 1`. A round that accepts nothing used
to project one position and now projects all `k` — the same single read of the
head tensor, `k` times its FLOPs. So it is a large win on high-accept rounds
against a small loss on low-accept ones, and the measured +1.96% on
Qwen3.8-27B-4bit was taken at 1.79 to 2.31 tokens per round. It narrows as
acceptance falls, and a pair that accepts poorly should be measured rather than
assumed to gain.

**One residual risk, recorded rather than fixed.** The substitution is
mathematically exact — RMSNorm is per-row, so slicing commutes with it — but the
head is now one GEMM at `M = k` where it was `k` GEMMs at `M = 1`, and a
differently tiled reduction can flip an argmax on a near-tie. In a speculative
loop that changes an emitted token. The evidence against it is one token digest
across 26 generations on two checkpoints plus the answer-equivalence suite; its
standing guard is that suite, which skips silently when the snapshots are absent
while `make ci-perf` reports green. Anyone touching the head path should run it
deliberately.

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
plus `mlx-community/Qwen3.6-35B-A3B-8bit` verifier. The round-0 first proposal
matches the mlx-vlm `_dflash_rounds` reference; that is a per-round alignment
check and says nothing about the full-run rate, which is prompt-dependent
(0.488-0.608 measured, see Reference Accept Rates) and a net decode loss at
every prompt class measured.

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
Applying `fcs` raises the measured greedy accept rate on the Dogacel checkpoint
by a factor of 1.04-1.52 depending on the prompt (0.263-0.362 with it,
0.173-0.292 without).

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

### Two-model drafter — a separate full model

**Source**: `speculative/mod.rs` (`SpeculativeDispatcher::spec_generate_greedy`)

The classic form: a smaller full model of the verifier's family proposes, the
verifier scores. Nothing hooks into the verifier's forward pass — the draft is
loaded as its own `Architecture`, keeps its own KV (and, on a GDN hybrid, its
own recurrent state), and is rolled back through the same
`rollback_round_caches` as the verifier. Any registered architecture can be the
draft, subject to the checks below; a pair of the same architecture is the
normal case (`gemma-4-e4b` drafted by `gemma-4-e2b`, `Qwen3.8-27B` drafted by
`ornith-1.0-9b`).

It is the one drafter kind with two acceptance rules. At `temperature == 0`
the loop is `spec_generate_greedy_cached`; above it, `spec_generate_stochastic_cached`
— Leviathan acceptance over the same post-sampling distributions the ordinary
sampler builds. The sidecar kinds are greedy only.
`crates/rmlx-models/tests/two_model_stochastic.rs` is the gate that the
stochastic loop runs at all: it pins that one seed reproduces one sequence, that
a second seed and `temperature == 0` do not, on the `gemma-4-e4b` / `gemma-4-e2b`
pair resolved by slug.

**Selecting it.** The kind is `two_model`, and it is read off the draft
snapshot's own `config.json` like every other kind: a registered architecture
there is a full model. `rmlx serve --model <verifier> --draft-model <draft>`
is the whole invocation; `--draft-kind two_model` is accepted and says the same
thing. `--draft-block-size` is the round block — the verifier's own token plus
the drafted ones, default 5 — so the draft proposes one fewer; it means the
same on every drafter kind, and `RoundStats.block_size` records that one
number whichever loop ran.

**What is checked at load, and why it is the tokenizer.** A mismatched pair
does not fail — it serves. The draft proposes ids, the verifier scores them as
indices into its own vocabulary, and if the two tokenizers disagree on what an
id means the verifier rejects nearly everything and the output is the
verifier's own, at a fraction of its plain speed, with no error anywhere.
`vocab_size` cannot see that: Gemma 3 and Gemma 4 both declare 262144 and share
no vocabulary (6207 of the ids differ). So `SpeculativeDispatcher::load_speculative`
compares the two `tokenizer.json` files id by id over every id both carry,
before any weight is read, and refuses the pair naming the first id whose
piece differs. A short tail of ids only one side carries is admitted, up to
128 — snapshots of one family ship one vocabulary with a different tail of
specials (the audio release of Qwen3.6 appends seven `<|audio_*|>` / `<tts_*>`
tokens the text release does not have), and neither side can propose an id
the other's logit row does not cover. The stop ids are not compared: the prompt
is tokenized and the stop decided by the verifier alone, and the draft only
ever sees ids. `vocab_size` equality is still asserted, for the stochastic
loop: `p` and `q` are indexed by one id and must be the same width.

**How the other engines do it.** Checked against their sources, September 2026.

- *llama.cpp* (`common/speculative.cpp`): `--spec-draft-model` / `-md` alone
  activates the draft-model path — `--spec-type` names only the n-gram methods.
  `common_speculative_are_compatible` requires the same vocab type, the same BOS
  and EOS (id and add-flag), a token count differing by at most
  `SPEC_VOCAB_MAX_SIZE_DIFFERENCE = 128`, and identical token text for every id
  from `SPEC_VOCAB_CHECK_START_TOKEN_ID = 5` up to the smaller count. Through
  mid-2026 an incompatible pair was *translated* rather than refused —
  detokenize the prompt with the target, retokenize with the draft, and back
  again for the proposals, with `--spec-replace` string substitutions and a
  warning that "tokens will be translated between the two"; current master
  refuses the pair with `draft model vocab type must match target model to use
  speculation`. rMLX's tail tolerance is the same 128 so the two engines admit
  the same pairs; rMLX compares from id 0 and skips the BOS/EOS check for the
  reason above.
- *mlx-lm* (`generate.py`, `server.py`): `--draft-model`; compares
  `draft_tokenizer.vocab_size` to the target's, raising in `generate` and only
  warning in the server. `--num-draft-tokens` is the fixed chain length.
- *vLLM* (`speculative_config`): `method` must be set to `draft_model` for a
  full draft — inference from the draft's own metadata covers EAGLE and MTP
  only. `num_speculative_tokens` is fixed. Vocabularies must match unless
  `use_heterogeneous_vocab` enables its token-level-intersection mapping,
  which does not combine with probabilistic draft sampling.

**Early stop on draft confidence.** llama.cpp has the knob:
`--spec-draft-p-min` (default 0.75) stops the current chain at the first token
whose draft top-1 probability falls below it — "only collect very
high-confidence draft tokens" — and `--spec-draft-n-min` (default 0) discards a
chain that came out shorter than `n_min`, so the verifier is not asked to
batch-score one or two tokens. Both are per-request in its server
(`speculative.p_min`, `speculative.n_min`). `--spec-draft-p-split` is declared
and unused. That is the idea of a chain whose depth follows the draft's own
confidence, arrived at independently and shipped since the late-2024
`common/speculative.cpp` refactor: the depth is decided token by token, inside
the round, by the draft's probability at each step. rMLX has no such knob on
any loop. The one depth policy it does run — DFlash's `dflash_next_block_size` —
is a different signal: it sets the *next* round's block from the accept rate of
the last eight rounds, per round and after the fact, not per token from the
draft's confidence. A `p_min` for the two-model loop would need the draft's
probability at each step; `draft_decode_n` today batches its argmaxes and
syncs once per round, so that is one host readback per draft token, the same
cost the stochastic loop already pays. Not implemented here.

**Measured** (temperature 0, `--kv-quant none`, `--max-ctx 8192`, block 5, 128
tokens, one warmup and three measured requests, `scripts/spec_bench.sh`), rows
in `runs.db` under `decode_config = two_model/block=5`: `gemma-4-e4b-it-mxfp8`
drafted by `gemma-4-e2b-it-mxfp8` runs at 72.95 TPS against 83.12 with no
drafter (accept rate 0.66, 3.56 tokens/round); `Qwen3.8-27B-mxfp8` drafted by
`ornith-1.0-9b-mxfp8-mlx` at 10.71 against 18.89 (accept rate 0.36, 51.5 ms of
rollback per round on the GDN pair). Neither pair pays at this block; both
reproduce the no-drafter arm's text.

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
desync by one position across a partial-accept verify-block rollback, so sizing
the mask from `cache_base_offset` produced a mask one
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

`SpeculativeDispatcher` is the top-level container for the two-model path.
The sidecar kinds (MTP, DFlash, EAGLE-3) each have their own round-loop
function and are dispatched from the serve layer on the resolved `DraftKind`.

Two constructors, because the two shapes hold different numbers of models:

- `load_speculative(verifier_dir, draft_dir, device)` — the `two_model` kind.
  Refuses a draft whose tokenizer is not the verifier's (§ "Two-model
  drafter") and, in `SpeculativeDispatcher::new`, a draft whose logit row is a
  different width. Naming the *same* directory on both sides is refused: it
  materialises the weights twice, and a draft that is the verifier costs
  exactly as much to run as the verifier it is meant to outrun.
- `load_verifier_only(verifier_dir, device)` — the sidecar path taken by the
  other three kinds. MTP / DFlash / EAGLE-3 drafters are small heads the serve
  layer loads and drives itself, so `draft` is `None` and the verifier weights
  are resident once. `spec_generate_*` refuses to run on such a dispatcher.

`spec_generate_greedy` is the public entry point:

- `sampler_cfg.sampling_active()` routes to `spec_generate_stochastic_cached`
  (Leviathan stochastic acceptance, temperature > 0).
- Otherwise routes to `spec_generate_greedy_cached` (deterministic argmax,
  temperature == 0).

Both paths share the same KV cache structure and rollback logic. There is no
re-prefill fallback: an architecture whose `forward_seq_last_k_with_cache` is
unwired surfaces that error.

## Accept-rate Gates and Verifier-prefill Chunking

The verifier and draft KV caches are allocated at round-loop entry with
`KvCache::with_quant_max_seq_window`, one cache per layer. Sliding-window
layers receive their layer-specific `window` value; full-attention layers
receive `max_seq`. The `max_seq` bound is the ceiling
`rmlx_models::context::resolve_context` produced for the pair — the verifier
owns the KV geometry, so its `ContextLimits` are what bound the round loop, and
`speculative::verifier_context` is the one wrapper all six drivers call. A
`--max-ctx` above the verifier's positional capacity is refused there, with the
same message the non-speculative paths give, instead of being taken verbatim
and overflowing a cache mid-round. See `docs/CLI.md` § "Context ceiling".

Verifier prefill (all paths) uses `prefill_chunked`, which gates how much
sequence length is dispatched per Metal command buffer. The chunk is the
verifier's own: `prefill_chunked_for_class` resolves the verifier's
`arch_class()` through `prefill_chunk::module_key_for_class`, so a verifier
prefills at the chunk that architecture's generate path uses, and a retuned
default or an `RMLX_PREFILL_CHUNK_<ARCH>` override reaches speculative prefill
too. The resolved size and the rule that produced it (`arch_default`,
`env_arch`, `env_global`, `adaptive`, `fallback`) are `debug!` fields here and
on the shared cold-prefill engine, so a run's log says what it chunked at
rather than leaving it to be inferred from timings. The hand-rolled per-arch
prefill loops are not covered — `docs/PROFILING.md` §10 lists them. Non-final chunks
forward with `forward_seq_last_k_with_cache` discarding the returned logits;
between chunks the KV cache state is flushed via `eval_prefill_state`. The
`enter_prefill` / `exit_prefill` bracket optimises cache memory layout.

GDN recurrent state is snapshotted before every draft round using
`snapshot_lin`. On partial acceptance (`accept < K`), `rollback_round_caches`
rolls the KV caches back to the pre-round offset, restores the pre-round GDN
state, and re-runs the forward over the kept token prefix **through those same
production caches** — which lands them on the truncated target and the GDN state
byte-consistent with them. It must not be a throwaway scratch stack; see the
partial-accept rollback section above for what that costs.

## CLI

Speculative decoding is activated on `rmlx serve` only, by supplying
`--draft-model`. `rmlx chat` has no draft flags. A `profiles.toml` profile can
carry `draft_model` too (`docs/CLI.md` § Profiles), and runs the same way.

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--draft-model <PATH>` | directory | (none) | The drafter snapshot: a sidecar head or a smaller full model. Which one it is is read from its `config.json`. |
| `--draft-kind <KIND>` | `mtp`, `dflash`, `dflash2`, `eagle3`, `two_model` | (from the snapshot) | Names the kind for a snapshot whose `config.json` declares none. Requires `--draft-model`. Refused when it contradicts what the snapshot declares. |
| `--draft-block-size <N>` | integer ≥ 2 | 5 | Round block: tokens the verifier scores per round, its own token included, so the drafter proposes one fewer. One meaning for every kind, and the `block_size` every `done` line and `decode_config` records. Refused below 2 at parse time. Upper-bounded by a sidecar's own `block_size`; an MTP sidecar config without that key takes the loader default of 3, which is what both shipped Qwen3.5-family sidecars do. |

Environment variable fallbacks: `MLX_VLM_DRAFT_KIND` and
`MLX_VLM_DRAFT_BLOCK_SIZE` for `--draft-kind` and `--draft-block-size`
respectively.

To disable speculative decoding omit `--draft-model`; there is no
`--draft-kind none`.

### Which drafter a snapshot is

`rmlx_models::Declared::from_snapshot` reads the draft's `architectures[0]`
and `model_type` — both, because export tools set one or the other:

| Declaration | Kind |
|---|---|
| `architectures[0]` contains `Eagle3` (`LlamaForCausalLMEagle3`, over `model_type = llama`) | `eagle3` |
| `architectures[0]` contains `DFlash2` (`DFlash2DraftModel`, over `model_type = qwen3`) | `dflash2` |
| `architectures[0]` contains `DFlash` (`DFlashDraftModel`, over `model_type = qwen3`) | `dflash` |
| `model_type = gemma4_assistant` / `Gemma4Assistant*`, or `qwen3_5_mtp` on either field (the Qwen3.5-family sidecars ship no `architectures` at all) | `mtp` |
| a registered generative architecture (`Gemma4ForConditionalGeneration`, `Qwen3_5ForConditionalGeneration`, …) | `two_model` — an inference from the registry, not a marker the snapshot carries |
| anything else, a registered encoder (`JinaEmbeddingsV4Model`) included | none — `--draft-kind` is required, and is the only reason the flag exists |

The order matters twice over. DFlash and EAGLE-3 declare a plain family
`model_type` under their own architecture name, so the architecture is read
first; and `DFlash2DraftModel` contains `DFlash`, so the generations are read
newest first — the older marker read first would make every DFlash 2 snapshot
a DFlash 1 one, and neither loader can build the other's checkpoint. A flag
that contradicts a sidecar marker is refused at load with both sides named
(`engine::speculative::decide_draft_kind`), because no loader can build a
snapshot as a kind it is not and the tensor-name error it would die with later
names neither. The registry inference is not a marker and yields to the flag:
the registry is edited whenever a model is supported, and an entry added for
some other reason must not turn a working `--draft-kind mtp` run into a
refusal.

Before this rule existed the kind came from `--draft-kind` alone, and
`--draft-model` required it; the two-model loops, selected by the *absence* of
a kind, could not be reached from the command line at all. Reading the kind
off the snapshot is what makes a bare `--draft-model` run and leaves no branch
to be orphaned by flag coupling again.

### `mtp` dispatch (arch-family routing)

`mtp` fronts **two structurally different drafter loaders**, and the serve
layer (`engine::speculative::classify_mtp_draft`) routes by the draft model's
**detected architecture family** (`architectures[0]` and `model_type`), never
a substring guess:

| Draft family (`model_type` / `architectures[0]`) | Drafter loaded | Notes |
|---|---|---|
| `qwen3_5_mtp`, MoE sidecar (`layers.0.mlp.switch_mlp.*` present) | `MtpDrafter` (Qwen3.5-family sidecar head) | The MTP head reuses the verifier's embedding, LM head, and one Qwen3.5 decoder layer with a sparse-MoE FFN. E.g. `Qwen3.6-35B-A3B-MTP-5bit`. |
| `qwen3_5_mtp`, dense sidecar (no `switch_mlp`, no `num_experts`) | `MtpDrafter` (same loader) | Same head; the reused decoder layer takes a plain SwiGLU FFN. E.g. `Qwen3.8-27B-MTP-mxfp8`. |
| `gemma4_assistant` / `Gemma4Assistant*` | `Gemma4AssistantDrafter` | The dedicated `*-it-assistant-bf16` snapshot — a small Gemma4 decoder stack that reads the verifier's own K/V cache. |
| anything else | **rejected at load** | Typed `Error::SpeculativePairing`, naming the family; never a fall-through to the Qwen3.5 sidecar loader, whose failure would name a missing MoE config instead of the mismatch. |

A **plain Gemma4 model** (`Gemma4ForConditionalGeneration`, e.g.
`gemma-4-e2b-it-mxfp8`) is a `two_model` draft, not an `mtp` one: it has no
MTP sidecar head and is not the assistant drafter. Given bare it drafts as a
full model; given with `--draft-kind mtp` it is refused by the contradiction
rule above before this table is consulted. The `*-it-assistant-bf16` snapshot
is the Gemma4 `mtp` drafter.

### Example invocations

```text
# Two full models: a Gemma4 verifier drafted by the smaller Gemma4. The kind
# is read off the draft's config.json; `--draft-kind two_model` says the same.
rmlx serve \
  --model   /path/to/gemma-4-e4b-it-mxfp8 \
  --draft-model /path/to/gemma-4-e2b-it-mxfp8 \
  --draft-block-size 5

# Gemma4 assistant speculative (verifier + dedicated assistant drafter):
# the draft is the *-assistant-bf16 snapshot, which declares itself `mtp`.
rmlx serve \
  --model   /path/to/gemma-4-e2b-it-mxfp8 \
  --draft-model /path/to/gemma-4-E2B-it-assistant-bf16 \
  --draft-block-size 6

# Qwen3.6-MoE + DFlash drafter:
rmlx serve \
  --model   /path/to/Qwen3.6-35B-A3B-8bit \
  --draft-model /path/to/Qwen3.6-35B-A3B-DFlash \
  --draft-block-size 16

# Qwen3.6-MoE + EAGLE-3 drafter, naming the kind explicitly (optional):
rmlx serve \
  --model   /path/to/Qwen3.6-35B-A3B-8bit \
  --draft-model /path/to/Qwen3.6-35B-A3B-Eagle3 \
  --draft-kind  eagle3 \
  --draft-block-size 5
```

## Reference Accept Rates

**Accept rate is a property of the (verifier, drafter, prompt) triple, not of
the engine.** The same pair swings from ~0.17 to ~0.90 across prompt classes
below, and decode throughput swings with it — so a single-prompt figure says
nothing about whether a drafter pays. Every row is quoted per prompt class for
that reason. Stochastic acceptance (temperature > 0) tends to reduce the rate by
5-15 percentage points for equivalent block sizes.

**Measurement basis.** Temperature 0, `--kv-quant none`, `--max-ctx 16384`,
200 completion tokens, one warmup plus three measured requests per cell, the
configurations run in palindromic order across two passes and pooled (n=6),
median reported. Decode throughput is the engine's own reading over the window
from the first emitted token to the last, prefill excluded — the same window
`rmlx baseline` reports, so the speculative and no-drafter arms mean the same
thing. Accept rate is read off the `<kind>_generate_greedy: done` serve-log
line; no done-line means the round loop never ran. Every cell below is also a
row in `runs.db` (metrics `accept_rate` and `decode_tps_warm`; runs recorded
since the round loop reported its own split also carry `tokens_per_round`,
`accepted_per_step` and the three `*_ms_per_round` figures).

**Tokens per round is not in the tables below.** Every row here predates the
round loop reporting it, and it is not recoverable from `accept_rate` and the
block for the DFlash rows — that drafter resizes its block. Re-running
`scripts/spec_bench.sh` for a cell fills it in `runs.db`, and
`rmlx metrics export --markdown` renders it in the speculative section.

That same `done` line carries a `decode_tps` field, on the same
first-token-to-last basis, for every one of the five round loops. It is an
`Option` and renders `Some(20.98)` or `None` — `None` when the run emitted
fewer than two tokens and there is no interval to measure, which is the honest
answer where a `0.0` would be averaged as a real rate.

**Every round loop closes a request with the same record.** One `done` line per
request, whatever stopped it — the stop token arriving before the first round
included, which used to leave no record at all and made a reader counting
records against requests served refuse the whole run.
`crates/rmlx-models/src/speculative/round_stats.rs` holds the counters, the
derivation and the log site, so a row from one drafter can be read the same way
as a row from another. The line carries the raw counters (`rounds`, `emitted`,
`total_draft`, `total_accept`, `prefill_ms`, `round_ms`, `draft_ms`,
`verifier_ms`, `block_size`), the figures derived from them
(`accept_rate`, `accepted_per_step`, `tokens_per_round`, `draft_ms_per_round`,
`verify_ms_per_round`, `loop_ms_per_round`) and the `decode_config` naming the
cell the request's rows belong to.

`tokens_per_round` is the figure a speculative result is read with: the tokens
the **rounds** produced, per round — accepted drafts plus the verifier's own
token. `1 + accept_rate × (block − 1)` recovers it only while every round drafts
the configured block — DFlash's does not, and never has — so it is recorded
rather than derived at read time.

The four sidecar loops argmax a bonus token out of the prefill forward and emit
it before the first round; the two-model loops emit nothing outside their loop.
That token is a product of the prefill, not of a verify round, so it does not
reach the figure — counting it reads high by `+1/rounds`, measured at +1.35% on
the Gemma4 assistant and +0.98% on the MTP sidecar.

Each loop counts what its rounds emit at its own emit site and reports that as
`emitted_in_rounds`; `seed_emitted` is what it had emitted before the first
round. With `emitted` those are three counts taken at three points, and the
engine and the reader both refuse a request where they do not add up. That is
what catches a seed captured on the wrong side of the pre-round emission — a
drift worth 0.5% in `tokens_per_round`, into an append-only table. An earlier
revision inferred it from an emission-budget inequality instead, which only
bites when a request's rounds exactly saturate `total_accept + rounds`: measured
on the four reachable loops at a fixed `--max-tokens`, that held for one.

The three `*_ms_per_round` figures partition one round's wall clock:
`round_ms` is the whole round loop, `draft_ms` and `verifier_ms` are disjoint
sub-spans of it, and `loop_ms_per_round` is the residual — rollback, snapshot
and restore, acceptance walks, sampling.

That partition can be checked against the engine's independently measured
decode rate: `tokens_per_round / (round_ms / rounds) * 1000` should be
`decode_tps`. Measured per request, it closes to within **0.5% on every loop**, and how much
tighter than that depends on the loop:

| loop | delta | explained by |
|---|---|---|
| DFlash, Qwen3.8-27B | −0.0001% to −0.0003% | — |
| MtpAssistant, gemma-4-e2b | ±0.0006% | — |
| Eagle3, Qwen3.6-35B | −0.053% to −0.059% | **not established** |
| MtpSidecar, Qwen3.8-27B | −0.416% to −0.425% | the last round's post-emit tail |

The sidecar's outlier has a measured mechanism: the decode window ends at the
last emitted token while `round_ms` ends after the rollback and GDN
snapshot/restore that follow it — 28.0 ms of a 6720 ms round loop there. That
mechanism does **not** account for Eagle3's 0.06%: the same tail is at most
0.01 ms on the other loops, which is under 0.001% of their round loops, and on
the assistant loop the residual implied by the identity is ±0.007 ms of a
1071 ms loop — noise about zero rather than a tail. Eagle3's 0.06% is recorded
as unexplained rather than attributed; it is two orders under the bound the
identity is used for and nobody has taken it apart.

`prefill_ms` is the same kind of figure and is looser still: on Eagle3 three
identical requests read 77.7, 799.8 and 809.7 ms. That is what a call-site wall
clock means under lazy evaluation. It is log-only and never reaches the DB. `draft_ms` and `verifier_ms` are the
wall-clock spans of their call sites and **not** the cost of the work those
calls issue: this engine evaluates lazily, so work issued in one span can be
paid for in another. They are reported as what they are. Inserting a blocking
evaluation to make them attributable prices the phases by changing them, and
that blocking evaluation is itself one of the costs the round loop is trying to
shed — which is why charging them is opt-in, why a charged request says so on
its `done` line, and why the bench refuses to file one. See § "Where a round's
time goes".

`scripts/lib/spec_round_log.py` is the only thing that reads that line, and
`scripts/spec_bench.sh` takes its speculative `decode_tps_warm` from there. It
also checks every event's derived fields against that event's own counters
before aggregating: the engine derives them per request and the reader derives
them per run, and two expressions of one formula drift silently otherwise. Its
no-drafter arm has no round-loop record, but the server times every generation's
inter-token gaps and publishes the aggregate at `GET /metrics/cache`, where
`1000 / step_mean_ms` is the same `(n - 1) / (t_last - t_first)` — that is the
no-drafter arm's figure, read through `scripts/lib/server_decode_tps.py`. Both
arms are then cross-checked against the same window timed client-side, and a
disagreement past the stated band stops the run rather than choosing between
them.
The `emitted` and `elapsed_ms` on the same line are **not** a second spelling of
it: `elapsed_ms` covers the prompt prefill, so `emitted / elapsed_ms` is a
different and lower number, and rows carrying that form are named in
`docs/METRICS_DB.md` under "Known-bad rows already in the DB". A reader that
finds a bare number in `decode_tps` rather than `Some(x)` / `None` is looking at
a log from before the field was corrected and is looking at exactly that lower
number; it must refuse it rather than read it.

Three prompt classes: `prose` and `code` are `prompts/spec_bench/{prose,code}.json`,
`4k` is `prompts/longctx_4k.json`, `paris` is the bare "What is the capital of
France?" probe.

### Qwen3.8-27B-mxfp8 — MTP sidecar (GDN hybrid, dense)

No-drafter baseline 18.98 (code) / 18.77 (prose) / 18.74 (4k) decode TPS.

| Block | Prompt | Accept rate | Decode vs no drafter |
|---|---|---|---|
| 2 | code | 0.877 | **1.36×** |
| 2 | 4k | 0.716 | **1.14×** |
| 2 | prose | 0.672 | **1.09×** |
| 3 | code | 0.728 | 1.23× |
| 3 | prose | 0.559 | 0.98× |
| 3 | 4k | 0.526 | 0.92× |

### Qwen3.8-27B-4bit — MTP sidecar (GDN hybrid, dense)

The published affine-4-bit verifier (`mlx-community/Qwen3.8-27B-4bit`, group 64)
with its own 4-bit sidecar (`mlx-community/Qwen3.8-27B-MTP-4bit`, `block_size: 3`).
Same backbone as the mxfp8 rows above at 15.5 GB of resident weights against
27.7, so the verifier's own step is 1.70× faster here (31.4 against 18.4 decode
TPS at a 3 892-token prompt, `--kv-quant none`, three runs each) and every fixed
cost in the round loop is a correspondingly larger share of a round.

No-drafter baseline 32.45 (code) / 32.53 (prose) / 32.03 (4k) decode TPS, each
measured in the same `scripts/spec_bench.sh` run as the speculative arm beside it.

| Block | Prompt | Accept rate | tokens/round | Decode vs no drafter |
|---|---|---|---|---|
| 2 | code | 0.730 | 1.72 | **1.19×** |
| 2 | 4k | 0.712 | 1.71 | **1.19×** |
| 2 | prose | 0.608 | 1.61 | 1.04× |
| 3 | code | 0.576 | 2.15 | 1.11× |
| 3 | 4k | 0.516 | 2.02 | 0.99× |
| 3 | prose | 0.477 | 1.95 | 0.95× |

**The drafter does care about the verifier's weight format.** The same sidecar
architecture against the mxfp8 verifier accepts 0.877 on code at block 2 and
returns 1.36×; here it accepts 0.730 and returns 1.19×. Acceptance is a
comparison against the *verifier's* argmax, so re-quantizing the verifier moves
the target the drafter is being scored against — and the sidecar was
re-quantized with it, which this pairing cannot separate from that. The speedup
falls further than the accept rate does, because a 1.70× faster verifier step
does not make the rollback, the GDN snapshot/restore and the acceptance walk any
faster: `loop_ms_per_round` is 14.6 ms of a 44.4 ms round at block 2 and 25.1 ms
of a 63.4 ms round at block 3, a third to two fifths of the round in a residual
that produces no tokens.

**One row already recorded against this checkpoint is not DFlash 2's.**
`z-lab/Qwen3.8-27B-DFlash2` is its own drafter kind (`dflash2`) with its own
loader: config, weights and shapes are read and validated, and 23 of its 81
tensors — a candidate selector and per-layer two-tap dynamic convolutions — are
weight families the DFlash 1 loader has no code for. That loader used to build
the earlier architecture out of the remaining 58 and serve: 0.530 accept, 2.59
tokens/round, 0.91× on code at block 8. Those figures are this engine's DFlash 1
drafter wearing the checkpoint's name, and `decode_config` recorded
`dflash/block=8` either way, so a row of them cannot be told from a row of the
real drafter afterwards. **They are not a DFlash 2 measurement and must not be
quoted as one.**

#### DFlash 2 and the MTP sidecar on that verifier, measured

Same harness, same three registered prompt classes, and a no-drafter arm
measured in every invocation beside the speculative one it is divided by:
`--kv-quant none`, `--max-ctx 8192`, temperature 0, seed 42, 128 max tokens, one
warmup and three measured requests. Every speculative row carries
`charged=false`, and every arm's engine-measured decode window agrees with the
client's reading of the same window to under 0.1%.

| Drafter | Block | Prompt | Accept rate | tokens/round | Decode vs no drafter |
|---|---|---|---|---|---|
| DFlash 2 | 5 | code | 0.981 | 4.88 | **1.93×** |
| DFlash 2 | 5 | structured | 0.824 | 4.23 | **1.54×** |
| DFlash 2 | 5 | prose | 0.515 | 3.02 | 0.94× |
| DFlash 2 | 8 | code | 0.865 | 7.06 | **1.85×** |
| DFlash 2 | 8 | structured | 0.682 | 5.52 | **1.38×** |
| DFlash 2 | 8 | prose | 0.405 | 3.74 | 0.87× |
| MTP-4bit | 3 | code | 0.576 | 2.15 | 1.14× |
| MTP-4bit | 3 | structured | 0.676 | 2.35 | 1.32× |
| MTP-4bit | 3 | prose | 0.477 | 1.95 | 0.97× |

The MTP rows reproduce the block-3 rows of the table above — same accept rate to
three places, same tokens per round — which is what makes them usable as the
comparison arm rather than a second, differently taken reading.

**The comparison is not at equal depth and cannot be.** `MtpDrafter::block_size`
is the sidecar's own `config.json` value (3 here) and the MTP round loop clamps
`block_total` to it, so `--draft-block-size 8` against that sidecar runs at 3 and
records `mtp/block=3`. The published comparison's "same 7 drafts" arm has no
counterpart on this checkpoint; each drafter is shown at the depth it can run.

The code column was taken DFlash 2 → MTP → MTP → DFlash 2, and the two DFlash 2
readings agree to 0.8% as do the two MTP ones, so the ranking is not this host's
slot drift. It is not run-to-run variation either: at temperature 0 the token
stream repeats exactly, so accept rate, round count and tokens per round come
back identical to the digit and only the timing moves.

**Block 8 is the trained block; block 5 is the faster one**, which is what
z-lab's own MLX guidance says (`block_size <= 5` against a quantized target and
draft). Acceptance is per chain, so a shorter chain has a larger accepted
fraction — 0.981 against 0.865 on code — and leaves less to roll back:
`loop_ms_per_round` is 3.0 ms at block 5 against 11.7 ms at block 8 on the same
prompt. Block 8 still wins on tokens per round and still loses on throughput,
because the round it buys them in is longer than the extra tokens pay for.

Where the round goes, against the 31.1 ms per token of the no-drafter arm:

| Block | draft ms | verify ms | loop ms | round ms | tokens/round |
|---|---|---|---|---|---|
| 3 (MTP) | 6.3 | 39.6 | 12.5 | 58.4 | 2.15 |
| 5 (DFlash 2) | 19.5 | 56.2 | 3.0 | 78.7 | 4.88 |
| 8 (DFlash 2) | 22.5 | 84.3 | 11.7 | 118.5 | 7.06 |

Verify against block: two more positions cost 16.5 ms and three more cost
27.3 ms, so a marginal verified position is 8.2–9.1 ms — **26–29% of a plain
decode step** where the bandwidth roofline is nearer 4%. That is the ceiling
every speculative arm on this machine meets, and this is a third drafter kind
measuring it.

Drafting costs 19.5–24.4 ms per round almost independently of block and of
prompt, against the sidecar's 5.9–6.9 ms. Two costs the port does not pay down
explain that floor: the forward re-projects the conditioning over as many rows as
the drafter's window reaches back over (2047 here) every round rather than over
the new rows only, and the drafter's 3.85 GB of bf16 weights are read every round.
Removing the loop residual entirely — more than the remaining accepted-prefix
replay work would do — takes the block-8 code round to 106.8 ms and 2.05×, and
also bringing drafting to the sidecar's cost takes it to 90.6 ms and 2.42×. Both
are computed from the rows above, not measured.

**The loop-overhead work is not landed** — its first half is, the accepted-prefix
replay fix is not. Every number in this section was taken in that state.

Greedy losslessness holds through all of it: on the code prompt the plain arm,
the block-5 arm and the block-8 arm return the same 429 characters under one
sha256, and the structured prompt's two arms likewise.

`rmlx_models::speculative::dflash2` binds every tensor at the shape the config
predicts and refuses a snapshot carrying one it does not read, and
`DFlash2Drafter::forward_hidden` runs the block through the stack — the
conditioning projection, the two-tap dynamic convolution around each of the two
sublayers, grouped-query attention over the conditioning window and the whole
block, RoPE and the MLP — and returns the block's final hidden states. That
forward is checked against the z-lab MLX reference
(`dflash/model_mlx.py`) on both a synthetic scale model
(`crates/rmlx-models/tests/fixtures/dflash2_scale`, to within one bf16 place)
and the published weights (`tests/dflash2_loader.rs`, bit-identical).

`DFlash2Drafter::select_chain` then turns those hidden states into one ordered
draft chain: the `selector_top_k` highest-scoring tokens are kept at each block
position, adjacent pairs are scored
`S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>` against the two rank-256
vocabulary codebooks, and the chain is traced left to right from the seed
token. `U_t` is the **verifier's** LM head over the drafter's hidden states —
4-bit on this pair — which the round loop passes in; the drafter has no head of
its own. `H(h_t)` is the context gate: it enters the pairwise term as a Hadamard
factor on the predecessor embedding, so the same token pair scores differently
depending on what the block is about at that position.

The chain is sequential but not synchronous. Position `t` needs the token chosen
at `t - 1`, and that dependency is carried in a device array rather than a host
integer: every gather, product and argmax is a lazy MLX op and the whole chain is
read back once, after the last position. The reference does the same. This is
where the DFlash 1 drafter differs — `greedy_block_tokens` evaluates and reads
back the argmax at every block position, which is `block_size - 1` device
synchronisations per round.

The selector is checked against the same reference on the same two models
(`selector_tests.rs` on the scale snapshot, `tests/dflash2_loader.rs` on the
published weights), chain for chain, and separately against the score formula
walked by hand in plain arithmetic. Each fixture's power is asserted rather than
assumed: both cases trace a chain that differs from the per-position argmax of
the logits and from the chain the pairwise term alone would trace, and the two
anchors trace different chains, so a selector that returned the argmax, dropped
the logits or ignored the seed fails rather than passing quietly.

`dflash2_generate_greedy` drives them. It prefills the whole prompt through
`forward_verify_capture_chunked`, keeping as many conditioning rows as the
drafter's window reaches back over (2047 here) — the depth the reference
conditions on, not the last prompt token alone — then per round drafts a block,
scores the carry token and every proposal in one verify forward, accepts the
agreed prefix through the shared `accept_prefix`, and rolls the caches back over
the rest through the shared `rollback_round_caches`. The block is the one the
drafter was trained at every round; only the token budget shortens it, so this
loop is not in `ADAPTIVE_DRAFTERS` and its rows are `dflash2/block=<n>`.

It is **greedy**, like every other sidecar loop here. The reference's sampled arm
accepts by rejection sampling restricted to the selector's own candidate set —
a different acceptance rule from the full-vocabulary one the two-model loop
implements — and `select_chain` traces a greedy chain and returns no candidate
distribution to sample against. A request above temperature 0 is served greedily,
which is what the serve layer already does for every sidecar.

`--draft-kind dflash` over that snapshot is refused as a flag contradicting the
declaration. `z-lab/Qwen3.6-35B-A3B-DFlash` is `dflash`, reads every tensor it
ships, and is unaffected.

The forward RoPEs its conditioning rows from position zero rather than from an
absolute offset. It recomputes the conditioning K/V on every call instead of
caching them across rounds, so all of one call's positions are rotated together
and only the query-key difference reaches the attention scores; a uniform shift
of every position is then not observable, which the reference's own answer
confirms — it moves by one bf16 place between two offsets that are
mathematically the same. **The round loop keeps that choice**: it carries the
committed hidden states forward and lets the forward re-derive the conditioning
K/V, where the reference carries a per-layer rotating K/V cache and feeds it only
each round's new rows. The two are the same answer — the cached rows are a
deterministic function of those hidden states — and adopting the cache would make
cached rows carry their own absolute RoPE, losing the invariance and the proof
that rests on it. The buffer is bounded by the drafter's window rather than
accumulated: unbounded it would grow by 50 KiB per emitted token.

Three scalars the reference applies to the drafter's logit path —
`input_embedding_scale`, `output_multiplier`, `final_logit_softcapping` — are
absent from this checkpoint, and the port applies none of them. A checkpoint that
moved one off the reference's default would otherwise be drafted through a
differently scaled head at no error, so the loader refuses one that does and
accepts one that spells its defaults out. Unlike every other refusal in that
loader this one fires on a key that is *present*: an unread key is the failure a
missing key cannot produce.

The two generations also read their config from different places, which is why
they do not share a loader: DFlash 1 carries `block_size` and `rope_theta` at
the top level of `config.json`, DFlash 2 carries `block_size` under
`dflash_config` (8) and its RoPE base under `rope_parameters` (1e7) and neither
at the top level. The DFlash 2 loader defaults nothing — a key it needs and
cannot find is a refusal naming the key, because a default is indistinguishable
from the checkpoint's own value once a run is recorded.

**That arm also did not reproduce its verifier's answer.** At temperature 0 on
the code prompt it diverged from the no-drafter arm at the fourth token and
stayed diverged, where the MTP sidecar on the same verifier and prompt is
byte-identical over 160 tokens. Greedy acceptance emits only the verifier's
argmax, so no drafter — however badly it proposes, and whatever tensors it was
built without — can change the answer; a changed answer is the round loop, and
the DFlash 1 loop is one of the three the answer-equivalence gate does not cover
(`docs/SPEC_ANSWER_EQUIVALENCE.md` § What it runs). That arm cannot be
reproduced on this pair at all now — the checkpoint declares itself `dflash2`
and no longer reaches the DFlash 1 loader — so the loop is reproduced where it
still runs: `z-lab/Qwen3.6-35B-A3B-DFlash` on its own verifier drives the same
`dflash_generate_greedy`. **The observation is about DFlash 1, not this
checkpoint**: the DFlash 2 loop on the same verifier is a pair in that gate and
agrees with plain greedy on six of six prompts.

### Qwen3.6-35B-A3B-8bit — three drafters (GDN hybrid, MoE)

No-drafter baseline 102.7 (code) / 102.7 (prose) / 100.0 (paris) / 98.7 (4k)
decode TPS.

| Drafter | Block | Prompt | Accept rate | Decode vs no drafter |
|---|---|---|---|---|
| MTP-5bit | 3 | code | 0.895 | **1.34×** |
| MTP-5bit | 3 | paris | 0.847 | **1.29×** |
| MTP-5bit | 3 | 4k | 0.809 | **1.23×** |
| MTP-5bit | 3 | prose | 0.653 | 1.02× |
| DFlash | 16 | code | 0.608 | 0.97× |
| DFlash | 16 | 4k | 0.524 | 0.84× |
| DFlash | 16 | paris | 0.491 | 0.86× |
| DFlash | 16 | prose | 0.488 | 0.78× |
| Eagle3 | 5 | paris | 0.362 | 0.74× |
| Eagle3 | 5 | code | 0.305 | 0.66× |
| Eagle3 | 5 | 4k | 0.270 | 0.61× |
| Eagle3 | 5 | prose | 0.263 | 0.62× |

### Gemma4-e4b-it-mxfp8 + E4B-it-assistant-bf16 — MTP assistant (full attention)

No-drafter baseline 83.6 (code) / 83.2 (prose) / 79.1 (4k) decode TPS.

| Block | Prompt | Accept rate | Decode vs no drafter |
|---|---|---|---|
| 6 | code | 0.731 | **1.90×** |
| 6 | 4k | 0.289 | 0.79× |
| 6 | prose | 0.238 | 0.88× |

### Where a round's time actually goes

Charged per-round split (`RUST_LOG=info,rmlx::spec::phase=trace`, see § "Where a
round's time goes"), code prompt, 200 tokens, `release-perf`, M5 Max. Milliseconds
per round. The plain step is that checkpoint's own `decode_profile
forward_per_step_ms` from a no-drafter run of the same prompt.

Qwen3.8-27B-4bit + MTP-4bit — GDN hybrid, plain step **30.77 ms** (32.5 t/s):

| Phase | block 2 | block 3 |
|---|---:|---:|
| verify forward | 34.01 | 41.28 |
| GDN replay, averaged over all rounds | 6.42 | 12.66 |
| — on the rounds that take it | 30.87 (21 % of rounds) | 31.99 (40 %) |
| drafting | 2.15 | 4.18 |
| acceptance walk | 0.000 | 0.000 |
| snapshot, emit, bookkeeping | 0.056 | 0.062 |
| **round total** | **42.63** | **58.18** |
| tokens per round | 1.793 | 2.314 |
| implied step cost vs plain | 1.39× | 1.89× |

Gemma4-e4b-it-mxfp8 + E4B assistant — full attention, plain step **11.9 ms**
(84 t/s), no recurrent state and so no replay:

| Phase | block 2 | block 6 |
|---|---:|---:|
| verify forward | 13.74 | 25.27 |
| rollback (K/V tail slice) | 0.023 | 0.038 |
| drafting | 1.08 | 4.21 |
| acceptance walk | 0.000 | 0.000 |
| emit, bookkeeping | 0.065 | 0.121 |
| **round total** | **14.92** | **29.63** |

Four things those say:

- **The GDN replay is the sidecar loop's entire residual.** A partial-accept
  round pays 31–32 ms for `rollback_round_caches`, a second full weight read of
  the verifier; a full-accept round pays 0.027 ms. Everything else the residual
  was ever suspected of — the 48-layer `LinearAttnCache` snapshot and restore,
  the emission, the cache truncation — is 0.06 ms together.
- **`kept` is never zero in these loops**, so the early return in
  `rollback_round_caches` is unreachable from them and every partial round
  replays. The retained prefix is `1 + accept`: the carry token is always kept.
- **Verifying one more position costs about a quarter of a plain step**, on both
  architectures: 7.27 ms against a 30.77 ms step on the 27B (0.236), 2.88 ms
  against an 11.9 ms step on e4b (0.242). A weight-bandwidth roofline says a
  block should be one weight read plus a few milliseconds of compute, so this is
  five to six times what that predicts, and it is the same fraction on a GDN
  hybrid at 4-bit and a full-attention model at mxfp8 — a property of verifying
  `k` positions through a quantised stack, not of either architecture. It is
  what makes a deeper block lose: block 3 on the 27B pays 7.27 ms more per round
  in the verify forward than block 2, comparable with the 6.24 ms more it pays in
  replay.
- **A verify forward's full-vocab logits are not a cost.** `forward_verify_capture`
  builds them for every block position, but the graph node is dropped
  unevaluated, so nothing is materialised until a caller asks. The cost that
  existed was the acceptance walk re-deriving the head one position at a time,
  which is why it now reads the block's argmax back once instead.

### What these say

- **MTP pays on both GDN hybrids and is the only drafter that does.** On the MoE
  it wins every prompt class at the shipped block size. On the dense 27B it wins
  every prompt class at block 2 and only the code class at block 3.
- **`--draft-block-size` is capped by the sidecar's own `block_size`, and every
  shipped Qwen3.5-family MTP sidecar declares that key as 3** — the Qwen3.6-35B
  5-bit one and both Qwen3.8-27B ones, which is also the loader's default when
  the key is absent. Any request above 3 is silently the same run. Block 2 and
  block 3 are the only two settings those pairs have, and 2 measured faster than
  3 on all three prompt classes for Qwen3.8-27B at both weight formats. The `block_size` field on the
  `mtp_generate_greedy: done` line reports the value actually used.
- **The step cost the round loop pays is 1.39× a plain step at block 2 and 1.89×
  at block 3** on Qwen3.8-27B-4bit, against 32.47 / 32.54 t/s no-drafter and
  39.70 / 37.00 t/s with the sidecar. The split above says where the growth goes:
  roughly half to the replay's rising partial-round fraction, roughly half to the
  verify forward's per-position cost.
- **DFlash and EAGLE-3 are net decode losses on this verifier at every prompt
  class measured**, DFlash by 3-22% and EAGLE-3 by 26-39%. Both run correctly and
  accept real tokens; neither clears its own round-loop overhead.
- **Two distinct MTP paths**: the Gemma4 assistant shared-K/V path
  (`mtp_assistant_generate_greedy`) and the Qwen3.5-family sidecar path
  (`mtp_generate_greedy` / `MtpDrafter`). The Gemma4 arm is full-attention and
  never touches the GDN rollback.
- **The `fcs` per-aux norms help EAGLE-3, but modestly.** With `fcs` active
  (Dogacel speculators-format checkpoint) the rate is 0.263-0.362 against
  0.173-0.292 with `RMLX_EAGLE3_NO_FCS=1` — a factor of 1.04-1.52 depending on
  the prompt, and never a doubling. The restricted-vocab hot path does not change
  the accepted token IDs; it only avoids materialising the full-vocab logit
  tensor for accepted positions.

## See also

- `docs/KV_CACHE.md` — KvCache asymmetric K/V quant, `truncate_to`, sliding
  window cache, and `LinearAttnCache` GDN rollback.
- `docs/MODELS.md` — architecture loading, `Architecture` trait surface, the
  verifier-side seams (`forward_verify_capture`, `forward_hidden_states_multi`,
  `embed_tokens_raw`, `hot_logits_from_final_hidden`).
