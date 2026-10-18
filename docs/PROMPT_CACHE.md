# Prompt Cache / Automatic Prefix Caching

The in-process prompt cache: block hashing, the per-arch caches, the consume
engine, eviction, the prefix index and the SSD handoff. The SSD tier itself is
in `docs/SSD_TIER.md`.

---

## Overview

A request whose prompt shares at least one whole 256-token block with a cached
prompt reuses that prompt's post-prefill K/V instead of prefilling it again.

- **`PromptCache<E>`** — a multi-slot LRU cache in RAM, one per architecture.
  Each slot holds a post-prefill snapshot keyed by chained block digests.
- **SSD tier** — when enabled, evicted slots are spilled to `.kvb` files and a
  RAM miss is retried against them. See `docs/SSD_TIER.md`.

---

## Block alignment

Matching is aligned to blocks of **256 tokens** (`BLOCK_TOKENS` in
`rmlx_kv_ssd::hashing`). Only whole blocks are stored and matched. A trailing
partial block is never cached and is always prefilled. A match of `k` blocks
reuses the first `k * 256` tokens; the caller prefills the rest on top.

---

## Block hashing

Block digests are **chained FNV-1a-64**:

```
offset_basis = 0xcbf29ce484222325  (FNV_OFFSET)
prime        = 0x00000100000001b3  (FNV_PRIME)

for each 256-token block b (0-indexed):
    h = prev_hash   # prev_hash = seed for b=0
    for each token id in block b (little-endian bytes):
        h ^= byte
        h  = h * FNV_PRIME
    digest[b] = h
    prev_hash  = h
```

Each digest folds in the one before it. So equal `digest[k]` proves the whole
`(k+1) * 256`-token prefix is identical, with no per-token rescan.

### The seed: `cache_seed`

The first block's seed comes from one function,
`rmlx_kv_ssd::hashing::cache_seed`:

```
h = FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt() ^ model_sig
for each per-layer codec q:
    h ^= q.cache_key_salt()
    h  = h * FNV_PRIME
seed = h
```

Four callers must produce the same digest stream: the RAM push, the
`find_best_prefix` query, the SSD spill key and the SSD hydrate probe. A push
seeded differently from its query is not a wrong answer. It is a cache that
never hits. So `cache_seed` lives in `rmlx-kv-ssd`, the deepest crate all four
can call. Do not re-derive the formula anywhere.

`ArchPromptCache::consume` computes the seed once per request through
`prompt_cache::request_cache_seed`. It hands the value to `find_best_prefix`
and to `hydrate_from_ssd`. The SSD source never seeds from state of its own:
one source serves every resident model of an arch.

The terms:

- **`model_sig`** — which model produced the K/V. The cache is one static per
  architecture, so two resident models of one arch share it (a multi-model
  registry, or a speculative pair). Token-id equality cannot separate them.
  It is derived from the snapshot directory name, so it survives a restart.
  On disk, `--project` puts several models in one namespace, so the directory
  is not a per-model partition either.
- **`layout_key`** — the SSD tier's shape key from
  `ssd_tier::compute_layout_key`, over the arch, the per-layer codec vector,
  `n_kv_heads`, `head_dim` and the codec. It is `0` when the SSD tier is off.
  It carries no model identity.
- **`kv_quant`** — the codec the stored K/V is packed under.
- **the per-layer codec vector** — what that codec resolves to under the
  current layer policy (`kv_layer_quants`). It is folded per request because
  `layout_key` is fixed at attach from the launch codec, and a request may run
  another codec. See "Codec namespacing" below.

All production callers pass a `cache_seed`. `chained_block_hashes(ids)` seeds
with the bare `FNV_OFFSET`; only tests call it.

The SSD index keys rows by `(hash, layout_key)`, so digests from two layouts
cannot collide there.

### Codec namespacing

One resident model can serve requests under different KV codecs (a per-request
`kv_quant`, no weight reload; see `docs/SERVER.md`). Cached K/V bytes are
codec-specific, so a prefix cached under `none` must not serve a `k8v4`
request. That is the `kv_quant.cache_key_salt()` term of
[`cache_seed`](#the-seed-cache_seed).

`KvQuant::cache_key_salt()` is an FNV-1a-64 hash of the codec's canonical
`Display` string (`"none"`, `"k8v4"`, `"mixed_k8g64_v4g64"`, …). It covers
every variant, payload-bearing ones included. The push, the
[`find_best_prefix`](#find_best_prefix-lookup) query and the SSD hydrate probe
all salt with the request's codec. The probe is handed the codec rather than
remembering the launch codec. Otherwise a request on another codec would probe
the wrong digest stream and reject its own stored rows.

Two requests with the same tokens and different codecs therefore produce
disjoint digest streams and occupy distinct slots. A codec switch is a clean
miss; the other codec's slot survives and stays reusable under its own codec.

With the SSD tier on, `layout_key` also folds the launch codec. The extra salt
is harmless, and it keeps a RAM-only run, where `layout_key = 0`, partitioned
by codec.

---

## Per-arch caches

Each architecture owns one process-global `static` of type
`ArchPromptCache<E>`:

| Arch | Static | Entry type | Reuse policy |
|---|---|---|---|
| `Gemma4ForConditionalGeneration` | `gemma4::prompt_cache::PROMPT_CACHE` | `Gemma4Entry` | `Partial` |
| `Gemma3ForConditionalGeneration` | `gemma3::prompt_cache::PROMPT_CACHE` | `Gemma3Entry` | `ExactOnly` |
| `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_moe::prompt_cache::PROMPT_CACHE` | `Qwen35MoeEntry` | `Partial` |
| `Qwen3VLMoeForConditionalGeneration` | `qwen3_vl_moe::prompt_cache::PROMPT_CACHE` | `Qwen3VlMoeEntry` | `ExactOnly` |
| `Qwen3ForCausalLM` | `QWEN3_PROMPT_CACHE` (in `qwen3.rs`) | `Qwen3Entry` | `ExactOnly` |
| `Qwen2ForCausalLM` | `qwen2::prompt_cache::PROMPT_CACHE` | `Qwen2Entry` | `ExactOnly` |
| `LagunaForCausalLM` | `laguna::prompt_cache::PROMPT_CACHE` | `LagunaEntry` | `ExactOnly` |
| `BitNetForCausalLM` | `bitnet::prompt_cache::PROMPT_CACHE` | `BitNetEntry` | `ExactOnly` |

### Entry types

Every entry holds the full prompt token ids, the chained block digests, the
post-prefill `kv_caches`, the first decode token, and the `KvQuant` in effect
when the snapshot was taken. Every arch that emits logprobs also holds
`first_logprobs`, the first token's record.

- **`Qwen35MoeEntry`** also holds `lin_caches`, the GatedDeltaNet recurrent
  states, and an `anchor`: a second set of them, saved inside the prompt (see
  "Qwen3.5 resume and the anchor"). A recurrent state is never truncated, since
  it has no sequence axis; the entry resumes from one it holds.

**First-token logprob on Exact hit.** An Exact hit replays the cached first
token without recomputing the prefix's last-position logits. To keep the
OpenAI contract of one `logprobs.content` entry per emitted token, the entry
stores the first token's top 20 logprobs when the snapshot is saved, whatever
the storing request asked for (`capture_first_logprobs` in `decode_loop.rs`).
A hit replays the record truncated to the replaying request's `top_logprobs`
(`replay_first_logprobs`), so it emits the same `token_logprob` a Miss would.
With logprobs off it emits `None`. On Qwen3.5 the resume path stores the record
too. An SSD-hydrated entry stores placeholder id 0 as its first token and has no
`first_logprobs`; it is never an Exact hit, so a repeat of its prompt prefills
again and emits its logprobs as a Miss does. The Laguna, Qwen2 and Qwen3-VL-MoE
entries hold no `first_logprobs`: those arches emit no logprobs on any path.

### `PromptCacheEntry` trait contract

| Method | Purpose |
|---|---|
| `prompt_token_ids()` | Full token sequence; the Exact arm compares it with the request. |
| `block_hashes()` | Chained 256-token block digests; read by `find_best_prefix`. |
| `deep_clone()` | Refcount clone of every MLX array; no tensor data is copied. |
| `kv_caches()` / `kv_caches_mut()` | The attention KV caches. |
| `lin_caches()` | Recurrent caches; `&[]` for pure-attention arches. Required. |
| `kv_quant()` | Codec of the snapshot; tags each spilled block. Required. |
| `is_ssd_hydrated()` | True for an entry rebuilt from the SSD tier. |
| `truncate_kv_to(len)` / `truncate_kv_to_block(n)` | Trim the KV caches; never touches `lin_caches`. |
| `kv_bytes()` | RAM estimate for the eviction budget. |
| `is_hydrate_complete()` | False when a hydrated entry lacks payload for an attended layer. |
| `is_reusable_prefix_of(..)` | Whether and how a cached prefix may be reused; default `None`. |
| `prepare_reuse(kind)` | Clone, and trim when the kind requires it. |

### `ArchPromptCache<E>`

The per-arch shell holds the arch name, its `ReusePolicy`, its cross-layer-KV
topology (it feeds the seed's per-layer vector), the cache itself (`None`
until the first request), and the SSD attach parameters (namespace,
`layout_key`, device). The attach parameters survive a cache rebuild.

The resident-KV byte counter is **not** here. It is per model instance
(`kv_bytes::KvBytesCounter`, a field on each arch's model struct). Two models
of one arch share this shell and would mix their byte totals in the `events`
table. See the `kv_cache_bytes` row of `docs/METRICS_SCHEMA.md` §4.

`ArchPromptCache::ensure(capacity)` runs once per generation. It is a no-op
when the cache already has that capacity, `0` included. Otherwise it rebuilds
the cache and reinstalls the SSD sinks. A capacity that never compared equal
would rebuild on every request: the snapshots and counters would be dropped
each time, and caching would look off.

### Zero slots

`--prompt-cache-slots 0` disables the cache as a real state. The cache object
is still built and counts its misses, but `push` refuses every entry. `slots`
stays empty, `find_best_prefix` can only miss, and every request prefills.
Nothing is clamped to one slot.

The SSD tier is disabled with it: `hydrate_from_ssd` returns before querying
the source, because a hydrated entry could only be refused. A zero-slot server
keeps `ssd_hits` at 0 and reads no `.kvb`.

An `X-Session-Id` header does not change this. Session reuse widens the slot
count by one per active session
(`session_cache::effective_prompt_cache_slots`), and leaves a configured `0`
alone.

To make one request miss without changing the configuration, use
`ArchPromptCache::clear()` (`Architecture::clear_prompt_cache`). It empties
the slots and resets the counters, and keeps the capacity and the SSD sinks.
`rmlx bench` does this. After `clear()`, a RAM miss can still be served from a
`.kvb` and counted in `ssd_hits`. A caller that needs a real prefill checks
`hits == 0 && ssd_hits == 0`.

---

## `find_best_prefix` lookup

```
find_best_prefix(prompt_ids, seed) -> Option<(slot_index, matched_blocks)>
```

The `seed` is the [`cache_seed`](#the-seed-cache_seed) the push side uses. A
slot stored by another model, under another codec or at another layout never
matches.

1. Compute the chained block digests of `prompt_ids` from `seed`.
2. Scan every slot (linear index) or query the radix tree (radix index).
3. Return the slot with the most leading equal digests, if it has at least
   one; otherwise `None`.
4. On a hit, stamp the slot most recently used.

It counts `hits`, `misses`, `block_hits`, `block_misses` and `partial_hits` (a
hit that matched fewer blocks than the prompt has). It never evicts.

---

## The consume engine

`ArchPromptCache::consume` makes the whole per-request decision and never
pushes. It returns `Consumed::Exact(entry)`, `Consumed::Reuse { entry, kind }`
or `Consumed::Miss(reason)`:

1. An image prompt misses at once (`has_image`). Its K/V depends on vision
   features, not on token ids alone.
2. Compute the seed (see above).
3. `find_best_prefix`; on a miss, `hydrate_from_ssd` and look again.
4. A stored codec that differs from the request's evicts the slot and misses
   (`quant_mismatch`).
5. **Exact**: an entry that is not SSD-hydrated and whose prompt equals the
   request is cloned and returned.
6. **Reuse**: a hydrated entry may be reused under any policy if it is
   complete; a RAM entry only under `Partial`. The entry's
   `is_reusable_prefix_of` picks the `ReuseKind`, and `prepare_reuse` clones
   (and trims).
7. A slot that declines is not the end. Every other slot that shares a block is
   asked in turn, most blocks first. An Exact hit wins outright; otherwise the
   longest reuse any slot allows is taken, the first of equals. The lookup is
   counted, and the slot stamped most recently used, for the slot that served.
   The SSD tier is not asked again after the RAM slots decline: a decline is the
   common outcome of a partial match, and the tier would be read on each one.
8. Anything else misses, with the best slot's reason.

Every miss carries a `MissReason` and logs one `debug!` event whose `branch`
field is the reason's label: `has_image`, `no_cache`, `no_match`,
`quant_mismatch`, `deep_clone_err`, `incomplete_hydrate`, `non_reusable`,
`hydrated_declined_to_exact`.

**Context ceiling.** The K/V of an entry carries the `--max-ctx` ceiling of the
request that filled it. Gemma4, Qwen3.5, Qwen3 dense and Qwen3-VL-MoE bind a
reused entry, Exact hits included, to the ceiling of the request that reuses it
(`context::rebind_to_ceiling`), so that request succeeds or fails as a cold run
of the same prompt would under its own ceiling: a prompt above it is refused
with `KvCeilingExceeded`, and a ceiling higher than the stored one lets the tail
and the decode grow. A ceiling that differs is never a miss.

A paged KV cache (`--paged-kv`) whose slots hold pages refuses its deep clone,
because the pages have no copy. The post-prefill snapshot of such a cache is
not stored (`snapshot_clone` logs one `debug!` with branch `deep_clone_err`),
an Exact or Reuse clone of it misses with `deep_clone_err`, and the spill hook
skips it with a `warn!`. A paged cache with no pages clones.

`ReuseKind` is `StrictPrefix { prefix_len }` (reuse the whole cached prefix,
prefill the rest) or `BlockTruncate { effective_blocks }` (trim to a block
boundary, prefill the rest: Gemma4's block truncate and the Qwen3.5 anchor).

---

## ReusePolicy

`ReusePolicy` is a runtime gate on the consume engine's reuse arm:

```rust
pub(crate) enum ReusePolicy {
    Partial,    // block-aligned partial-prefix reuse allowed
    ExactOnly,  // full-token-equality reuse only
}
```

**`Partial`** (Gemma4, Qwen3.5): a RAM entry may be reused as a prefix. The
entry's hook decides which prefix. See the Gemma4 and Qwen3.5 sections below.

**`ExactOnly`** (every other arch): a RAM entry is reused only on an Exact
hit; any partial match misses. A hydrated entry is still reused as a strict
prefix where the arch's hook allows it.

### Judging a resume arm

**A resume arm cannot be judged by byte equality against a cold baseline, and
a warm arm that agrees with one has not necessarily run.**

- *A resume is not a re-prefill.* Restoring at a block boundary and forwarding
  the tail is the same arithmetic as a single-shot prefill, chunked
  differently. The rows agree to bf16 noise, not bit for bit. Over a wide
  vocabulary a row can hold an exact tie, and one flipped argmax decodes an
  unrelated stream. A stream comparison reports that as corruption.

  Judge the tail logits instead: the argmax at every tail position first, and
  a per-logit bound second. The bound is `TAIL_LOGIT_NOISE_BOUND` (2.0) in
  `crates/rmlx-models/src/qwen3_5_moe/tests.rs`, whose doc comment records
  what it separates. `tests/qwen3_5_moe_forward_seq_last_k.rs` repeats it as a
  literal.

  The decoded stream can be compared token for token only against a cold
  prefill **split where the resume splits** (`set_prefill_chunk`). That pair
  is byte-identical, logits included. It is the only baseline that reaches
  what the tail leaves behind: the recurrent state after the tail, and the
  first KV append on a resumed offset.
- *A Miss agrees with the cold baseline for free.* It prefills the same
  prompt. A test that pushes an entry by hand must seed its block digests with
  `request_cache_seed`, the seed `consume` queries with. Otherwise the entry
  is invisible, every warm arm is a Miss, and the comparison passes while
  exercising nothing. Assert the branch the engine reached. Where that branch
  is `Miss`, first show that the entry was findable.

---

## Gemma4 prefix reuse

Gemma4 mixes full-attention layers and sliding-window (SWA) layers. An SWA
layer is a `RotatingKvCache` ring of the last `sliding_window` tokens.
`Gemma4Entry::is_reusable_prefix_of` tries two kinds, in this order.

**Strict prefix** (`is_strict_prefix_of`): the cached prompt is at least
`BLOCK_TOKENS` long, shorter than the request, and equal to the request's
first tokens. The snapshot is reused whole and only the new tail is
prefilled. Nothing is truncated, so this is correct even when an SWA ring has
wrapped: the snapshot already holds what the tail attends to.

**Block truncate**: otherwise take the matched blocks, less the last one when
they cover the whole block-aligned prompt, since the prefilled tail must not
be empty. Reuse them only if `can_truncate_to_block` holds: every layer that
holds anything passes `KvCache::can_truncate_to(blocks * BLOCK_TOKENS)`. A
ring past its wrap cannot give a position back, so the reuse misses instead
of desyncing the layers. A ring that still holds the window the shorter
prefix attends over may be rolled back; that rollback is lossless by the rule
in `docs/KV_CACHE.md` § "Rolling the SWA ring back".
`gemma4_kv_cache_equivalence.rs` checks that the reused prefix reproduces a
full prefill.

Gemma4's SWA rings are not written to the SSD tier. A hydrated entry with a
payload-less attended layer fails `is_hydrate_complete` and misses.

---

## Qwen3.5 resume and the anchor

A recurrent state has no sequence axis, so it cannot be cut back to a shorter
prefix the way K/V can. `Qwen35MoeEntry::is_reusable_prefix_of` therefore
resumes a request only from a state the entry holds. It tries two, in this
order.

**Strict prefix**: the stored prompt is shorter than the request and equal to
its first tokens. The entry is reused whole, its end state included, and only
the tail is prefilled. A RAM entry and an SSD-hydrated one are alike here, since
a hydrated block's caches sit exactly at the prefix it claims. The strict
less-than leaves an equal prompt to the Exact arm.

**Anchor**: the entry also keeps one recurrent state saved inside its prompt, at
`Anchor::pos_for(len)` = `floor((len - 1) / 256) * 256`, the last full block
that leaves a non-empty tail. A prompt under one block saves none. The request
resumes there when it agrees with the stored prompt up to the anchor, has a
token past it, and every K/V cache passes `can_truncate_to_block`.
`prepare_reuse` then clones the entry, cuts its K/V to the anchor with
`truncate_kv_to_block` and puts the anchor's state in place of the end state
(`ReuseKind::BlockTruncate`). A request that first differs from the stored prompt
at its last token resumes here. A chat does that when its template renders the
previous generation prompt differently in the next request. A request that
differs before the anchor misses (`non_reusable`).

**Capture.** The anchor is taken while the prompt is prefilled
(`qwen3_5_moe/anchor_capture.rs`). A chunk that holds the anchor position is
forwarded in two pieces. `prefill_chunk::prefill_pieces` is the one producer of
that split. The recurrent state is cloned and evaluated before the piece that
starts at the anchor.

Every cold prefill of an architecture with recurrent state splits at the anchor
of the prompt length. The split does not depend on the request. A request that
stores no entry (no prompt-cache slots, or paged KV) splits there too, and only
skips the clone and the store. The speculative prefill forwards the prompt less
its last token. It splits at the anchor of the whole prompt, so it forwards the
pieces of the plain prefill, less the last token. The split moves where the
prefill cuts, so its logits can differ from those of a prefill that does not
split there by bf16 noise. A cold prefill forwards the same pieces whatever
`--prompt-cache-slots` or `--paged-kv` is set to. A resume and an exact hit
still differ from a cold prefill by bf16 noise. The DFlash 2 and EAGLE-3
prefills also cut every 1024 tokens, so their pieces are not the plain ones.

A resumed request stores an entry of its own. It takes the anchor of its prompt
from its tail when the tail reaches that block, and inherits the resumed entry's
anchor otherwise. A tail that starts at the anchor takes the state before its
first piece.

**Cost.** One anchor per entry, the newest. The K/V is shared with the entry and
cut on reuse, not copied. The state is a fixed size per model whatever the
prompt length, and it counts in `kv_bytes`, so the RAM cap and the eviction see
it. The engine logs a `debug!` event `qwen3_5moe: saved the recurrent state at
the anchor` per capture, and an `info!` event `qwen3_5moe: prompt-cache resume`
with `kind` (`strict_prefix` or `anchor`), `prefix_len` and `tail_len` per
resume.

`crates/rmlx-models/src/qwen3_5_moe/anchor_resume_tests.rs` judges each arm as
"Judging a resume arm" says, on the dense checkpoint: the consume branch first,
then the forwards that built the resumed state against the forwards of the cold
baseline, read off the attention calls and required equal (and equal to the
pieces the chunk size and the anchor position predict), then the argmax at every
tail position and a tail-logit difference of exactly zero between the two.
One forward over the whole prompt is compared as well, within the noise bound.

**Not covered.** The sparse MoE arch (`Qwen3_5MoeForConditionalGeneration`) shares
the hook, the capture and the split, and has no cell in the resume gate. The
Qwen3.6 checkpoint is no longer a test target. It was measured to disagree
with itself across forward lengths by more than the dense checkpoint did.
The dense checkpoint is not exact across forward lengths either. On the
pinned MLX the speculative refold test measures a disagreement between one
verify forward and stepped decode
(`crates/rmlx-models/src/speculative/round_common_tests.rs`). On the pinned
MLX, attention over several query rows takes a different kernel once the rows
pass a limit that depends on the row count and the head grouping. The
quantized matmul takes a different kernel from two rows on. Each kernel
differs from the one-row result by the same order. So a
forward of several rows is not the same function as the same rows stepped one
at a time. A resume on the sparse arch can therefore differ from a cold
prefill by more than the dense gate shows.

---

## LRU eviction

`PromptCache<E>` keeps a monotonic `seq` counter. Every hit and every `push`
advance it, and each slot carries the `seq` of its last use.

`push` runs, in order:

1. **Zero capacity**: a cache with no slots stores nothing; `push` returns
   `None`. See "Zero slots" above.
2. **Over-cap admission**: an entry whose KV alone exceeds `max_bytes` is not
   admitted and nothing is evicted; `push` returns `None`.
3. **RAM cap**: while the stored bytes plus the new entry exceed `max_bytes`,
   evict the least recently used slot.
4. **Slot count**: if the cache is full, evict the least recently used slot.

Each eviction increments `evictions` and offers the entry to the SSD spill
sink, if one is attached. `push` returns `Some(slot_index)` when it stores.

### Over-cap admission

An oversized snapshot is refused rather than stored over the cap. The RAM-cap
loop evicts only other slots, so without the refusal an empty cache would
store one entry far over the cap. Reusing such an entry is also harmful: an
Exact hit clones the snapshot, and the first decode append copies the whole
KV on write, a second full-size copy. At long context that can exceed physical
RAM and stall decode. With the refusal the repeat request prefills like the
cold one.

The guard compares `entry.kv_bytes()` with the cap and names no arch or
codec. It keeps the existing slots. An SSD hydrate whose rebuilt entry is over
the cap is a miss.

### RAM cap

| Mechanism | Detail |
|---|---|
| CLI flag | `--prompt-cache-ram-gb <f64>`, in GiB. A negative or non-finite value is refused at startup. |
| Config file | `ram_prompt_cache_gb` under `[global]` in `<RMLX_HOME>/projects.toml`, used when the flag is absent. |
| Default | 2 GiB (`DEFAULT_MAX_BYTES`), when neither is set. |
| Scope | Process-global `OnceLock`, set by `install_ram_cap` from `rmlx serve` before any model loads. A second call with another value is dropped with a `warn!`. |

The RAM cap and the slot count (`--prompt-cache-slots`) are independent.
Either can evict first on a given `push`.

---

## PrefixIndex (linear / radix)

`PrefixIndex` is the longest-prefix index over chained block digests. The
strategy is process-global, set once at serve startup by
`--prefix-index {linear|radix}` (default `linear`). The prompt cache keeps the
index in step with its slots on every push and eviction. It passes
`layout_key = 0`, since the seed already partitions by layout.

**LinearScan** (default): `find_best_prefix` walks `slots` directly. The index
is kept only so the two strategies can be compared.

**PositionalRadixTree** (`--prefix-index radix`): a port of NVIDIA Dynamo's
`PositionalRadixTree`. Each node stores `(block_hash, layout_key)` and the
`(slot_id, leaf_depth)` of every entry whose path passes through it. A lookup
walks one block at a time and stops at the first mismatch; the deepest node
with an entry wins. Removal prunes empty nodes; the node vector is
append-only until `clear`. If the tree returns a slot id no slot carries, the
lookup warns and falls back to a linear scan.

`differential_linear_vs_radix_1000_prompts` checks the two against each other
over 1000 random prompts.

---

## KVBM block manager

`crates/rmlx-models/src/block_manager/` ports NVIDIA Dynamo's `kvbm-logical`
layer. It is compiled and unit-tested. No production path calls it.

- **TinyLFU** (`tinylfu`): a 4-bit Count-Min Sketch with halving decay, over
  four FNV-1a-64 streams (the reference uses xxh3; FNV adds no dependency).
  It decays every `capacity * 10` increments with mask
  `0x7777_7777_7777_7777`.
- **MultiLruBackend** (`multi_lru`): four LRU pools keyed by TinyLFU bin,
  thresholds `[3, 8, 15]`; eviction drains the coldest pool first.
- **BlockStore** (`store`): one mutex over the slots and their state
  machine: `Reset → Mutable → Staged → Primary`; `Duplicate` beside a
  `Primary` of the same hash; `Inactive` at refcount 0; evicted to `Reset`.
- **Events** (`events`): `EventReleaseHandle` fires `Remove` once, when the
  last clone drops. `PowerOfTwoPolicy` keeps events for blocks at
  power-of-two positions only.
- **`OverflowSink`** (`overflow`): called when tier 0 evicts a block; must
  not block.
- **`BlockManager`** (`manager`): the facade — `allocate_blocks`,
  `register_blocks`, `match_blocks`, `scan_matches`.

Its seed is `CacheKey::chained_seed`, which folds `layout_key` and an
optional `lora_salt` and `mm_hash`, and no model or codec term. Its digests
are not `cache_seed`'s and cannot address `.kvb` rows. Lock order is
`attachments → store`, never reversed.

---

## SSD handoff

The SSD tier adds two hooks to `PromptCache<E>`. One blanket impl of each
serves every arch.

- **`SpillSink<E>`** (`impl SpillSink<E> for SsdSpiller` in
  `prompt_cache.rs`) — called on every eviction. It refcount-clones the
  entry's `kv_caches` and `lin_caches` (a hybrid entry whose prompt ends
  inside a block spills its anchor instead of its end state, see
  `docs/SSD_TIER.md`), evaluates them on the inference thread, and `try_send`s
  a job to a drain thread that writes the `.kvb` and its index row. A full channel drops the job with a `warn!`; eviction always
  proceeds.
- **`SsdHydrate<E>`** (`impl SsdHydrate<E> for SsdHydrator` in
  `rmlx-kv-ssd`) — called on a RAM miss. It finds the longest matching block
  prefix in the index, reads the `.kvb`, checks its metadata, and rebuilds
  the entry under the request's seed, codec and `DispatchPolicy`. An arch
  states only what a restored block becomes, as `HydratedEntry`
  (`docs/SSD_TIER.md` § "The per-arch entry impls"). Corruption deletes the
  file and row, logs a `warn!` and reads as a miss.

The sinks are attached at model load by `ssd_tier::attach_at_load` when
`--kv-ssd-cache-gb` is set. Without them an evicted entry is dropped.
`ensure` reinstalls them whenever it rebuilds the cache.

`crates/rmlx-server/tests/ssd_cache_restart.rs` drives spill, server restart
and hydrate end to end against a real `rmlx serve`.
