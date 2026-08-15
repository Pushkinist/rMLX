# Prompt Cache / Automatic Prefix Caching

Reference for the in-process prompt cache layer: how blocks are hashed, how
per-arch caches are structured, how the block manager works, and how the SSD
spill tier integrates.

---

## Overview

rMLX implements automatic prefix caching (APC) entirely in Rust, inside the
inference process. When two requests share a common prefix longer than 256
tokens, the second request reuses the KV tensors computed during the first
instead of re-prefilling from scratch.

The design has three layers:

1. **Per-arch `PromptCache<E>`** — multi-slot LRU cache in RAM. Each slot
   holds a post-prefill snapshot keyed by a chained block-hash fingerprint.
   Evicted slots are offered to the SSD tier (when enabled) before being
   dropped.

2. **SSD tier** — evicted entries are written to `.kvb` safetensors files on
   NVMe. On a RAM miss the hydrator reads the longest matching prefix block
   back into RAM. Documented separately in `docs/SSD_TIER.md`.

3. **KVBM block manager** (logical layer, currently unwired in production) — a
   port of NVIDIA Dynamo's `kvbm-logical` providing TinyLFU + multi-tier LRU
   eviction for fine-grained per-block management. Available as a building
   block for the per-arch swap follow-up.

---

## Block alignment

All prefix matching is aligned to blocks of **256 tokens** (`BLOCK_TOKENS`).
Only complete 256-token blocks are stored and matched; a trailing partial block
is never cached and is always re-prefilled.

The 256-token floor matches oMLX's `prefix_cache.py` and represents the
empirical break-even point where the prefill savings outweigh the overhead of
a cache lookup and a tail re-prefill.

A match of `k` blocks reuses the first `k * 256` tokens of the prompt. The
caller then re-prefills tokens `[k*256 .. len)` on top of the restored
snapshot.

---

## Block hashing

Block fingerprints use **chained FNV-1a-64**.

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

Because each block's digest folds in the previous block's digest as its seed,
comparing `digest[k]` proves the entire `(k+1)*256`-token prefix is
byte-identical — no per-token rescan is needed at lookup time.

### Layout-key salt

When the SSD tier is active, the starting seed is salted:

```
seed = FNV_OFFSET ^ layout_key
```

where `layout_key` is a stable FNV-1a-64 hash over
`(arch, n_layers, n_kv_heads, head_dim, kv_quant)` computed in
`ssd_tier::compute_layout_key`. Salting ensures that two snapshots of the
same prompt at different KV layouts (e.g. `k8v8` vs `k8v4`) produce disjoint
digest streams and cannot accidentally share cache blocks.

When the SSD tier is off (no salt, `layout_key = 0`), the seed is
`FNV_OFFSET ^ 0 = FNV_OFFSET` — the same as the bare un-salted form. RAM-only
behaviour is therefore byte-identical to the legacy pre-SSD path.

The seeded variant is `chained_block_hashes_seeded(ids, seed)`. The un-seeded
convenience wrapper `chained_block_hashes(ids)` calls it with the bare
`FNV_OFFSET`. All production callers pass the salted seed; the bare wrapper
is retained for tests and for backward-compat verification.

The SSD index also stores `(hash, layout_key)` as a composite primary key,
providing defence-in-depth against hash collisions across layouts.

### Codec namespacing (issue #26)

A single resident model can serve requests under **different KV codecs**
(per-request `kv_quant` hot-swap, no weight reload — see `docs/SERVER.md`).
Because cached K/V bytes are codec-specific, a prefix cached under `none` (bf16)
**must not** serve a `k8v4` request. The block-hash seed therefore folds in a
per-codec salt as well:

```
seed = FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt()
```

`KvQuant::cache_key_salt()` is a stable FNV-1a-64 hash over the codec's
canonical `Display` string (`"none"`, `"k8v4"`, `"mixed_k8g64_v4g64"`, …), so it
covers every variant including payload-bearing ones (`Mixed`, `RotK`,
`RotorK*Asym`). Both the push side (storing a slot) and the
[`find_best_prefix`](#find_best_prefix-lookup) query side salt with the same
value, so two requests with identical tokens but different codecs produce
**disjoint digest streams** and occupy **distinct cache slots**. A codec switch
is a clean cross-codec miss — the other codec's slot survives and is reusable
again under its own codec, rather than being thrash-evicted.

On a single-codec RAM-only run (`layout_key = 0`, constant codec) the salt is a
constant XOR for every request, so hit/miss behaviour is identical to the legacy
stream for that codec — zero regression.

> The SSD `layout_key` already mixes `kv_quant` into its hash, so on an
> SSD-active run the codec is salted twice (`layout_key` *and*
> `cache_key_salt`); the extra XOR is harmless (still a deterministic salt) and
> keeps the RAM-only path correct, where `layout_key = 0` would otherwise leave
> the codec un-partitioned.

---

## Per-arch caches

Each architecture owns a process-global `static` of type
`ArchPromptCache<E>`:

| Arch | Static | Entry type | Reuse policy |
|---|---|---|---|
| `Gemma4ForConditionalGeneration` | `gemma4::prompt_cache::PROMPT_CACHE` | `Gemma4Entry` | `Partial` |
| `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_moe::prompt_cache::PROMPT_CACHE` | `Qwen35MoeEntry` | `ExactOnly` |
| `Qwen3ForCausalLM` | `QWEN3_PROMPT_CACHE` (in `qwen3.rs`) | `Qwen3Entry` | `ExactOnly` |

### Entry types

**`Gemma4Entry`** — pure-attention arch. Holds:
- `prompt_token_ids: Vec<u32>` — full prompt used to build this snapshot.
- `block_hashes: Vec<u64>` — chained block digests computed at construction.
- `kv_caches: Vec<KvCache>` — post-prefill KV tensors for all decoder layers.
- `first_id / first_piece` — argmax token from the first decode step (used by
  the Exact fast path to skip one decode round).
- `kv_quant: Option<KvQuant>` — KV codec in effect when the snapshot was taken.
  A mismatch between the stored value and the runtime `KvQuant` triggers
  `evict_slot` and a fall-through to full re-prefill.

**`Qwen35MoeEntry`** — hybrid GDN arch. Same fields as `Gemma4Entry` plus:
- `lin_caches: Vec<LinearAttnCache>` — GatedDeltaNet recurrent states. Not
  truncatable: `truncate_kv_to_block` trims only `kv_caches` and deliberately
  leaves `lin_caches` intact, because recurrent state cannot be reconstructed
  from a block-truncated KV prefix. This is the direct cause of `ExactOnly`.

**`Qwen3Entry`** — pure-attention dense arch (Bonsai). Identical shape to
`Gemma4Entry` minus the SWA helpers, plus a `first_logprobs:
Option<TokenLogprobs>` field. Uses `ExactOnly` for simplicity (the dominant
workload is identical-prompt warm-TTFT, which is an Exact hit).

**First-token logprob on Exact hit.** The cached `first_id` token is
replayed on a hit without re-running the prefix's last-position logits, so its
logprob is not otherwise recomputable. To keep the OpenAI contract (exactly one
`logprobs.content` entry per emitted token), `Qwen3Entry` carries
`first_logprobs`: the prefill token's top-k logprobs captured from the raw
prefill logits at store time, at the OpenAI `top_logprobs` ceiling (20),
independent of the storing request's `top_logprobs_k`. On an Exact hit with
`logprobs` enabled the engine replays this record truncated to the replaying
request's `top_logprobs_k`, yielding the SAME `token_logprob` the Miss path
would emit (true value, not a placeholder). The `lp_k == 0` path emits `None`
so the zero-overhead decode stays byte-identical. SSD-hydrated entries store no
first decode token, so their `first_logprobs` is `None`.

### `PromptCacheEntry` trait contract

Every entry type implements:

| Method | Purpose |
|---|---|
| `prompt_token_ids() -> &[u32]` | Full token sequence; used for Exact-path verification and token-identity gate. |
| `block_hashes() -> &[u64]` | Chained 256-token block digests; used by `find_best_prefix`. |
| `deep_clone() -> Result<Self>` | Refcount-increment clone of all MLX arrays (copy-on-write; no tensor data copied). |
| `truncate_kv_to(prefix_len)` | Trim KV caches to `prefix_len` positions in-place. |
| `truncate_kv_to_block(block_count)` | Block-aligned variant: equivalent to `truncate_kv_to(block_count * 256)`. |
| `kv_bytes() -> u64` | Best-effort RAM estimate for eviction budget. |

### `ArchPromptCache<E>` generic shell

`ArchPromptCache<E>` collapses all per-arch boilerplate into one type:

- `inner: Mutex<Option<PromptCache<E>>>` — the actual cache; `None` until
  `ensure` initialises it for the first request.
- `attach: Mutex<Option<AttachParams>>` — SSD tier attachment parameters
  (namespace, `kv_quant`, `layout_key`, device). Recorded so they survive a
  capacity-change cache re-creation.
- `policy: ReusePolicy` — read by the generate loop; enforced at runtime, not
  by a comment.

The resident-KV byte counter is **not** here. It is per model *instance*
(`kv_bytes::KvBytesCounter`, a field on each arch's model struct) because this
shell is per arch *type*: two models of the same architecture sharing it would
cross-attribute each other's byte totals into the append-only `events` table.
See `docs/METRICS_DB.md` §`kv_cache_bytes`.

`ArchPromptCache::ensure(capacity)` is a no-op when the existing cache already
has the correct capacity. If the capacity changes (e.g. `--prompt-cache-slots`
changes between model loads), the cache is rebuilt and the SSD sinks are
re-installed from the recorded `attach` params.

`ensure` runs once per generation on every arch, so the comparison is against
what the cache actually stores and holds for every value, `0` included. A
capacity that never compares equal to itself would rebuild on every request —
discarding snapshots, resetting the hit/miss counters and re-installing the SSD
sinks each time. That reads as "caching is off" from the outside, and it zeroes
any measurement taken as `after - before` around a generation.

### Zero slots

`--prompt-cache-slots 0` disables the cache as a real state. The cache object is
still built and still counts its misses, but `push` refuses every entry, so
`slots` stays empty, `find_best_prefix` can only miss, and every request runs a
full prefill. Nothing is clamped to one slot.

The SSD tier is disabled with it: `hydrate_from_ssd` returns before querying the
source, because a hydrated entry could only be refused admission. So a zero-slot
server keeps `ssd_hits` at 0 and does no `.kvb` reads — the "every request runs a
full prefill" above holds literally, not just for the RAM tier.

A request carrying an `X-Session-Id` header does not change this. Session
KV-reuse widens the configured slot count by one slot per active session
(`session_cache::effective_prompt_cache_slots`), and a configured `0` is left
alone: a header must not switch on a cache the operator switched off, and
alternating capacities would rebuild the cache on every request.

To make a *single* request miss without changing the configuration, use
`ArchPromptCache::clear()` (`Architecture::clear_prompt_cache`), which empties
the slots and resets the counters while keeping the capacity and the installed
SSD sinks. That is what `rmlx bench` does: measuring a zero-slot cache would
time a cache no operator runs.

For `clear()` — unlike zero slots — a RAM miss is still not the same as a
prefill: it leaves the SSD source attached, so the next request can be served
from a `.kvb` and recorded as `ssd_hits`. A caller that needs a real prefill
checks `hits == 0 && ssd_hits == 0` rather than trusting the clear.

---

## `find_best_prefix` lookup

```
find_best_prefix(prompt_ids, seed) -> Option<(slot_index, matched_blocks)>
```

The `seed` (issue #26) is the `FNV_OFFSET ^ layout_key ^ codec_salt` the caller
also uses on the push side, so the query digest stream partitions by
`(layout, codec)` — a slot stored under a different KV codec or SSD layout never
matches. Pass the bare `FNV_OFFSET` for the legacy un-salted stream (RAM-only,
single-codec, `layout_key = 0`).

1. Compute chained block hashes for `prompt_ids` using `seed`.
2. Scan all slots (Linear path) or query the radix tree (Radix path).
3. For each slot count how many leading block digests match.
4. Return `Some((best_slot_index, best_block_count))` where
   `best_block_count >= 1`. Return `None` if no slot shares at least one
   full block.
5. On a hit, advance the winning slot's `last_used_seq` (MRU stamp).

Block-count statistics are accumulated: `block_hits`, `block_misses`,
`partial_hits` (hit where `best_blocks < want_blocks`).

Eviction is never triggered inside `find_best_prefix`. The caller calls
`push` after the lookup to store the new snapshot.

---

## ReusePolicy

`ReusePolicy` is a hard runtime gate (not a comment) on the generate loop's
`CacheLookup` arm:

```rust
pub(crate) enum ReusePolicy {
    Partial,    // Gemma4: block-aligned partial-prefix reuse allowed
    ExactOnly,  // Qwen3 / Qwen3.5-MoE: full-token-equality only
}
```

**`Partial`** (Gemma4): a block-aligned partial hit (`best_blocks <
want_blocks`) may be taken. The generate loop deep-clones the slot, calls
`truncate_kv_to_block(best_blocks)`, and re-prefills the tail. There is an
additional per-slot gate: `Gemma4Entry::can_truncate_to_block` checks whether
every layer cache is trimmable (`KvCache::is_trimmable`). A SWA
`RotatingKvCache` whose ring buffer has wrapped is not trimmable; in that
case the partial path falls back to Miss.

**`ExactOnly`** (Qwen3, Qwen3.5-MoE): any block-level match that is not a
full-token-equality Exact hit is routed to `CacheLookup::Miss` and triggers a
full re-prefill. The Exact path verifies token identity by comparing
`entry.prompt_token_ids()` byte-for-byte with the incoming prompt.

---

## LRU eviction

`PromptCache<E>` maintains a monotonic `seq: u64` counter (no syscall). Every
`find_best_prefix` hit and every `push` advance it. Each slot carries a
`last_used_seq` stamped at its last access.

`push` runs an admission guard, then two eviction passes, before appending the
new entry:

0. **Zero-capacity guard**: a cache configured with no slots stores nothing —
   `push` returns `None`. See "Zero slots" above.
1. **Over-cap admission guard**: if the incoming entry's KV alone exceeds
   `max_bytes`, the entry is **not admitted** — `push` returns `None` without
   evicting any existing slot. See "Over-cap admission" below.
2. **RAM cap**: while `total_kv_bytes + new_entry_bytes > max_bytes`, evict
   the slot with the smallest `last_used_seq`.
3. **Slot count cap**: if `slots.len() == capacity`, evict the slot with the
   smallest `last_used_seq`.

Either cap triggers independently; the smaller cap wins. Each eviction
increments `stats.evictions` and calls `spill_evicted` (the SSD sink
hook, if attached). `push` returns `Some(slot_index)` when the entry is stored
and `None` when the admission guard rejects it.

### Over-cap admission

A single snapshot whose resident KV alone exceeds `max_bytes` is **refused
admission** rather than stored above the cap. Without this guard, an empty (or
near-empty) cache would happily store one entry many times larger than the cap —
the RAM-cap eviction loop only evicts *other* slots and never refuses the
incoming entry — silently violating the documented cap.

The refusal is not just bookkeeping hygiene: admitting an over-cap snapshot is
what caused the large-KV **warm-cache decode stall**. The next identical (warm)
request takes the Exact path, which `deep_clone`s the stored snapshot
(refcount-shared, no copy) and then, on the first decode append, triggers MLX
copy-on-write of the whole KV — a *second* full-size residency. For a
bf16-mirror KV codec at long context (tens of GB), that doubling pushes total
residency past physical RAM and stalls decode with a single multi-hundred-second
pause (steady-state `itl` stays healthy; only the aggregate craters).

Refusing admission bounds peak residency to one live copy: the repeat request
re-prefills exactly like the cold request instead of reusing an over-cap slot,
so warm decode ≈ cold decode. The guard is model- and codec-agnostic — it keys
off `entry.kv_bytes()` versus the cap, never an arch or codec name. Existing
(smaller, valid) slots are left intact; the guard never evicts to make room for
something that still could not fit. An SSD hydrate whose reconstructed block is
over-cap is likewise treated as a miss (the caller re-prefills).

The RAM cap is set once at process start:

- CLI `--prompt-cache-ram-gb` (takes precedence).
- Env `RMLX_PROMPT_CACHE_MAX_BYTES` (bytes, decimal; silent fallback).
- Default: 2 GiB.

`install_ram_cap(cli_gib)` is called once from `rmlx serve` before any model
loads. A second call with a different value is dropped with a `warn!`
(idempotent — matches the `ssd_tier::install_config` pattern).

---

## KVBM block manager

The `block_manager` module (`crates/rmlx-models/src/block_manager/`) is a
port of NVIDIA Dynamo's `kvbm-logical` layer, providing fine-grained per-block
management as a building block for the per-arch swap follow-up. It is compiled
and unit-tested but not yet wired into production inference paths.

### Layers

**TinyLFU** (`tinylfu`): 4-bit Count-Min Sketch with halving-decay aging.
Four independent FNV-1a-64 hash streams (derived from four stable `u64` seeds)
replace the reference's xxh3 with 192-byte secrets. Same algorithmic
properties: 4 independent CMS slots, 4-bit ceiling, `decay_threshold =
capacity * 10` increments, decay mask `0x7777_7777_7777_7777`.

**MultiLruBackend** (`multi_lru`): 4-tier LRU keyed by TinyLFU bin.
Default bin thresholds `[3, 8, 15]` map TinyLFU counts to pools 0–3 (0 =
coldest). Eviction drains pool 0 first, then 1, 2, 3. Match lookups walk all
four pools. Each pool is a `VecDeque<BlockHash>`; `touch` moves a block to the
back of its current (or re-binned) pool. Single-threaded by the surrounding
store mutex.

**BlockStore** (`store`): single-mutex store + slot state machine.

Slot lifecycle:

```
Reset ──allocate──> Mutable ──register──> Staged ──commit──> Primary
│
└─dup─> Duplicate
Primary ──refcount=0──> Inactive ──reuse──> Primary
│
└──evict──> Reset  (payload offered to OverflowSink)
```

States:
- `Reset` — free, in the reset pool.
- `Mutable` — allocated, no hash assigned; writable.
- `Staged` — hash assigned but not committed to `active_by_hash`.
- `Primary` — committed, live references outstanding.
- `Duplicate` — same hash as a Primary; lives parallel until refcount drops.
- `Inactive` — refcount 0; eligible for resurrection or eviction.

**`EventReleaseHandle`**: RAII `Arc`-backed token. The `Remove` event fires
exactly once when the last clone drops, matching the reference's
`Arc<Inner> + Drop` semantics. Multiple `ImmutableBlock`s cloning the same
registration share one handle.

**`PowerOfTwoPolicy`**: filters event batches keeping only blocks at
power-of-two positions (1-indexed: 1, 2, 4, 8, …), keeping event volume
O(log N) per batch.

**`OverflowSink`**: trait called when the inactive index evicts a block from
tier 0. The production implementation wraps `SsdSpiller` and translates the
payload to a spill job. `offer_evicted` must not block the decode thread.

**`BlockManager`** (`manager`): public facade over `BlockStore`. Provides
`allocate_blocks`, `register_blocks`, `match_blocks`, and `scan_matches`.

**Hash family**: FNV-1a-64 with `layout_key` mixing (`FNV_OFFSET ^
layout_key`), consistent with `chained_block_hashes_seeded`. The reference
uses xxh3; rMLX uses FNV to avoid adding a new dependency.

**Lock order**: `attachments → store`, never reversed. The store mutex is
never held while calling into an `OverflowSink` (non-blocking `try_send`).

---

## PrefixIndex (linear / radix)

`PrefixIndex` is a pluggable longest-prefix index over chained block hashes.
The active strategy is a process-global set once at serve startup via
`--prefix-index {linear|radix}` (default: `linear`).

### LinearScan (default)

O(slots × n_blocks) walk. Maintains a parallel `Vec<LinearEntry>` mirroring
every `(chained_hashes, layout_key)` in `PromptCache::slots`. On
`find_best_prefix` the linear path ignores this index and walks `slots`
directly — the index is maintained in lockstep for differential-testing parity
with the Radix path, not for the lookup itself.

### PositionalRadixTree (`--prefix-index radix`)

Port of NVIDIA Dynamo's `PositionalRadixTree`. Arena-allocated node vector;
children are `Vec<NodeId>` (small `u32`). Each node stores `(block_hash,
layout_key)` and a list of `(slot_id, leaf_depth)` tuples for every entry
whose chained-hash path passes through it.

Lookup (`match_best`): walk one block at a time, descending to the child
matching `(next_block_hash, layout_key)`. The deepest visited node with a
tuple wins. Stops on the first mismatch.

Complexity: O(n_blocks · avg_fanout · avg_entries_per_node). Fanout stays
small in practice (bounded by the number of distinct continuations under any
shared prefix).

Insert: walk or create nodes along the chained-hash sequence, stamp
`(slot_id, leaf_depth)` at every node. Overwrite at the same key with a
different `slot_id` first removes the prior path tuples (`evict_slot_path`)
then inserts the new ones — mirrors LinearScan's overwrite contract.

Remove: walk the path leaf-to-root, remove the `(slot_id, leaf_depth)` tuple
from each node, prune payload-less child-less nodes. The node vector is
append-only (orphaned nodes leak until `clear`), bounded by the working set.

Layout disambiguation: entries with the same `chained_hashes` but different
`layout_key` are stored on separate branches — the same composite PK rule as
the SQLite SSD index.

The Radix tree is verified against LinearScan by a 1 000-random-prompt
differential test that asserts Some/None parity, `n_matched_blocks` equality,
and (when the prefix is unambiguous) `slot_id` identity.

---

## SWA snapshot/restore (Gemma4)

Gemma4 mixes full-attention and sliding-window attention (SWA) layers. SWA
layers use `RotatingKvCache`, which is a ring buffer of the last
`sliding_window` K/V tokens.

**`can_truncate_to_block(block_count)`**: returns true iff every layer cache
passes `KvCache::is_trimmable`. A SWA cache whose ring has wrapped (i.e.
`offset >= sliding_window`) is not trimmable — block-aligned truncation
would silently no-op on SWA layers while trimming full-attention layers,
desyncing the caches. When `can_truncate_to_block` returns false, the partial
prefix hit degrades to Miss and the request falls back to full re-prefill.

**`is_strict_prefix_of(prompt_ids)`**: returns true iff this entry's full
token sequence is a strict prefix of `prompt_ids` (i.e. the new prompt
extends the cached prompt by at least one token). Requirements:
- `cached_len >= BLOCK_TOKENS` (256-token worthwhile floor).
- `cached_len < prompt_ids.len()` (strict extension, not equal-length).
- `cached_tokens[..cached_len] == prompt_ids[..cached_len]` (byte-identical).

When `is_strict_prefix_of` is true the generate loop takes the B1 strict-prefix
path: deep-clone the slot, do not truncate, re-prefill only the tail
`prompt_ids[cached_len..]`. This path is correct for wrapped-SWA because the
snapshot already holds the last `sliding_window` K/V tokens of the cached
prefix — exactly what the tail attends to — and no truncation is performed.

---

## Partial-prefix hit / `truncate_kv_to_block`

When `find_best_prefix` returns `(slot_idx, k)` where `k < want_blocks` and
the arch policy is `Partial`:

1. Call `deep_clone` on the winning slot's entry (refcount-increment, no tensor
   copy).
2. Gate on `can_truncate_to_block(k)` (Gemma4 only; fails if any SWA ring is
   wrapped).
3. Call `truncate_kv_to_block(k)` on the clone — trims all KV caches to
   `k * 256` sequence positions. For Qwen3.5-MoE, `lin_caches` are NOT
   trimmed; but the `ExactOnly` policy means this method is never called in
   production for that arch.
4. Re-prefill tokens `[k*256 .. len)` on top of the trimmed clone.
5. After the request completes, `push` the new full snapshot into the cache.

---

## SSD handoff

The SSD tier adds two hooks to `PromptCache<E>`:

- `SpillSink<E>` — called when a slot is evicted (RAM-cap or slot-count). The
  production implementation (`SsdSpiller`) refcount-clones the evicted caches,
  materializes GPU buffers on the inference thread (`eval_for_spill`), and
  `try_send`s a `SpillJob` onto a bounded channel. A dedicated drain thread
  serializes the job to a `.kvb` safetensors file and records the block in
  `SsdKvIndex`. Back-pressure drops the job with a `warn!`; eviction from RAM
  always proceeds regardless.

- `SsdHydrate<E>` — called by `hydrate_from_ssd` on a RAM-cache miss. Queries
  the `SsdKvIndex` for the longest matching block-hash prefix, reads the `.kvb`
  file, verifies `model_id` and `kv_quant` metadata, and reconstructs the arch
  entry. Corruption (bad read, metadata mismatch, missing file) is handled
  internally (delete + `warn!`) and surfaces as `Ok(None)` — the caller falls
  through to full re-prefill.

Both sinks are arch-specific because the spill job differs:
- Gemma4 / Qwen3: `SpillJob { kv_caches, lin_caches: vec![] }`.
- Qwen3.5-MoE: `SpillJob { kv_caches, lin_caches }` (both fields populated).

The sinks are attached at model load by `ssd_tier::attach_at_load` → per-arch
`attach_ssd_tier`, gated by `--kv-ssd-cache-gb`. When not attached, eviction
drops entries silently (the pre-SSD behavior).

`ArchPromptCache::ensure` reinstalls the sinks from the recorded `AttachParams`
whenever the cache is rebuilt (capacity change), so a capacity bump never
silently drops the tier.

### `ssd_cache_restart` smoke test

`crates/rmlx-server/tests/ssd_cache_restart.rs` is an end-to-end integration
test (`#[ignore]`, env-gated on `RMLX_TEST_MODEL`) that proves the full
spill → restart → hydrate chain:

1. Start `rmlx serve` with `--kv-ssd-cache-gb 1 --prompt-cache-slots 1`.
2. Send a long prompt A; confirm the response is coherent.
3. Send a second request to force eviction of A to SSD.
4. Kill the server, clear the Metal claim file.
5. Restart with the same `--kv-ssd-cache-gb` and same `RMLX_HOME`.
6. Send prompt A again; assert the response is byte-identical (SSD hydration).

Each run uses a fresh `RMLX_HOME` tempdir for hermeticity.

---

## RAM cap

| Mechanism | Detail |
|---|---|
| CLI flag | `--prompt-cache-ram-gb <f64>` — value in GiB, converted to bytes. |
| Env fallback | `RMLX_PROMPT_CACHE_MAX_BYTES` — bytes, decimal (undocumented compat). |
| Default | 2 GiB. |
| Scope | Process-global `OnceLock`; first call to `install_ram_cap` wins. |
| Admission guard | `push`: an entry whose KV alone exceeds `max_bytes` is refused (`push` → `None`), no eviction. See "Over-cap admission". |
| Eviction trigger | `push`: evicts LRU slots until `total_kv_bytes + new_entry_bytes <= max_bytes`. |

The RAM cap and the slot count cap (`--prompt-cache-slots`) are independent.
Either can trigger eviction first on a given `push`. The over-cap admission
guard runs first: a snapshot larger than the whole cap is never stored (it would
both violate the cap and cause the warm-cache decode stall on reuse).

---

## See also

- `docs/KV_CACHE.md` — KV quantization codec reference (`--kv-quant`,
  `--cache-type-k` / `--cache-type-v`).
- `docs/SSD_TIER.md` — SSD cache tier: `.kvb` file format, `SsdKvIndex`
  schema, eviction budget, namespace layout.
- `docs/MODELS.md` — per-arch generate loop, `CacheLookup` match arms,
  prefill / decode pipeline.
