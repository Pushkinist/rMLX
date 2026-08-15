# SSD KV-Cache Tier

rMLX operates a three-tier KV-cache hierarchy. The SSD tier is the third and
lowest level: it persists prompt-cache blocks evicted from RAM to disk so that
they can be reloaded on subsequent requests without a full re-prefill.

> **Crate ownership:** The SSD tier lives in the workspace member crate
> **`rmlx-kv-ssd`** — `crates/rmlx-kv-ssd/`. It owns the index, spill,
> hydrate, block-I/O, layout-key salt, the 5 Prometheus hook globals, the
> `SsdHydrate<E>` trait, and the chained FNV-1a-64 block-digest helpers
> (`BLOCK_TOKENS`, `chained_block_hashes`, `chained_block_hashes_seeded`,
> `FNV_OFFSET`, `FNV_PRIME`). The per-arch `attach_ssd_tier` dispatcher
> (Gemma4 / Qwen3 / Qwen3.5-MoE) stays in `rmlx-models::ssd_tier` because the
> arch-specific `SsdHydrate<Entry>` / `SpillSink<Entry>` impls live in
> `rmlx-models`. Dep edge: `rmlx-models → rmlx-kv-ssd → rmlx-kv-quant`
> (acyclic; no back-edge).

## Overview

The primary purpose is **cross-restart prompt reuse**. Because the SSD index
survives process boundaries, a long system prompt or repeated document context
that was prefilled in a previous `rmlx serve` session is available immediately
on the next request, subject to the on-disk budget.

The tier is **off by default**. It activates when `--kv-ssd-cache-gb > 0` is
passed at startup. When off, the spill and hydrate hooks are never installed and
decode is byte-identical to the RAM-only path.

Supported architectures (spill + hydrate wired):
- `Gemma4ForConditionalGeneration`
- `Qwen3ForCausalLM`
- `Qwen3_5MoeForConditionalGeneration`

Other architectures silently remain RAM-only when the tier is enabled; the SSD
flag itself is not an error.

---

## Architecture — 3-Tier Hierarchy

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 1 — GPU RAM (Metal)                                           │
│  KvStorage: K8V4 / K8V8 / Planar / Paged / Mixed / RotK            │
│  Resident in the per-arch PromptCache (LRU, RAM-capped).            │
│  All active decode sessions read/write here.                        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ RAM-cap eviction
                                ▼ (SpillSink → try_send → bounded channel)
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 2 — Drain Thread (host CPU / RAM transit)                     │
│  SsdSpiller: bounded sync_channel (depth 16).                       │
│  Hot path: refcount-clone only (no tensor copy).                    │
│  Drain thread: block_io::write_caches (MLX eval → host bytes).     │
│  Serializes to .kvb (safetensors) + records in SsdKvIndex.         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ flush to disk
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Tier 3 — SSD                                                       │
│  <RMLX_HOME>/cache/kv/<namespace>/                                  │
│    <hash>.kvb          — safetensors block file (one per KV slot)   │
│    index.db            — SsdKvIndex v3 (SQLite, WAL mode)           │
│                                                                     │
│  On RAM miss: SsdHydrator reads .kvb → GPU upload → promote to     │
│  Tier 1 without re-prefill.                                         │
└─────────────────────────────────────────────────────────────────────┘
```

Block granularity: **256 tokens per block** (`BLOCK_TOKENS = 256`). Only whole
blocks are stored; trailing partial blocks are always re-prefilled.

---

## SsdTierConfig

`SsdTierConfig` is the process-global config set once at serve startup by
`ssd_tier::install_config`. Fields are frozen; cross-crate construction in
`rmlx-cli::serve` must name all four.

| Field | Type | Meaning |
|---|---|---|
| `per_namespace_budget_bytes` | `u64` | Per-namespace on-disk cap (bytes). Maps from `--kv-ssd-cache-gb`. |
| `global_budget_bytes` | `u64` | Cross-namespace pool ceiling across all `<RMLX_HOME>/cache/kv/*` namespaces. `0` = no global cap. Maps from `--kv-ssd-global-gb`. |
| `default_namespace` | `Option<String>` | Override namespace name from `--project`. `None` → falls back to model id at `attach_at_load`. |
| `per_project_budgets` | `BTreeMap<String, u64>` | Per-project budget overrides from `projects.toml`. Empty when the file is absent or has no matching section. |

`install_config` is a `OnceLock`: the first call wins. A second call panics in
debug builds; in release builds it logs a warning and is ignored. Both budget
fields zero → tier OFF; the `OnceLock` stores `None`.

At startup, `install_config` runs two operations before any model loads:

1. **Pre-release schema wipe** — scans every `<RMLX_HOME>/cache/kv/<ns>/`
   directory. If `index.db` holds a `kv_blocks` table at any `schema_version`
   other than the current one (a missing `schema_version` table reads as the v1
   layout), the entire namespace directory is removed (`fs::remove_dir_all`)
   and a `ssd_cache_pre_release_wipe` tracing event records the dropped bytes.
   Idempotent: a clean boot at the current schema is a no-op. A directory with
   no `kv_blocks` table is never touched.

2. **Cross-namespace LRU enforcement** — when `global_budget_bytes > 0`,
   calls `evict_pool_lru_until` to bring the pool under the global ceiling
   before any namespace-level startup maintenance runs.

Effective per-namespace ceiling when both budgets are set:
`min(per_namespace_budget_bytes, global_budget_bytes)`.

---

## SsdKvIndex — Schema v3

Each namespace has one SQLite database at
`<RMLX_HOME>/cache/kv/<namespace>/index.db`. The database runs in WAL mode
with `synchronous = NORMAL` and a 5-second busy timeout.

```text
kv_blocks (
    hash        TEXT    NOT NULL,   -- chained FNV-1a-64 digest, hex string
    layout_key  INTEGER NOT NULL,   -- stable u64 from compute_layout_key()
    path        TEXT    NOT NULL,   -- absolute path to the .kvb file
    model_id    TEXT    NOT NULL,   -- "<arch>/<snapshot>" identity string
    kv_quant    TEXT    NOT NULL,   -- KvQuant Display string (e.g. "k8v4")
    byte_size   INTEGER NOT NULL,   -- on-disk byte size of the .kvb file
    last_used   INTEGER NOT NULL,   -- unix epoch MICROSECONDS (see below)
    PRIMARY KEY (hash, layout_key)
)

schema_version (
    version     INTEGER PRIMARY KEY NOT NULL   -- 3
)
```

### last_used is the LRU total order

Eviction sorts on `last_used` and nothing else, so the column has to be a
**total order over accesses**, not just an approximate one. `record` and
`touch` stamp it from `ssd_index::now_stamp_us`, which reads the wall clock in
microseconds and clamps the result above the highest value the process has
already issued. That covers the two ways a raw clock stops being usable here:
several writes inside one microsecond, and a clock stepped backwards by NTP.

Seconds do not work. Under any realistic request rate many blocks land in the
same second, `ORDER BY last_used ASC` has no tiebreak, and the victim becomes
whichever row SQLite happens to yield first — a block used 900 ms ago can be
discarded ahead of one used 100 ms ago, so the tier can throw away exactly the
prefix it is about to be asked for. The degradation toward random replacement
grows with request rate, i.e. it is worst when the tier matters most.

It stays a wall-clock value rather than a plain counter for two reasons: the
cross-namespace sweep merges rows from independent namespace databases and has
to compare them (a per-database counter means nothing outside its own
database), and the ordering has to survive a restart.

Because the stamp source is already a total order, no tiebreak column is
needed and the `kv_blocks_last_used` index stays single-column — the index that
keeps eviction from sorting the whole table.

### Composite Primary Key

The `(hash, layout_key)` composite PK is defence-in-depth. The chained digests
stored under a given `layout_key` are already disjoint from those of any other
layout (the salt enters the FNV seed — see layout_key salt section). The
composite PK catches any upstream regression where the salt is accidentally
omitted.

Two distinct `layout_key` values for the same arch and project may hold blocks
from the same prompts without ever colliding on a row. Two snapshots of the
same arch with identical weights at the same `kv_quant` share the same
`layout_key` and therefore share their cached blocks, which is the intended
behaviour — `layout_key` is weight-independent on purpose.

### Schema Version Enforcement

`SsdKvIndex::open` inspects the DB before touching any table:

- File absent → creates the current schema, inserts `schema_version = 3`.
- File present, `schema_version` table missing → `SchemaMismatch` (pre-release v1 DB; should have been wiped by `install_config`).
- File present, `schema_version != 3` → `SchemaMismatch` (a schema this binary cannot interpret).
- File present, `schema_version == 3` → open succeeds.

### Migration

rMLX is unreleased, so every schema transition is a **one-time wipe**, not a
row-by-row migration. The tier holds nothing that is not regenerable, so the
cost of a wipe is one re-prefill per dropped block.

| From | What changed | What happens to existing data |
|---|---|---|
| v1 (no `schema_version` table, `hash` is the sole PK) | `layout_key` column + composite PK + `schema_version` table | Namespace dir removed at startup |
| v2 | `last_used` unit: seconds → microseconds | Namespace dir removed at startup |

`install_config` removes the entire namespace directory for any `index.db`
whose version is not the current one, before any `SsdKvIndex::open` call runs
in-process. A v2 table could not simply be reused: the shape is identical, so
new microsecond rows and old second rows would coexist three orders of
magnitude apart, reporting nonsense to any operator query and leaving the old
rows as coarsely ordered among themselves as before.

The mismatch is wiped in **either** direction, not only for older versions. A
namespace this binary cannot read has exactly two possible fates — reclaim the
bytes, or strand them while `SsdKvIndex::open` disables the namespace for the
run — and nothing else reclaims a stranded namespace.

---

## layout_key Salt

`compute_layout_key` produces a stable `u64` from the five-tuple
`(arch, n_layers, n_kv_heads, head_dim, kv_quant)`:

```text
input  = arch.as_bytes() + ":<n_layers>:<n_kv_heads>:<head_dim>:<kv_quant>"
h      = FNV_OFFSET  (0xcbf29ce484222325)
for each byte b in input:
    h ^= u64(b)
    h  = h * FNV_PRIME  (0x00000100000001b3)
layout_key = h
```

The key is deterministic across runs and distinct with overwhelming probability
for every distinct input tuple. It is **not** weight-dependent: reloading the
same architecture at the same `kv_quant` from a different snapshot yields the
same `layout_key` and shares the cache, which is correct.

The key salts the chained block hash stream at the call site:

```text
seed    = FNV_OFFSET ^ layout_key
digests = chained_block_hashes_seeded(ids, seed)
```

XOR mixing keeps the existing FNV avalanche behaviour intact. Different layouts
produce disjoint digest streams for the same token sequence, so a block cached
under one KV layout cannot collide with the same block cached under another.

`layout_key` is resolved and logged (as a 16-char hex string, field
`layout_key`) at `attach_at_load` before the index is opened, so operators can
correlate log rows with the active layout during debugging.

## Live reconfiguration

Per-request KV **codec** and **context ceiling** are hot-swappable on a resident
model (see `docs/SERVER.md` § "Per-request KV-config hot-swap"). The SSD tier is
**deliberately excluded** from that per-request surface: it stays launch-fixed
(`--kv-ssd-cache-gb`, attached once at `attach_at_load`).

**Why it is not a per-request override.** The tier is wired into the prompt cache
via a per-namespace spiller + hydrator keyed by `(namespace, layout_key)`, opened
against an on-disk `ssd_index` and `.kvb` block files at load. A per-request SSD
toggle would have to tear that down and re-attach it mid-flight (close/reopen the
index, respawn the spiller, re-key the hydrator) — architecturally heavy and
orthogonal to the read-only-weights insight that makes codec/ctx hot-swap cheap.

The codec partitioning applied to the **RAM** prompt cache
(`KvQuant::cache_key_salt()` XOR'd into the block-hash seed) composes correctly
with `layout_key`: on an SSD-active run the seed mixes both, so RAM slots are
codec-partitioned even though the SSD tier itself stays single-codec for the
resident lifetime.

### What a live *budget* change would take

The per-namespace budget is now a live quantity: the spill drain thread evicts to
it after every block (§ "Evict-to-budget (runtime)"), so changing it changes tier
behaviour immediately rather than at the next model load. What is missing is only
the surface to change it through — the value is resolved once at
`SsdSpiller::spawn` from the `OnceLock` config, and there is no control-plane
route for it.

Such a change would **not** need a drain of in-flight requests. Eviction is
already concurrency-safe by construction (same argument as § "Evict-to-budget
(runtime)"), and the budget has no bearing on how a block is keyed, read, or
decoded.

### What a live *enable / disable* would take

Detaching the tier on a resident model means dropping the spiller + hydrator from
the live `PromptCache<E>` **and** clearing the recorded `AttachParams`, so that
`ArchPromptCache::ensure` does not re-install them the next time the cache is
rebuilt for a capacity change.

This too is not corruption-shaped, for a structural reason worth stating: within
one resident model, `layout_key` can only change if `kv_quant` changes, and
`layout_key` is *lookup namespacing*, not data layout. Because the block digests
are chained from the seed, a query computed under one seed cannot partially match
a slot stored under another — block 0 already differs, `find_best_prefix` returns
`None`, and the request re-prefills. A mid-flight enable or disable can therefore
lose spills and strand slots as unfindable, but cannot produce a wrong-length or
wrong-codec reuse.

What it *would* need is a decision on the control-plane surface. The existing
lifecycle routes are per-model (`POST /v1/models/{id}/load` / `/unload`,
`GET /v1/models/{id}/status`); the SSD tier is per-*namespace*, and a namespace
can be shared by several models via `--project`. Neither is a subset of the
other, so the route shape is a design call, not an implementation detail.

---

## Hydrate Path (RAM Miss → SSD Load → Promote)

`SsdHydrator` is the read side of the SSD tier. One instance per loaded model,
constructed at `attach_at_load` and installed via
`PromptCache::set_ssd_source`.

On a RAM miss the hydrator executes the following phases on the **request
thread** (the cold path that would otherwise pay a full re-prefill):

```text
Phase 1 — SQLite prefix lookup
    Compute: chained = chained_block_hashes_seeded(prompt_ids, FNV_OFFSET ^ layout_key)
    Call:    SsdKvIndex::lookup_longest_prefix(chained, layout_key)
    Result:  (block_count, KvBlockRow) or miss → return None

Phase 2 — File read
    block_io::read_caches(&row.path, device, &model_id, kv_quant)
    Verifies model_id + kv_quant header before deserializing.
    On mismatch or I/O error: delete .kvb + delete index row + warn! + return None (graceful miss).

Phase 3 — Dequant
    Reconstruct KvStorage tensors from codes/scales/rotations as stored.

Phase 4 — GPU upload
    Move reconstructed tensors to the Metal device.

Phase 5 — SQLite touch
    index.touch(hash, layout_key) — updates last_used to now (LRU bookkeeping).
```

The hydrated block is returned as a `HydratedBlock` containing the matched
token-ID prefix (`block_count * 256` tokens) and reconstructed `Vec<KvCache>` +
`Vec<LinearAttnCache>`. The arch impl wraps this into its concrete prompt-cache
entry type.

Corruption is always handled as a graceful miss: the bad file and row are
deleted, a `warn!` is emitted, and the caller falls through to a full prefill.
The hydrator never panics.

---

## Spill Path (RAM Eviction → SSD Write)

`SsdSpiller` is the write side. One instance per loaded model, installed via
`PromptCache::set_spill_sink`.

When `PromptCache::push` evicts an entry (RAM cap exceeded or slot count
exceeded), the spill hook receives the evicted entry. The spiller:

1. **Hot path (request thread):** refcount-clones the KV caches (no tensor
   copy), builds a `SpillJob`, and `try_send`s it onto a bounded
   `sync_channel` (depth 16). If the channel is full, the job is dropped and a
   `warn!` is logged — back-pressure never stalls decode.

   The clone is `KvCache::try_deep_clone`. For a rotor K-only cache (`k_rotor3`
   / `k_rotor4`) this is also the point where a **ring-only decode tail** is
   reconciled: the fused decode path keeps the CPU `blocks` frozen at the
   prefill prefix while the GPU ring carries the decode tail (see
   `docs/KV_QUANT.md` § "Ring eligibility"), so the clone materialises the tail
   back into complete CPU blocks from the ring. `block_io::write_quant_rotor_k{3,4}`
   serialize `blocks`, so without this the spill would persist a store truncated
   at the last CPU block. The serializer additionally refuses to write any rotor
   K store whose blocks fall short of `shape[2]`
   (`ensure_rotor_k_blocks_cover_shape` → `BlockIoError::TruncatedStore`) — the
   invariant is enforced at the persistence boundary, not only at the codec.

2. **Drain thread (`rmlx-kv-spill`):** receives jobs from the channel.
   - Serializes via `block_io::write_caches` — forces MLX arrays to host
     bytes (CPU eval), writes to `<namespace_dir>/<hash>.kvb`.
   - Records the block in `SsdKvIndex` via `index.record`.
   - On any error: `warn!`, remove partial `.kvb` if it exists, drop job.
   - On a block that reached the index, runs the **evict-to-budget pass**
     (below).

The drain thread opens the `SsdKvIndex` once on startup. If the index cannot be
opened, the thread drains the channel (to unblock senders) and exits; the
spiller silently drops subsequent jobs. The cache continues working with spill
effectively disabled.

### Evict-to-budget (runtime)

`SsdSpiller::spawn` resolves the namespace byte ceiling once, from the installed
config via `ssd_tier::effective_namespace_budget` — the same figure the
attach-time maintenance pass evicts to. After each block it records, the drain
thread calls `enforce_namespace_budget`, which evicts LRU-first until the
namespace is back within that ceiling and republishes `rmlx_ssd_bytes_used` /
`rmlx_ssd_evict_total`. That is the same routine the attach path runs, so the
two cannot disagree about what the configured budget means.

The drain thread owns this because it is the only writer that grows the tier, it
already holds the index handle, and it runs off the inference path. Without it
the budget was enforced **only at attach**: a `serve` that stayed up between
model loads grew past `--kv-ssd-cache-gb` for its whole lifetime, and the
`rmlx_ssd_bytes_used` gauge stayed frozen at the value measured when the model
loaded. (Measured on a 4-request session: gemma-4-e2b at a 0.02 GiB ceiling held
47.1 MB, Ternary-Bonsai-8B at a 0.3 GiB ceiling held 980.6 MB.)

`effective_namespace_budget` resolves the ceiling as: no per-namespace budget →
the global pool ceiling governs the namespace alone; no global pool → the
per-namespace budget stands alone; both set → the tighter of the two. It
therefore only yields `0` when the tier is off, so a zero can never mean
"ceiling of zero bytes". `enforce_namespace_budget` treats `0` as "no ceiling
configured" for every caller and evicts nothing — the literal "keep nothing"
reading stays on the raw `SsdKvIndex::evict_lru_until` API.

Eviction reads rows in `last_used` order and runs after every spilled block, so
it must not be O(rows): the scan stops as soon as the running total is back under
the ceiling, an index on `last_used` keeps SQLite from sorting the table to find
the oldest row, and the post-eviction footprint is returned rather than re-summed
by the caller. Measured at 100k rows (≈200 GiB of 2 MiB blocks), one block over
budget: ~41 ms → ~3.1 ms per spilled block, the remainder being the one
`SUM(byte_size)` that decides whether there is anything to do.

The deletes run in a single transaction, so a SQLite failure part-way leaves the
index untouched instead of dropping rows whose files the caller is never told to
unlink — those files would be unreclaimable, since `prune_missing` drops rows
whose file vanished and not the inverse. Only rows this call actually deleted are
returned, so the eviction count published to `rmlx_ssd_evict_total` cannot
double-count a row a second evictor removed first.

The pass is safe against in-flight hydrates on the same namespace, in this
specific sense: **no reader is handed a block other than the one it asked for**.
Rows are deleted before their `.kvb` files, and a block is only reachable through
its own `(hash, layout_key)` row, so a concurrent lookup either does not find the
row (a plain miss) or finds its file already gone and falls through to a full
prefill. It is not a claim that no block is lost — the hydrate-side cleanup can
still drop a block that a re-spill recreated underneath it (see below).

The job carries the last chained-block digest of the entry's prompt as `hash`
(the block's identity), plus `layout_key`, `model_id`, `kv_quant`, and the
refcount-cloned caches. The `.kvb` filename stem is the hex-formatted hash.

A `SsdSpillEvent` is emitted per drained job via the process-global
`EventRecorder` with per-phase timing (`dur_serialize_us`, `dur_write_us`,
`dur_index_us`) and `byte_size`.

---

## Cross-Namespace LRU — evict_pool_lru_until

When `--kv-ssd-global-gb > 0`, the global pool ceiling is enforced across all
namespaces. `evict_pool_lru_until` runs at startup (inside `install_config`) and
can be called independently.

Algorithm:

```text
1. Scan every directory under <RMLX_HOME>/cache/kv/.
   Open each namespace's index.db (SsdKvIndex::open_at).
   Collect all kv_blocks rows into a merged list with (last_used, namespace, hash, layout_key, byte_size, path).
   Sum pool_bytes.

2. If pool_bytes <= global_budget_bytes: return EvictionReport (no-op).

3. Sort merged list ascending by last_used (oldest first; the namespace + hash
   tie-break is dead weight now that stamps are a total order, and is kept only
   to make the sort itself deterministic).

4. Walk oldest-first, accumulate evictions until running_total <= global_budget_bytes.

5. Group eviction candidates by namespace.
   For each (namespace, rows):
     - Remove .kvb file from disk (best-effort; warn! on error, continue).
     - Delete index row via SsdKvIndex::delete(hash, layout_key).

6. Return EvictionReport { bytes_freed, blocks_evicted, namespaces_touched }.
```

I/O errors on a single namespace are `warn!`ed and the walk continues. The
function emits a single `ssd_pool_lru_eviction` tracing event on completion
with `bytes_freed`, `blocks_evicted`, `namespaces_touched`, and pool size
before/after.

`EvictionReport` fields:

| Field | Meaning |
|---|---|
| `bytes_freed` | Sum of `byte_size` over all evicted rows. |
| `blocks_evicted` | Count of `kv_blocks` rows removed across all namespaces. |
| `namespaces_touched` | Count of distinct namespaces that contributed at least one eviction. |

---

## Block IO — .kvb File Format

`.kvb` files are **safetensors** format. Named tensors plus a JSON
`__metadata__` header make the format self-describing and debuggable.

### Metadata header keys

| Key | Value |
|---|---|
| `model_id` | `<arch>/<snapshot>` identity string. |
| `kv_quant` | `KvQuant` Display string. |
| `n_layers` | Number of attention layers serialized. |
| `seq_len` | Token count at the time of serialization. |
| `n_linear` | Number of GDN linear-attention layers (0 for pure-attention archs). |
| `l{idx}.geom` | Per-layer geometry JSON (shape, page count for paged storage). |

The reader verifies `model_id` and `kv_quant` against the loaded model before
deserializing any tensors. A mismatch returns `BlockIoError::ModelIdMismatch`
or `BlockIoError::KvQuantMismatch` — never a silently-wrong cache.

### Tensor layout per KV variant

| `KvStorage` variant | Tensors written |
|---|---|
| `K8V4` | `l{i}.k.codes` `l{i}.k.scales` `l{i}.v.codes` `l{i}.v.scales` |
| `K8V8` | same as K8V4 (V is also q8_0) |
| `Planar` | K8V4 tensors plus `l{i}.v.rotations` |
| `None` (filled bf16) | `l{i}.k.bf16` `l{i}.v.bf16` — the off-storage bf16 prefix; geom tag `none_bf16` |
| `None` (no payload) | geometry only, geom tag `none` — see note below |
| `Mixed` | `l{i}.k.codes` `l{i}.k.scales` `l{i}.k.biases` + V equivalents |
| `Paged` | gathered contiguous codes/scales/rotations + page geometry metadata |
| `LinearAttn` | `l{i}.conv_state` + `l{i}.delta_state` (recurrent state, whole, never truncated) |

The round trip is byte-exact on codes. GDN state has no sequence axis and is
serialized in full.

#### `None` payload spill (bf16)

`KvStorage::None` (the `--kv-quant none` path) does **not** hold its live K/V in
the storage buffer — the bf16 K/V live on the parent `KvCache`
(`decode_fp16_{k,v}`, the same buffers `exit_prefill` seeds). The spill bridge
(`write_caches`) reads those buffers and, when present, persists them under
`l{i}.k.bf16` / `l{i}.v.bf16` with the geometry tag `none_bf16`. On hydrate the
reader restores the pair and re-seeds the reconstructed cache via
`KvCache::with_decode_fp16_seed`, so an exact-hit SSD replay reads the real K/V.
bf16 round-trips bit-for-bit, so the restored prefix equals the pre-spill value.

A `None` layer with **no** bf16 pair (a never-filled cache, or a hydrated SWA
layer whose rotating ring was never serialised) falls back to the geometry-only
`none` tag and re-prefills its prefix on reuse — it carries no spillable payload.
This is the distinguishing signal: presence of `decode_fp16_{k,v}` means real KV
exists and must travel; absence means there is nothing to persist.

#### SWA layers are not spilled — hydrated entries degrade to re-prefill

Gemma4's sliding-window-attention (SWA) layers run a bf16 `RotatingKvCache`
ring that is **not** serialised to the SSD tier (the rotating ring layout is not
expressed in the `.kvb` format). On hydrate those layers are reconstructed as
payload-less `KvStorage::None` — empty, carrying neither quantized storage nor a
restored bf16 seed.

Reusing such a hydrated entry as a **prefix** (re-prefilling only the tail on top
of the hydrated block) would leave every SWA layer's prefix empty, giving them
the wrong attention context and corrupting the output for a non-block-aligned
prompt. `KvCache::is_trimmable()` returns `true` for a `None` layer, so it is the
wrong predicate to detect this.

The consume path therefore evaluates a per-entry **hydrate-completeness** guard
(`Gemma4Entry::is_hydrate_complete`) before the strict-prefix / block-truncate
reuse arms: an entry is complete only if every attended layer holds payload
(persistent storage, or a restored `decode_fp16_{k,v}` bf16 seed). A hydrated
entry with an empty SWA `None` layer fails the check and is **degraded to a full
re-prefill** (Miss), which recomputes the SWA prefix correctly. The guard is a
no-op for RAM-resident snapshots (every layer holds payload), so non-SSD prefix
reuse is unaffected. Serialising the SWA ring so hydrated entries could be reused
as prefixes is a future enhancement.

---

## CLI Flags

All flags are on the `rmlx serve` subcommand.

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--kv-ssd-cache-gb <GIB>` | `f64` | `0.0` | Per-namespace on-disk budget in GiB. `0` = tier off. |
| `--project <NAME>` | `String` | none | Namespace name. Requires `--kv-ssd-cache-gb > 0`. Absent → namespace defaults to model id. |
| `--kv-ssd-global-gb <GIB>` | `f64` | `0.0` | Cross-namespace pool ceiling in GiB. `0` = no global cap. |
| `--prompt-cache-ram-gb <GIB>` | `f64` | none | RAM cap for the in-process prompt cache. Tier 1 ceiling. |
| `--paged-kv` | flag | false | Route K8V4 / K8V8 / Planar caches through the paged block-table allocator. |
| `--paged-kv-page-tokens <N>` | `i32` | 32 | Page size in tokens. Requires `--paged-kv`. |
| `--prefix-index <KIND>` | `linear\|radix` | `linear` | Prompt-cache prefix lookup strategy. See below. |

### Validation rules

- `--kv-ssd-cache-gb` must be `>= 0`. Negative value is an exit-2 error.
- `--project` requires `--kv-ssd-cache-gb > 0`. Passing a project name with the
  tier off is an exit-2 error.
- `--kv-ssd-global-gb` must be `>= 0`. Negative value is an exit-2 error.
- When `--kv-ssd-cache-gb > --kv-ssd-global-gb > 0`, the per-namespace ceiling
  is clamped to the global budget at startup and a `warn!` is emitted.
- `--paged-kv` with `--kv-quant bf16` or `--kv-quant none` is rejected.
- `--paged-kv` with `--cache-type-k rot_k*` is rejected (RotK / RotKTq4V are
  not paged-compatible).

### --prefix-index

Controls the prompt-cache longest-prefix lookup data structure.

| Kind | Lookup complexity | Notes |
|---|---|---|
| `linear` (default) | O(slots × n_blocks) | Byte-identical to the pre-radix path. The radix tree is still maintained in parallel for benchmarking but unused on lookup. |
| `radix` | O(n_blocks) | NVIDIA Dynamo positional radix tree port. Opt-in pending the bench gate (≥2× linear at N≥32 with <5% memory overhead). |

The installed kind is a process-global `OnceLock` set by `install_prefix_index_kind`.
Every `PromptCache<E>` constructed after that call uses the selected strategy.

---

## projects.toml Integration

`projects.toml` provides stable per-project budget defaults without long CLI
lines. See `docs/PROJECTS_CONFIG.md` for the full file shape and precedence
rules. The relevant fields for the SSD tier:

| TOML key | CLI equivalent | Scope |
|---|---|---|
| `[global].ssd_pool_gb` | `--kv-ssd-global-gb` | All namespaces |
| `[project.<name>].ssd_cap_gb` | `--kv-ssd-cache-gb` (scoped) | Named namespace |

Precedence: `CLI flag > [project.<name>] > [global] > built-in default`.

Unknown `--project` names fall back to `[global]`. A malformed `projects.toml`
is a startup error (exit 2).

---

## Operational Rules

**Startup sequence (per serve invocation)**

1. `install_config` is called once before any model loads.
2. Pre-release v1 namespace wipe runs across all `<RMLX_HOME>/cache/kv/` dirs.
3. If `global_budget_bytes > 0`, cross-namespace LRU runs (`evict_pool_lru_until`).
4. For each model loaded: `attach_at_load` → per-namespace startup maintenance
   (prune missing blocks, evict to per-namespace budget), then spiller +
   hydrator are installed onto the per-arch `PromptCache`.

**Startup maintenance per namespace**

- `prune_missing`: drops index rows whose `.kvb` file has been deleted outside
  the process (manual cleanup, OS eviction). Returns count of pruned rows.
- `enforce_namespace_budget(effective_budget)`: deletes oldest rows + their
  `.kvb` files until the index total is within budget, then republishes the
  `ssd_bytes_used` gauge and the `ssd_evict_total` counter.

**Runtime maintenance per namespace**

- The spill drain thread runs `enforce_namespace_budget` after every block it
  records, so the ceiling holds for the life of the process and not only at
  attach. See § "Evict-to-budget (runtime)".
- `prune_missing` stays attach-only: a row whose file vanished mid-run is
  repaired on the read path (hydrate drops it and re-prefills), so a periodic
  scan of every row would buy nothing.

**Budget accounting**

- `SsdKvIndex::total_bytes()` sums `byte_size` over all indexed rows.
- `byte_size` is the on-disk size written at spill time and stored in the index.
  It does not account for filesystem fragmentation.
- The budget is a **ceiling on the indexed total**, checked after each spill. A
  single block larger than the whole ceiling is written and then immediately
  evicted; the tier does not refuse an over-sized block up front.

**Failure containment**

- Spill errors are always `warn!`-logged and dropped; the inference path is
  unaffected.
- Hydrate errors (corrupt file, metadata mismatch) are `warn!`-logged; the
  request falls through to a full re-prefill. A block whose file is simply
  *gone* is not one of these: LRU eviction unlinks blocks whose rows it has
  already deleted, so a hydrate finding no file is the routine outcome of a
  racing eviction. That case is `debug!`-logged and treated as a miss, so a tier
  running normally at its ceiling does not stream corruption warnings.
- Both cleanup paths delete the index row before unlinking the file, matching
  the eviction order. An interrupted cleanup therefore leaves an unreferenced
  row, which `prune_missing` reclaims at the next attach, rather than an
  unreferenced file, which nothing reclaims. A re-spill of the same hash landing
  between a failed read and its cleanup can still cost that block: the row and
  the file are the same for identical content, so the cleanup cannot tell the
  re-spilled row from the one it read. (`last_used` distinguishes them, but the
  reader captured no stamp to compare against, so acting on it would need a
  compare-and-delete the cleanup path does not have.)
- Index open failures disable the tier for the affected namespace; other
  namespaces and the RAM cache are unaffected.

**Do not**

- Hard-code paths to `<RMLX_HOME>/cache/kv/`. Always use
  `rmlx_core::paths::kv_cache_dir(namespace)`.
- Manually edit or delete `index.db` rows while the server is running (WAL
  is consistent, but row deletion will cause hydrate misses or double-delete
  errors).
- Call `install_config` more than once in the same process.

---

## Public API

The SSD tier lives in its own crate (`rmlx-kv-ssd`). Several types that were
`pub(crate)` inside `rmlx_models::kv_cache` were **promoted to `pub`** so
per-arch trait impls in `rmlx-models` (Gemma4 / Qwen3 / Qwen3.5-MoE
`attach_ssd_tier`) can reach across the crate boundary:

| Item                         | Previous visibility           | New crate          |
|------------------------------|-------------------------------|--------------------|
| `SsdHydrate<E>` (trait)      | `pub(crate)` in `prompt_cache` | `rmlx_kv_ssd::traits` (pub) |
| `SsdSpiller`, `SpillJob`     | `pub(crate)` in `kv_cache`    | `rmlx_kv_ssd::spill` (pub)  |
| `SsdHydrator`, `HydratedBlock` | `pub(crate)` in `kv_cache`  | `rmlx_kv_ssd::hydrate` (pub) |
| `SsdKvIndex` + helpers       | already `pub`                  | `rmlx_kv_ssd::ssd_index`    |
| `KvBlockWriter`, `KvBlockReader`, `write_caches`, `BlockIoError` | already `pub` | `rmlx_kv_ssd::block_io` |
| `SsdTierConfig`, `install_config`, `active`, `compute_layout_key` | already `pub` | `rmlx_kv_ssd::ssd_tier` |
| `set_ssd_event_recorder`, `set_ssd_{spill_prom,hydrate_prom,bytes_used,evict_total}_hook` | `pub` | `rmlx_kv_ssd::hooks` |
| `BLOCK_TOKENS`, `FNV_OFFSET`, `FNV_PRIME`, `chained_block_hashes`, `chained_block_hashes_seeded` | `pub(crate)` in `prompt_cache` | `rmlx_kv_ssd::hashing` (pub) |

The promoted structs (`SpillJob`, `HydratedBlock`) carry an
`#[allow(clippy::exhaustive_structs, reason = "promoted: …")]`
header — they are closed internal bridge structs whose field list is the
contract between the spill/hydrate hot path and the per-arch impls; adding a
field is a coordinated update across `rmlx-kv-ssd` and `rmlx-models`.

## Import paths

The `rmlx_models::*` / `rmlx_models::kv_cache::*` / `rmlx_models::ssd_tier::*`
re-export shims for SSD-tier items have been dropped. Callers in `rmlx-cli`
/ `rmlx-server` / tests import directly from `rmlx-kv-ssd`. Only the per-arch
dispatch (`attach_at_load`) stays in `rmlx-models` because its trait impls
live there.

| Caller import path                              | Owning crate                       |
|-------------------------------------------------|------------------------------------|
| `rmlx_kv_ssd::set_ssd_event_recorder(…)`        | `rmlx-kv-ssd` (`hooks::*`)         |
| `rmlx_kv_ssd::write_caches(…)`                  | `rmlx-kv-ssd` (`block_io::*`)      |
| `rmlx_kv_ssd::ssd_index::SsdKvIndex`            | `rmlx-kv-ssd` (`ssd_index::*`)     |
| `rmlx_kv_ssd::ssd_tier::install_config(…)`      | `rmlx-kv-ssd` (`ssd_tier::*`)      |
| `rmlx_kv_ssd::ssd_tier::SsdTierConfig`          | `rmlx-kv-ssd` (`ssd_tier::*`)      |
| `rmlx_models::ssd_tier::attach_at_load(…)`      | `rmlx-models` (thin arch dispatcher; calls `rmlx_kv_ssd::prepare_attach`) |
| `crate::prompt_cache::FNV_OFFSET`               | `rmlx-models::prompt_cache` (`pub(crate) use rmlx_kv_ssd::hashing::*`) |
| `crate::prompt_cache::SsdHydrate`               | `rmlx-models::prompt_cache` (`pub(crate) use rmlx_kv_ssd::traits::SsdHydrate`) |

In-crate `crate::prompt_cache::FNV_OFFSET` / `crate::prompt_cache::SsdHydrate`
remain `pub(crate) use` re-exports — they are crate-internal aliases for
`rmlx-models` consumers (not cross-crate shims). The `RUST_LOG` filter
string in `crates/rmlx-server/tests/ssd_cache_restart.rs` targets
`rmlx_kv_ssd::spill` and `rmlx_kv_ssd::ssd_tier` directly.

## Contract B — codec dispatch

Every `KvStorage` variant is dispatched in two places, both now living in
`rmlx_kv_ssd::block_io`:

1. `KvBlockWriter::write_caches` — serialise the per-layer storage to the
   safetensors `.kvb` payload (one match arm per variant).
2. `KvBlockReader::read_caches` — reconstruct the storage on hydrate (one
   match arm per variant).

Both arms are matched on the runtime `KvQuant` plus the variant tag carried
in the per-layer geometry header (`l{idx}.geom`). The active
[`compute_layout_key`] folds `kv_quant.to_string()` into the
`layout_key` salt, which participates in the SSD-index composite
`(hash, layout_key)` primary key — so two different codecs can never alias
the same on-disk row even if the chained block-hash digests collide.

Adding a new codec MUST:

1. add the new `KvQuant` discriminant + `KvStorage` variant in `rmlx-kv-quant`;
2. extend the geometry header serialiser in `KvBlockWriter::write_caches`
   with the new buffers;
3. add the matching reconstructor arm in `KvBlockReader::read_caches`;
4. add a per-codec roundtrip test in `block_io_tests.rs` that closes the
   loop on disk.

---

## See also

- `docs/KV_CACHE.md` — KV quantization variants and storage types (K8V4, K8V8,
  Planar, Mixed, RotK, RotKTq4V, Paged).
- `docs/KV_QUANT.md` — per-quant codec details and byte sizes.
- `docs/PROJECTS_CONFIG.md` — `projects.toml` reference (per-project budget
  overrides, precedence chain).
- `docs/METRICS_DB.md` — metrics persistence; SSD-tier events land in the
  `events` table via `SsdSpillEvent` and `SsdHydrateEvent`.
