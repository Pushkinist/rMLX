# SSD KV-Cache Tier

rMLX operates a three-tier KV-cache hierarchy. The SSD tier is the third and
lowest level: it persists prompt-cache blocks evicted from RAM to disk so that
they can be reloaded on subsequent requests without a full re-prefill.

> **Crate ownership:** The SSD tier lives in the workspace member crate
> **`rmlx-kv-ssd`** — `crates/rmlx-kv-ssd/`. It owns the index, spill,
> hydrate, block-I/O, layout-key salt, the 5 Prometheus hook globals, the
> `SsdHydrate<E>` trait, the chained FNV-1a-64 block-digest helpers
> (`BLOCK_TOKENS`, `chained_block_hashes`, `chained_block_hashes_seeded`,
> `FNV_OFFSET`, `FNV_PRIME`) and the `cache_seed` those helpers are seeded
> with — which the RAM prompt cache in `rmlx-models` re-exports rather than
> redefines, so both tiers hash under one formula. The per-arch
> `attach_ssd_tier` dispatcher
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

Supported architectures (spill + hydrate wired). These are **resolved** classes
(`Architecture::arch_class()`), not the checkpoint's declared `architectures[0]`:
- `Gemma4ForConditionalGeneration` (also serves the 12B unified snapshot, which
  declares `Gemma4UnifiedForConditionalGeneration` and resolves to this class)
- `Gemma3ForConditionalGeneration`
- `Qwen2ForCausalLM`
- `Qwen3ForCausalLM`
- `Qwen3_5MoeForConditionalGeneration`
- `Qwen3_5ForConditionalGeneration` (dense Qwen3.5 — same loader, model struct
  and `PROMPT_CACHE` static as the sparse-MoE class)
- `Qwen3VLMoeForConditionalGeneration`
- `BitNetForCausalLM`

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

1. **Stale-schema wipe** — scans every `<RMLX_HOME>/cache/kv/<ns>/` directory.
   If `index.db` holds a `kv_blocks` table at a `schema_version` this binary
   supersedes (a missing `schema_version` table reads as the v1 layout), the
   entire namespace directory is removed (`fs::remove_dir_all`) and a
   `ssd_cache_pre_release_wipe` tracing event records the dropped bytes plus
   the version each namespace was found at. A **newer** version is left in
   place and warned about instead (see "Namespaces from a newer binary").
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

### last_used is the LRU ordering key

Eviction sorts on `last_used` and nothing else, so the column has to order
accesses, not merely approximate them. `record` and `touch` stamp it from
`ssd_index::now_stamp_us`, which reads the wall clock in microseconds and
clamps the result above the highest value **this process** has issued.

Seconds do not work. Under any realistic request rate many blocks land in the
same second, `ORDER BY last_used ASC` has no tiebreak, and the victim becomes
whichever row SQLite happens to yield first — a block used 900 ms ago can be
discarded ahead of one used 100 ms ago, so the tier can throw away exactly the
prefix it is about to be asked for. The degradation toward random replacement
grows with request rate, i.e. it is worst when the tier matters most.

It stays a wall-clock value rather than a plain counter because the
cross-namespace sweep merges rows from independent namespace databases and has
to compare them; a per-database counter means nothing outside its own database.

**Durability across restarts.** The clamp is process state and starts at zero,
so `SsdKvIndex::open_at` seeds it from `MAX(last_used)` in the namespace it is
opening (`adopt_persisted_stamps`, one indexed b-tree descent). Without that,
a restart that follows a backwards clock step — NTP correction, a manual date
change, RTC drift across sleep — writes blocks that sort *below* every row
already on disk and evicts the newest data first, for as long as the clock
lags. That inversion is persistent, and therefore worse than the same-second
tie it replaced.

**Scope: one process.** The stamps are a total order within a server, not
across two. The Metal claim file is keyed by port (`/tmp/rmlx.<port>.claim`),
so two `rmlx serve` instances on different ports each run `install_config` and
can write the same pool; two writes landing in the same microsecond from
different processes still tie. Seeding the clamp at every open narrows the
window (each open adopts the other's stamps) but does not close it, and
`evict_lru_until`'s `ORDER BY last_used ASC` has no tiebreak. The pool sweep's
`(namespace, hash)` tiebreak is what keeps *that* path deterministic.

No tiebreak column was added, so the `kv_blocks_last_used` index stays
single-column — that is the index that keeps eviction from sorting the whole
table.

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
- File present, `schema_version != 3` → `SchemaMismatch` (a schema this binary cannot interpret; older ones are wiped by `install_config` first, newer ones are deliberately left in place).
- File present, `schema_version == 3` → open succeeds.

### Migration

rMLX is unreleased, so every schema transition is a **one-time wipe**, not a
row-by-row migration. The tier holds nothing that is not regenerable, so the
cost of a wipe is one re-prefill per dropped block.

| From | What changed | What happens to existing data |
|---|---|---|
| v1 (no `schema_version` table, `hash` is the sole PK) | `layout_key` column + composite PK + `schema_version` table | Namespace dir removed at startup |
| v2 | `last_used` unit: seconds → microseconds | Namespace dir removed at startup |

`install_config` removes the entire namespace directory for any `index.db` at a
version this binary **supersedes**, before any `SsdKvIndex::open` call runs
in-process.

The v2 case is a deliberate choice, not a forced one. The table shape did not
change, so the rows are convertible in a single statement (`UPDATE kv_blocks
SET last_used = last_used * 1000000`). They are wiped anyway because rMLX is
pre-release and carrying migration code — plus the branch in the wipe pass that
would have to let a v2 DB through to be converted — buys nothing that a warm
cache does not rebuild in minutes. What is *not* an option is leaving v2 rows
in place unconverted: they would sit three orders of magnitude below every new
row and be evicted first regardless of use.

### Namespaces from a newer binary

A namespace at a version **newer** than this binary's is left alone and
`warn!`ed (`event = "ssd_cache_newer_schema_skipped"`, with `found_version` and
`expected_version`). Two rMLX builds at different schema versions on one
machine is ordinary — a tap-installed release beside a dev build — and wiping
forward would have the older binary destroy the newer one's whole pool on every
alternate boot. This build cannot use such a namespace; reclaiming its bytes is
an operator action, and the warning names the directory.

A directory whose `index.db` has no `kv_blocks` table, or cannot be read at
all, is never touched.

---

## layout_key Salt

`compute_layout_key` produces a stable `u64` from
`(arch, layer_quants, n_kv_heads, head_dim, kv_quant)`, where `layer_quants` is
the **effective per-layer codec vector** — one entry per decoder layer, exactly
what the caller builds its `KvCache`s from — and `n_layers` is its length:

```text
input  = arch.as_bytes() + ":<n_layers>:<n_kv_heads>:<head_dim>:<kv_quant>"
                         + ":<layer_quants[0]>" + … + ":<layer_quants[n-1]>"
h      = FNV_OFFSET  (0xcbf29ce484222325)
for each byte b in input:
    h ^= u64(b)
    h  = h * FNV_PRIME  (0x00000100000001b3)
layout_key = h
```

The vector is folded, and not just the requested `kv_quant`, because the
per-layer assignment is a **policy** decision — `kv_quant_for_layer` promotes
boundary layers of a quantizing base codec to `K8V8` — and that policy can
change between builds at an unchanged base codec and unchanged geometry. With
only the base folded, a block spilled under one mixture hydrates into a request
running another: each layer deserializes from its own storage tag, so nothing
errors, and the request silently runs per-layer codecs it did not ask for. The
vector makes such a block a miss. `rmlx-kv-ssd` does not compute the mixture
(the policy lives above it, in `rmlx_models::kv_cache`); the caller supplies it
in `rmlx_models::ssd_tier::attach_at_load`, from the same `kv_quant_for_layer`
call its cache construction uses. `prepare_attach` refuses to attach on an
empty vector rather than key a zero-length layout.

The key is deterministic across runs and distinct with overwhelming probability
for every distinct input tuple. It is **not** weight-dependent: reloading the
same architecture at the same layout from a different snapshot yields the same
`layout_key` and shares the cache, which is correct.

> **One-time invalidation — every namespace, every codec.** The *formula*
> changed, not just one codec's vector: the key now appends a term per layer, so
> `k8v8`, `k8v4`, `planar`, `mixed_*` and `none` alike hash to new values, and
> the per-request seed below changed for the same reason. Every previously
> spilled `.kvb` is unreachable after upgrading, on every namespace. Expect one
> cold pass — full re-prefill — per prompt, then steady state; no error and no
> action needed.
>
> **Unreachable blocks still occupy the namespace budget.** Nothing prunes them
> proactively: `startup_maintenance` removes rows whose file vanished, not rows
> whose layout no longer matches, so they sit in `--kv-ssd-cache-gb` until
> eviction reaches them. They are never `touch`ed, so LRU picks them first —
> but only once the namespace is at its budget and something has to go. Until
> then the effective cache is smaller by whatever the stale set occupies. A
> namespace well under budget can hold them indefinitely; `rm -rf` of the
> namespace directory is the operator's fast path if that matters.
>
> For the `none` codec specifically the invalidation is also *load-bearing*
> rather than incidental: `--kv-quant none` used to be built as a bf16/K8V8
> mixture and is uniform bf16 now, so without the key moving, a stale block
> would have been served to a `none` request and quietly restored the promotion
> for it.

`arch` is the **resolved** class. It was previously the declared
`architectures[0]`, which could name a model that was not built — the salt is
supposed to describe the layout, and a declaration is not evidence of one.

> **One-time invalidation.** The 12B unified Gemma4 snapshot declares
> `Gemma4UnifiedForConditionalGeneration` and resolves to
> `Gemma4ForConditionalGeneration`, so its `layout_key` changes. Blocks spilled
> by an earlier version become unreachable and keep counting against the
> namespace budget until LRU evicts them: expect **one cold pass** for that
> snapshot after upgrading, then steady state. No error, no action required,
> and only when the tier is enabled (it is off by default). The alias was not
> preserved in the salt on purpose — two snapshots that resolve to the same
> architecture with the same geometry have the same layout, and keeping the
> declared string here would re-introduce the dependence on an unvalidated,
> model-side name that keying off the resolved class exists to remove.

> **One-time invalidation (iso4 V).** `layout_key` is derived from
> `(arch, layer_quants, n_kv_heads, head_dim, kv_quant)` and the per-layer block
> header carries only `{tag, max_seq, shape}`. Neither moves when the byte
> *orientation* inside a block changes, so a layout change of that kind has to
> move the **layer tag** or it is invisible on disk. The iso4 V GPU append stored
> its block head-major while `QuantIsoV4::dequant` reads sequence-major; fixing
> the append changed the bytes, so `ISOV4_LAYOUT_TAG` and
> `ISO_SYM_4_LAYOUT_TAG` became `iso_v_4_v2` / `iso_sym_4_v2`. A `.kvb` written
> before that matches no read arm and fails its hydrate with
> `unknown layer tag` rather than being decoded with the new orientation —
> loud, and one cold pass for `--kv-quant iso4` / `iso4_sym` users. Pinned by
> `block_io_tests::iso4_layout_tags_are_versioned_past_the_head_major_payload`.
> The iso3 tags are deliberately **not** bumped: that append always reordered
> heads↔seq, so its bytes are unchanged.

The key is one of the four terms of the block-hash seed, built at the call site
by `rmlx_kv_ssd::hashing::cache_seed`:

```text
seed    = FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt() ^ model_sig
          then, per layer: seed ^= layer_quant.cache_key_salt(); seed *= FNV_PRIME
digests = chained_block_hashes_seeded(ids, seed)
```

XOR mixing keeps the existing FNV avalanche behaviour intact. Different layouts
produce disjoint digest streams for the same token sequence, so a block cached
under one KV layout cannot collide with the same block cached under another.

**Why the per-layer term is here and not only in `layout_key`.** The layout key
is computed once, at attach, from the *launch* codec. A request need not run
that codec: `kv_quant_for_ctx` picks one by prompt length in auto mode, and the
OpenAI route accepts a per-request `kv_quant` override. For those requests the
attach-time vector describes a layout they are not running, so a guarantee
resting on the key alone ("a per-layer policy change invalidates stored
blocks") would hold only for requests that happened to run the launch default.
The seed is where the request's own codec already enters, so it is where the
request's own mixture belongs. `rmlx-models` supplies it through
`prompt_cache::request_cache_seed(layout_key, kv_quant, n_layers, model_sig)`,
which expands `n_layers` with `kv_layer_quants` — the same single producer the
arch cache-construction loops use — because the policy lives above this crate.

`cache_seed` is a single function, in `rmlx-kv-ssd` — below both the RAM prompt
cache in `rmlx-models` and the SSD hydrate probe in `rmlx-kv-ssd` — so neither
side can define the seed independently. The RAM push, the RAM query, the spill
key (`SpillJob::hash` is the last digest of the pushed entry) and the hydrate
probe are four sites that must agree bit for bit; a probe that omits one term is
not a wrong answer, it is a tier that silently never hits.

**The seed is computed once per request and passed down, not recomputed.**
`ArchPromptCache::consume` builds it, hands the same `u64` to
`find_best_prefix` and to `hydrate_from_ssd`, and the hydrator probes the index
and recomputes the promoted entry's block hashes from that value. The four
sites therefore share one variable rather than four evaluations that have to
keep agreeing.

That is a correctness requirement, not tidiness. `SsdHydrator` is installed on
the **per-architecture** prompt cache and outlives the model that installed it:
`--max-loaded-models` lets several models of one arch be resident, and
`attach_ssd_tier` overwrites the single attach slot on each load, so the
installed hydrator belongs to whichever model attached last. A hydrator that
remembered a `model_sig` would seed every other resident model's probe with the
wrong identity — the same silent 0-hit, reintroduced. The same argument applies
to `kv_quant`: it is per *request* (`kv_quant` in the request body hot-swaps the
codec), so an attach-time codec would mis-seed every hot-swapped request and
then reject the rows those requests wrote as header mismatches.

`SsdHydrator` therefore holds only namespace-scoped state — the index, the
directory, and the `layout_key` stamped on the rows in it — which the spiller
filling that namespace is installed from the same parameters, so the two always
agree. `layout_key` is a *shape* key and carries no model identity, and
`--project` collapses several models onto one namespace directory, so the seed's
model term is the only thing that keeps two models' blocks apart on disk.

`layout_key` is resolved and logged (as a 16-char hex string, field
`layout_key`) at `attach_at_load` before the index is opened, so operators can
correlate log rows with the active layout during debugging.

### Blocks written under an older seed

The key is content-addressed, so changing a seed term does not corrupt anything
— it re-partitions. Rows written before a term existed are simply digests that
nothing will ever probe for: still readable, still valid, permanently cold.

They are **not** purged, and deliberately so.

A targeted purge is not expressible: nothing in a `kv_blocks` row distinguishes
"written under an older seed" from "written for a prompt that has not come back
yet". A `SCHEMA_VERSION` bump *would* clear them — `wipe_stale_schema_namespaces`
runs from `install_config` before any `SsdKvIndex::open` and removes any
namespace whose recorded version is below this binary's, tagged or untagged, so
`open` then recreates the namespace empty (see
`wipe_removes_superseded_schema_namespace`). It is rejected for what it costs,
not because it would not work: it discards every **live** block in the namespace
along with the cold ones, to reclaim rows that are already first in line for
reclamation.

That line is the eviction path already in place: cold rows carry the oldest
`last_used`, and both `startup_maintenance` (at attach) and the spill drain
thread (after every recorded block) evict oldest-first until the namespace is
inside `--kv-ssd-cache-gb`. So they are the first thing dropped once the budget
binds, and until then they cost disk inside a ceiling the operator set — never a
wrong hit, never a stall. An operator who wants the space back immediately can
delete `<RMLX_HOME>/cache/kv/<namespace>/`; the tier rebuilds it.

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
with `layout_key` and `model_sig`: on an SSD-active run `cache_seed` mixes all
three, so RAM slots are codec- and model-partitioned even though the SSD tier
itself stays single-codec for the resident lifetime.

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
    Given:   seed  — the caller's `cache_seed(...)`, the same u64 the RAM query ran
             kv_quant — the request's codec
             policy   — the `DispatchPolicy` the request's caches dispatch under
    Compute: chained = chained_block_hashes_seeded(prompt_ids, seed)
    Call:    SsdKvIndex::lookup_longest_prefix(chained, layout_key)
    Result:  (block_count, KvBlockRow) or miss → return None

Phase 2 — File read
    block_io::read_caches(&row.path, device, &model_id, kv_quant, policy)
    Verifies model_id + kv_quant header before deserializing. Every
    reconstructed KvCache carries the caller's `policy`, so a hydrated set
    dispatches through the same kernel paths the live set did.
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

3. Sort merged list ascending by last_used (oldest first, tie-break by
   namespace + hash — the stamp source only totally orders one process, so this
   is what keeps the victim deterministic when two servers share the pool).

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

What a layer writes is decided by what it **holds**, not by the codec name.
`KvStorage::geometry_only_max_seq()` answers that in one place: a layer with no
packed payload takes the bf16 / geometry-only route below, everything else its
codec's tensors.

| `KvStorage` variant | Tensors written |
|---|---|
| `K8V4` | `l{i}.k.codes` `l{i}.k.scales` `l{i}.v.codes` `l{i}.v.scales` |
| `K8V8` | same as K8V4 (V is also q8_0) |
| `Planar` | K8V4 tensors plus `l{i}.v.rotations` |
| any variant with **no** packed payload, bf16 mirror live | `l{i}.k.bf16` `l{i}.v.bf16` — the off-storage bf16 prefix; geom tag `none_bf16` |
| any variant with no packed payload and no mirror | geometry only, geom tag `none` — see note below |
| `Mixed` | `l{i}.k.codes` `l{i}.k.scales` `l{i}.k.biases` + V equivalents |
| `Paged` | gathered contiguous codes/scales/rotations + page geometry metadata |
| `LinearAttn` | `l{i}.conv_state` + `l{i}.delta_state` (recurrent state, whole, never truncated) |

The round trip is byte-exact on codes. GDN state has no sequence axis and is
serialized in full.

#### bf16 payload spill (`none_bf16`)

Two kinds of layer hold their live K/V on the parent `KvCache`
(`decode_fp16_{k,v}`, the buffers `exit_prefill` seeds) rather than in a packed
store, and both spill the same way:

* `KvStorage::None` — the `--kv-quant none` path, where those buffers **are**
  the cache;
* every **bf16-mirror codec** (`K8V4`, `K8V8`, `Planar*`, `PlanarK`,
  `K8VTurbo*`, `TurboSym*`, `Iso3/4`, `Rotor3/4`, `RotorK*Asym`), whose decode
  reads only the mirror and whose `exit_prefill` therefore builds no store at
  all (`docs/KV_CACHE.md` §9.6 F3).

The exception is the same codec *with* codes: a K8V4/K8V8/Planar layer that
really carries a payload — a hydrated store-backed cache, or one that never
bracketed a prefill — takes its own row in the table above and spills codes.
The device does not decide this; a `Device::Cpu` run that brackets a prefill
gets the same absent store as a GPU one.

The spill bridge (`write_caches`) reads the mirror and persists it under
`l{i}.k.bf16` / `l{i}.v.bf16` with the geometry tag `none_bf16`. It **refuses**
any mirror longer than the cache's `offset`, failing the whole block rather than
writing that layer: a decode-expanded buffer is grown to the `max_seq` ceiling
and its tail is zeros, the writer takes the buffer's own `shape[2]` as the
layer's `seq_len`, and compacting it would need a device the drain thread does
not have. The mirror is already row-major — `exit_prefill` stores it through
`Array::contiguous` — which is what makes the persisted bytes mean what the
shape says. On hydrate the reader restores
the pair and re-seeds the reconstructed cache via
`KvCache::with_decode_fp16_seed`, so an exact-hit SSD replay reads the real K/V
and decodes off exactly the bytes the spilling cache held — the disk-served
prompt-cache hit and the RAM-served one produce the same tokens. bf16
round-trips bit-for-bit.

The row-major step is load-bearing, not hygiene: a live mirror is normally a
**slice view** over the larger prefill/decode buffer, and the serialiser reads
the raw allocation by linear offset (`Array::to_bytes` ignores strides).
Persisting the view directly writes the parent buffer's leading bytes under the
slice's shape, which for `kv_h > 1` is head 0's whole row window in place of
every head — it reads back as one head of real KV and zeros for the rest, with
no error anywhere. Pinned by
`block_io_tests::roundtrip_mirror_codec_spills_and_hydrates_as_bf16`, which
drives the mirror through a real prefill (a fixture that installs an
already-compact array cannot see this failure), and by
`decode_expanded_mirror_is_refused_rather_than_spilled_with_its_tail` and
`expanded_mirror_in_one_layer_fails_the_whole_block` for the refusal.

Cost: a bf16 block is ~2× the bytes of the q8 block the mirror codecs used to
write, against the `--kv-ssd-cache-gb` budget. That is the price of the block
meaning the same thing as the RAM cache it came from.

A layer with **no** payload and **no** bf16 pair (a never-filled cache, or a
hydrated SWA layer whose rotating ring was never serialised) falls back to the
geometry-only `none` tag and re-prefills its prefix on reuse. This is the
distinguishing signal: presence of `decode_fp16_{k,v}` means real KV exists and
must travel; absence means there is nothing to persist.

Blocks written before the mirror codecs moved to `none_bf16` are **not** served.
The layer tag is authoritative on read, so such a block would hydrate happily as
a store-backed cache and decode through the codec body — dequantised numbers for
a request whose RAM-served twin decodes bf16, with nothing to signal it. The
`(hash, layout_key)` key did not change, so the block would hit. What keeps it
out is `SsdKvIndex::SCHEMA_VERSION`, bumped to 4 for exactly this transition:
every pre-change namespace is reclaimed by the `wipe_stale_schema_namespaces`
pass at model load. Pinned by
`ssd_tier_tests::wipe_removes_the_pre_none_bf16_block_format_namespace`.

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
2. Stale-schema namespace wipe runs across all `<RMLX_HOME>/cache/kv/` dirs.
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
| `cache_seed` | `pub(crate)` in `prompt_cache` | `rmlx_kv_ssd::hashing` (pub); `prompt_cache` re-exports it |

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
