# SSD KV-Cache Tier

The SSD tier persists prompt-cache entries that the RAM cache evicts, so a
later request, in this process or the next one, can reload them instead of
re-prefilling. It is off by default and turns on when `--kv-ssd-cache-gb` or
`--kv-ssd-global-gb` is above `0`. The flags are in `docs/CLI.md`; the RAM
prompt cache it sits under is in `docs/PROMPT_CACHE.md`.

## Crates

`rmlx-kv-ssd` (`crates/rmlx-kv-ssd/`) owns the tier:

| Module | Holds |
|---|---|
| `ssd_tier` | `SsdTierConfig`, `install_config`, `active`, `compute_layout_key`, `prepare_attach`, the budget helpers, the stale-schema wipe |
| `ssd_index` | `SsdKvIndex`, `evict_pool_lru_until`, `SCHEMA_VERSION` |
| `spill` | `SsdSpiller`, `SpillJob` |
| `hydrate` | `SsdHydrator`, `HydratedBlock` |
| `traits` | `SsdHydrate<E>`, `HydratedEntry`, and the one blanket impl joining them |
| `block_io` | `.kvb` read and write, `BlockIoError` |
| `hashing` | `BLOCK_TOKENS`, `FNV_OFFSET`, `FNV_PRIME`, `chained_block_hashes`, `chained_block_hashes_seeded`, `cache_seed` |
| `hooks` | the five process-global hooks (below) |

The crate root re-exports these. It depends on `rmlx-core`, `rmlx-mlx`,
`rmlx-kv-quant` and `rmlx-metrics`, never on `rmlx-models`.

`rmlx-models` keeps what is per-arch. `ssd_tier::attach_at_load` dispatches on
the resolved class to each arch's `PROMPT_CACHE` static. The spill side is one
blanket `impl<E: PromptCacheEntry> SpillSink<E> for SsdSpiller` in
`prompt_cache.rs`. The hydrate side is one `HydratedEntry` impl per arch entry.
The RAM cache re-exports the `hashing` items, so both tiers hash with one
formula.

**The five hooks.** `set_ssd_event_recorder` installs the `EventRecorder` that
receives `SsdSpillEvent` and `SsdHydrateEvent` rows. `set_ssd_spill_prom_hook`,
`set_ssd_hydrate_prom_hook`, `set_ssd_bytes_used_hook` and
`set_ssd_evict_total_hook` feed the server's Prometheus series, among them
`rmlx_ssd_bytes_used`. Each is a `OnceLock`, set once at serve startup.

**Architectures.** `attach_at_load` wires every generative class: Gemma4 (the
unified 12B alias included), Gemma3, Qwen2, Qwen3, both Qwen3.5 classes,
Qwen3-VL MoE, Laguna and BitNet. Any other class logs that it stays RAM-only.

---

## Layout on disk

```text
<RMLX_HOME>/cache/kv/<namespace>/
    <hash>.kvb     one safetensors block file per spilled entry
    index.db       SsdKvIndex (SQLite)
```

The namespace is `--project` when given, else the model id. A block covers
whole 256-token blocks of the prompt (`BLOCK_TOKENS`); a trailing partial
block is always re-prefilled. Paths go through
`rmlx_core::paths::kv_cache_dir(namespace)`.

## SsdTierConfig

`install_config` stores the config in a `OnceLock` once, before any model
loads. A second call returns `Error::SsdTierAlreadyInstalled`.

| Field | Meaning |
|---|---|
| `per_namespace_budget_bytes` | `--kv-ssd-cache-gb`, in bytes |
| `global_budget_bytes` | `--kv-ssd-global-gb`, in bytes; `0` is no pool cap |
| `default_namespace` | `--project`, or `None` for the model id |
| `per_project_budgets` | filled by `serve` from `projects.toml`; read by nothing |

Both budgets `0` store `None`: the tier is off, and no spiller or hydrator is
installed. `install_config` then:

1. runs the stale-schema wipe, whether the tier is on or off;
2. when `global_budget_bytes > 0`, runs `evict_pool_lru_until` on the pool.

`effective_namespace_budget` gives the ceiling every namespace is held to:
the per-namespace budget alone, the global one alone, or the smaller of the
two. It is `0` only when the tier is off.

`projects.toml` resolution (flag, then project section, then global section)
happens in `serve` before `install_config`; see `docs/PROJECTS_CONFIG.md`.

---

## SsdKvIndex

One SQLite database per namespace, in WAL mode with `synchronous = NORMAL`
and a 5 s busy timeout.

```text
kv_blocks (
    hash        TEXT    NOT NULL,   -- chained FNV-1a-64 digest, hex
    layout_key  INTEGER NOT NULL,   -- compute_layout_key()
    path        TEXT    NOT NULL,   -- the .kvb file
    model_id    TEXT    NOT NULL,
    kv_quant    TEXT    NOT NULL,   -- KvQuant Display string
    byte_size   INTEGER NOT NULL,   -- on-disk size of the .kvb
    last_used   INTEGER NOT NULL,   -- unix epoch microseconds
    PRIMARY KEY (hash, layout_key)
)
schema_version ( version INTEGER PRIMARY KEY NOT NULL )
INDEX kv_blocks_last_used ON kv_blocks (last_used)
```

**`last_used` is the LRU key.** `record` and `touch` stamp it from
`ssd_index::now_stamp_us`: wall-clock microseconds, clamped above the highest
stamp this process has issued. `SsdKvIndex::open_at` seeds the clamp from the
namespace's `MAX(last_used)`, so a clock stepped backwards across a restart
cannot sort new blocks below old ones. The order is total within one process.
Two servers sharing a pool can still tie within a microsecond. The pool sweep
breaks ties on `(namespace, hash)`; per-namespace eviction has no tiebreak.

**The composite key.** Digests under different `layout_key` values are
already disjoint, because the key enters the digest seed. The composite
primary key is a second guard. Two snapshots of one architecture at one
layout share their blocks: `layout_key` does not depend on the weights.

### Schema version

`ssd_index::SCHEMA_VERSION` is the one source; its doc comment records what
each version changed. `SsdKvIndex::open`:

- no file: create the schema at the current version;
- no `schema_version` table, or a different version: `SchemaMismatch`;
- the current version: open.

Every schema change is a wipe, never a row migration: the tier holds nothing
that a prefill cannot rebuild. The stale-schema wipe removes every namespace
directory whose `index.db` holds `kv_blocks` at a version this binary
supersedes. It logs an `ssd_cache_pre_release_wipe` event with the dropped
bytes. A namespace at a newer version is left in place and logged as
`ssd_cache_newer_schema_skipped`, since two builds can share one machine. A
directory with no readable `kv_blocks` table is never touched.

`layout_key` folds codec names and shapes, not the bytes inside a block. A
change to a stored layout or a geometry tag therefore needs a
`SCHEMA_VERSION` bump, or old blocks would still hit.

---

## layout_key Salt

`compute_layout_key` hashes, with FNV-1a-64 from `FNV_OFFSET`:

```text
arch + ":<n_layers>:<n_kv_heads>:<head_dim>:<kv_quant>"
     + ":<layer_quants[0]>" + … + ":<layer_quants[n-1]>"
```

- `arch` is the resolved class (`Architecture::arch_class()`), not the
  declared string.
- `layer_quants` is the nominal per-layer codec vector from
  `kv_layer_quants(n_layers, kv_quant, shares_kv)`, the producer every
  arch's cache loop uses. `n_layers` is its length. It carries the
  boundary-layer promotion, so a change to that policy makes old blocks miss.
- It is nominal: windowed layers run the bf16 ring and shared-KV consumers
  own no cache, but neither filter is folded.
- It is fixed at attach, from the launch codec. `prepare_attach` refuses an
  empty vector.

`prepare_attach` logs the key as 16 hex characters in the `layout_key` field.

**The per-request seed.** A request may run a codec other than the launch one.
Its block digests are therefore seeded per request:

```text
seed    = FNV_OFFSET ^ layout_key ^ kv_quant.cache_key_salt() ^ model_sig
for each layer quant q:  seed ^= q.cache_key_salt();  seed *= FNV_PRIME
digests = chained_block_hashes_seeded(ids, seed)
```

`rmlx_kv_ssd::hashing::cache_seed` computes it, and
`prompt_cache::request_cache_seed` supplies the request's own layer vector.
`ArchPromptCache::consume` computes it once and passes the same value to
`find_best_prefix` and `hydrate_from_ssd`. The RAM push, the RAM query, the
spill key and the hydrate probe all read that one value. A probe seeded
differently never hits and reports no error.

**The hydrator holds no model or codec identity.** One `SsdHydrator` is
installed per architecture and outlives the model that attached it.
`--max-loaded-models` can keep several models of one arch resident. A
remembered `model_sig` would mis-seed every other model's probe. A remembered
`kv_quant` would mis-seed every hot-swapped request. The hydrator holds the
index, the directory, the namespace, the `layout_key` and the device.
`layout_key` carries no model identity, and `--project` can put several models
in one namespace. The seed's `model_sig` term keeps their blocks apart.

**Blocks under an older seed.** A changed seed term leaves old rows that no
probe asks for. Nothing can tell them from rows whose prompt has not returned,
so they are not purged. They carry the oldest `last_used`, so eviction takes
them first once the budget binds. Deleting the namespace directory reclaims
them at once; the tier rebuilds it.

## Live reconfiguration

A request can override its KV codec and context ceiling (`docs/SERVER.md`
§ "Per-request KV config"). The SSD tier is not part of that surface.
It is attached once per model, at load, by `attach_at_load`, and its budgets
are fixed at launch. There is no route to change them or to detach the tier.

The per-request seed keeps the RAM cache partitioned by codec and by model on
an SSD-active run, while the tier's `layout_key` stays the launch codec's.

---

## Hydrate path (RAM miss, SSD load, promote)

`SsdHydrator` is installed on the arch's prompt cache with `set_ssd_source`.
On a RAM miss it runs on the request thread:

1. **Lookup.** `chained_block_hashes_seeded(prompt_ids, seed)`, then
   `SsdKvIndex::lookup_longest_prefix(digests, layout_key)`. No row is a miss.
2. **Read.** `block_io::read_caches` checks the block's `model_id` (the
   namespace) and `kv_quant` metadata before reading tensors. Every rebuilt
   `KvCache` takes the request's `DispatchPolicy`.
3. **Touch.** `index.touch(hash, layout_key)` updates `last_used`.

It returns a `HydratedBlock`: the matched block-aligned token prefix, the
`KvCache`s and the `LinearAttnCache`s.

A corrupt block (bad read, metadata mismatch) is a miss: the row, then the
file, are deleted with a `warn!`. A missing file is the normal result of an
eviction racing the read. It drops the row with a `debug!` and is a miss. The
hydrator never panics and never returns an error to the caller.

### The per-arch entry impls

`SsdHydrate<E>` has one production implementation, the blanket
`impl<E: HydratedEntry> SsdHydrate<E> for SsdHydrator` in
`crates/rmlx-kv-ssd/src/traits.rs`. It runs the probe and hands the block to
`E::from_hydrated`:

```rust
pub trait HydratedEntry: Sized {
    /// The arch's `SHARES_KV_ACROSS_LAYERS`.
    const SHARES_KV: bool;
    fn from_hydrated(block: HydratedBlock, block_hashes: Vec<u64>, kv_quant: KvQuant) -> Self;
}
```

| File | Entry |
|---|---|
| `crates/rmlx-models/src/bitnet/prompt_cache.rs` | `BitNetEntry` |
| `crates/rmlx-models/src/gemma3/prompt_cache.rs` | `Gemma3Entry` |
| `crates/rmlx-models/src/gemma4/prompt_cache.rs` | `Gemma4Entry` |
| `crates/rmlx-models/src/laguna/prompt_cache.rs` | `LagunaEntry` |
| `crates/rmlx-models/src/qwen2/prompt_cache.rs` | `Qwen2Entry` |
| `crates/rmlx-models/src/qwen3.rs` | `Qwen3Entry` |
| `crates/rmlx-models/src/qwen3_vl_moe/prompt_cache.rs` | `Qwen3VlMoeEntry` |
| `crates/rmlx-models/src/qwen3_5_moe/prompt_cache.rs` | `Qwen35MoeEntry` |

`Qwen35MoeEntry` keeps the block's `lin_caches`; the other seven drop them.
Every entry sets `is_ssd_hydrated` and the placeholder `first_id` /
`first_piece`, and destructures `HydratedBlock` exhaustively. `SHARES_KV`
names the arch's own `SHARES_KV_ACROSS_LAYERS`. It lands on every restored
cache and decides the bf16 mirror a tail extension builds.

**Why the blanket impl lives in `rmlx-kv-ssd`.** In `rmlx-models` both
`SsdHydrate` and `SsdHydrator` are foreign and `E` is uncovered, so the orphan
rule rejects the impl (`E0210`). An arch entry is local to `rmlx-models`, so
its `HydratedEntry` impl is allowed there.

**Tests.** `crates/rmlx-models/src/ssd_hydrate_tests.rs` spills a block per
entry under a unique namespace and hydrates it back, on `Device::Cpu`. It
compares per-layer digests of the K and V dequants and checks the token
prefix. It also holds the population guards:

- `every_arch_that_hydrates_has_a_fixture_here`;
- `every_hydrating_entry_has_a_topology_assertion`;
- `every_entry_declares_its_archs_cross_layer_kv_topology`;
- `every_entry_names_its_archs_topology_constant`;
- `every_entry_destructures_the_block_exhaustively`;
- `no_production_entry_bypasses_the_blanket_impl`.

Value pins in `crates/rmlx-kv-ssd/src/ssd_tier_tests.rs` and
`crates/rmlx-kv-ssd/src/hashing_tests.rs` fix the output of
`compute_layout_key` and `cache_seed`. None of these reads the effect of
`SHARES_KV` on a cache extended after hydrate. Reading blocks a previous
process wrote is the job of `crates/rmlx-server/tests/ssd_cache_restart.rs`
and `docs/SSD_CANARY.md`.

---

## Spill path (RAM eviction, SSD write)

`SsdSpiller` is installed on the arch's prompt cache with `set_spill_sink`.
When `PromptCache::push` evicts an entry:

1. **Request thread.** An entry with no full block, or no known codec, is not
   spilled. Otherwise the spill sink deep-clones the caches and evaluates the
   clones (`eval_for_spill`) while the Metal context is its own. It then
   `try_send`s a `SpillJob` on a bounded channel of depth 16. A full channel
   drops the job with a `warn!`; decode never waits.
2. **Drain thread (`rmlx-kv-spill`).** `block_io::write_caches` serialises the
   job to `<namespace>/<hash>.kvb` and `index.record` adds the row. On error
   it logs, removes any partial file and drops the job. After a recorded block
   it runs evict-to-budget.

The job carries the last chained digest of the entry's prompt as `hash`, plus
`layout_key`, `model_id`, `kv_quant` and the cloned caches. An
`SsdSpillEvent` per job records the serialise, write and index times and the
byte size. If the drain thread cannot open the index, it drains the channel
and exits, and spill is off for that namespace.

**Iso and rotor stores.** Once a store's GPU ring is live, decode drops its
CPU blocks (`drop_blocks_when_ring_live_*`); the ring is then the only copy.
The store's `try_deep_clone` rebuilds complete CPU blocks from the ring, and
`block_io` writes those blocks. The writer refuses an iso or rotor store whose
blocks hold a token count other than the product of `shape[0..3]`
(`BlockIoError::TruncatedStore`).

### Evict-to-budget (runtime)

`SsdSpiller::spawn` reads the namespace ceiling once, from
`effective_namespace_budget`. After each recorded block the drain thread calls
`enforce_namespace_budget`, the routine attach also runs:

- `SsdKvIndex::evict_lru_until` deletes the oldest rows until the namespace
  fits, in one transaction, and returns only the rows it deleted. The scan
  stops once under the ceiling and uses the `last_used` index.
- The files of those rows are then unlinked.
- `rmlx_ssd_bytes_used` and the eviction counter are republished.

A ceiling of `0` means none is configured, and nothing is evicted. The budget
bounds the indexed total after each spill: a block larger than the ceiling is
written and then evicted.

Rows go before files, so a concurrent hydrate finds either no row or no file,
and misses. A reader never gets a block other than the one it asked for. A
block re-spilled between a failed read and its cleanup can still be lost: the
cleanup cannot tell the new row from the one it read.

---

## Cross-namespace LRU (evict_pool_lru_until)

With `global_budget_bytes > 0`, `install_config` bounds the whole pool:

1. Open every namespace index under `<RMLX_HOME>/cache/kv/`. One that fails to
   open is warned about and skipped.
2. Sum the pool. At or under the budget, return.
3. Sort all rows by `(last_used, namespace, hash)` and take the oldest until
   the pool fits.
4. Per namespace, unlink each file, then delete its row. Errors are warned
   about and the walk continues.

It logs one `ssd_pool_lru_eviction` event and returns an `EvictionReport`
(`bytes_freed`, `blocks_evicted`, `namespaces_touched`).

The sweep runs only in `install_config`. While a server runs, each namespace is
held to its own ceiling and nothing bounds the pool as a whole. Several
namespaces can therefore exceed the global budget until the next start.

## Attach and maintenance

At each model load, `attach_at_load` expands the layer vector and calls
`prepare_attach`. That resolves the namespace and `layout_key`, then runs the
namespace maintenance:

- `prune_missing` drops rows whose `.kvb` is gone;
- `enforce_namespace_budget` evicts to the ceiling.

The arch's `attach_ssd_tier` then installs the spiller and hydrator.
`prune_missing` runs only at attach; a file lost mid-run is repaired on the
read path.

Spill and hydrate errors never reach the inference path. An index that will
not open disables the tier for that namespace only. Editing `index.db` while a
server runs causes misses.

---

## Block IO (.kvb)

A `.kvb` is a safetensors file. Its `__metadata__`:

| Key | Value |
|---|---|
| `model_id` | the namespace: `--project`, or the model id |
| `kv_quant` | `KvQuant` Display string |
| `n_layers` | attention layers serialised |
| `seq_len` | tokens at serialisation |
| `n_linear` | GDN layers (`0` on pure attention) |
| `l{idx}.geom` | per-layer geometry: storage tag, `max_seq`, shape |

A reader whose `model_id` or `kv_quant` differs fails with
`BlockIoError::ModelIdMismatch` or `KvQuantMismatch` before any tensor read.

Each hydrated layer gets the codec the arch builder gives that layer
(`kv_layer_quants`), not the block's base codec. The caller passes that vector
to `SsdHydrator::lookup`, and a vector whose length is not the block's layer
count is an error, which the tier treats as a corrupt block. A boundary layer
holds the boundary floor, not the base codec
(`docs/KV_LAYER_POLICY.md` § "Layer-adaptive overrides"), and decode reads its
widths from the codec.

**What a layer writes follows what it holds.**
`KvStorage::is_geometry_only()` decides:

- a layer with a packed store writes its codec's tensors under
  `l{i}.k.*` / `l{i}.v.*` (codes, scales, and per codec biases, norms,
  rotations or rotor tables); a paged store writes its gathered pages;
- a layer with no packed store but a bf16 mirror writes `l{i}.k.bf16` and
  `l{i}.v.bf16` under the tag `none_bf16`;
- a layer with neither writes geometry only, tag `none`, and re-prefills on
  reuse;
- a GDN layer writes `lin{i}.conv_state` and `lin{i}.delta_state` whole.

The `none_bf16` case covers `--kv-quant none` and every mirror-fed codec,
whose `exit_prefill` builds no store (`docs/KV_CACHE.md` §9.6). The writer
refuses a mirror whose length differs from the cache `offset`, failing the
whole block: a decode-grown buffer is zero past `offset`, and a short one would
claim rows it does not hold. `Array::to_bytes` reads a strided
view in logical order (`docs/FFI.md` §"Data readback"). On hydrate the reader
re-seeds the pair with `KvCache::with_decode_fp16_seed`, so a disk hit decodes
from the bytes a RAM hit would. Codes and bf16 round-trip bit for bit.

#### SWA layers are not spilled — hydrated entries degrade to re-prefill

The bf16 rotating ring of a sliding-window layer has no `.kvb` form. On
hydrate, Gemma3 and Gemma4 SWA layers come back as payload-less
`KvStorage::None`.

Reusing such an entry as a prefix would give every SWA layer an empty
context. `KvCache::is_trimmable()` is `true` for a `None` layer, so it cannot
detect this. `Gemma3Entry::is_hydrate_complete` and
`Gemma4Entry::is_hydrate_complete` therefore require every attended layer to
hold a store or a bf16 seed. An incomplete hydrated entry falls back to a full
re-prefill. A RAM-resident entry always passes.

## Adding a codec

`block_io::write_layer` matches every `KvStorage` variant and writes its
geometry tag. `block_io::read_layer` matches every geometry tag. A new codec:

1. adds its `KvQuant` and `KvStorage` variants in `rmlx-kv-quant`;
2. adds its arm and geometry tag in `write_layer`;
3. adds the matching arm in `read_layer`;
4. adds a round-trip test in `block_io_tests.rs`.

A change to an existing codec's stored bytes or tag also bumps
`SCHEMA_VERSION`.

---

## See also

- `docs/PROMPT_CACHE.md` — the RAM prompt cache and its consume engine.
- `docs/KV_CACHE.md`, `docs/KV_CODECS.md` — the stores a block carries.
- `docs/PROJECTS_CONFIG.md` — `projects.toml`.
- `docs/METRICS_SCHEMA.md` §3.6 — the `events` table where `SsdSpillEvent` and
  `SsdHydrateEvent` land.
- `docs/SSD_CANARY.md` — the cross-restart probe.
