//! `model.safetensors.index.json` parsing + single-file fallback.
//! Also provides `ShardHandle` (mmap-backed read-only file handle) and
//! `ShardSet` (collection of all open shard handles for a model).

// unsafe_code: safetensors mmap view — unsafe { Mmap::map(&file) } for zero-copy tensor loading,
// and the in-place remap that releases a copied tensor's pages
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_json::Value;
use tracing::{error, warn};

use rmlx_core::mapped_pages::{inner_page_range, page_size, remap_in_place};
use rmlx_core::{Error, Result};

// Bounded against malformed/adversarial model.safetensors.index.json.
// HuggingFace repos typically have <10k tensors; 65536 is a generous but defensive ceiling.
pub(crate) const MAX_TENSORS: usize = 65_536;

// Bounded against malformed/adversarial model.safetensors.index.json.
// HuggingFace repos typically use <100 shards; 512 is a generous but defensive ceiling.
pub(crate) const MAX_SHARDS: usize = 512;

/// The shard index of a multi-file snapshot.
pub(crate) const SHARD_INDEX: &str = "model.safetensors.index.json";

/// The shard file of a snapshot that has no index.
pub(crate) const SINGLE_SHARD: &str = "model.safetensors";

/// Returns `true` when `name` is one plain file name: no directory part, not
/// absolute, not `.` or `..`, no NUL byte. A name read from a snapshot file
/// must be one, or opening it would reach outside the snapshot directory.
///
/// The one component must be the whole name: `Path::components` drops a
/// trailing `/` and an inner `.`, so `x/` and `x/.` give the component `x`.
pub(crate) fn is_plain_file_name(name: &str) -> bool {
    let mut parts = Path::new(name).components();
    !name.contains('\0')
        && matches!(
            (parts.next(), parts.next()),
            (Some(Component::Normal(part)), None) if part == name
        )
}

// ── ShardHandle ──────────────────────────────────────────────────────────────

/// A memory-mapped, read-only handle to one `.safetensors` shard file.
///
/// The file is mapped lazily — OS pages are faulted in only when accessed.
/// No full file copy is made. The mapping is dropped automatically when this
/// handle is dropped. While the handle lives, [`release_mapped`] can drop the
/// resident pages of a tensor already copied out of it.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal RAII type — fields are the complete mmap-backed shard-handle contract; adding a field requires updating ShardHandle::open and all shard consumers"
)]
pub struct ShardHandle {
    /// Shard filename as it appears in `model.safetensors.index.json`.
    pub filename: String,
    /// Absolute on-disk path to the shard file.
    pub abs_path: PathBuf,
    // SAFETY invariant: `mmap` borrows from the underlying file for the
    // lifetime of this handle. We open the file read-only and never mutate
    // the mapped region; `release_mapped` only maps the same file range over
    // a sub-range of it, and the whole-range `munmap` on drop covers those.
    // Concurrent writes to the file from another process would be UB, but
    // model weights on disk are never modified while the server is running.
    mmap: memmap2::Mmap,
}

impl std::fmt::Debug for ShardHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardHandle")
            .field("filename", &self.filename)
            .field("abs_path", &self.abs_path)
            .field("mmap_len", &self.mmap.len())
            .finish()
    }
}

impl ShardHandle {
    /// Open `<model_dir>/<filename>` as a read-only memory-mapped file.
    pub fn open(model_dir: &Path, filename: &str) -> Result<Self> {
        let abs_path = model_dir.join(filename);
        let file = File::open(&abs_path)
            .map_err(|e| Error::Loader(format!("cannot open {}: {e}", abs_path.display())))?;

        // SAFETY: We open the file read-only. The mmap is read-only (Mmap, not MmapMut).
        // The file must not be truncated or externally modified while mapped; for
        // static model weight files this invariant always holds.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Loader(format!("mmap failed for {}: {e}", abs_path.display())))?;

        if !mmap.is_empty() {
            live_mappings().insert(
                mmap.as_ptr() as usize,
                LiveMapping {
                    len: mmap.len(),
                    file,
                },
            );
        }

        Ok(ShardHandle {
            filename: filename.to_owned(),
            abs_path,
            mmap,
        })
    }

    /// Borrows the entire mmap'd byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Parses the safetensors header and returns a view over `as_bytes()`.
    ///
    /// This reads only the compact JSON header (typically a few KB), not the
    /// full tensor data. Tensor bytes are accessed on demand via the returned
    /// view's `data()` slices.
    pub fn safetensors(&self) -> Result<safetensors::SafeTensors<'_>> {
        safetensors::SafeTensors::deserialize(self.as_bytes()).map_err(|e| {
            Error::Loader(format!(
                "cannot parse safetensors header in {}: {e}",
                self.filename
            ))
        })
    }
}

impl Drop for ShardHandle {
    fn drop(&mut self) {
        // Deregister before the fields drop and `munmap` runs, under the lock a
        // release holds, so no release can remap a range that is being unmapped.
        if !self.mmap.is_empty() {
            live_mappings().remove(&(self.mmap.as_ptr() as usize));
        }
    }
}

// ── release_mapped ───────────────────────────────────────────────────────────

/// A live shard mapping: `len` bytes of `file` from offset 0. The entry keeps
/// the file open for as long as its handle is registered.
struct LiveMapping {
    len: usize,
    file: File,
}

/// Every live [`ShardHandle`] mapping, by start address.
static LIVE_MAPPINGS: Mutex<BTreeMap<usize, LiveMapping>> = Mutex::new(BTreeMap::new());

/// Set after the first failed release, so a failure that repeats for every
/// tensor warns once.
static RELEASE_FAILED: AtomicBool = AtomicBool::new(false);

fn live_mappings() -> MutexGuard<'static, BTreeMap<usize, LiveMapping>> {
    LIVE_MAPPINGS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Releases the resident pages of `bytes` once they have been copied out of a
/// shard mapping, and returns how many bytes were released.
///
/// Only the whole pages inside `bytes` are released, so neighbouring tensors
/// keep theirs. The range stays mapped to the same file bytes: `bytes`, and
/// any other view of the range, still read the same values, faulted back in
/// from the page cache. Returns 0 when `bytes` lies in no live shard mapping
/// or holds no whole page. A failed release is logged and returns 0.
pub fn release_mapped(bytes: &[u8]) -> usize {
    let Some(page) = page_size() else {
        return 0;
    };
    let start = bytes.as_ptr() as usize;
    let end = start + bytes.len();
    let Some(pages) = inner_page_range(start, end, page) else {
        return 0;
    };
    let mappings = live_mappings();
    let Some((&base, mapping)) = mappings.range(..=start).next_back() else {
        return 0;
    };
    if end > base + mapping.len {
        return 0;
    }
    // SAFETY: `pages` is page-aligned and lies inside `bytes`, which lies
    // inside the mapping registered at `base`. `Mmap::map` maps the whole file
    // read-only and shared from offset 0, so the byte at `pages.start` is file
    // byte `pages.start - base`. The lock is held for the call and
    // `ShardHandle::drop` deregisters under it before unmapping, so the mapping
    // is live. Only pages inside the copied tensor are replaced, so a read of
    // another tensor of the shard is unaffected.
    let released = unsafe {
        remap_in_place(
            pages.start,
            pages.len(),
            &mapping.file,
            (pages.start - base) as u64,
        )
    };
    match released {
        Ok(()) => pages.len(),
        Err(e) => {
            if !RELEASE_FAILED.swap(true, Ordering::Relaxed) {
                warn!(
                    error = %e,
                    bytes = pages.len(),
                    "cannot release a copied tensor's mapped pages; the load continues with them resident"
                );
            }
            0
        }
    }
}

// ── ShardSet ─────────────────────────────────────────────────────────────────

/// All open shard handles for a model, keyed by filename.
#[derive(Debug)]
pub struct ShardSet {
    handles: BTreeMap<String, ShardHandle>,
}

impl ShardSet {
    /// Open every distinct shard listed in `idx.weight_map`.
    ///
    /// Shards are opened in parallel using rayon when there are
    /// multiple shards (single-shard models skip rayon entirely). Each
    /// shard open is `open(2) + mmap(2)` — fully independent, I/O-bound,
    /// and safe to overlap. Thread count is capped at `min(4, n_shards)`
    /// to leave cores available for MLX during model-load.
    pub fn open(model_dir: &Path, idx: &ShardIndex) -> Result<Self> {
        let filenames: Vec<String> = {
            let set: std::collections::BTreeSet<&String> = idx.weight_map.values().collect();
            set.into_iter().cloned().collect()
        };
        Self::open_filenames(model_dir, filenames)
    }

    /// Open the named shards from `model_dir`, deduplicating nothing (callers
    /// pass an already-distinct list) and parallelizing with a capped rayon pool
    /// for multi-shard sets. Shared open body behind [`open`](ShardSet::open) and
    /// [`open_dir`](ShardSet::open_dir).
    fn open_filenames(model_dir: &Path, filenames: Vec<String>) -> Result<Self> {
        let n_shards = filenames.len();

        if n_shards <= 1 {
            // Fast path: avoid rayon overhead for single-shard models.
            let mut handles = BTreeMap::new();
            for filename in filenames {
                let handle = ShardHandle::open(model_dir, &filename)?;
                handles.insert(filename, handle);
            }
            return Ok(ShardSet { handles });
        }

        // Multi-shard path: open shards in parallel, capped at 4 threads.
        let thread_cap = n_shards.min(4);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_cap)
            .build()
            .map_err(|e| Error::Loader(format!("rayon pool build failed: {e}")))?;

        let model_dir_owned = model_dir.to_path_buf();
        let results: Vec<Result<(String, ShardHandle)>> = pool.install(|| {
            filenames
                .into_par_iter()
                .map(|filename| {
                    let handle = ShardHandle::open(&model_dir_owned, &filename)?;
                    Ok((filename, handle))
                })
                .collect()
        });

        let mut handles = BTreeMap::new();
        for result in results {
            let (name, handle) = result?;
            handles.insert(name, handle);
        }

        Ok(ShardSet { handles })
    }

    /// Open every `*.safetensors` shard in `model_dir` without consulting an index.
    ///
    /// For snapshots whose `model.safetensors.index.json` is absent or
    /// untrustworthy (the gemma3 / medgemma class — the index omits sibling
    /// tensors or mis-assigns shards) and for index-less layouts (qwen3_vl_moe,
    /// whose shipped index references a pre-sanitize shard layout). Files are
    /// discovered by directory glob and opened with the same mmap/handle path as
    /// [`open`](ShardSet::open).
    ///
    /// Shards are opened in parallel using rayon when there are multiple shards
    /// (single-shard discovery skips rayon entirely), capped at `min(4, n)` to
    /// leave cores for MLX during model-load — identical to [`open`](ShardSet::open).
    pub fn open_dir(model_dir: &Path) -> Result<Self> {
        // Propagate `read_dir` enumeration errors instead of silently shrinking
        // the shard set — a dropped shard surfaces later as "tensor not found",
        // the exact masking this resolver exists to avoid.
        let mut filenames: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(model_dir)
            .map_err(|e| Error::Loader(format!("cannot read dir {}: {e}", model_dir.display())))?
        {
            let entry = entry.map_err(|e| {
                Error::Loader(format!("read_dir entry in {}: {e}", model_dir.display()))
            })?;
            match entry.file_name().into_string() {
                Ok(n) if n.ends_with(".safetensors") => filenames.push(n),
                Ok(_) => {}
                Err(os) => {
                    tracing::warn!(name = ?os, "skipping non-UTF8 filename during shard discovery");
                }
            }
        }
        filenames.sort();

        if filenames.is_empty() {
            return Err(Error::Loader(format!(
                "no *.safetensors shards found in {}",
                model_dir.display()
            )));
        }

        if filenames.len() > MAX_SHARDS {
            error!(
                got = filenames.len(),
                max = MAX_SHARDS,
                dir = %model_dir.display(),
                "shard count exceeds MAX_SHARDS bound — possible malformed snapshot directory"
            );
            return Err(Error::Loader(format!(
                "{} holds {} *.safetensors shards (max {MAX_SHARDS})",
                model_dir.display(),
                filenames.len()
            )));
        }

        Self::open_filenames(model_dir, filenames)
    }

    /// Look up a shard handle by filename.
    pub fn get(&self, filename: &str) -> Option<&ShardHandle> {
        self.handles.get(filename)
    }

    /// Iterate over `(filename, handle)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ShardHandle)> {
        self.handles.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of shards in the set.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Returns `true` when the set has no shards.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }
}

/// Parsed shard index.
///
/// `weight_map` maps tensor name to the shard filename that contains it.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed loader struct — fields are the complete shard-index contract; adding a field requires updating load_shard_index and all index consumers"
)]
#[derive(Debug, Clone)]
pub struct ShardIndex {
    /// Raw `metadata` block from the index JSON (or `Value::Null` for the
    /// synthetic single-file case where the field is absent).
    pub metadata: Value,
    /// tensor-name → shard-filename, sorted for deterministic output.
    pub weight_map: BTreeMap<String, String>,
}

/// Parse `<model_dir>/model.safetensors.index.json`.
///
/// If that file is absent but `<model_dir>/model.safetensors` exists, returns
/// a synthetic `ShardIndex` whose `weight_map` maps every tensor in that
/// single shard to `"model.safetensors"`.
pub fn load_shard_index(model_dir: &Path) -> Result<ShardIndex> {
    let index_path = model_dir.join(SHARD_INDEX);
    let single_path = model_dir.join(SINGLE_SHARD);

    if index_path.exists() {
        load_from_index_file(&index_path)
    } else if single_path.exists() {
        load_synthetic_from_single(model_dir, SINGLE_SHARD)
    } else {
        Err(Error::Loader(format!(
            "neither {} nor {} found in {}",
            index_path.display(),
            single_path.display(),
            model_dir.display()
        )))
    }
}

/// Returns a `BTreeMap<shard_filename, tensor_count>`.
pub fn count_tensors_per_shard(idx: &ShardIndex) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for shard in idx.weight_map.values() {
        *counts.entry(shard.clone()).or_insert(0) += 1;
    }
    counts
}

// ── private ──────────────────────────────────────────────────────────────────

fn load_from_index_file(path: &Path) -> Result<ShardIndex> {
    let data = std::fs::read(path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;

    let mut root: serde_json::Map<String, Value> = serde_json::from_slice(&data)
        .map_err(|e| Error::Loader(format!("malformed {}: {e}", path.display())))?;

    let metadata = root.remove("metadata").unwrap_or(Value::Null);

    let raw_map = root
        .remove("weight_map")
        .ok_or_else(|| Error::Loader(format!("missing 'weight_map' in {}", path.display())))?;

    let map_obj = raw_map.as_object().ok_or_else(|| {
        Error::Loader(format!(
            "'weight_map' is not an object in {}",
            path.display()
        ))
    })?;

    // Reject oversized weight_maps before building the BTreeMap.
    let raw_len = map_obj.len();
    if raw_len > MAX_TENSORS {
        error!(
            got = raw_len,
            max = MAX_TENSORS,
            file = "model.safetensors.index.json",
            "weight_map exceeds MAX_TENSORS bound — possible malformed or adversarial index"
        );
        return Err(Error::Loader(format!(
            "weight_map has {raw_len} entries (max {MAX_TENSORS}) in {}",
            path.display()
        )));
    }

    let mut weight_map = BTreeMap::new();
    for (tensor, shard_val) in map_obj {
        let shard = shard_val.as_str().ok_or_else(|| {
            Error::Loader(format!(
                "weight_map value for '{tensor}' is not a string in {}",
                path.display()
            ))
        })?;
        if !is_plain_file_name(shard) || !shard.ends_with(".safetensors") {
            return Err(Error::Loader(format!(
                "weight_map value for '{tensor}' is {shard:?}, not a .safetensors file name \
                 in the snapshot directory, in {}",
                path.display()
            )));
        }
        weight_map.insert(tensor.clone(), shard.to_owned());
    }

    // Reject oversized shard counts after deduplication.
    let n_shards = weight_map
        .values()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>()
        .len();
    if n_shards > MAX_SHARDS {
        error!(
            got = n_shards,
            max = MAX_SHARDS,
            file = "model.safetensors.index.json",
            "shard count exceeds MAX_SHARDS bound — possible malformed or adversarial index"
        );
        return Err(Error::Loader(format!(
            "index references {n_shards} distinct shards (max {MAX_SHARDS}) in {}",
            path.display()
        )));
    }

    Ok(ShardIndex {
        metadata,
        weight_map,
    })
}

/// The index of a one-file snapshot: every tensor of the shard's header maps
/// to the shard. The shard is mapped, not read, so only its header pages are
/// touched.
fn load_synthetic_from_single(model_dir: &Path, filename: &str) -> Result<ShardIndex> {
    let shard = ShardHandle::open(model_dir, filename)?;
    let weight_map = shard
        .safetensors()?
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), filename.to_owned()))
        .collect();

    Ok(ShardIndex {
        metadata: Value::Null,
        weight_map,
    })
}

#[cfg(test)]
#[path = "shards_tests.rs"]
mod tests;
