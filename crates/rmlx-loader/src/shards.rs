//! `model.safetensors.index.json` parsing + single-file fallback.
//! Also provides `ShardHandle` (mmap-backed read-only file handle) and
//! `ShardSet` (collection of all open shard handles for a model).

// unsafe_code: safetensors mmap view — unsafe { Mmap::map(&file) } for zero-copy tensor loading
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde_json::Value;
use tracing::error;

use rmlx_core::{Error, Result};

// Bounded against malformed/adversarial model.safetensors.index.json.
// HuggingFace repos typically have <10k tensors; 65536 is a generous but defensive ceiling.
pub(crate) const MAX_TENSORS: usize = 65_536;

// Bounded against malformed/adversarial model.safetensors.index.json.
// HuggingFace repos typically use <100 shards; 512 is a generous but defensive ceiling.
pub(crate) const MAX_SHARDS: usize = 512;

// ── ShardHandle ──────────────────────────────────────────────────────────────

/// A memory-mapped, read-only handle to one `.safetensors` shard file.
///
/// The file is mapped lazily — OS pages are faulted in only when accessed.
/// No full file copy is made. The mapping is dropped automatically when this
/// handle is dropped.
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
    // the mapped region. Concurrent writes to the file from another process
    // would be UB, but model weights on disk are never modified while the
    // server is running.
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
        let file = std::fs::File::open(&abs_path)
            .map_err(|e| Error::Loader(format!("cannot open {}: {e}", abs_path.display())))?;

        // SAFETY: We open the file read-only. The mmap is read-only (Mmap, not MmapMut).
        // The file must not be truncated or externally modified while mapped; for
        // static model weight files this invariant always holds.
        let mmap = unsafe { memmap2::Mmap::map(&file) }
            .map_err(|e| Error::Loader(format!("mmap failed for {}: {e}", abs_path.display())))?;

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
        let filenames: Vec<&String> = {
            let set: std::collections::BTreeSet<&String> = idx.weight_map.values().collect();
            set.into_iter().collect()
        };
        let n_shards = filenames.len();

        if n_shards <= 1 {
            // Fast path: avoid rayon overhead for single-shard models.
            let mut handles = BTreeMap::new();
            for filename in filenames {
                let handle = ShardHandle::open(model_dir, filename)?;
                handles.insert(filename.clone(), handle);
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
                    let handle = ShardHandle::open(&model_dir_owned, filename)?;
                    Ok((filename.clone(), handle))
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
    let index_path = model_dir.join("model.safetensors.index.json");
    let single_path = model_dir.join("model.safetensors");

    if index_path.exists() {
        load_from_index_file(&index_path)
    } else if single_path.exists() {
        load_synthetic_from_single(&single_path)
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

fn load_synthetic_from_single(path: &Path) -> Result<ShardIndex> {
    let data = std::fs::read(path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;

    // Use safetensors to enumerate tensor names without loading tensor data.
    let st = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| Error::Loader(format!("cannot parse {}: {e}", path.display())))?;

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.safetensors")
        .to_owned();

    let weight_map = st
        .names()
        .into_iter()
        .map(|name| (name.to_owned(), filename.clone()))
        .collect();

    Ok(ShardIndex {
        metadata: Value::Null,
        weight_map,
    })
}

#[cfg(test)]
#[path = "shards_tests.rs"]
mod tests;
