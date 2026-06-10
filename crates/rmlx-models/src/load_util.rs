//! Unified tensor fetch over a snapshot's shards — index-first for speed,
//! header-truth for correctness.
//!
//! Two failure modes drive the design:
//!
//! - **Index-first speed.** Well-formed snapshots (gemma4's 31b model) carry an
//!   accurate `model.safetensors.index.json`; an index view locates each tensor's
//!   shard in O(log n) without touching every header.
//! - **Header truth.** Some snapshots ship an index that lies — medgemma's index
//!   omits ~240 sibling tensors and mis-assigns ~255 to the wrong shard. For those
//!   the index view returns [`TensorLookup::NotInIndex`] /
//!   [`TensorLookup::WrongShard`] and we fall back to scanning every open shard's
//!   header. Existence checks ([`Weights::has`]) ALWAYS scan headers and never
//!   consult the index, because the index cannot be trusted to list every sibling.
//!
//! A corrupt/truncated shard header is never confused with "not here": it
//! propagates as `Err`, never masked by the header-scan fallback.
//!
//! This helper wires into nothing yet — per-arch loaders adopt it incrementally.

use std::cell::OnceCell;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{
    view_discriminated, ShardHandle, ShardIndex, ShardSet, TensorLookup, TensorView,
};
use rmlx_mlx::Array;
use safetensors::SafeTensors;
use tracing::debug;

use crate::layers::{Linear, QuantMode, QuantParams};

/// Unified tensor fetch over a snapshot's shards.
///
/// Index-first when an index is available (fast path); header-scan fallback for
/// index lies. Borrows the `ShardSet` (and optional `ShardIndex`) for its
/// lifetime — building a `Weights` opens no files.
///
/// Parsed safetensors headers are memoized per shard in `headers`, a vector of
/// empty `OnceCell`s indexed parallel to `shards.iter()` (deterministic
/// `BTreeMap` order). `has`/`scan_raw` parse a shard header at most once across
/// the whole load; a first-touch parse failure propagates and is NEVER cached
/// as "absent". Construction stays allocation-light — the `OnceCell`s are empty
/// until first touch.
///
/// Per-arch loaders adopt this incrementally; until then the only constructors
/// are the unit tests, hence the crate-local `dead_code` allow.
#[derive(Debug)]
#[allow(
    dead_code,
    reason = "shared loader helper — per-arch loaders adopt it incrementally; unwired until the loader migration lands"
)]
pub(crate) struct Weights<'a> {
    shards: &'a ShardSet,
    idx: Option<&'a ShardIndex>,
    /// Lazily-parsed safetensors header per shard, indexed parallel to
    /// `shards.iter()`. Empty until first touched by `header`.
    headers: Box<[OnceCell<SafeTensors<'a>>]>,
}

#[allow(
    dead_code,
    reason = "shared loader helper — per-arch loaders adopt it incrementally; unwired until the loader migration lands"
)]
impl<'a> Weights<'a> {
    /// Index-first fetch: the index locates each tensor's shard, with a
    /// header-scan fallback when the index misses or lies.
    ///
    /// The header-scan fallback reaches only shards the `ShardSet` actually
    /// opened. For snapshots whose index omits or mis-assigns tensors, pair with
    /// [`ShardSet::open_dir`] — an index-driven [`ShardSet::open`] may not open
    /// the shard holding an index-omitted tensor.
    pub(crate) fn new(shards: &'a ShardSet, idx: &'a ShardIndex) -> Self {
        Weights {
            headers: Self::empty_headers(shards),
            shards,
            idx: Some(idx),
        }
    }

    /// Header-scan-only fetch for snapshots without a usable index (the
    /// gemma3 / medgemma class) or index-less layouts (qwen3_vl_moe). Pair with
    /// [`ShardSet::open_dir`], which discovers shards by directory glob.
    pub(crate) fn scan_only(shards: &'a ShardSet) -> Self {
        Weights {
            headers: Self::empty_headers(shards),
            shards,
            idx: None,
        }
    }

    /// One empty `OnceCell` per open shard — the lazy header-memo backing store.
    fn empty_headers(shards: &ShardSet) -> Box<[OnceCell<SafeTensors<'a>>]> {
        (0..shards.len()).map(|_| OnceCell::new()).collect()
    }

    /// Parsed safetensors header for the shard at iteration index `i`, parsing
    /// (and memoizing) it on first touch.
    ///
    /// A parse failure is a corrupt/truncated shard — it propagates as `Err` and
    /// is never cached, so a later touch re-attempts and re-propagates rather
    /// than masking the corruption as "tensor absent".
    fn header(&self, i: usize, handle: &'a ShardHandle) -> Result<&SafeTensors<'a>> {
        // `i` indexes the cell parallel to this shard; `headers` is sized to
        // `shards.len()`, so `get` never misses — the error is defensive only.
        let cell = self
            .headers
            .get(i)
            .ok_or_else(|| Error::Loader(format!("shard header index {i} out of range")))?;
        if let Some(st) = cell.get() {
            return Ok(st);
        }
        // Empty cell (single-threaded access) → `get_or_init` parses once here.
        let st = handle.safetensors()?;
        Ok(cell.get_or_init(|| st))
    }

    /// Locate `name`, copy its bytes into a freshly allocated MLX [`Array`].
    ///
    /// When an index is present the index view runs first ([`view_discriminated`]):
    /// - [`TensorLookup::Found`] → build the array from the located view.
    /// - [`TensorLookup::NotInIndex`] / [`TensorLookup::WrongShard`] → fall back to
    ///   a header scan over every open shard (the `WrongShard` warning is already
    ///   logged at the lookup source).
    /// - `Err(...)` (corrupt header / I/O / alloc) → propagate, never fall back.
    ///
    /// Index-less ([`scan_only`](Weights::scan_only)) goes straight to the scan.
    pub(crate) fn array(&self, name: &str) -> Result<Array> {
        if let Some(idx) = self.idx {
            match view_discriminated(self.shards, idx, name)? {
                TensorLookup::Found(tv) => return Array::from_safetensor_view(&tv),
                // Index miss or index lies — safe to fall back to a header scan.
                TensorLookup::NotInIndex | TensorLookup::WrongShard => {}
            }
        }
        let (bytes, shape, dtype) = self.scan_raw(name)?.ok_or_else(|| {
            Error::Loader(format!(
                "tensor '{name}' not found in any open shard header"
            ))
        })?;
        // The view borrows `name` (call-site lifetime) and `bytes` (shard `'a`);
        // it is consumed immediately, so the call-site borrow suffices.
        let tv = TensorView {
            name,
            dtype,
            shape,
            bytes,
        };
        Array::from_safetensor_view(&tv)
    }

    /// Header-based existence check — NEVER consults the index.
    ///
    /// The index omits sibling tensors on the medgemma class of snapshots, so a
    /// `.scales`/`.biases` presence test must read shard headers. A corrupt
    /// header propagates as `Err` and is never masked as "absent" — masking it
    /// would let `linear()` wrap a quantized weight in `Linear::Plain` while the
    /// corrupt shard (which may hold the `.scales` sibling) is never touched.
    pub(crate) fn has(&self, name: &str) -> Result<bool> {
        for (i, (_filename, handle)) in self.shards.iter().enumerate() {
            let st = self.header(i, handle)?;
            if st.tensor(name).is_ok() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Raw bytes + shape + dtype for `name`, using the same index-first /
    /// header-scan-fallback resolution as [`array`](Weights::array).
    ///
    /// Returns an owned byte copy so the caller may outlive the borrow — the PARO
    /// byte-math path consumes this.
    pub(crate) fn raw(&self, name: &str) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
        if let Some(idx) = self.idx {
            match view_discriminated(self.shards, idx, name)? {
                TensorLookup::Found(tv) => {
                    return Ok((tv.bytes.to_vec(), tv.shape, tv.dtype));
                }
                TensorLookup::NotInIndex | TensorLookup::WrongShard => {}
            }
        }
        let (bytes, shape, dtype) = self.scan_raw(name)?.ok_or_else(|| {
            Error::Loader(format!(
                "tensor '{name}' not found in any open shard header"
            ))
        })?;
        Ok((bytes.to_vec(), shape, dtype))
    }

    /// Assemble a [`Linear`] for `<base>` from its `.weight` and (optional)
    /// `.scales` / `.biases` siblings.
    ///
    /// Sibling presence is detected via [`has`](Weights::has) (header-based, not
    /// the index). `qp` receives `has_biases` and resolves the quant params for
    /// this tensor (typically a closure over `resolve_quant`); it may hard-error
    /// on a config/data contradiction. With no `.scales` sibling the layer is
    /// [`Linear::Plain`]; otherwise [`Linear::Quantized`] with `biases: Some(_)`
    /// iff a `.biases` sibling exists.
    pub(crate) fn linear(
        &self,
        base: &str,
        qp: impl FnOnce(bool) -> Result<QuantParams>,
    ) -> Result<Linear> {
        let scales_name = format!("{base}.scales");
        if !self.has(&scales_name)? {
            // No scales sibling → plain bf16 weight.
            let weight = self.array(&format!("{base}.weight"))?;
            return Ok(Linear::Plain { weight });
        }

        let has_biases = self.has(&format!("{base}.biases"))?;
        let params = qp(has_biases)?;

        let weight = self.array(&format!("{base}.weight"))?;
        let scales = self.array(&scales_name)?;
        let biases = if has_biases {
            Some(self.array(&format!("{base}.biases"))?)
        } else {
            None
        };

        Ok(Linear::Quantized {
            weight,
            scales,
            biases,
            group_size: params.group_size,
            bits: params.bits,
            mode: QuantMode::from(params.mode.as_str()),
        })
    }

    /// Scan every open shard's header for `name`, returning the first match as a
    /// `(bytes, shape, dtype)` triple borrowing the shard mmap (`'a`).
    ///
    /// `Ok(Some(..))` — found; `Ok(None)` — absent from every shard (the caller
    /// decides whether that is an error). A header that fails to parse is a real
    /// corruption and propagates as `Err` — never silently treated as "absent".
    ///
    /// Returning the raw triple (not a `TensorView`) keeps the borrowed `bytes`
    /// lifetime (`'a`, the shard mmap) independent of the looked-up `name`'s
    /// call-site lifetime — `array` rebuilds a short-lived `TensorView` from it.
    fn scan_raw(&self, name: &str) -> Result<Option<(&'a [u8], Vec<usize>, safetensors::Dtype)>> {
        for (i, (filename, handle)) in self.shards.iter().enumerate() {
            // A parse failure is a corrupt/truncated shard — `header` propagates
            // it, never masking it as "not here, look elsewhere".
            let st = self.header(i, handle)?;
            if let Ok(t) = st.tensor(name) {
                debug!(
                    tensor = name,
                    shard = filename,
                    "tensor located by header scan"
                );
                return Ok(Some((t.data(), t.shape().to_vec(), t.dtype())));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
#[path = "load_util_tests.rs"]
mod load_util_tests;
