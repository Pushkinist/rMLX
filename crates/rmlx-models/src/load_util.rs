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
// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret
// of the i32 packed-pair buffer for Array::from_bytes (PARO assembly).
#![allow(unsafe_code)]

use std::cell::OnceCell;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{
    view_discriminated, ShardHandle, ShardIndex, ShardSet, TensorLookup, TensorView,
};
use rmlx_mlx::{Array, Dtype};
use rmlx_quant::awq::{
    convert_awq_qweight, convert_awq_qzeros_to_biases, f16_bits_to_f32, f32_to_f16_bits,
    quantize_f16_affine_int4,
};
use safetensors::SafeTensors;
use tracing::debug;

use crate::layers::{Embedding, Linear, QuantMode, QuantParams};

/// Read `config.json` under `model_dir` as raw JSON, preserving all keys —
/// including nested blocks (e.g. per-tensor `quantization` overrides) that
/// typed config structs drop.
pub(crate) fn read_raw_config(model_dir: &std::path::Path) -> Result<serde_json::Value> {
    let path = model_dir.join("config.json");
    let data = std::fs::read(&path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_slice(&data).map_err(|e| Error::Loader(format!("malformed config.json: {e}")))
}

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
#[derive(Debug)]
pub(crate) struct Weights<'a> {
    shards: &'a ShardSet,
    idx: Option<&'a ShardIndex>,
    /// Lazily-parsed safetensors header per shard, indexed parallel to
    /// `shards.iter()`. Empty until first touched by `header`.
    headers: Box<[OnceCell<SafeTensors<'a>>]>,
}

/// Reclassify an MLX error raised during the weights-load phase as
/// [`Error::Oom`] when its message carries an unambiguous allocation-failure
/// signature.
///
/// SCOPE / HONESTY: mlx-c surfaces every failure through one opaque string
/// channel, so OOM is NOT reliably distinguishable from a shape / kernel-compile
/// error at the status-code level. This classifier is therefore deliberately
/// bounded to the weights-load phase only — where a "[malloc_or_wait] Unable to
/// allocate" / "out of memory" string is overwhelmingly allocation, never a
/// shape mismatch (tensors come straight from a validated safetensors index).
/// It is intentionally NOT applied on the decode / forward path, where the same
/// substrings could plausibly come from a non-OOM failure and a false
/// `Error::Oom` would be worse than an honest 503.
fn classify_load_oom(e: Error) -> Error {
    let Error::Mlx(ref msg) = e else {
        return e;
    };
    let lower = msg.to_ascii_lowercase();
    let is_alloc_failure = lower.contains("out of memory")
        || lower.contains("failed to allocate")
        || lower.contains("unable to allocate")
        || lower.contains("insufficient memory");
    if is_alloc_failure {
        Error::Oom {
            phase: rmlx_core::OomPhase::LoadWeights,
            requested_bytes: None,
            peak_alloc_mb: None,
            msg: msg.clone(),
        }
    } else {
        e
    }
}

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
                TensorLookup::Found(tv) => {
                    return Array::from_safetensor_view(&tv).map_err(classify_load_oom);
                }
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
        Array::from_safetensor_view(&tv).map_err(classify_load_oom)
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
                    // Owned host byte copy for the PARO byte-math path. No OOM
                    // classification here: this is a `Vec<u8>` copy, not an MLX
                    // `Array` allocation, so `classify_load_oom` (which maps
                    // device-alloc error strings) does not apply — PARO's
                    // `from_bytes` call sites surface any device-alloc failure.
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

    /// Assemble an [`Embedding`] for `<base>` from its `.weight` and (optional)
    /// `.scales` / `.biases` siblings.
    ///
    /// Mirrors the structure and contract of [`linear`](Weights::linear): sibling
    /// presence is detected via [`has`](Weights::has) (header-based, not the
    /// index). `qp` receives `has_biases` and resolves the quant params for this
    /// tensor; it may hard-error on a config/data contradiction. With no `.scales`
    /// sibling the result is [`Embedding::Plain`]; otherwise [`Embedding::Quantized`]
    /// with `biases: Some(_)` iff a `.biases` sibling exists.
    ///
    /// All fetches go through [`array`](Weights::array), so load-phase OOM errors
    /// are classified uniformly.
    pub(crate) fn embedding(
        &self,
        base: &str,
        qp: impl FnOnce(bool) -> Result<QuantParams>,
    ) -> Result<Embedding> {
        let scales_name = format!("{base}.scales");
        if !self.has(&scales_name)? {
            // No scales sibling → plain bf16 weight.
            let weight = self.array(&format!("{base}.weight"))?;
            return Ok(Embedding::Plain { weight });
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

        Ok(Embedding::Quantized {
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

// ---------------------------------------------------------------------------
// PARO + embedding-quant assembly (Array-side; stays in rmlx-models because it
// needs `crate::paroquant_msl::pack_pairs_cpu`).
// ---------------------------------------------------------------------------

/// Reconstructed parts for one PARO INT4 linear layer.
///
/// Holds the raw MLX arrays a caller needs to build its own per-arch
/// `Linear::Paro` (the gemma4 and qwen3_5_moe `Linear` enums differ, so this
/// shared helper returns the arrays + scalars rather than a constructed layer).
pub(crate) struct ParoParts {
    /// Packed INT4 codes `[out, in*4/32]` U32.
    pub weight: Array,
    /// Per-group scales `[out, num_groups]` F16.
    pub scales: Array,
    /// Per-group biases (zero-points) `[out, num_groups]` F16.
    pub biases: Array,
    /// I32 packed pair indices `[krot, hidden/2]`.
    pub packed_pairs: Array,
    /// F16 cosine values `[krot, hidden/2]`.
    pub cos_theta: Array,
    /// F16 sine values `[krot, hidden/2]`.
    pub sin_theta: Array,
    /// F16 per-channel scales `[1, hidden]`.
    pub channel_scales: Array,
    /// Actual krot for this layer.
    pub krot: usize,
    /// Group size used by both the rotation kernel and the INT4 matmul.
    pub group_size: usize,
}

/// Reconstruct one PARO INT4 linear layer from its six raw checkpoint tensors.
///
/// Fetches `<base>.{qweight,scales,qzeros,theta,pairs,channel_scales}` via
/// [`Weights::raw`], converts AWQ packing to MLX layout, pre-computes cos/sin
/// from theta, and packs the rotation pairs via
/// [`crate::paroquant_msl::pack_pairs_cpu`]. Returns the assembled [`ParoParts`];
/// the caller wraps them in its arch-specific `Linear::Paro`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(crate) fn load_paro_parts(w: &Weights<'_>, base: &str, group_size: usize) -> Result<ParoParts> {
    let (qweight_bytes, qweight_shape, _) = w.raw(&format!("{base}.qweight"))?;
    let (scales_bytes, scales_shape, _) = w.raw(&format!("{base}.scales"))?;
    let (qzeros_bytes, _, _) = w.raw(&format!("{base}.qzeros"))?;
    let (theta_bytes, theta_shape, _) = w.raw(&format!("{base}.theta"))?;
    let (pairs_bytes, _, _) = w.raw(&format!("{base}.pairs"))?;
    let (channel_scales_bytes, _, _) = w.raw(&format!("{base}.channel_scales"))?;

    if scales_shape.len() != 2 || qweight_shape.len() != 2 {
        return Err(Error::Loader(format!(
            "load_paro_parts '{base}': unexpected tensor rank"
        )));
    }

    let num_groups = scales_shape[0];
    let out_features = scales_shape[1];
    let in_features = qweight_shape[0];

    let mlx_weight_bytes = convert_awq_qweight(&qweight_bytes, in_features, out_features, 4)?;
    let weight = Array::from_bytes(
        &mlx_weight_bytes,
        &[out_features as i32, (in_features * 4 / 32) as i32],
        Dtype::U32,
    )?;

    let (scales_bytes_t, biases_bytes_t) =
        convert_awq_qzeros_to_biases(&qzeros_bytes, &scales_bytes, num_groups, out_features, 4)?;
    let scales = Array::from_bytes(
        &scales_bytes_t,
        &[out_features as i32, num_groups as i32],
        Dtype::F16,
    )?;
    let biases = Array::from_bytes(
        &biases_bytes_t,
        &[out_features as i32, num_groups as i32],
        Dtype::F16,
    )?;

    if theta_shape.len() != 2 {
        return Err(Error::Loader(format!(
            "load_paro_parts '{base}': theta shape unexpected: {theta_shape:?}"
        )));
    }
    let krot = theta_shape[0];
    let half_hidden = theta_shape[1];
    let hidden = half_hidden * 2;

    let n_theta = krot * half_hidden;
    if theta_bytes.len() != n_theta * 2 {
        return Err(Error::Loader(format!(
            "load_paro_parts '{base}': theta bytes length {} != expected {}",
            theta_bytes.len(),
            n_theta * 2
        )));
    }
    let mut cos_bytes = vec![0u8; n_theta * 2];
    let mut sin_bytes = vec![0u8; n_theta * 2];
    for i in 0..n_theta {
        let th_bits = u16::from_le_bytes([theta_bytes[i * 2], theta_bytes[i * 2 + 1]]);
        let th_f32 = f16_bits_to_f32(th_bits);
        let cos_f16 = f32_to_f16_bits(th_f32.cos());
        let sin_f16 = f32_to_f16_bits(th_f32.sin());
        cos_bytes[i * 2..i * 2 + 2].copy_from_slice(&cos_f16.to_le_bytes());
        sin_bytes[i * 2..i * 2 + 2].copy_from_slice(&sin_f16.to_le_bytes());
    }
    let cos_theta = Array::from_bytes(&cos_bytes, &[krot as i32, half_hidden as i32], Dtype::F16)?;
    let sin_theta = Array::from_bytes(&sin_bytes, &[krot as i32, half_hidden as i32], Dtype::F16)?;

    let packed = crate::paroquant_msl::pack_pairs_cpu(&pairs_bytes, krot, hidden, group_size)?;
    // SAFETY: reinterpret the i32 packed-pair buffer as bytes for Array::from_bytes;
    // the slice is consumed immediately and never aliased mutably.
    let packed_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) };
    let packed_pairs =
        Array::from_bytes(packed_bytes, &[krot as i32, half_hidden as i32], Dtype::I32)?;

    let channel_scales =
        Array::from_bytes(&channel_scales_bytes, &[1i32, hidden as i32], Dtype::F16)?;

    Ok(ParoParts {
        weight,
        scales,
        biases,
        packed_pairs,
        cos_theta,
        sin_theta,
        channel_scales,
        krot,
        group_size,
    })
}

/// Quantize a stored-F16 embedding/lm_head weight `<name>.weight` to MLX affine
/// INT4 at load time (PARO checkpoints store these as F16).
///
/// Returns `(weight U32 [vocab, hidden*4/32], scales F16 [vocab, num_groups],
/// biases F16 [vocab, num_groups])` — the caller wraps them in its arch-specific
/// `Embedding::Quantized` / `Linear::Quantized`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(crate) fn quantize_embedding_int4(
    w: &Weights<'_>,
    name: &str,
    group_size: usize,
) -> Result<(Array, Array, Array)> {
    let (w_bytes, w_shape, _) = w.raw(&format!("{name}.weight"))?;
    if w_shape.len() != 2 {
        return Err(Error::Loader(format!(
            "{name}.weight: expected 2-D, got shape {w_shape:?}"
        )));
    }
    let vocab = w_shape[0];
    let hidden = w_shape[1];
    let num_groups = hidden / group_size;
    let (wq_bytes, sc_bytes, bi_bytes) =
        quantize_f16_affine_int4(&w_bytes, vocab, hidden, group_size)?;
    let weight = Array::from_bytes(
        &wq_bytes,
        &[vocab as i32, (hidden * 4 / 32) as i32],
        Dtype::U32,
    )?;
    let scales = Array::from_bytes(&sc_bytes, &[vocab as i32, num_groups as i32], Dtype::F16)?;
    let biases = Array::from_bytes(&bi_bytes, &[vocab as i32, num_groups as i32], Dtype::F16)?;
    Ok((weight, scales, biases))
}

#[cfg(test)]
#[path = "load_util_tests.rs"]
mod load_util_tests;
