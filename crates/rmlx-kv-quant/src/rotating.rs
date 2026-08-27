// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Rotating (ring-buffer) KV cache for SWA layers.
//!
//! Byte-for-byte port of mlx-lm's `RotatingKVCache`
//! (`mlx_lm/models/cache.py:410-592`). Pre-allocates a `[B, kv_h, max_size, D]`
//! buffer once and rotates writes modulo `max_size`. After the buffer fills,
//! attention reads at most `max_size` tokens — no per-step trimming, no per-step
//! window mask.
//!
//! Used by SWA layers (e.g. Gemma3/Gemma4) under **every** `KvQuant`. The
//! rotating branch is selected on `window > 0` alone
//! ([`crate::KvCache::with_quant_max_seq_window`]), so a quantized codec on an
//! SWA layer runs this bf16 ring and its [`crate::KvStorage`] is never
//! allocated; mlx-lm's reference also keeps RotatingKVCache as bf16
//! (`to_quantized` raises `NotImplementedError`).
//!
//! Algorithm (mirrors mlx-lm exactly):
//! - `update_and_fetch(K_new, V_new)`:
//! - If new sequence length is 1 (decode), call `_update_in_place`.
//! - Otherwise (prefill), call `_update_concat`.
//! - `_update_in_place` grows the buffer in `step` chunks until it reaches
//!   `max_size`, then rotates writes modulo `max_size`.
//! - `_update_concat` puts the cache in temporal order, trims, and concatenates
//!   the new keys/values — used during prefill to absorb a long prompt.
//!
//! `keep` is fixed at 0 for Gemma SWA layers (mlx-lm `gemma4_text.py::Model.make_cache`
//! line 686: `RotatingKVCache(max_size=sliding_window)` — no keep arg).

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, zeros, Array, Device, Dtype};

/// Step size for buffer growth, mirrors mlx-lm `RotatingKVCache.step = 256`.
const ROTATING_STEP: i32 = 256;

/// Owned snapshot of a [`RotatingState`], the rMLX equivalent of mlx-lm's
/// `RotatingKVCache.state` (the ring buffer arrays) + `meta_state`
/// (`(keep, max_size, offset, _idx)`).
///
/// Produced by [`RotatingState::snapshot`] and consumed by
/// [`RotatingState::restore`]. Holds refcount clones of the ring buffers
/// (no tensor data copied — MLX copy-on-write applies on the next write) plus
/// the four scalar meta fields. Storing this in the prompt-cache entry lets a
/// later turn whose prompt extends the cached prefix RESUME the SWA ring
/// instead of re-prefilling it (B1 fix).
#[allow(missing_debug_implementations)]
pub(super) struct RotatingSnapshot {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) offset: i32,
    pub(super) max_size: i32,
    pub(super) keep: i32,
    pub(super) idx: i32,
}

/// A restart-safe, host-owned representation of one rotating KV ring tensor.
///
/// `bytes` are the row-major bytes of the *physical* ring buffer, not a
/// temporal reorder.  Keeping the physical layout together with the ring
/// position makes the next append exactly equivalent after restoration.
#[allow(clippy::exhaustive_structs)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotatingTensorSnapshot {
    /// Row-major tensor payload.
    pub bytes: Vec<u8>,
    /// Tensor shape in MLX order.
    pub shape: Vec<i32>,
    /// Tensor element dtype.
    pub dtype: Dtype,
}

/// Lossless, process-independent snapshot of a [`RotatingState`].
///
/// The K/V payload is copied to host memory, so this value can be serialized
/// by the caller and restored in a later process.  `idx` is the next physical
/// ring position (and is deliberately retained separately from `valid_len`),
/// while `offset` is the absolute number of tokens accepted by the cache.
#[allow(clippy::exhaustive_structs)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotatingStateSnapshot {
    /// Physical K ring payload.
    pub keys: Option<RotatingTensorSnapshot>,
    /// Physical V ring payload.
    pub values: Option<RotatingTensorSnapshot>,
    /// Absolute number of accepted tokens.
    pub offset: i32,
    /// Configured ring capacity.
    pub max_size: i32,
    /// Prefix retained during rotation.
    pub keep: i32,
    /// Number of valid logical positions.
    pub valid_len: i32,
    /// Next physical write position.
    pub idx: i32,
}

/// Internal state for a rotating bf16 KV cache.
#[allow(missing_debug_implementations)]
pub struct RotatingState {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) offset: i32,
    pub(super) max_size: i32,
    pub(super) keep: i32,
    pub(super) idx: i32,
}

impl RotatingState {
    /// Resident bytes of the SWA ring, counting the filled prefix of each
    /// buffer.
    ///
    /// The ring holds at most `max_size` live positions and is allocated to
    /// that window; early in a sequence only `filled` of them carry K/V.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile. Reaching the ring's
    /// buffers by field access from the caller would leave a third buffer
    /// silently uncounted, which is the whole class this accounting closes.
    pub(super) fn byte_size(&self, filled: u64) -> u64 {
        let Self {
            keys,
            values,
            // Ring bookkeeping, not allocations.
            offset: _,
            max_size: _,
            keep: _,
            idx: _,
        } = self;
        crate::bytes::opt_filled_seq_bytes(keys.as_ref(), filled)
            + crate::bytes::opt_filled_seq_bytes(values.as_ref(), filled)
    }

    /// Construct an empty ring with the given capacity.
    pub fn new(max_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
            keep: 0,
            idx: 0,
        }
    }

    /// Configured sliding-window capacity.
    pub fn max_size(&self) -> i32 {
        self.max_size
    }

    pub(super) fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.idx = 0;
    }

    /// Lossless rollback by `n` positions, mirroring mlx-lm
    /// `RotatingKVCache.is_trimmable` + `trim` (`cache.py:542-549`).
    ///
    /// Returns the number of positions actually rolled back. mlx-lm's
    /// `is_trimmable` is `self.offset < self.max_size` — the ring buffer is
    /// rollable iff it has not yet wrapped. After a wrap, the original
    /// pre-wrap K/V have been overwritten; mlx-lm's `trim_prompt_cache`
    /// silently returns 0 in that case (`cache.py:109-111`). This port
    /// matches that exact behaviour: returns 0 (no-op) when not trimmable.
    pub(super) fn trim_lossless(&mut self, n: i32) -> i32 {
        if self.offset >= self.max_size {
            // Not trimmable — see mlx-lm `is_trimmable`. Caller treats this
            // as `trim_prompt_cache returned 0`.
            return 0;
        }
        let n = n.min(self.offset);
        self.offset -= n;
        self.idx -= n;
        n
    }

    /// Deep (refcount) clone of the ring.
    ///
    /// Implemented as `snapshot` followed by a `restore` into a fresh state so
    /// the SWA prompt-cache reuse path (B1) and the existing entry deep-clone
    /// share ONE snapshot/restore implementation — `snapshot`/`restore` are
    /// the single source of truth for ring state transfer (no parallel copy
    /// logic to drift out of sync).
    pub(super) fn try_deep_clone(&self) -> Result<Self> {
        let snap = self.snapshot()?;
        let mut out = Self::new(self.max_size);
        out.restore(&snap)?;
        Ok(out)
    }

    /// Snapshot the ring state, mirroring mlx-lm `RotatingKVCache.state`
    /// + `meta_state` (`cache.py:520-540`).
    ///
    /// mlx-lm's `state` getter returns the raw ring buffer arrays *as stored*
    /// (rotated layout — NOT temporal order) and `meta_state` carries
    /// `(keep, max_size, offset, _idx)` so the buffer interpretation is fully
    /// preserved. The `state.setter` does a plain `self.keys, self.values = v`
    /// (no temporal reorder) — round-trip exactness depends entirely on the
    /// meta being restored alongside the buffers. This port captures exactly
    /// that contract: a deep refcount-clone of the buffer arrays plus the four
    /// scalar meta fields. A subsequent `update_and_fetch` on the restored
    /// state behaves bit-identically to the un-snapshotted cache because
    /// `_update_in_place` / `_update_concat` derive everything they need from
    /// `offset`, `idx`, `max_size`, `keep`, and `keys.shape[2]`.
    pub(super) fn snapshot(&self) -> Result<RotatingSnapshot> {
        Ok(RotatingSnapshot {
            keys: match &self.keys {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            values: match &self.values {
                Some(a) => Some(a.try_clone()?),
                None => None,
            },
            offset: self.offset,
            max_size: self.max_size,
            keep: self.keep,
            idx: self.idx,
        })
    }

    /// Capture a restart-safe snapshot. Unlike [`Self::snapshot`], this
    /// evaluates each ring array on the calling (inference-owner) thread and
    /// copies it to host bytes, so the result does not retain MLX device
    /// allocations or process-local array handles.
    pub fn snapshot_persistent(&self) -> Result<RotatingStateSnapshot> {
        fn tensor(a: &Array) -> Result<RotatingTensorSnapshot> {
            Ok(RotatingTensorSnapshot {
                bytes: a.to_bytes()?,
                shape: a.shape(),
                dtype: a.dtype(),
            })
        }
        Ok(RotatingStateSnapshot {
            keys: self.keys.as_ref().map(tensor).transpose()?,
            values: self.values.as_ref().map(tensor).transpose()?,
            offset: self.offset,
            max_size: self.max_size,
            keep: self.keep,
            valid_len: self.offset.min(self.max_size),
            idx: self.idx,
        })
    }

    /// Restore the ring from a [`RotatingSnapshot`], rebuilding the exact
    /// pre-snapshot state (mlx-lm `state.setter` + `meta_state.setter`).
    ///
    /// After this call the ring is bit-identical to the cache the snapshot was
    /// taken from: subsequent appends/attention behave exactly as if the
    /// snapshot had never been taken. Round-trip (`snapshot` → `restore`) is
    /// exact.
    pub(super) fn restore(&mut self, snap: &RotatingSnapshot) -> Result<()> {
        self.keys = match &snap.keys {
            Some(a) => Some(a.try_clone()?),
            None => None,
        };
        self.values = match &snap.values {
            Some(a) => Some(a.try_clone()?),
            None => None,
        };
        self.offset = snap.offset;
        self.max_size = snap.max_size;
        self.keep = snap.keep;
        self.idx = snap.idx;
        Ok(())
    }

    /// Restore a persistent snapshot onto `device`.
    ///
    /// Shape, dtype, and ring metadata are checked before MLX arrays are
    /// created.  K and V must describe the same batch/head/ring dimensions;
    /// their final head dimensions may differ, as in the live cache.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape length is checked before indexing and slicing"
    )]
    pub fn restore_persistent(
        &mut self,
        snap: &RotatingStateSnapshot,
        device: Device,
    ) -> Result<()> {
        fn array(t: &RotatingTensorSnapshot, device: Device) -> Result<Array> {
            let host = Array::from_bytes(&t.bytes, &t.shape, t.dtype)?;
            host.astype(t.dtype, device)
        }

        if snap.max_size <= 0
            || snap.keep < 0
            || snap.keep > snap.max_size
            || snap.offset < 0
            || snap.valid_len < 0
            || snap.valid_len > snap.max_size
            || snap.idx < 0
        {
            return Err(Error::Mlx("invalid rotating snapshot metadata".into()));
        }
        if snap.valid_len != snap.offset.min(snap.max_size) {
            return Err(Error::Mlx(
                "rotating snapshot valid_len is inconsistent with offset".into(),
            ));
        }
        match (&snap.keys, &snap.values) {
            (None, None) if snap.offset == 0 && snap.valid_len == 0 => {}
            (None, None) => {
                return Err(Error::Mlx(
                    "rotating snapshot is missing K/V payload for a non-empty cache".into(),
                ))
            }
            (Some(k), Some(v)) => {
                let valid = |t: &RotatingTensorSnapshot| {
                    t.shape.len() == 4
                        && t.shape[0] > 0
                        && t.shape[1] > 0
                        && t.shape[2] > 0
                        && t.shape[3] > 0
                        && t.bytes.len()
                            == t.shape.iter().map(|&x| x as usize).product::<usize>()
                                * t.dtype.itemsize()
                };
                if !valid(k)
                    || !valid(v)
                    || k.shape[..3] != v.shape[..3]
                    || k.dtype != v.dtype
                    || snap.idx > k.shape[2]
                {
                    return Err(Error::Mlx(
                        "invalid rotating snapshot tensor invariants".into(),
                    ));
                }
            }
            _ => {
                return Err(Error::Mlx(
                    "rotating snapshot must contain both K and V".into(),
                ))
            }
        }
        self.keys = snap.keys.as_ref().map(|t| array(t, device)).transpose()?;
        self.values = snap.values.as_ref().map(|t| array(t, device)).transpose()?;
        self.offset = snap.offset;
        self.max_size = snap.max_size;
        self.keep = snap.keep;
        self.idx = snap.idx;
        Ok(())
    }

    /// Port of mlx-lm `RotatingKVCache.update_and_fetch` (`cache.py:512-515`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn update_and_fetch(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let s = new_k.shape()[2];
        if s == 1 {
            self.update_in_place(new_k, new_v, device)
        } else {
            self.update_concat(new_k, new_v, device)
        }
    }

    // ── helpers (byte-for-byte mlx-lm port) ────────────────────────────────

    /// Port of `RotatingKVCache._trim` (`cache.py:421-429`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn trim(
        &self,
        trim_size: i32,
        v: &Array,
        append: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let shape = v.shape();
        let ndim = shape.len();
        if ndim != 4 {
            return Err(Error::Mlx(format!(
                "RotatingKvCache::trim expected 4-D tensor, got ndim={ndim}"
            )));
        }
        let strides = vec![1i32; 4];

        let mut to_cat: Vec<Array> = Vec::with_capacity(3);
        if trim_size > 0 {
            let keep_start = vec![0i32; 4];
            let keep_stop: Vec<i32> = [shape[0], shape[1], self.keep, shape[3]].into();
            let keep_slice = v.slice(&keep_start, &keep_stop, &strides, device)?;
            to_cat.push(keep_slice);

            let mut tail_start = vec![0i32; 4];
            tail_start[2] = trim_size + self.keep;
            let tail_stop: Vec<i32> = [shape[0], shape[1], shape[2], shape[3]].into();
            let tail_slice = v.slice(&tail_start, &tail_stop, &strides, device)?;
            to_cat.push(tail_slice);
        } else {
            to_cat.push(v.try_clone()?);
        }
        if let Some(a) = append {
            to_cat.push(a.try_clone()?);
        }
        let refs: Vec<&Array> = to_cat.iter().collect();
        concatenate(&refs, 2, device)
    }

    /// Port of `RotatingKVCache._temporal_order` (`cache.py:431-447`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn temporal_order(&self, v: &Array, device: Device) -> Result<Array> {
        let shape = v.shape();
        let strides = vec![1i32; 4];

        if self.idx == shape[2] {
            return v.try_clone();
        } else if self.idx < self.offset {
            let keep_start = vec![0i32; 4];
            let keep_stop: Vec<i32> = [shape[0], shape[1], self.keep, shape[3]].into();
            let part_keep = v.slice(&keep_start, &keep_stop, &strides, device)?;

            let mut new_start = vec![0i32; 4];
            new_start[2] = self.idx;
            let new_stop: Vec<i32> = [shape[0], shape[1], shape[2], shape[3]].into();
            let part_new = v.slice(&new_start, &new_stop, &strides, device)?;

            let mut old_start = vec![0i32; 4];
            old_start[2] = self.keep;
            let old_stop: Vec<i32> = [shape[0], shape[1], self.idx, shape[3]].into();
            let part_old = v.slice(&old_start, &old_stop, &strides, device)?;

            let parts = [&part_keep, &part_new, &part_old];
            concatenate(&parts, 2, device)
        } else {
            let start = vec![0i32; 4];
            let stop: Vec<i32> = [shape[0], shape[1], self.idx, shape[3]].into();
            v.slice(&start, &stop, &strides, device)
        }
    }

    /// Port of `RotatingKVCache._update_concat` (`cache.py:449-467`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_concat(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        if self.keys.is_none() {
            self.keys = Some(new_k.try_clone()?);
            self.values = Some(new_v.try_clone()?);
        } else {
            let kt = self.temporal_order(self.keys.as_ref().unwrap(), device)?;
            let vt = self.temporal_order(self.values.as_ref().unwrap(), device)?;
            self.keys = Some(kt);
            self.values = Some(vt);
            self.idx = self.keys.as_ref().unwrap().shape()[2];

            let trim_size = self.idx - self.max_size + 1;
            let kt2 = self.trim(trim_size, self.keys.as_ref().unwrap(), Some(new_k), device)?;
            let vt2 = self.trim(
                trim_size,
                self.values.as_ref().unwrap(),
                Some(new_v),
                device,
            )?;
            self.keys = Some(kt2);
            self.values = Some(vt2);
        }
        self.offset += new_k.shape()[2];
        self.idx = self.keys.as_ref().unwrap().shape()[2];
        Ok((
            self.keys.as_ref().unwrap().try_clone()?,
            self.values.as_ref().unwrap().try_clone()?,
        ))
    }

    /// Port of `RotatingKVCache._update_in_place` (`cache.py:469-510`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn update_in_place(
        &mut self,
        new_k: &Array,
        new_v: &Array,
        device: Device,
    ) -> Result<(Array, Array)> {
        let k_shape = new_k.shape();
        let v_shape = new_v.shape();
        let b = k_shape[0];
        let n_kv_heads = k_shape[1];
        let s = k_shape[2];
        let k_head_dim = k_shape[3];
        let v_head_dim = v_shape[3];
        let dtype = new_k.dtype();
        let prev = self.offset;

        let need_grow = match &self.keys {
            None => true,
            Some(k) => prev >= k.shape()[2] && k.shape()[2] < self.max_size,
        };
        if need_grow {
            let new_size = ROTATING_STEP.min(self.max_size - prev);
            let k_alloc_shape = [b, n_kv_heads, new_size, k_head_dim];
            let v_alloc_shape = [b, n_kv_heads, new_size, v_head_dim];
            let new_k_buf = zeros(&k_alloc_shape, dtype, device)?;
            let new_v_buf = zeros(&v_alloc_shape, dtype, device)?;
            if self.keys.is_some() {
                let cur_k = self.keys.as_ref().unwrap();
                let cur_v = self.values.as_ref().unwrap();
                let kc = concatenate(&[cur_k, &new_k_buf], 2, device)?;
                let vc = concatenate(&[cur_v, &new_v_buf], 2, device)?;
                self.keys = Some(kc);
                self.values = Some(vc);
            } else {
                self.keys = Some(new_k_buf);
                self.values = Some(new_v_buf);
            }
            self.idx = prev;
        }

        let buf_len = self.keys.as_ref().unwrap().shape()[2];
        let trim_size = buf_len - self.max_size;
        if trim_size > 0 {
            let kt = self.trim(trim_size, self.keys.as_ref().unwrap(), None, device)?;
            let vt = self.trim(trim_size, self.values.as_ref().unwrap(), None, device)?;
            self.keys = Some(kt);
            self.values = Some(vt);
            self.idx = self.max_size;
        }

        if self.idx == self.max_size {
            self.idx = self.keep;
        }

        let strides = vec![1i32; 4];
        let mut start = vec![0i32; 4];
        start[2] = self.idx;
        let stop_k: Vec<i32> = [b, n_kv_heads, self.idx + s, k_head_dim].into();
        let stop_v: Vec<i32> = [b, n_kv_heads, self.idx + s, v_head_dim].into();
        let kbuf = self.keys.as_ref().unwrap();
        let vbuf = self.values.as_ref().unwrap();
        let k_updated = kbuf.slice_update(new_k, &start, &stop_k, &strides, device)?;
        let v_updated = vbuf.slice_update(new_v, &start, &stop_v, &strides, device)?;
        self.keys = Some(k_updated);
        self.values = Some(v_updated);
        self.offset += s;
        self.idx += s;

        let _ = self.keys.as_ref().unwrap().async_eval();
        let _ = self.values.as_ref().unwrap().async_eval();

        if self.offset < self.max_size {
            let kshape = self.keys.as_ref().unwrap().shape();
            let vshape = self.values.as_ref().unwrap().shape();
            let zero4 = vec![0i32; 4];
            let k_stop: Vec<i32> = [kshape[0], kshape[1], self.offset, kshape[3]].into();
            let v_stop: Vec<i32> = [vshape[0], vshape[1], self.offset, vshape[3]].into();
            let k_full = self
                .keys
                .as_ref()
                .unwrap()
                .slice(&zero4, &k_stop, &strides, device)?;
            let v_full = self
                .values
                .as_ref()
                .unwrap()
                .slice(&zero4, &v_stop, &strides, device)?;
            return Ok((k_full, v_full));
        }
        Ok((
            self.keys.as_ref().unwrap().try_clone()?,
            self.values.as_ref().unwrap().try_clone()?,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests — snapshot/restore round-trip exactness (B1)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "rotating_tests.rs"]
mod tests;
