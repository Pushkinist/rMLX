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
use rmlx_mlx::{concatenate, zeros, Array, Device};

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
    pub(super) stream: Option<Device>,
}

/// Internal state for a rotating bf16 KV cache.
#[allow(missing_debug_implementations)]
pub(super) struct RotatingState {
    pub(super) keys: Option<Array>,
    pub(super) values: Option<Array>,
    pub(super) offset: i32,
    pub(super) max_size: i32,
    pub(super) keep: i32,
    pub(super) idx: i32,
    /// The stream the buffers above were last written on. A rollback slices
    /// them and runs on the same one, which is why the ring records it instead
    /// of every `truncate_to` caller carrying a device it otherwise never uses.
    /// `None` until the first write, when there is nothing to slice.
    pub(super) stream: Option<Device>,
}

/// The first `keep` positions of a `[B, kv_h, S, D]` ring buffer.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn slice_leading(v: &Array, keep: i32, device: Device) -> Result<Array> {
    let shape = v.shape();
    if shape.len() != 4 {
        return Err(Error::Mlx(format!(
            "RotatingKvCache::slice_leading expected 4-D tensor, got ndim={}",
            shape.len()
        )));
    }
    v.slice(
        &[0, 0, 0, 0],
        &[shape[0], shape[1], keep, shape[3]],
        &[1i32; 4],
        device,
    )
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
            stream: _,
        } = self;
        crate::bytes::opt_filled_seq_bytes(keys.as_ref(), filled)
            + crate::bytes::opt_filled_seq_bytes(values.as_ref(), filled)
    }

    pub(super) fn new(max_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            max_size,
            keep: 0,
            idx: 0,
            stream: None,
        }
    }

    pub(super) fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.idx = 0;
        self.stream = None;
    }

    /// Whether [`Self::trim`] can roll this ring back by `n` positions without
    /// losing a position the window would still need.
    ///
    /// Two regimes are rollable, and they are the two the ring can be left in:
    ///
    /// - **Before the first wrap** (`offset < max_size`), every position ever
    ///   written is still in the buffer at its own slot, so any rollback is a
    ///   move of the write pointer. This is mlx-lm's `is_trimmable`
    ///   (`cache.py:542-543`).
    /// - **After a wrap, while the buffer is in temporal order** — which is the
    ///   state [`Self::update_concat`] leaves it in, holding `max_size + s - 1`
    ///   positions for a write of `s`. Those are the newest positions in order,
    ///   so dropping the last `n` of them is lossless exactly while what is left
    ///   still covers the window the rolled-back offset needs. A block-verify
    ///   write of `s` positions can therefore always be rolled back over its own
    ///   rejected tail, which is at most `s - 1` long.
    ///
    /// A ring left in rotated order by a single-token write
    /// (`update_in_place` past the wrap) is **not** rollable: the newest slots
    /// hold the positions they overwrote, and those are gone.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn can_trim(&self, n: i32) -> bool {
        if n <= 0 {
            return true;
        }
        if n > self.offset {
            return false;
        }
        if self.offset < self.max_size {
            return true;
        }
        let (Some(keys), Some(_)) = (self.keys.as_ref(), self.stream) else {
            return false;
        };
        let len = keys.shape()[2];
        self.idx == len && len - n >= (self.offset - n).min(self.max_size)
    }

    /// Roll the ring back by `n` positions. Returns `false` and leaves the ring
    /// untouched when that cannot be done losslessly — see [`Self::can_trim`].
    ///
    /// Not to be confused with [`Self::trim`], the mlx-lm `_trim` port that
    /// drops the ring's *oldest* positions to make room for a write.
    ///
    /// mlx-lm's `trim_prompt_cache` silently returns 0 for the case this
    /// refuses (`cache.py:109-111`). A silent no-op is safe for its caller,
    /// which re-prefills what it could not roll back; it is not safe for a
    /// speculative round loop, where the other layers do roll back and the ring
    /// is left holding the rejected drafts at an offset the rest of the stack
    /// has left behind.
    pub(super) fn roll_back(&mut self, n: i32) -> Result<bool> {
        if !self.can_trim(n) {
            return Ok(false);
        }
        if n <= 0 {
            return Ok(true);
        }
        if self.offset >= self.max_size {
            let Some(device) = self.stream else {
                return Ok(false);
            };
            // Temporal order: the newest `n` positions are the last `n` slots.
            let keep = self.idx - n;
            self.keys = match &self.keys {
                Some(k) => Some(slice_leading(k, keep, device)?),
                None => None,
            };
            self.values = match &self.values {
                Some(v) => Some(slice_leading(v, keep, device)?),
                None => None,
            };
            self.idx = keep;
        } else {
            self.idx -= n;
        }
        self.offset -= n;
        Ok(true)
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
            stream: self.stream,
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
        self.stream = snap.stream;
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
        self.stream = Some(device);
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
