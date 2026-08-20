// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
// unsafe_code: mlx-rs Array zero-copy view
#![allow(unsafe_code)]

//! Mixed-precision KV buffer types: `MixedTuple` and `MixedKvState`.

use crate::rot_k_msl::rot_k_fwht_quantize_gpu;
use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{quantize, zeros, Array, Device, Dtype};

/// Pre-allocation step size — matches `MixedQuantKVCache.step = 256`.
const STEP: i32 = 256;

/// One set of quantized buffers — codes (U32), scales, biases.
///
/// Mirrors the 3-tuple returned by `mx.quantize(..., mode="affine")`.
#[allow(missing_debug_implementations)]
pub struct MixedTuple {
    pub codes: Array,
    pub scales: Array,
    pub biases: Array,
}

impl MixedTuple {
    /// Resident bytes of this quantized 3-tuple.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            codes,
            scales,
            biases,
        } = self;
        crate::bytes::array_bytes(codes)
            + crate::bytes::array_bytes(scales)
            + crate::bytes::array_bytes(biases)
    }

    /// Slice each of the three arrays along axis=2 to `[..., :off, :]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn slice_seq_to(&self, off: i32, device: Device) -> Result<Self> {
        let s = self.codes.shape();
        let b = s[0];
        let kv_h = s[1];
        let codes_d = s[3];
        let scale_s = self.scales.shape();
        let scales_d = scale_s[3];

        let codes = self.codes.slice(
            &[0, 0, 0, 0],
            &[b, kv_h, off, codes_d],
            &[1, 1, 1, 1],
            device,
        )?;
        let scales = self.scales.slice(
            &[0, 0, 0, 0],
            &[b, kv_h, off, scales_d],
            &[1, 1, 1, 1],
            device,
        )?;
        let biases = self.biases.slice(
            &[0, 0, 0, 0],
            &[b, kv_h, off, scales_d],
            &[1, 1, 1, 1],
            device,
        )?;
        Ok(Self {
            codes,
            scales,
            biases,
        })
    }

    /// In-place slice_update of all three arrays at `[..., prev:off, :]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn write_at(
        &mut self,
        src: &MixedTuple,
        prev: i32,
        off: i32,
        device: Device,
    ) -> Result<()> {
        let s = self.codes.shape();
        let b = s[0];
        let kv_h = s[1];
        let codes_d = s[3];
        let scales_d = self.scales.shape()[3];

        let codes_updated = self.codes.slice_update(
            &src.codes,
            &[0, 0, prev, 0],
            &[b, kv_h, off, codes_d],
            &[1, 1, 1, 1],
            device,
        )?;
        let scales_updated = self.scales.slice_update(
            &src.scales,
            &[0, 0, prev, 0],
            &[b, kv_h, off, scales_d],
            &[1, 1, 1, 1],
            device,
        )?;
        let biases_updated = self.biases.slice_update(
            &src.biases,
            &[0, 0, prev, 0],
            &[b, kv_h, off, scales_d],
            &[1, 1, 1, 1],
            device,
        )?;
        self.codes = codes_updated;
        self.scales = scales_updated;
        self.biases = biases_updated;
        Ok(())
    }

    pub fn try_clone(&self) -> Result<Self> {
        Ok(Self {
            codes: self.codes.try_clone()?,
            scales: self.scales.try_clone()?,
            biases: self.biases.try_clone()?,
        })
    }

    pub fn force_eval(&self) -> Result<()> {
        self.codes.eval()?;
        self.scales.eval()?;
        self.biases.eval()
    }
}

/// Mixed-precision quantized KV cache state.
///
/// Owned by [`KvCache`](super::super::kvcache::KvCache) when the parent is configured
/// with [`KvQuant::Mixed`](super::super::KvQuant::Mixed).
#[allow(missing_debug_implementations)]
pub struct MixedKvState {
    pub k_bits: i32,
    pub v_bits: i32,
    pub k_group_size: i32,
    pub v_group_size: i32,
    pub offset: i32,
    pub keys: Option<MixedTuple>,
    pub values: Option<MixedTuple>,
    /// K-side rotation flag. When `true`, K is rotated by a Hadamard
    /// matrix `R` before `mx.quantize` (stored in the rotated basis, never
    /// inverse-rotated) and the SDPA helper pre-rotates Q by the same `R` so the
    /// rotations cancel. `false` for plain Mixed — the hot path pays nothing.
    pub rotate_k: bool,
    /// The `[D, D]` rotation matrix `R`, built lazily on the first K encode
    /// (head_dim is only known then). `None` until built / when `rotate_k`.
    pub k_rotation: Option<Array>,
}

impl MixedKvState {
    /// Resident bytes held by this state: both quantized 3-tuples plus the
    /// optional per-layer Hadamard rotation matrix.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    #[must_use]
    pub fn byte_size(&self) -> u64 {
        let Self {
            keys,
            values,
            k_rotation,
            // Codec parameters / bookkeeping, not allocations.
            k_bits: _,
            v_bits: _,
            k_group_size: _,
            v_group_size: _,
            offset: _,
            rotate_k: _,
        } = self;
        keys.as_ref().map_or(0, MixedTuple::byte_size)
            + values.as_ref().map_or(0, MixedTuple::byte_size)
            + k_rotation.as_ref().map_or(0, crate::bytes::array_bytes)
    }

    pub fn new(k_bits: i32, v_bits: i32, k_group_size: i32, v_group_size: i32) -> Self {
        Self {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
            offset: 0,
            keys: None,
            values: None,
            rotate_k: false,
            k_rotation: None,
        }
    }

    /// construct a Mixed state whose K side is rotated by a Hadamard
    /// matrix before quantization (RotK codec). K is fixed at 8-bit/group=64.
    pub fn new_rotated(v_bits: i32, v_group_size: i32) -> Self {
        Self {
            k_bits: 8,
            v_bits,
            k_group_size: 64,
            v_group_size,
            offset: 0,
            keys: None,
            values: None,
            rotate_k: true,
            k_rotation: None,
        }
    }

    /// Rotate K by the stored rotation when RotK is active; identity otherwise.
    /// Builds the `[D, D]` Hadamard matrix on first use (`D = keys.head_dim`).
    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    pub fn maybe_rotate_k(&mut self, keys: &Array, device: Device) -> Result<Array> {
        if !self.rotate_k {
            return keys.try_clone();
        }
        if self.k_rotation.is_none() {
            let d = *keys.shape().last().expect("rotate K: empty K shape");
            self.k_rotation = Some(super::super::rot_k::hadamard_rotation(
                d as usize,
                Dtype::F32,
                device,
            )?);
        }
        let r = self.k_rotation.as_ref().expect("rotation built above");
        super::super::rot_k::rotate_last_axis(keys, r, device)
    }

    /// rotate K and quantize in one step (fused FWHT kernel when the caller's
    /// policy selects it).
    pub fn rotate_k_and_quantize(
        &mut self,
        keys: &Array,
        device: Device,
        policy: DispatchPolicy,
    ) -> Result<(Array, Array, Array)> {
        if !self.rotate_k {
            return quantize(keys, self.k_group_size, self.k_bits, device);
        }

        if policy.rot_k_fused {
            let d = *keys
                .shape()
                .last()
                .ok_or_else(|| Error::Mlx("rotate_k_and_quantize: empty K shape".into()))?
                as usize;
            if crate::rot_k_msl::is_supported_d(d) {
                match rot_k_fwht_quantize_gpu(keys, device) {
                    Ok(triple) => {
                        if self.k_rotation.is_none() {
                            self.k_rotation = Some(super::super::rot_k::hadamard_rotation(
                                d,
                                Dtype::F32,
                                device,
                            )?);
                        }
                        return Ok(triple);
                    }
                    Err(e) => {
                        tracing::warn!(
                            reason = %e,
                            "rot_k_fwht_quantize_gpu failed; falling back to v1 matmul path"
                        );
                    }
                }
            }
        }

        let keys_rot = self.maybe_rotate_k(keys, device)?;
        quantize(&keys_rot, self.k_group_size, self.k_bits, device)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.keys = None;
        self.values = None;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        k_bits: i32,
        v_bits: i32,
        k_group_size: i32,
        v_group_size: i32,
        offset: i32,
        keys: Option<MixedTuple>,
        values: Option<MixedTuple>,
        rotate_k: bool,
    ) -> Self {
        Self {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
            offset,
            keys,
            values,
            rotate_k,
            k_rotation: None,
        }
    }

    pub fn try_deep_clone(&self) -> Result<Self> {
        Ok(Self {
            k_bits: self.k_bits,
            v_bits: self.v_bits,
            k_group_size: self.k_group_size,
            v_group_size: self.v_group_size,
            offset: self.offset,
            keys: match &self.keys {
                Some(t) => Some(t.try_clone()?),
                None => None,
            },
            values: match &self.values {
                Some(t) => Some(t.try_clone()?),
                None => None,
            },
            rotate_k: self.rotate_k,
            k_rotation: match &self.k_rotation {
                Some(r) => Some(r.try_clone()?),
                None => None,
            },
        })
    }

    pub fn eval_gpu_state(&self) -> Result<()> {
        if let Some(k) = &self.keys {
            k.force_eval()?;
        }
        if let Some(v) = &self.values {
            v.force_eval()?;
        }
        Ok(())
    }

    fn init_quant(
        b: i32,
        n_kv_heads: i32,
        n_steps: i32,
        dim: i32,
        group_size: i32,
        bits: i32,
        scales_dtype: Dtype,
        device: Device,
    ) -> Result<MixedTuple> {
        let el_per_int = 32 / bits;
        if dim % el_per_int != 0 {
            return Err(Error::Mlx(format!(
                "MixedKvState::init_quant: dim={dim} not divisible by el_per_int={el_per_int} (bits={bits})"
            )));
        }
        if dim % group_size != 0 {
            return Err(Error::Mlx(format!(
                "MixedKvState::init_quant: dim={dim} not divisible by group_size={group_size}"
            )));
        }
        let codes = zeros(
            &[b, n_kv_heads, n_steps, dim / el_per_int],
            Dtype::U32,
            device,
        )?;
        let scales = zeros(
            &[b, n_kv_heads, n_steps, dim / group_size],
            scales_dtype,
            device,
        )?;
        let biases = zeros(
            &[b, n_kv_heads, n_steps, dim / group_size],
            scales_dtype,
            device,
        )?;
        Ok(MixedTuple {
            codes,
            scales,
            biases,
        })
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn bulk_init_from_fp16(
        &mut self,
        keys: &Array,
        values: &Array,
        device: Device,
        policy: DispatchPolicy,
    ) -> Result<()> {
        let k_shape = keys.shape();
        let v_shape = values.shape();
        let b = k_shape[0];
        let n_kv_heads = k_shape[1];
        let num_steps = k_shape[2];
        let k_dim = k_shape[3];
        let v_dim = v_shape[3];
        let scales_dtype = keys.dtype();

        let (k_codes, k_scales, k_biases) = self.rotate_k_and_quantize(keys, device, policy)?;
        let (v_codes, v_scales, v_biases) =
            quantize(values, self.v_group_size, self.v_bits, device)?;

        let el_per_int_k = 32 / self.k_bits;
        let el_per_int_v = 32 / self.v_bits;
        debug_assert_eq!(k_codes.shape()[2], num_steps);
        debug_assert_eq!(k_codes.shape()[3], k_dim / el_per_int_k);
        debug_assert_eq!(v_codes.shape()[3], v_dim / el_per_int_v);

        let _ = (b, n_kv_heads, scales_dtype);

        self.keys = Some(MixedTuple {
            codes: k_codes,
            scales: k_scales,
            biases: k_biases,
        });
        self.values = Some(MixedTuple {
            codes: v_codes,
            scales: v_scales,
            biases: v_biases,
        });
        self.offset = num_steps;
        Ok(())
    }

    pub fn bulk_init_k_from_fp16(
        &mut self,
        keys: &Array,
        device: Device,
        policy: DispatchPolicy,
    ) -> Result<(Array, Array, Array)> {
        self.rotate_k_and_quantize(keys, device, policy)
    }

    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn update_k_and_fetch(
        &mut self,
        new_k: &Array,
        device: Device,
        policy: DispatchPolicy,
    ) -> Result<MixedTuple> {
        let prev = self.offset;
        let k_shape = new_k.shape();
        let b = k_shape[0];
        let n_kv_heads = k_shape[1];
        let num_steps = k_shape[2];
        let k_dim = k_shape[3];
        let scales_dtype = new_k.dtype();

        let need_alloc = match &self.keys {
            None => true,
            Some(k) => prev + num_steps > k.codes.shape()[2],
        };
        if need_alloc {
            let n_increment = ((STEP + num_steps - 1) / STEP) * STEP;
            if let Some(k) = self.keys.take() {
                let cur_seq = k.codes.shape()[2];
                let k_trim = if prev % STEP != 0 && prev < cur_seq {
                    k.slice_seq_to(prev, device)?
                } else {
                    k
                };
                self.keys = Some(expand_quant(&k_trim, b, n_kv_heads, n_increment, device)?);
            } else {
                self.keys = Some(Self::init_quant(
                    b,
                    n_kv_heads,
                    n_increment,
                    k_dim,
                    self.k_group_size,
                    self.k_bits,
                    scales_dtype,
                    device,
                )?);
            }
        }

        self.offset = prev + num_steps;
        let off = self.offset;

        let (k_codes, k_scales, k_biases) = self.rotate_k_and_quantize(new_k, device, policy)?;
        let k_new = MixedTuple {
            codes: k_codes,
            scales: k_scales,
            biases: k_biases,
        };

        let k_buf = self.keys.as_mut().expect("keys initialised above");
        k_buf.write_at(&k_new, prev, off, device)?;

        k_buf.slice_seq_to(off, device)
    }

    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn update_and_fetch(
        &mut self,
        keys: &Array,
        values: &Array,
        device: Device,
        policy: DispatchPolicy,
    ) -> Result<(MixedTuple, MixedTuple)> {
        let prev = self.offset;
        let k_shape = keys.shape();
        let v_shape = values.shape();
        let b = k_shape[0];
        let n_kv_heads = k_shape[1];
        let num_steps = k_shape[2];
        let k_dim = k_shape[3];
        let v_dim = v_shape[3];
        let scales_dtype = keys.dtype();

        let need_alloc = match &self.keys {
            None => true,
            Some(k) => prev + num_steps > k.codes.shape()[2],
        };
        if need_alloc {
            let n_increment = ((STEP + num_steps - 1) / STEP) * STEP;
            if let (Some(k), Some(v)) = (self.keys.take(), self.values.take()) {
                let cur_seq = k.codes.shape()[2];
                let (k_trim, v_trim) = if prev % STEP != 0 && prev < cur_seq {
                    (k.slice_seq_to(prev, device)?, v.slice_seq_to(prev, device)?)
                } else {
                    (k, v)
                };
                self.keys = Some(expand_quant(&k_trim, b, n_kv_heads, n_increment, device)?);
                self.values = Some(expand_quant(&v_trim, b, n_kv_heads, n_increment, device)?);
            } else {
                self.keys = Some(Self::init_quant(
                    b,
                    n_kv_heads,
                    n_increment,
                    k_dim,
                    self.k_group_size,
                    self.k_bits,
                    scales_dtype,
                    device,
                )?);
                self.values = Some(Self::init_quant(
                    b,
                    n_kv_heads,
                    n_increment,
                    v_dim,
                    self.v_group_size,
                    self.v_bits,
                    scales_dtype,
                    device,
                )?);
            }
        }

        self.offset = prev + num_steps;
        let off = self.offset;

        let (k_codes, k_scales, k_biases) = self.rotate_k_and_quantize(keys, device, policy)?;
        let (v_codes, v_scales, v_biases) =
            quantize(values, self.v_group_size, self.v_bits, device)?;
        let k_new = MixedTuple {
            codes: k_codes,
            scales: k_scales,
            biases: k_biases,
        };
        let v_new = MixedTuple {
            codes: v_codes,
            scales: v_scales,
            biases: v_biases,
        };

        let k_buf = self.keys.as_mut().expect("keys initialised above");
        let v_buf = self.values.as_mut().expect("values initialised above");
        k_buf.write_at(&k_new, prev, off, device)?;
        v_buf.write_at(&v_new, prev, off, device)?;

        let k_view = k_buf.slice_seq_to(off, device)?;
        let v_view = v_buf.slice_seq_to(off, device)?;
        Ok((k_view, v_view))
    }
}

/// `_expand_quant`: concatenate `new_steps` zero rows along axis=2.
///
/// Byte-for-byte port of mixed_quant_cache.py:59-65. Each of (codes, scales,
/// biases) is appended with `new_steps` zero rows along axis=2; the resulting
/// total length is `src.shape[2] + new_steps`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn expand_quant(
    src: &MixedTuple,
    b: i32,
    n_kv_heads: i32,
    new_steps: i32,
    device: Device,
) -> Result<MixedTuple> {
    use rmlx_mlx::concatenate;

    let scales_dtype = src.scales.dtype();
    let codes_d = src.codes.shape()[3];
    let scales_d = src.scales.shape()[3];

    let codes_zeros = zeros(&[b, n_kv_heads, new_steps, codes_d], Dtype::U32, device)?;
    let scales_zeros = zeros(&[b, n_kv_heads, new_steps, scales_d], scales_dtype, device)?;
    let biases_zeros = zeros(&[b, n_kv_heads, new_steps, scales_d], scales_dtype, device)?;

    let codes = concatenate(&[&src.codes, &codes_zeros], 2, device)?;
    let scales = concatenate(&[&src.scales, &scales_zeros], 2, device)?;
    let biases = concatenate(&[&src.biases, &biases_zeros], 2, device)?;
    Ok(MixedTuple {
        codes,
        scales,
        biases,
    })
}
