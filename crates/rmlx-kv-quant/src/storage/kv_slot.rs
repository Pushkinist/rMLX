//! `KvStore` and `KvSlot`: the per-store operations that the `KvStorage` sites
//! reach through [`KvStorage::view`](super::KvStorage::view) and
//! [`KvStorage::view_mut`](super::KvStorage::view_mut).
//!
//! `KvStore` is implemented once per store type (the width stores once for all
//! widths, a boxed store through its box). Every `Option<Store>` slot gets its
//! `KvSlot` from the one blanket impl. Only `MixedKvState` has its own `KvSlot`
//! impl, because its payload is not an `Option`.
//!
//! None of these calls is on the per-token decode math: they serve the byte
//! total, the graph flush, the SSD spill decision, the test probes, the cache
//! reset, the truncation, the payload clear and the deep clone.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use super::{
    QuantIsoK, QuantIsoV, QuantK, QuantKTurbo, QuantPlanarK, QuantPlanarV, QuantRotorK,
    QuantRotorV, QuantV,
};
use crate::mixed_quant::MixedKvState;
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};

/// The operations every KV store type has.
pub(crate) trait KvStore {
    /// True when an empty slot of this store, as a layer's first slot, leaves
    /// the layer with only its geometry to persist. The paged writer
    /// serialises its own empty state, so the paged stores say `false`.
    const EMPTY_IS_GEOMETRY_ONLY: bool;
    /// Bytes the store holds. GPU buffers count their full allocation.
    fn bytes(&self) -> u64;
    /// Evaluate the pending MLX graph of the GPU arrays the store holds.
    fn eval(&self) -> Result<()>;
    /// Dequantize the filled sequence to flat f32. `None` when the store has
    /// no CPU dequant.
    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>>;
    /// Empty the sequence for the next request. The flat and block stores keep
    /// their GPU buffers (`truncate_to(0)`); the iso, rotor and paged stores
    /// run their own `reset`, which keeps only their layer-static tables.
    fn reset_sequence(&mut self);
    /// Cut the sequence to `n >= 0` positions, with the store's own clamping.
    fn truncate_to(&mut self, n: i32);
    /// An independent copy of the store. The paged stores refuse when they
    /// hold pages.
    fn try_deep_clone(&self) -> Result<Self>
    where
        Self: Sized;
}

/// One store slot of a `KvStorage` variant, filled or empty.
pub(crate) trait KvSlot {
    /// True when the slot holds a payload.
    fn is_filled(&self) -> bool;
    /// True when the layer that holds this slot as its first slot has only
    /// its geometry to persist.
    fn geometry_only(&self) -> bool;
    /// Bytes the slot holds; an empty slot holds none.
    fn resident_bytes(&self) -> u64;
    /// Evaluate the pending MLX graph of the slot's GPU arrays.
    fn eval(&self) -> Result<()>;
    /// `None` when the slot has no CPU-dequantizable store. `Some(Err(..))`
    /// when the store exists and its dequant refused.
    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>>;
    /// Empty the sequence and keep the payload's allocation.
    fn reset(&mut self);
    /// Cut the sequence to `n >= 0` positions.
    fn truncate_to(&mut self, n: i32);
    /// Drop the payload.
    fn clear(&mut self);
    /// An independent copy of the slot, filled or empty.
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized;
}

impl<T: KvStore> KvSlot for Option<T> {
    fn is_filled(&self) -> bool {
        self.is_some()
    }

    fn geometry_only(&self) -> bool {
        self.is_none() && T::EMPTY_IS_GEOMETRY_ONLY
    }

    fn resident_bytes(&self) -> u64 {
        self.as_ref().map_or(0, KvStore::bytes)
    }

    fn eval(&self) -> Result<()> {
        self.as_ref().map_or(Ok(()), KvStore::eval)
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        self.as_ref().and_then(|store| store.dequant_f32(device))
    }

    fn reset(&mut self) {
        if let Some(store) = self {
            store.reset_sequence();
        }
    }

    fn truncate_to(&mut self, n: i32) {
        if let Some(store) = self {
            KvStore::truncate_to(store, n);
        }
    }

    fn clear(&mut self) {
        *self = None;
    }

    fn try_clone(&self) -> Result<Self> {
        self.as_ref().map(KvStore::try_deep_clone).transpose()
    }
}

/// `Mixed` fills its capacity buffers in place: `offset` is its fill marker.
impl KvSlot for MixedKvState {
    fn is_filled(&self) -> bool {
        self.offset > 0
    }

    fn geometry_only(&self) -> bool {
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        self.eval_gpu_state()
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }

    fn reset(&mut self) {
        MixedKvState::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        MixedKvState::truncate_to(self, n);
    }

    /// The payload is not an `Option`, so clearing it is its `reset`.
    fn clear(&mut self) {
        MixedKvState::reset(self);
    }

    fn try_clone(&self) -> Result<Self> {
        self.try_deep_clone()
    }
}

impl<T: KvStore> KvStore for Box<T> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = T::EMPTY_IS_GEOMETRY_ONLY;

    fn bytes(&self) -> u64 {
        T::bytes(self)
    }

    fn eval(&self) -> Result<()> {
        T::eval(self)
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        T::dequant_f32(self, device)
    }

    fn reset_sequence(&mut self) {
        T::reset_sequence(self);
    }

    fn truncate_to(&mut self, n: i32) {
        T::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        T::try_deep_clone(self).map(Box::new)
    }
}

// The paged pages are evaluated by the `slice_update` chain of `write_page`,
// so `eval` has nothing to flush, and no probe dequantizes them.

impl KvStore for PagedKStorage {
    const EMPTY_IS_GEOMETRY_ONLY: bool = false;

    fn bytes(&self) -> u64 {
        self.resident_bytes()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl KvStore for PagedVStorage {
    const EMPTY_IS_GEOMETRY_ONLY: bool = false;

    fn bytes(&self) -> u64 {
        self.resident_bytes()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl KvStore for PagedPlanarVStorage {
    const EMPTY_IS_GEOMETRY_ONLY: bool = false;

    fn bytes(&self) -> u64 {
        self.resident_bytes()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

fn eval_arrays(arrays: &[&Option<Array>]) -> Result<()> {
    for array in arrays.iter().copied().flatten() {
        array.eval()?;
    }
    Ok(())
}

impl KvStore for QuantK {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        Some(
            self.dequantize_choice(device, Dtype::F32)
                .map(|(flat, _)| flat),
        )
    }

    fn reset_sequence(&mut self) {
        Self::truncate_to(self, 0);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl KvStore for QuantV {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        Some(
            self.dequantize_choice(device, Dtype::F32)
                .map(|(flat, _)| flat),
        )
    }

    fn reset_sequence(&mut self) {
        Self::truncate_to(self, 0);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl<const BITS: u8> KvStore for QuantKTurbo<BITS> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        Some(
            self.dequantize_choice(device, Dtype::F32)
                .map(|(flat, _)| flat),
        )
    }

    fn reset_sequence(&mut self) {
        Self::truncate_to(self, 0);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl KvStore for QuantPlanarK {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[
            &self.gpu_codes_buf,
            &self.gpu_scales_buf,
            &self.gpu_rotations_buf,
        ])
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        Some(
            self.dequantize_choice(device, Dtype::F32)
                .map(|(flat, _)| flat),
        )
    }

    fn reset_sequence(&mut self) {
        Self::truncate_to(self, 0);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl KvStore for QuantPlanarV {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[
            &self.gpu_codes_buf,
            &self.gpu_scales_buf,
            &self.gpu_rotations_buf,
        ])
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        Some(
            self.dequantize_choice(device, Dtype::F32)
                .map(|(flat, _)| flat),
        )
    }

    fn reset_sequence(&mut self) {
        Self::truncate_to(self, 0);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

// The iso and rotor stores dequantize on the CPU from their blocks. `eval` does
// not flush their GPU buffers (the iso V buffers, the rotor K ring): MLX
// evaluates them when a kernel reads them.

impl<const BITS: u8> KvStore for QuantIsoK<BITS> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        Some(self.dequant())
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl<const BITS: u8> KvStore for QuantIsoV<BITS> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        Some(self.dequant())
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl<const BITS: u8> KvStore for QuantRotorK<BITS> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        Some(self.dequant())
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}

impl<const BITS: u8> KvStore for QuantRotorV<BITS> {
    const EMPTY_IS_GEOMETRY_ONLY: bool = true;

    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        Some(self.dequant())
    }

    fn reset_sequence(&mut self) {
        Self::reset(self);
    }

    fn truncate_to(&mut self, n: i32) {
        Self::truncate_to(self, n);
    }

    fn try_deep_clone(&self) -> Result<Self> {
        Self::try_deep_clone(self)
    }
}
