//! `KvStore` and `KvSlot`: the per-store operations that the read-only
//! `KvStorage` sites reach through [`KvStorage::view`](super::KvStorage::view).
//!
//! `KvStore` is implemented once per store type (the width stores once for all
//! widths). Every `Option<Store>` slot gets its `KvSlot` from the one blanket
//! impl. Only `MixedKvState` and the three paged slot types have their own
//! `KvSlot` impl, because an empty slot of theirs is not geometry-only.
//!
//! None of these calls is on the per-token decode math: they serve the byte
//! total, the graph flush, the SSD spill decision and the test probes.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use super::{
    QuantIsoK, QuantIsoV, QuantK, QuantKTurbo, QuantPlanarK, QuantPlanarV, QuantRotorK,
    QuantRotorV, QuantV,
};
use crate::mixed_quant::MixedKvState;
use crate::paged::{PagedKStorage, PagedPlanarVStorage, PagedVStorage};

/// The operations every packed KV store type has.
pub(crate) trait KvStore {
    /// Bytes the store holds. GPU buffers count their full allocation.
    fn bytes(&self) -> u64;
    /// Evaluate the pending MLX graph of the GPU arrays the store holds.
    fn eval(&self) -> Result<()>;
    /// Dequantize the filled sequence to flat f32.
    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>>;
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
}

impl<T: KvStore> KvSlot for Option<T> {
    fn is_filled(&self) -> bool {
        self.is_some()
    }

    fn geometry_only(&self) -> bool {
        self.is_none()
    }

    fn resident_bytes(&self) -> u64 {
        self.as_ref().map_or(0, KvStore::bytes)
    }

    fn eval(&self) -> Result<()> {
        self.as_ref().map_or(Ok(()), KvStore::eval)
    }

    fn dequant_f32(&self, device: Device) -> Option<Result<Vec<f32>>> {
        self.as_ref().map(|store| store.dequant_f32(device))
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
}

// The paged pages are evaluated by the `slice_update` chain of `write_page`,
// so `eval` has nothing to flush.

impl KvSlot for Option<PagedKStorage> {
    fn is_filled(&self) -> bool {
        self.is_some()
    }

    fn geometry_only(&self) -> bool {
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.as_ref().map_or(0, PagedKStorage::resident_bytes)
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }
}

impl KvSlot for Option<Box<PagedVStorage>> {
    fn is_filled(&self) -> bool {
        self.is_some()
    }

    fn geometry_only(&self) -> bool {
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.as_ref().map_or(0, |store| store.resident_bytes())
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }
}

impl KvSlot for Option<Box<PagedPlanarVStorage>> {
    fn is_filled(&self) -> bool {
        self.is_some()
    }

    fn geometry_only(&self) -> bool {
        false
    }

    fn resident_bytes(&self) -> u64 {
        self.as_ref().map_or(0, |store| store.resident_bytes())
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Option<Result<Vec<f32>>> {
        None
    }
}

fn eval_arrays(arrays: &[&Option<Array>]) -> Result<()> {
    for array in arrays.iter().copied().flatten() {
        array.eval()?;
    }
    Ok(())
}

impl KvStore for QuantK {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>> {
        self.dequantize_choice(device, Dtype::F32)
            .map(|(flat, _)| flat)
    }
}

impl KvStore for QuantV {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>> {
        self.dequantize_choice(device, Dtype::F32)
            .map(|(flat, _)| flat)
    }
}

impl<const BITS: u8> KvStore for QuantKTurbo<BITS> {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        eval_arrays(&[&self.gpu_codes_buf, &self.gpu_scales_buf])
    }

    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>> {
        self.dequantize_choice(device, Dtype::F32)
            .map(|(flat, _)| flat)
    }
}

impl KvStore for QuantPlanarK {
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

    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>> {
        self.dequantize_choice(device, Dtype::F32)
            .map(|(flat, _)| flat)
    }
}

impl KvStore for QuantPlanarV {
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

    fn dequant_f32(&self, device: Device) -> Result<Vec<f32>> {
        self.dequantize_choice(device, Dtype::F32)
            .map(|(flat, _)| flat)
    }
}

// The iso and rotor stores dequantize on the CPU from their blocks. `eval` does
// not flush their GPU buffers (the iso V buffers, the rotor K ring): MLX
// evaluates them when a kernel reads them.

impl<const BITS: u8> KvStore for QuantIsoK<BITS> {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Result<Vec<f32>> {
        self.dequant()
    }
}

impl<const BITS: u8> KvStore for QuantIsoV<BITS> {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Result<Vec<f32>> {
        self.dequant()
    }
}

impl<const BITS: u8> KvStore for QuantRotorK<BITS> {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Result<Vec<f32>> {
        self.dequant()
    }
}

impl<const BITS: u8> KvStore for QuantRotorV<BITS> {
    fn bytes(&self) -> u64 {
        self.byte_size()
    }

    fn eval(&self) -> Result<()> {
        Ok(())
    }

    fn dequant_f32(&self, _device: Device) -> Result<Vec<f32>> {
        self.dequant()
    }
}
