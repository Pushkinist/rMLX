//! The weight file of a Whisper snapshot: `weights.safetensors` when the
//! directory has it, else `weights.npz`. Both give the same [`WeightMap`]: the
//! tensor names, shapes and dtypes are those of the file, except that int64
//! becomes int32 and a boolean or an int8 becomes uint8.
//!
//! An error names the file and never its directory: the server sends it to
//! the client.

use std::path::Path;

use rmlx_loader::{release_mapped, SafetensorsDtype, ShardHandle};
use rmlx_mlx::{dtype_from_safetensors, Array, Dtype};
use tracing::{debug, error};

use super::{load_npz, WhisperError};
use crate::npz::{int64_to_i32_bytes, WeightMap};

const SAFETENSORS_FILE: &str = "weights.safetensors";
const NPZ_FILE: &str = "weights.npz";

/// Load the weights of the snapshot in `dir`.
pub(super) fn load_weights(dir: &Path) -> Result<WeightMap, WhisperError> {
    let file = [SAFETENSORS_FILE, NPZ_FILE]
        .into_iter()
        .find(|file| std::fs::symlink_metadata(dir.join(file)).is_ok())
        .ok_or_else(|| {
            WhisperError::WeightFile(format!(
                "neither {SAFETENSORS_FILE} nor {NPZ_FILE} was found"
            ))
        })?;
    let bytes = std::fs::metadata(dir.join(file)).map_or(0, |m| m.len());
    debug!(stage = "weights", file, bytes, "loading Whisper weights");
    if file == SAFETENSORS_FILE {
        load_safetensors(dir)
    } else {
        load_npz(dir.join(file))
    }
}

fn load_safetensors(dir: &Path) -> Result<WeightMap, WhisperError> {
    let shard = ShardHandle::open(dir, SAFETENSORS_FILE).map_err(|e| {
        error!(error = %e, "cannot open the Whisper weight file");
        WhisperError::WeightFile(format!("{SAFETENSORS_FILE}: cannot be opened"))
    })?;
    let tensors = shard
        .safetensors()
        .map_err(|e| WhisperError::WeightFile(e.to_string()))?;
    tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            let array =
                array_of(tensor.dtype(), tensor.shape(), tensor.data()).map_err(|cause| {
                    WhisperError::WeightFile(format!("{SAFETENSORS_FILE}: tensor {name}: {cause}"))
                })?;
            release_mapped(tensor.data());
            Ok((name, array))
        })
        .collect()
}

/// The array of a tensor. As in `weights.npz`, int64 becomes int32, and a
/// boolean or an int8 becomes uint8.
fn array_of(dtype: SafetensorsDtype, shape: &[usize], bytes: &[u8]) -> Result<Array, String> {
    let dims = shape
        .iter()
        .map(|&dim| i32::try_from(dim))
        .collect::<Result<Vec<i32>, _>>()
        .map_err(|_| "a dimension does not fit in int32".to_owned())?;
    let (dtype, bytes) = if dtype == SafetensorsDtype::I64 {
        let narrowed = int64_to_i32_bytes(bytes)
            .map_err(|value| format!("int64 value {value} does not fit in int32"))?;
        (Dtype::I32, narrowed)
    } else if dtype == SafetensorsDtype::BOOL || dtype == SafetensorsDtype::I8 {
        (Dtype::U8, bytes.to_vec())
    } else {
        let supported =
            dtype_from_safetensors(dtype).map_err(|_| format!("dtype {dtype} is not supported"))?;
        (supported, bytes.to_vec())
    };
    Array::from_bytes(&bytes, &dims, dtype).map_err(|e| e.to_string())
}

#[cfg(test)]
#[path = "weight_file_tests.rs"]
mod weight_file_tests;
