//! Safetensors loader for `mlx-community` snapshots.
//!
//! Handles: shard index, config.json, sibling resolver (`<n>.weight` + `.scales` + `.biases`).
//! ParoQuant rotation-tensor recognition: `<n>.pairs` + `.theta` + `.channel_scales`.

#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        // disallowed_methods is a separate lint from unwrap_used;
        // test code (bucket-B) is already exempted for unwrap_used, extend here.
        clippy::disallowed_methods,
    )
)]

pub mod calibration;
pub mod calibration_writer;
pub mod config;
pub mod head_budgets;
pub mod model_size;
pub mod shards;
pub mod tensors;

pub use calibration::{
    bytes_to_f32, calibrate_model, detect_kv_weight_pattern, f16_to_f32, layer_key_from_pattern,
    top_k_norm_indices,
};
pub use calibration_writer::{
    discover_kv_calibration, outlier_count_for, read_kv_calibration, recipe_to_internal,
    write_kv_calibration, CalibrationMeta, CodebookOverride, KvCalibration, LayerCalib,
};
pub use config::{load_config, ModelConfig, ParoQuantConfig, QuantConfig, TextConfig};
pub use head_budgets::{load_head_budgets, write_head_budgets, HeadBudgetCalibration, HeadBudgets};
pub use model_size::estimate_params_billions;
pub use shards::{count_tensors_per_shard, load_shard_index, ShardHandle, ShardIndex, ShardSet};
pub use tensors::{
    resolve, resolve_paro, try_exact_then_suffix, view, view_discriminated, ParoQuantParams,
    ParoQuantState, ResolvedTensor, TensorKind, TensorLookup, TensorView,
};
