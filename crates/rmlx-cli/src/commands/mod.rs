//! CLI command implementations.

pub(crate) mod baseline;
pub(crate) mod calibration_softmax;
pub(crate) mod eval;
pub(crate) mod healthcheck;
pub(crate) mod info;
pub(crate) mod kv_calibrate;
pub(crate) mod metrics;
pub(crate) mod parse;
pub(crate) mod preset_table;
pub(crate) mod profile;
pub(crate) mod serve;
pub(crate) mod transcribe;

pub(crate) use baseline::run_baseline;
pub(crate) use eval::run_ppl;
pub(crate) use healthcheck::run_healthcheck;
pub(crate) use info::run_info;
pub(crate) use kv_calibrate::run_kv_calibrate;
pub(crate) use parse::{
    acquire_claim_for_device, build_cache_type_spec, parse_device, parse_kv_bits_combo,
    parse_kv_bits_fractional, parse_kv_preset, parse_kv_quant, parse_max_ctx,
    parse_max_prompt_tokens, resolve_model_flags, resolve_preset_arg,
};
pub(crate) use profile::run_profile_list;
pub(crate) use serve::run_serve;
