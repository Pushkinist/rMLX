//! Weight + KV-cache quantization kernels.
//!
//! Stage 1: `affine`, `mxfp8`, `mxfp4`, `nvfp4` decode.
//! Stage 2: TurboQuant (port from TheTom MSL kernels).
//! Stage 3: PlanarQuant (port from johndpope / write MSL).
//!
//! Module per family. Empty stubs in Stage 0.

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

pub mod affine;
pub mod bf16;
pub mod fp4;
pub mod fp8;
pub mod mxfp;

// TurboQuant and PlanarQuant codecs were extracted to the
// `rmlx-kv-quant` crate (KV-side codecs no longer live alongside the
// weight-quant families). The original `rmlx_quant::turboquant::*` /
// `rmlx_quant::planarquant::*` re-export was attempted here but would
// create a workspace dependency cycle through `rmlx-loader` (loader →
// rmlx-quant, rmlx-mlx → loader, rmlx-kv-quant → rmlx-mlx). The two
// remaining external call sites (`rmlx_models::kv_cache::block_io.rs` and
// its test sibling — both SSD modules) have been updated to import directly from
// `rmlx_kv_quant::{turboquant,planarquant}::*`.

pub use fp4::e2m1_decode;
pub use fp8::{e4m3_decode, e8m0_decode, ue4m3_decode};
pub use mxfp::{dequant_to_f32, dequant_vec, MxFamily, MxParams};
