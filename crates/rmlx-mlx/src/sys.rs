//! Raw bindgen-generated FFI bindings for mlx-c.
//!
//! This module is the lowest layer of the rmlx-mlx crate. It `include!`s the
//! bindgen output from `build.rs` inside a suppression wrapper (`mod ffi`) and
//! re-exports all symbols crate-internally via `pub(crate) use ffi::*`.
//!
//! # Public API
//!
//! None. The `ffi` module's symbols are re-exported as `pub(crate)` only.
//! All external callers must go through the safe wrappers in [`super::ops`]
//! and [`super::fast_ops`].
//!
//! # Invariants
//!
//! - **Do not expose these types in any public API.** All callers must go
//!   through the safe wrappers in [`super::ops`] and [`super::fast_ops`].
//! - The `ffi` mod suppresses every clippy and rustdoc lint; that is
//!   intentional — the generated file is not under our control.
//! - **`mod ffi` is private, deliberately.** `pub(crate) use ffi::*` below
//!   re-exports every symbol as `sys::mlx_*`, so that is the single spelling
//!   for an mlx-c call anywhere in the crate. Making the module itself
//!   `pub(crate)` would also admit `sys::ffi::mlx_*`, a second spelling of the
//!   same function that the `check-eval-lock` gate's `sys::mlx_*` anchor does
//!   not match — an unguarded evaluation could then pass the gate.
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    unreachable_pub,
    missing_debug_implementations,
    clippy::all,
    clippy::pedantic
)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub(crate) use ffi::*;
