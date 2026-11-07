//! Coverage for the two parses `build.rs` does: the MLX version and the
//! mlx-c C API.
//!
//! The helpers are `include!`d from the same file the build script includes: a
//! build script cannot be imported. The first parse decides what the runtime
//! version-skew warning compares the loaded library against; the second decides
//! which SDPA and `gather_qmm` argument lists are compiled.
//!
//! The MLX / mlx-c **pin** is not checked in a build script and is not covered
//! here — see `crates/rmlx-mlx/src/pin_tests.rs`.

include!("../build_support.rs");

#[test]
fn mlx_version_comes_from_the_header_macros() {
    let header = "#pragma once\n\
                  #define MLX_VERSION_MAJOR 0\n\
                  #define MLX_VERSION_MINOR 31\n\
                  #define MLX_VERSION_PATCH 2\n";
    assert_eq!(read_mlx_version(header), "0.31.2");
}

#[test]
fn mlx_version_degrades_to_unknown() {
    // An unreadable header means "cannot verify", not "mismatch" — the skew
    // warning keys off this string to stay quiet rather than warn wrongly.
    assert_eq!(read_mlx_version(""), "unknown");
    assert_eq!(read_mlx_version("#define MLX_VERSION_MAJOR 0\n"), "unknown");
    assert_eq!(
        read_mlx_version("#define MLX_VERSION_MAJOR 0\n#define MLX_VERSION_MINOR 31\n"),
        "unknown"
    );
}

/// One difference between the C APIs, in the bindings as bindgen writes them.
#[derive(Clone, Copy, PartialEq)]
enum Feature {
    /// `force_fused` in front of the SDPA stream.
    ForceFused,
    /// The 0.7 marker function.
    Marker0_7,
    /// `global_scale` in front of the `gather_qmm` `sorted_indices`.
    GlobalScale,
    /// The 0.8 marker function.
    Marker0_8,
}

use Feature::{ForceFused, GlobalScale, Marker0_7, Marker0_8};

const V0_6: &[Feature] = &[];
const V0_7: &[Feature] = &[ForceFused, Marker0_7];
const V0_8: &[Feature] = &[ForceFused, Marker0_7, GlobalScale, Marker0_8];

fn bindings(features: &[Feature]) -> String {
    let text = |feature: Feature, text: &'static str| {
        if features.contains(&feature) {
            text
        } else {
            ""
        }
    };
    let marker_0_7 = text(
        Marker0_7,
        "    pub fn mlx_compile_cache_new() -> mlx_compile_cache;\n",
    );
    let marker_0_8 = text(
        Marker0_8,
        "    pub fn mlx_matrix_transpose(res: *mut mlx_array, a: mlx_array, s: mlx_stream) \
         -> ::std::os::raw::c_int;\n",
    );
    let force_fused = text(ForceFused, "        force_fused: bool,\n");
    let global_scale = text(GlobalScale, "        global_scale: mlx_array,\n");
    format!(
        "extern \"C\" {{\n{marker_0_7}{marker_0_8}    pub fn mlx_fast_scaled_dot_product_attention(\n        \
         res: *mut mlx_array,\n        queries: mlx_array,\n        keys: mlx_array,\n        \
         values: mlx_array,\n        scale: f32,\n        \
         mask_mode: *const ::std::os::raw::c_char,\n        mask_arr: mlx_array,\n        \
         sinks: mlx_array,\n{force_fused}        s: mlx_stream,\n    ) -> ::std::os::raw::c_int;\n    \
         pub fn mlx_gather_qmm(\n        res: *mut mlx_array,\n        x: mlx_array,\n        \
         w: mlx_array,\n        scales: mlx_array,\n        biases: mlx_array,\n        \
         lhs_indices: mlx_array,\n        rhs_indices: mlx_array,\n        transpose: bool,\n        \
         group_size: mlx_optional_int,\n        bits: mlx_optional_int,\n        \
         mode: *const ::std::os::raw::c_char,\n{global_scale}        sorted_indices: bool,\n        \
         s: mlx_stream,\n    ) -> ::std::os::raw::c_int;\n}}\n"
    )
}

#[test]
fn the_three_mlx_c_c_apis_are_told_apart_by_two_arities_and_two_markers() {
    assert_eq!(mlxc_c_api(&bindings(V0_6)), Ok(MlxcApi::V0_6));
    assert_eq!(mlxc_c_api(&bindings(V0_7)), Ok(MlxcApi::V0_7));
    assert_eq!(mlxc_c_api(&bindings(V0_8)), Ok(MlxcApi::V0_8));
}

/// A combination no known C API has stops the build: the runtime probe reads
/// the markers, so it is only sound when the markers and the arities agree.
/// The refusal names all three C APIs.
#[test]
fn an_unknown_mlx_c_c_api_is_an_error() {
    let mixed: [&[Feature]; 7] = [
        // SDPA 10 without the 0.7 marker, and the reverse.
        &[ForceFused],
        &[Marker0_7],
        // gather_qmm 14 without the 0.8 marker, and the reverse.
        &[ForceFused, Marker0_7, GlobalScale],
        &[ForceFused, Marker0_7, Marker0_8],
        // The 0.8 marker on a 0.6 shape.
        &[Marker0_8],
        // A 14-parameter gather_qmm beside a 0.6 SDPA, with and without the marker.
        &[GlobalScale],
        &[GlobalScale, Marker0_8],
    ];
    for features in mixed {
        let result = mlxc_c_api(&bindings(features));
        assert!(
            result.as_ref().is_err_and(|e| {
                [MLXC_0_7_MARKER, MLXC_0_8_MARKER, "0.6", "0.7", "0.8"]
                    .iter()
                    .all(|needle| e.contains(needle))
            }),
            "an unknown C API must not pass, and the refusal must name the three: {result:?}"
        );
    }
    assert!(mlxc_c_api("pub fn mlx_array_new() -> mlx_array;").is_err());
    let no_qmm = bindings(V0_7).replace("mlx_gather_qmm", "mlx_gather_other");
    assert!(mlxc_c_api(&no_qmm).is_err());
}
