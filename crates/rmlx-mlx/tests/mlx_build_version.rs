//! Coverage for the two parses `build.rs` does: the MLX version and the
//! mlx-c C API.
//!
//! The helpers are `include!`d from the same file the build script includes: a
//! build script cannot be imported. The first parse decides what the runtime
//! version-skew warning compares the loaded library against; the second decides
//! which SDPA argument list is compiled.
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

/// The SDPA declaration as bindgen writes it, with `extra` in front of the
/// stream parameter.
fn sdpa_bindings(extra: &str, marker: bool) -> String {
    let marker = if marker {
        "    pub fn mlx_compile_cache_new() -> mlx_compile_cache;\n"
    } else {
        ""
    };
    format!(
        "extern \"C\" {{\n{marker}    pub fn mlx_fast_scaled_dot_product_attention(\n        \
         res: *mut mlx_array,\n        queries: mlx_array,\n        keys: mlx_array,\n        \
         values: mlx_array,\n        scale: f32,\n        \
         mask_mode: *const ::std::os::raw::c_char,\n        mask_arr: mlx_array,\n        \
         sinks: mlx_array,\n{extra}        s: mlx_stream,\n    ) -> ::std::os::raw::c_int;\n}}\n"
    )
}

#[test]
fn the_two_mlx_c_c_apis_are_told_apart_by_the_sdpa_arity_and_the_marker() {
    assert_eq!(sdpa_takes_force_fused(&sdpa_bindings("", false)), Ok(false));
    assert_eq!(
        sdpa_takes_force_fused(&sdpa_bindings("        force_fused: bool,\n", true)),
        Ok(true)
    );
}

/// A combination neither C API has stops the build: the runtime probe reads
/// the marker, so it is only sound when the marker and the arity agree.
#[test]
fn an_unknown_mlx_c_c_api_is_an_error() {
    for (extra, marker) in [
        ("        force_fused: bool,\n", false),
        ("", true),
        ("        a: bool,\n        b: bool,\n", true),
    ] {
        let result = sdpa_takes_force_fused(&sdpa_bindings(extra, marker));
        assert!(
            result.as_ref().is_err_and(|e| e.contains(MLXC_0_7_MARKER)),
            "an unknown C API must not pass: {result:?}"
        );
    }
    assert!(sdpa_takes_force_fused("pub fn mlx_array_new() -> mlx_array;").is_err());
}
