//! Coverage for the one parse `build.rs` still does.
//!
//! The helper is `include!`d from the same file the build script includes: a
//! build script cannot be imported, and this parse decides what the runtime
//! version-skew warning compares the loaded library against.
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
