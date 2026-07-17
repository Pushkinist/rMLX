//! Build script for `rmlx-mlx`: generates Rust FFI bindings from `mlx-c`
//! headers via `bindgen` and repacks them into `bindings.rs`.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    // disallowed_methods is a separate lint from unwrap_used;
    // build scripts are bucket-A (OUT_DIR always set by Cargo, abort is correct).
    clippy::disallowed_methods,
)]

use std::env;
use std::path::PathBuf;

/// Parse `MLX_VERSION_{MAJOR,MINOR,PATCH}` out of MLX's `version.h`.
///
/// The header is the authoritative source for the version of the tree we
/// compile against — the Cellar directory name is only a Homebrew convention.
/// Returns `"unknown"` when the header cannot be read or parsed; the runtime
/// check treats that as "cannot verify" and stays quiet rather than crying
/// wolf on a non-Homebrew layout.
fn read_mlx_version(header: &str) -> String {
    let Ok(src) = std::fs::read_to_string(header) else {
        return "unknown".to_owned();
    };
    let field = |name: &str| -> Option<String> {
        src.lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .map(|v| v.trim().to_owned())
    };
    match (
        field("MLX_VERSION_MAJOR"),
        field("MLX_VERSION_MINOR"),
        field("MLX_VERSION_PATCH"),
    ) {
        (Some(a), Some(b), Some(c)) => format!("{a}.{b}.{c}"),
        _ => "unknown".to_owned(),
    }
}

fn main() {
    // Rebuild triggers.
    println!("cargo:rerun-if-env-changed=MLX_C_PREFIX");
    println!("cargo:rerun-if-env-changed=MLX_PREFIX");
    println!("cargo:rerun-if-changed=wrapper.h");

    // Target guard: Apple Silicon only.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    assert!(
        !(target_arch != "aarch64" || target_os != "macos"),
        "rmlx-mlx: Apple Silicon (aarch64-apple-darwin) required. \
         Got arch={target_arch}, os={target_os}. \
         MLX runs only on Apple Silicon Metal."
    );

    // Prefix resolution: env var → brew canonical path.
    let mlx_c_prefix = env::var("MLX_C_PREFIX")
        .unwrap_or_else(|_| "/opt/homebrew/Cellar/mlx-c/0.6.0_2".to_owned());
    let mlx_prefix =
        env::var("MLX_PREFIX").unwrap_or_else(|_| "/opt/homebrew/Cellar/mlx/0.31.2".to_owned());

    // Verify the dylibs exist, give a helpful message if not.
    let mlxc_lib = format!("{mlx_c_prefix}/lib/libmlxc.dylib");
    let mlx_lib = format!("{mlx_prefix}/lib/libmlx.dylib");
    assert!(
        std::path::Path::new(&mlxc_lib).exists(),
        "rmlx-mlx: libmlxc.dylib not found at {mlxc_lib}. \
         Install with: brew install mlx-c. \
         Or set MLX_C_PREFIX to the correct cellar path."
    );
    assert!(
        std::path::Path::new(&mlx_lib).exists(),
        "rmlx-mlx: libmlx.dylib not found at {mlx_lib}. \
         Install with: brew install mlx. \
         Or set MLX_PREFIX to the correct cellar path."
    );

    // Record the MLX version we compile against, so the runtime can detect a
    // build-vs-runtime skew.
    //
    // This matters because the linked dylib's install name is the Homebrew
    // `opt` symlink (`/opt/homebrew/opt/mlx/lib/libmlx.dylib`), not the Cellar
    // path resolved above. The symlink follows whatever version Homebrew has
    // linked *now*, so upgrading MLX silently swaps the library this binary
    // loads — a different dylib and a different `mlx.metallib` — with no
    // rebuild and no error. The two versions can differ in ABI and in kernel
    // throughput, so the skew must not pass unnoticed.
    let version_header = format!("{mlx_prefix}/include/mlx/version.h");
    println!("cargo:rerun-if-changed={version_header}");
    let build_version = read_mlx_version(&version_header);
    println!("cargo:rustc-env=RMLX_MLX_BUILD_VERSION={build_version}");

    // Link search paths.
    println!("cargo:rustc-link-search=native={mlx_c_prefix}/lib");
    println!("cargo:rustc-link-search=native={mlx_prefix}/lib");

    // Link the dylibs.
    println!("cargo:rustc-link-lib=dylib=mlxc");
    println!("cargo:rustc-link-lib=dylib=mlx");

    // rpath so the binary finds dylibs at runtime without DYLD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_c_prefix}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{mlx_prefix}/lib");

    // Run bindgen.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{mlx_c_prefix}/include"))
        .clang_arg(format!("-I{mlx_prefix}/include"))
        // Only emit mlx_* symbols — avoids polluting with C stdlib types.
        .allowlist_function("mlx_.*")
        .allowlist_function("_mlx_.*")
        .allowlist_type("mlx_.*")
        .allowlist_var("MLX_.*")
        // Emit Rust enums instead of raw C int constants.
        .rustified_enum("mlx_dtype_")
        .rustified_enum("mlx_device_type_")
        // Derived impls & no layout tests (avoids alignment tests that
        // don't compile cleanly with opaque-struct patterns).
        .derive_default(true)
        .layout_tests(false)
        // Note: do NOT add `raw_line("#![...]")` here. The output is
        // wrapped via `include!()` in `src/sys.rs`, where inner attrs are
        // not permitted. The `#[allow(...)]` outer attr on `mod ffi` in
        // sys.rs covers all the lints we care about.
        .generate()
        .expect(
            "rmlx-mlx build.rs: bindgen failed to generate bindings. \
             Check that mlx-c headers are present at the configured prefix \
             and that clang is available (xcode-select --install).",
        );

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings_path = out_path.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("rmlx-mlx build.rs: could not write bindings.rs to OUT_DIR");

    // Strip any inner attributes (`#![...]`) that bindgen 0.71 emits at
    // file head — they are illegal in the `include!()` context used by
    // sys.rs. Outer attrs on `mod ffi` already cover the lints.
    let raw = std::fs::read_to_string(&bindings_path)
        .expect("rmlx-mlx build.rs: could not re-read bindings.rs for post-process");
    let cleaned: String = raw
        .lines()
        .filter(|line| !line.trim_start().starts_with("#!["))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&bindings_path, cleaned)
        .expect("rmlx-mlx build.rs: could not rewrite bindings.rs after strip");
}
