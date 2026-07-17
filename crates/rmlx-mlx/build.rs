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

// Pure resolution/parsing logic, shared with `tests/mlx_pin.rs`.
include!("build_support.rs");

/// Declares the MLX / mlx-c pair this build is validated against.
const PIN_FILE: &str = "mlx-pin.txt";

/// The GEMM kernel family whose presence in `mlx.metallib` is the capability
/// the pin exists to guarantee. Missing entirely in some bottles.
const NAX_GEMM_KERNEL: &str = "steel_gemm_fused_nax";

/// Read size for the metallib scan.
const SCAN_CHUNK: usize = 1 << 20;

/// Resolve the install prefix of a Homebrew formula.
///
/// Order: explicit env override -> `brew --prefix <formula>` -> the
/// conventional `opt` symlink.
///
/// Resolving to the `opt` path rather than a Cellar path is deliberate: the
/// linked dylibs' install names *are* `opt` paths, so that is what gets loaded
/// at run time no matter what this build points at. Compiling against the same
/// path the loader will use is what keeps build and runtime the same file; a
/// hard-coded Cellar path silently drifts from it on every upgrade.
fn resolve_prefix(env_key: &str, formula: &str) -> String {
    println!("cargo:rerun-if-env-changed={env_key}");
    if let Ok(prefix) = env::var(env_key) {
        return prefix;
    }
    if let Some(prefix) = brew_prefix(formula) {
        return prefix;
    }
    format!("/opt/homebrew/opt/{formula}")
}

/// Declare a rebuild trigger for a path, but only if it exists.
///
/// Cargo treats a listed path that is absent as permanently dirty, which would
/// re-run this script — and bindgen — on every single build for anyone whose
/// MLX is not laid out like a Homebrew keg. The probes already tolerate a
/// missing file by going quiet, so the trigger must tolerate it too.
///
/// Note what this can and cannot catch: cargo compares mtimes and re-runs only
/// for a *newer* one, and it stats through symlinks. Repointing `opt/mlx` at an
/// older keg therefore moves the observed mtime backwards and does **not**
/// trigger a re-run — recovering from a bad stack needs an explicit
/// `cargo clean -p rmlx-mlx`, and the runtime version-skew warning is the
/// backstop for a binary that was never rebuilt.
fn rerun_if_present(path: &str) {
    if std::path::Path::new(path).exists() {
        println!("cargo:rerun-if-changed={path}");
    }
}

/// Ask Homebrew where a formula lives. `None` when brew is absent, the formula
/// is not installed, or the answer does not exist on disk.
fn brew_prefix(formula: &str) -> Option<String> {
    let out = std::process::Command::new("brew")
        .args(["--prefix", formula])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let prefix = String::from_utf8(out.stdout).ok()?.trim().to_owned();
    (!prefix.is_empty() && std::path::Path::new(&prefix).exists()).then_some(prefix)
}

/// Compare the resolved MLX / mlx-c against the pinned pair, and check that the
/// metallib actually carries the fast GEMM kernels.
///
/// Warn, never fail. The pin records what was validated here; it is not a
/// statement that anything else is broken. A correct non-bottle build of
/// another version must still build, so a hard error would be wrong.
fn check_mlx_pin(mlx_prefix: &str, mlx_c_prefix: &str, mlx_version: &str) {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let pin_path = format!("{manifest_dir}/{PIN_FILE}");
    println!("cargo:rerun-if-changed={pin_path}");

    let pin_src = std::fs::read_to_string(&pin_path)
        .unwrap_or_else(|e| panic!("rmlx-mlx: cannot read the MLX pin at {pin_path}: {e}"));
    let pin = parse_pin(&pin_src).unwrap_or_else(|| {
        panic!(
            "rmlx-mlx: {pin_path} must declare one `mlx <version>` line and one \
             `mlx-c <version>` line (plus `#` comments). The two are pinned as a \
             pair — see docs/FFI.md."
        )
    });
    let (pin_mlx, pin_mlx_c) = (pin.mlx.as_str(), pin.mlx_c.as_str());

    // mlx-c ships no version header, so its identity is the keg directory name
    // — which is also the only place the Homebrew revision suffix appears, and
    // that suffix is the whole point (0.6.0_2 and 0.6.0_3 are the same upstream
    // release built against different mlx).
    let mlx_c_version = std::fs::canonicalize(mlx_c_prefix)
        .ok()
        .and_then(|real| keg_version_from(&real, "mlx-c"))
        .unwrap_or_else(|| "unknown".to_owned());

    // The capability itself, read off the library that will be loaded. This is
    // the ground truth the version pin only proxies for, and it keeps passing
    // on its own once a fixed bottle ships. `None` = no metallib to inspect.
    let metallib = format!("{mlx_prefix}/lib/mlx.metallib");
    rerun_if_present(&metallib);
    let fast_gemm = std::fs::File::open(&metallib)
        .ok()
        .and_then(|f| contains_needle(f, NAX_GEMM_KERNEL.as_bytes(), SCAN_CHUNK).ok());

    let drift = pin_drift(mlx_version, &mlx_c_version, &pin);

    if fast_gemm == Some(false) {
        println!(
            "cargo:warning=MLX at {mlx_prefix} (mlx {mlx_version}, mlx-c {mlx_c_version}) ships \
             no {NAX_GEMM_KERNEL} kernels in lib/mlx.metallib."
        );
        println!(
            "cargo:warning=  Those are the Neural-Accelerator GEMM path. Without them GPU matmul \
             throughput measured ~3.8x lower and prefill 2.2-3.7x slower on \
             Neural-Accelerator-class hardware. Decode is bandwidth-bound and looks normal, so \
             the symptom mimics a model-code defect."
        );
        println!(
            "cargo:warning=  rMLX pins the validated pair mlx {pin_mlx} + mlx-c {pin_mlx_c}, \
             whose metallib ships them. Some Homebrew bottles omit the family entirely; a \
             non-bottle build of the same version can be fine."
        );
        println!(
            "cargo:warning=  Fix (Homebrew; both kegs must already be in the Cellar — check with \
             `ls /opt/homebrew/Cellar/mlx`):"
        );
        println!(
            "cargo:warning=    ln -sfn ../Cellar/mlx/{pin_mlx} /opt/homebrew/opt/mlx && ln -sfn \
             ../Cellar/mlx-c/{pin_mlx_c} /opt/homebrew/opt/mlx-c && brew pin mlx mlx-c && cargo \
             clean -p rmlx-mlx"
        );
        println!(
            "cargo:warning=  Repoint both: mlx and mlx-c are ABI-coupled and a mismatched pair \
             aborts at load. The `cargo clean` is required, not tidiness: repointing to an older \
             keg moves file times backwards, and cargo only re-runs a build script for a *newer* \
             one — so a plain rebuild would silently keep these stale bindings. Pin: \
             crates/rmlx-mlx/{PIN_FILE}; rationale and un-pin steps: docs/FFI.md."
        );
    } else if drift {
        println!(
            "cargo:warning=MLX pin drift: resolved mlx {mlx_version} + mlx-c {mlx_c_version}, but \
             rMLX pins mlx {pin_mlx} + mlx-c {pin_mlx_c} (crates/rmlx-mlx/{PIN_FILE})."
        );
        println!(
            "cargo:warning=  The {NAX_GEMM_KERNEL} kernels the pin exists for are present, so GEMM \
             throughput should be unaffected. mlx and mlx-c are ABI-coupled though: an unvalidated \
             pair can abort at load with a dyld \"Symbol not found\". If this pair checks out, bump \
             both pin lines together. See docs/FFI.md."
        );
    }
}

fn main() {
    // Rebuild triggers. (The prefix env vars are declared by resolve_prefix.)
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build_support.rs");

    // Target guard: Apple Silicon only.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    assert!(
        !(target_arch != "aarch64" || target_os != "macos"),
        "rmlx-mlx: Apple Silicon (aarch64-apple-darwin) required. \
         Got arch={target_arch}, os={target_os}. \
         MLX runs only on Apple Silicon Metal."
    );

    // Prefix resolution: env var → brew → the `opt` symlink the loader uses.
    let mlx_c_prefix = resolve_prefix("MLX_C_PREFIX", "mlx-c");
    let mlx_prefix = resolve_prefix("MLX_PREFIX", "mlx");

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
    rerun_if_present(&version_header);
    let build_version =
        read_mlx_version(&std::fs::read_to_string(&version_header).unwrap_or_default());
    println!("cargo:rustc-env=RMLX_MLX_BUILD_VERSION={build_version}");

    // Report a resolved stack that is not the validated one, or that cannot
    // reach the fast GEMM path at all.
    check_mlx_pin(&mlx_prefix, &mlx_c_prefix, &build_version);

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
