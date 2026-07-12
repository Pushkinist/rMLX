//! Stamps the real Cargo profile name into the binary.
//!
//! `cfg!(debug_assertions)` cannot tell `release`, `release-perf` and
//! `release-debug` apart — all three build with debug-assertions off, so it
//! collapses them to a single "release" label. Cargo's `PROFILE` env var is no
//! better: for custom profiles it only ever reports `"debug"` or `"release"`.
//!
//! `OUT_DIR` does carry the truth. It is laid out as
//! `<target-dir>/[<triple>/]<profile-dir>/build/<pkg>-<hash>/out`, and the
//! profile dir is named after the profile. Take the path component directly
//! before `build` and we have the real name.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").unwrap_or_default();
    println!(
        "cargo:rustc-env=RMLX_BUILD_PROFILE={}",
        profile_from_out_dir(&out_dir)
    );
}

/// Pull the profile-dir name out of an `OUT_DIR` path.
///
/// Returns `"unknown"` rather than guessing when the layout is not recognised —
/// an honest unknown beats a wrong label in a metrics row.
fn profile_from_out_dir(out_dir: &str) -> String {
    let parts: Vec<String> = Path::new(out_dir)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();

    parts
        .iter()
        .position(|c| c == "build")
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| parts.get(i))
        .filter(|p| !p.is_empty())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}
