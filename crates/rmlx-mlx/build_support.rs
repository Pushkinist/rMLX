// Pure helpers behind `build.rs`: the MLX version and the mlx-c C API it
// compiles against.
//
// `include!`d by `build.rs` and by `tests/mlx_build_version.rs` rather than
// imported: a build script cannot depend on the crate it builds, and this
// parse decides what the runtime version-skew warning compares against, so it
// needs coverage that `cargo test` actually runs. Everything here is pure —
// all I/O and all `cargo:` directives stay in `build.rs`.
//
// The MLX / mlx-c pin is deliberately *not* checked here. A build script only
// re-runs when a `rerun-if-changed` path is newer than the last run, so
// repointing a package manager's `opt` symlink at an older keg moves the
// observed mtime backwards, the script does not re-run, and cargo replays its
// cached output — reporting a stale verdict about a stack that has since
// changed. The pin gate lives in `src/pin.rs`, where it can observe the
// library the process actually loaded.
//
// Paths and traits are fully qualified: an `include!`d file cannot own imports
// without colliding with whichever file pulled it in.

/// Parse `MLX_VERSION_{MAJOR,MINOR,PATCH}` out of the text of MLX's `version.h`.
///
/// The header is authoritative for the tree we compile against — a Cellar
/// directory name is only a Homebrew convention, and MLX is also installed
/// other ways. Returns `"unknown"` when the header cannot be parsed; callers
/// treat that as "cannot verify" and stay quiet rather than crying wolf.
fn read_mlx_version(src: &str) -> String {
    let field = |name: &str| -> Option<String> {
        src.lines()
            .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")))
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

/// The function the mlx-c 0.7 C API adds in the same upstream commit
/// (ml-explore/mlx-c `d4afaec5cc`) that gives
/// `mlx_fast_scaled_dot_product_attention` its `force_fused` parameter.
///
/// The build checks that the headers it compiles against declare it exactly
/// when that parameter is there. At run time `src/c_api.rs` looks it up in the
/// loaded `libmlxc.dylib`: a C signature is not visible at run time, an export
/// is.
const MLXC_0_7_MARKER: &str = "mlx_compile_cache_new";

/// Whether the generated bindings are the mlx-c 0.7 C API (`Ok(true)`) or the
/// 0.6 C API (`Ok(false)`).
///
/// The two differ in one function rMLX calls: 0.7 adds `bool force_fused` in
/// front of the stream of `mlx_fast_scaled_dot_product_attention`, 9 parameters
/// become 10. Any other combination of that arity and [`MLXC_0_7_MARKER`] is an
/// mlx-c this crate does not know, and the build must stop rather than guess
/// which argument list to pass.
fn sdpa_takes_force_fused(bindings: &str) -> Result<bool, String> {
    const SDPA: &str = "pub fn mlx_fast_scaled_dot_product_attention(";
    let params = bindings
        .split_once(SDPA)
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(list, _)| list.split(',').filter(|p| !p.trim().is_empty()).count())
        .ok_or_else(|| format!("the bindings declare no `{SDPA}..)`"))?;
    let marker = bindings.contains(&format!("pub fn {MLXC_0_7_MARKER}("));
    match (params, marker) {
        (9, false) => Ok(false),
        (10, true) => Ok(true),
        _ => Err(format!(
            "mlx_fast_scaled_dot_product_attention takes {params} parameters and \
             `{MLXC_0_7_MARKER}` is {}declared. rmlx-mlx knows two mlx-c C APIs: \
             0.6 (9 parameters, not declared) and 0.7 (10 parameters, declared)",
            if marker { "" } else { "not " }
        )),
    }
}
