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

/// The function the mlx-c 0.8 C API adds (`mlx_matrix_transpose`, declared in
/// `ops.h`, the header of `mlx_gather_qmm`), in the release that gives
/// `mlx_gather_qmm` its `global_scale` parameter.
///
/// The build checks that the headers it compiles against declare it exactly
/// when that parameter is there. At run time `src/c_api.rs` looks it up in the
/// loaded `libmlxc.dylib`.
const MLXC_0_8_MARKER: &str = "mlx_matrix_transpose";

/// An mlx-c C API that `rmlx-mlx` compiles against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MlxcApi {
    V0_6,
    V0_7,
    V0_8,
}

/// The number of parameters bindgen declared for `name`, or `None` when the
/// bindings do not declare it.
fn declared_arity(bindings: &str, name: &str) -> Option<usize> {
    bindings
        .split_once(&format!("pub fn {name}("))
        .and_then(|(_, rest)| rest.split_once(')'))
        .map(|(list, _)| list.split(',').filter(|p| !p.trim().is_empty()).count())
}

/// The mlx-c C API of the generated bindings.
///
/// The C APIs differ in two functions rMLX calls: 0.7 adds `bool force_fused`
/// in front of the stream of `mlx_fast_scaled_dot_product_attention` (9
/// parameters become 10), and 0.8 adds `global_scale` in front of
/// `sorted_indices` in `mlx_gather_qmm` (13 become 14). Each also declares its
/// marker function. Any other combination of the two arities and the two
/// markers is an mlx-c this crate does not know, and the build must stop rather
/// than guess which argument lists to pass.
fn mlxc_c_api(bindings: &str) -> Result<MlxcApi, String> {
    const SDPA: &str = "mlx_fast_scaled_dot_product_attention";
    const QMM: &str = "mlx_gather_qmm";
    let arity = |name: &str| {
        declared_arity(bindings, name).ok_or_else(|| format!("the bindings declare no `{name}`"))
    };
    let (sdpa, qmm) = (arity(SDPA)?, arity(QMM)?);
    let declares = |marker: &str| bindings.contains(&format!("pub fn {marker}("));
    let (marker_0_7, marker_0_8) = (declares(MLXC_0_7_MARKER), declares(MLXC_0_8_MARKER));
    match (sdpa, marker_0_7, qmm, marker_0_8) {
        (9, false, 13, false) => Ok(MlxcApi::V0_6),
        (10, true, 13, false) => Ok(MlxcApi::V0_7),
        (10, true, 14, true) => Ok(MlxcApi::V0_8),
        _ => Err(format!(
            "{SDPA} takes {sdpa} parameters, `{MLXC_0_7_MARKER}` is {}declared, {QMM} takes \
             {qmm} parameters and `{MLXC_0_8_MARKER}` is {}declared. rmlx-mlx knows three \
             mlx-c C APIs: 0.6 (9 parameters, not declared, 13, not declared), 0.7 (10, \
             declared, 13, not declared) and 0.8 (10, declared, 14, declared)",
            if marker_0_7 { "" } else { "not " },
            if marker_0_8 { "" } else { "not " },
        )),
    }
}
