// Pure helpers behind `build.rs`'s MLX version resolution.
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
