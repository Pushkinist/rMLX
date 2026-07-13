// Shared with `src/build_info.rs` via `include!` — see that file for why.
// No `use` statements here: whatever includes this must bring `Path` itself
// into scope, since `include!` splices raw tokens with no module boundary.

/// Pull the profile-dir name out of an `OUT_DIR` path.
///
/// `OUT_DIR` is laid out as `<target-dir>/[<triple>/]<profile-dir>/build/
/// <pkg>-<hash>/out`. The profile dir is the component directly before the
/// LAST `build` component — using the first `build` component misfires
/// whenever the target dir path contains an earlier one (a checkout path or
/// `CARGO_TARGET_DIR` under a directory literally named `build/`).
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
        .rposition(|c| c == "build")
        .and_then(|i| i.checked_sub(1))
        .and_then(|i| parts.get(i))
        .filter(|p| !p.is_empty())
        .cloned()
        .unwrap_or_else(|| "unknown".to_string())
}
