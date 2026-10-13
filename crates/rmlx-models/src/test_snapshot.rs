//! Snapshot resolution for the model-gated unit tests under `src/`.
//!
//! `tests/common` is a module of the integration-test binaries and cannot be
//! reached from a lib unit test, so the join is made here: the slug under
//! `RMLX_O_MODELS_ROOT`, with a per-architecture variable as the override for a
//! root that does not hold it. (`crates/rmlx-cli/src/commands/kv_calibrate_tests.rs`
//! reads `RMLX_TEST_MODEL_BONSAI` first and the slug second.) A cell with no
//! such variable uses the slug alone (`snapshot_by_slug`).
//!
//! A resolver that returns a snapshot also calls [`release_cached_buffers`],
//! because a caller normally loads that snapshot right after.
//!
//! **An unset variable is not a stand-down.** A machine holding the snapshot
//! runs the cell. Most of these cells are GPU tests that `make gpu-test`
//! selects; a few (for example `quantize_paro_embed_row760`,
//! `integration_qwen3_5_moe_35b`) are `#[ignore]`d CPU tests that reach no
//! `Device::Gpu` and so run under no gate — they need `cargo test -- --ignored
//! <name>` by hand. Either way, a cell that returned before asserting on a
//! host holding its snapshot would pass without testing anything.

use std::path::{Path, PathBuf};

/// Root holding every snapshot, addressed by slug.
const MODELS_ROOT_VAR: &str = "RMLX_O_MODELS_ROOT";

/// Free the device buffers MLX keeps for reuse after a model is dropped.
///
/// The library tests share one process. Without this call, the buffers of each
/// dropped model stay in memory under the next model.
pub(crate) fn release_cached_buffers() {
    assert!(rmlx_mlx::mlx_clear_cache(), "mlx_clear_cache failed");
}

/// The snapshot `slug` names, or `None` after announcing why this cell stood
/// down.
///
/// `archs` is what the cell was written against. A snapshot declaring another
/// belongs to a different cell, so this stands down rather than running the
/// wrong model — the variable is per architecture, but a path is a path.
///
/// Every return of `None` prints `SKIP <test>: <why>`. A cell that returns
/// silently is invisible to `scripts/run_gpu_tests.sh` and to the
/// shader-validation census, and libtest reports it as a pass.
pub(crate) fn snapshot(test: &str, var: &str, slug: &str, archs: &[&str]) -> Option<PathBuf> {
    let dir = match under_root(slug) {
        Some(dir) => dir,
        None => match std::env::var(var).ok().filter(|v| !v.is_empty()) {
            Some(named) if Path::new(&named).is_dir() => PathBuf::from(named),
            Some(named) => {
                println!("SKIP {test}: {var}={named} is not an existing directory");
                return None;
            }
            None => {
                println!(
                    "SKIP {test}: {MODELS_ROOT_VAR} does not hold {slug}, and {var} names \
                     nothing either"
                );
                return None;
            }
        },
    };
    declaring(test, dir, archs)
}

/// The snapshot `slug` names under `RMLX_O_MODELS_ROOT`, with no variable to
/// override it, or `None` after announcing why this cell stood down. Same
/// identity check, `SKIP` rule and cache release as [`snapshot`].
pub(crate) fn snapshot_by_slug(test: &str, slug: &str, identities: &[&str]) -> Option<PathBuf> {
    let Some(dir) = under_root(slug) else {
        println!("SKIP {test}: {MODELS_ROOT_VAR} does not hold {slug}");
        return None;
    };
    declaring(test, dir, identities)
}

fn under_root(slug: &str) -> Option<PathBuf> {
    std::env::var(MODELS_ROOT_VAR)
        .ok()
        .filter(|root| !root.is_empty())
        .map(|root| Path::new(&root).join(slug))
        .filter(|dir| dir.is_dir())
}

/// `dir` when it declares one of `identities`, else `None` after a `SKIP`.
fn declaring(test: &str, dir: PathBuf, identities: &[&str]) -> Option<PathBuf> {
    let identity = declared_identity(&dir);
    if !identities.contains(&identity.as_str()) {
        println!(
            "SKIP {test}: {} declares {identity:?}, and this cell is written against \
             {identities:?}",
            dir.display()
        );
        return None;
    }
    release_cached_buffers();
    Some(dir)
}

/// The first entry of a snapshot's `architectures` array, or its `model_type`
/// when it names no architecture (a Prism Hadamard pack), or the empty string
/// when the file is missing or unreadable — which no cell's list contains, so
/// callers read it as a mismatch.
fn declared_identity(dir: &Path) -> String {
    let Ok(data) = std::fs::read(dir.join("config.json")) else {
        return String::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return String::new();
    };
    v.get("architectures")
        .and_then(|a| a.get(0))
        .or_else(|| v.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned()
}
