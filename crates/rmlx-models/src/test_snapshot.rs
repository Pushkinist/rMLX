//! Snapshot resolution for the model-gated unit tests under `src/`.
//!
//! `tests/common` is a module of the integration-test binaries and cannot be
//! reached from a lib unit test, so the join is made here: the slug under
//! `RMLX_O_MODELS_ROOT`, with a per-architecture variable as the override for a
//! root that does not hold it. (`crates/rmlx-cli/src/commands/kv_calibrate_tests.rs`
//! reads `RMLX_TEST_MODEL_BONSAI` first and the slug second.)
//!
//! **An unset variable is not a stand-down.** A machine holding the snapshot
//! runs the cell: these cells are GPU tests that `make gpu-test` selects, and
//! a cell that returned before asserting on a host holding its snapshot would
//! pass without testing anything.

use std::path::{Path, PathBuf};

/// Root holding every snapshot, addressed by slug.
const MODELS_ROOT_VAR: &str = "RMLX_O_MODELS_ROOT";

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
    let by_slug = std::env::var(MODELS_ROOT_VAR)
        .ok()
        .filter(|root| !root.is_empty())
        .map(|root| Path::new(&root).join(slug))
        .filter(|dir| dir.is_dir());

    let dir = match by_slug {
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

    let arch = declared_arch(&dir);
    if !archs.contains(&arch.as_str()) {
        println!(
            "SKIP {test}: {} declares arch {arch:?}, and this cell is written against {archs:?}",
            dir.display()
        );
        return None;
    }
    Some(dir)
}

/// The first entry of a snapshot's `architectures` array, or the empty string
/// when the file is missing or unreadable — which no cell's list contains, so
/// callers read it as a mismatch.
fn declared_arch(dir: &Path) -> String {
    let Ok(data) = std::fs::read(dir.join("config.json")) else {
        return String::new();
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&data) else {
        return String::new();
    };
    v.get("architectures")
        .and_then(|a| a.get(0))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("")
        .to_owned()
}
