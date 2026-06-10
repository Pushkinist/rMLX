//! Per-arch golden-token decode gate + loader sibling-parity invariant for
//! medgemma (Gemma3ForConditionalGeneration).
//!
//! Two server-free `#[ignore]` tests over the medgemma snapshot:
//!
//! 1. `medgemma_golden_tokens_k8v8` — temp=0 greedy decode of a fixed prompt
//!    must reproduce the committed golden token-id sequence exactly. Catches
//!    genuine decode regressions without server/metrics noise.
//!
//! 2. `medgemma_loader_sibling_parity` — every `.scales` / `.biases` tensor
//!    present in the shard *headers* must be loaded by the gemma3 load path.
//!    medgemma's `model.safetensors.index.json` lists ZERO sibling tensors and
//!    points 255 plain entries at the wrong shard, so the scan-only loader is
//!    the only thing standing between us and silently dropped weights. This
//!    invariant FAILS the instant the loader stops scanning headers (e.g. a
//!    future index-truth migration that trusts the lying index).
//!
//! Model: `mlx-community__medgemma-1.5-4b-it-8bit` (Gemma3ForConditionalGeneration).
//! KV quant: K8V8 — the deprecated `for_arch_default` fallback the in-process
//! golden harness (`arch::generate_greedy`) uses. Production serving resolves
//! gemma3 to `Planar` via `resolve_default`; this golden gates the weight-load
//! + greedy-decode path (KV-codec-independent), not the Planar KV codec itself.
//!
//! Record the golden once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/medgemma-1.5-4b-it-8bit \
//! cargo test -p rmlx-models --test medgemma_golden_tokens -- --ignored
//! Then gate (re-run without regen):
//! RMLX_KV_TEST_MODEL=/path/to/medgemma-1.5-4b-it-8bit \
//! cargo test -p rmlx-models --test medgemma_golden_tokens -- --ignored

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
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use std::collections::BTreeSet;

use rmlx_kv_quant::KvQuant;
use rmlx_loader::{load_shard_index, ShardSet};
use rmlx_mlx::Device;
use rmlx_models::arch;

/// Architectures this golden was recorded against. Any other arch is skipped.
const EXPECTED_ARCHS: &[&str] = &["Gemma3ForConditionalGeneration"];

/// KV quant pinned for the golden. K8V8 is the deprecated `for_arch_default`
/// fallback the in-process harness uses; production `resolve_default` returns
/// `Planar` for gemma3. The golden gates weight loading + greedy decode (both
/// KV-codec-independent), so K8V8 is a sound, reproducible pin here.
const GOLDEN_KV_QUANT: KvQuant = KvQuant::K8V8;

#[ignore]
#[test]
fn medgemma_golden_tokens_k8v8() {
    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(&model_path, "medgemma_golden_tokens_k8v8", EXPECTED_ARCHS) {
        return;
    }
    common::run_golden_test("medgemma_4b_k8v8", GOLDEN_KV_QUANT);
}

/// Loader sibling-parity invariant.
///
/// Enumerates every `.scales` / `.biases` tensor straight from the shard
/// *headers* (NOT the index — the index lies), loads the model via the normal
/// gemma3 path, and asserts the loaded sibling count equals the header-truth
/// count. This must PASS on the current scan-only loader and FAIL if any future
/// loader silently drops index-omitted siblings or mis-resolves wrong-shard
/// entries.
#[ignore]
#[test]
fn medgemma_loader_sibling_parity() {
    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(
        &model_path,
        "medgemma_loader_sibling_parity",
        EXPECTED_ARCHS,
    ) {
        return;
    }

    // --- Header truth: enumerate `.scales` / `.biases` from the shard headers.
    // Open every shard listed in the index, then read each shard's safetensors
    // header (KB-sized JSON, no tensor data) and collect sibling tensor names.
    // The index `weight_map` is only used to discover which shard FILES exist;
    // the sibling NAMES come from the headers, since the index omits them all.
    let idx = load_shard_index(&model_path).expect("load_shard_index");
    let shards = ShardSet::open(&model_path, &idx).expect("ShardSet::open");

    // Shard-file discovery sanity: the index must reference every `.safetensors`
    // file on disk. The header scan below opens only index-referenced shards, so
    // an orphan shard (present on disk, named by no index entry) would be
    // invisible to BOTH the header scan and the loader — a mutually-invisible
    // drop. Asserting set-equality with a directory glob closes that residual.
    let dir_shards: BTreeSet<String> = std::fs::read_dir(&model_path)
        .expect("read model dir")
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|f| f.ends_with(".safetensors"))
        .collect();
    let index_shards: BTreeSet<String> = idx.weight_map.values().cloned().collect();
    assert_eq!(
        dir_shards, index_shards,
        "shard-file mismatch: the index must reference every `.safetensors` on disk \
         (an unreferenced shard's siblings would be invisible to both the header scan and the loader)"
    );

    let mut header_siblings: BTreeSet<String> = BTreeSet::new();
    for (_filename, handle) in shards.iter() {
        let st = handle.safetensors().expect("parse safetensors header");
        for name in st.names() {
            if name.ends_with(".scales") || name.ends_with(".biases") {
                header_siblings.insert(name.to_owned());
            }
        }
    }
    let header_sibling_count = header_siblings.len();

    // The text loader (and `loaded_sibling_count`) owns only the `language_model.`
    // namespace. Assert every header sibling lives there, so a future snapshot
    // that also quantizes the vision tower fails with this clear "revisit"
    // message instead of a misleading parity mismatch.
    for n in &header_siblings {
        assert!(
            n.starts_with("language_model."),
            "header sibling `{n}` is outside `language_model.` — the snapshot now quantizes \
             tensors the text-loader parity check does not cover; revisit this test"
        );
    }

    eprintln!(
        "[medgemma_loader_sibling_parity] header `.scales`/`.biases` siblings: {header_sibling_count} \
         (over {} shard headers)",
        shards.len()
    );

    // The index lies: it lists ZERO of these siblings. Assert that, so the test
    // documents *why* header-scanning is mandatory and fails loudly if a future
    // snapshot/index ever starts listing siblings (which would change the
    // invariant's meaning).
    let index_sibling_count = idx
        .weight_map
        .keys()
        .filter(|n| n.ends_with(".scales") || n.ends_with(".biases"))
        .count();
    eprintln!(
        "[medgemma_loader_sibling_parity] index `.scales`/`.biases` entries: {index_sibling_count} \
         (index omits siblings — header scan is the only source of truth)"
    );
    assert_eq!(
        index_sibling_count, 0,
        "expected medgemma index to omit ALL sibling tensors (the documented index lie); \
         got {index_sibling_count} — the invariant's premise has changed, revisit this test"
    );
    assert!(
        header_sibling_count > 0,
        "expected the shard headers to carry `.scales`/`.biases` siblings; got 0 — \
         snapshot layout changed, the parity check would be vacuous"
    );

    // --- Loaded truth: load via the normal gemma3 path and count materialised
    // siblings. The load path itself scans headers (index-omitted siblings are
    // resolved by `has_tensor` / `load_array` over the open shards), so a
    // correct loader materialises exactly the header-truth sibling set.
    let device = Device::Gpu;
    let model = arch::load_model(&model_path, device, &arch::LoadOpts::default())
        .expect("arch::load_model");
    let gemma3 = model
        .as_gemma3()
        .expect("medgemma must load as a Gemma3 model");
    let loaded_sibling_count = gemma3.loaded_sibling_count();

    eprintln!(
        "[medgemma_loader_sibling_parity] loaded siblings: {loaded_sibling_count} \
         (embed_tokens + decoder projections + lm_head)"
    );

    assert_eq!(
        header_sibling_count, loaded_sibling_count,
        "loader sibling-parity FAILED: {header_sibling_count} `.scales`/`.biases` tensors in the \
         shard headers but {loaded_sibling_count} loaded into the model. The scan-only loader \
         dropped index-omitted siblings or mis-resolved wrong-shard entries — index-truth regression."
    );
}
