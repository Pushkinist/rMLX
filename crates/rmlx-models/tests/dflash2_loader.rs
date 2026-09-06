//! DFlash 2 loader against the real `z-lab/Qwen3.8-27B-DFlash2` checkpoint.
//!
//! The unit tests beside the loader run on a JSON literal and on synthetic name
//! sets; neither can tell whether the names and shapes in the loader are the
//! ones the checkpoint on disk actually ships. This is where that is settled:
//! the snapshot is opened, every tensor is bound at the shape the config
//! predicts, and the twenty-three weights DFlash 2 adds over DFlash 1 are
//! pinned by name — including the two vocabulary codebooks, which carry no
//! `.weight` suffix and are found by nothing that looks for one.
//!
//! The snapshot resolves by slug from `RMLX_O_MODELS_ROOT`, so `make gpu-test`
//! runs this wherever the snapshots are and skips with a reason where they are
//! not.
//!
//! Run: `cargo test -p rmlx-models --test dflash2_loader -- --ignored --nocapture`

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason,
    clippy::items_after_statements
)]

mod common;

use std::path::PathBuf;

use rmlx_mlx::Device;
use rmlx_models::layers::Linear;
use rmlx_models::speculative::dflash2::DFlash2Drafter;

const DRAFT_SLUG: &str = "z-lab__Qwen3.8-27B-DFlash2";

/// The width of `mlx-community/Qwen3.8-27B-4bit`, the verifier this drafter is
/// published against. The drafter's `fc` reads that verifier's hidden states.
const VERIFIER_HIDDEN: usize = 5120;

/// Every tensor the checkpoint ships, and nothing else.
const TENSOR_COUNT: usize = 81;

/// The drafter snapshot by slug, or the reason this test stands down.
///
/// The golden harness's `slug_snapshot` cannot be used here: it requires a
/// `tokenizer.json`, which a drafter sidecar does not ship and never will — it
/// runs against the verifier's. Routing through it made all three tests below
/// announce a skip on a machine that holds the checkpoint, which is a green run
/// that asserted nothing.
///
/// The three outcomes match the harness's, and for its reasons: a root that is
/// set but is not a directory is one keystroke that disarms the test, so it
/// fails; a root that simply does not hold this slug skips, because nobody
/// holds every snapshot; an unset root skips with the variable named.
fn snapshot(slug: &str) -> Result<PathBuf, String> {
    let Some(root) = std::env::var(common::MODELS_ROOT_VAR)
        .ok()
        .filter(|r| !r.is_empty())
    else {
        return Err(format!(
            "no snapshot configured — set {} (holding {slug})",
            common::MODELS_ROOT_VAR
        ));
    };
    let root_path = PathBuf::from(&root);
    assert!(
        root_path.is_dir(),
        "{}={root} is not an existing directory",
        common::MODELS_ROOT_VAR
    );
    let path = root_path.join(slug);
    if path.join("config.json").is_file() && path.join("model.safetensors").is_file() {
        return Ok(path);
    }
    Err(format!(
        "{}={root} does not hold a {slug} with a config.json and a model.safetensors; \
         put the snapshot, or a symlink to it, there",
        common::MODELS_ROOT_VAR
    ))
}

fn linear_shape(l: &Linear) -> Vec<i32> {
    match l {
        Linear::Plain { weight } => weight.shape(),
        Linear::Quantized { .. } | Linear::Paro { .. } => {
            panic!("the DFlash 2 checkpoint ships bf16 weights; a quantized one is a bug")
        }
    }
}

fn norm_shape(n: &rmlx_models::layers::RmsNorm) -> Vec<i32> {
    n.weight.as_ref().expect("RmsNorm carries a weight").shape()
}

/// One decoder layer's tensors, at the shapes the next chunk's forward reads.
/// `kernel_projection` is 2 sides x 2 taps x (5120 / 16) groups = 1280 rows.
fn assert_layer_shapes(i: usize, layer: &rmlx_models::speculative::dflash2::DFlash2Layer) {
    for (which, conv) in [
        ("attention_conv", &layer.attention_conv),
        ("mlp_conv", &layer.mlp_conv),
    ] {
        assert_eq!(
            conv.base_kernel.shape(),
            vec![2, 2, 5120],
            "layers.{i}.{which}.base_kernel"
        );
        assert_eq!(
            linear_shape(&conv.kernel_projection),
            vec![1280, 5120],
            "layers.{i}.{which}.kernel_projection.weight"
        );
    }
    assert_eq!(linear_shape(&layer.q_proj), vec![4096, 5120]);
    assert_eq!(linear_shape(&layer.k_proj), vec![1024, 5120]);
    assert_eq!(linear_shape(&layer.v_proj), vec![1024, 5120]);
    assert_eq!(linear_shape(&layer.o_proj), vec![5120, 4096]);
    assert_eq!(norm_shape(&layer.q_norm), vec![128]);
    assert_eq!(norm_shape(&layer.k_norm), vec![128]);
    assert_eq!(norm_shape(&layer.input_layernorm), vec![5120]);
    assert_eq!(norm_shape(&layer.post_attention_layernorm), vec![5120]);
    assert_eq!(linear_shape(&layer.mlp.gate_proj), vec![17408, 5120]);
    assert_eq!(linear_shape(&layer.mlp.up_proj), vec![17408, 5120]);
    assert_eq!(linear_shape(&layer.mlp.down_proj), vec![5120, 17408]);
}

/// The checkpoint on disk carries exactly the tensors the loader reads, at
/// exactly the shapes it predicts from the config.
///
/// A successful load is a two-way proof of the name set: every name the loader
/// asks for must resolve or the load fails, and the shared unread-tensor
/// refusal fails the load on any name the snapshot ships that the loader did
/// not ask for. The count below pins the third thing neither of those can — that
/// this is still the 81-tensor checkpoint and not a re-export.
#[ignore]
#[test]
fn the_published_checkpoint_loads_whole() {
    let dir = match snapshot(DRAFT_SLUG) {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP dflash2_loader: {why}");
            return;
        }
    };

    // Count the tensors on disk independently of the loader, so a loader that
    // stopped asking for a family cannot also lower the bar it is measured
    // against.
    let shards = rmlx_loader::ShardSet::open_dir(&dir).expect("open the snapshot");
    let mut names: Vec<String> = Vec::new();
    for (_, handle) in shards.iter() {
        let st = handle.safetensors().expect("safetensors header");
        names.extend(st.names().into_iter().map(ToOwned::to_owned));
    }
    names.sort();
    assert_eq!(
        names.len(),
        TENSOR_COUNT,
        "the checkpoint ships {TENSOR_COUNT} tensors; got {}",
        names.len()
    );

    let drafter = DFlash2Drafter::load(&dir, VERIFIER_HIDDEN, Device::Gpu)
        .expect("the published DFlash 2 checkpoint must load whole");

    let cfg = &drafter.cfg;
    assert_eq!(cfg.block_size, 8, "block_size comes from dflash_config");
    assert!(
        (cfg.rope_theta - 1.0e7).abs() < 1.0,
        "rope base comes from rope_parameters: {}",
        cfg.rope_theta
    );
    assert_eq!(cfg.conv_group_size, 16);
    assert_eq!(cfg.conv_kernel_size, 2);
    assert_eq!(cfg.selector_rank, 256);
    assert_eq!(cfg.selector_top_k, 16);
    assert_eq!(cfg.mask_token_id, 248_070);
    assert_eq!(cfg.target_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.sliding_window, 2048);
    assert!(!cfg.is_causal);

    // The shared trunk, at the shapes the next chunk's forward will consume.
    assert_eq!(linear_shape(&drafter.fc), vec![5120, 25600]);
    assert_eq!(norm_shape(&drafter.hidden_norm), vec![5120]);
    assert_eq!(norm_shape(&drafter.norm), vec![5120]);
    assert_eq!(drafter.layers.len(), 5);

    for (i, layer) in drafter.layers.iter().enumerate() {
        assert_layer_shapes(i, layer);
    }

    // And the selector. The codebooks are the reason this test exists: they are
    // stored as bare parameters, so a loader that appended `.weight` — as it
    // must for every other tensor in the file — would not find them.
    assert!(
        names.contains(&"candidate_selector.predecessor_codebook".to_owned())
            && names.contains(&"candidate_selector.successor_codebook".to_owned()),
        "the codebooks carry no .weight suffix in this checkpoint"
    );
    assert!(
        !names.iter().any(|n| n.ends_with("_codebook.weight")),
        "a codebook with a .weight suffix would mean the checkpoint layout moved"
    );
    assert_eq!(
        linear_shape(&drafter.selector.hidden_projection),
        vec![256, 5120]
    );
    assert_eq!(
        drafter.selector.predecessor_codebook.shape(),
        vec![248_320, 256]
    );
    assert_eq!(
        drafter.selector.successor_codebook.shape(),
        vec![248_320, 256]
    );
}

/// The DFlash 1 loader still refuses this checkpoint, on the real file rather
/// than on a synthetic name set.
///
/// Serve routes it by declaration and never reaches that loader, so this is not
/// a path a user takes; it is the measurement that the twenty-three tensors are
/// real and that the shared refusal counts them. If this ever passes, the two
/// checkpoints have become interchangeable and the separate kind is dead
/// weight.
#[ignore]
#[test]
fn the_dflash_1_loader_still_refuses_this_checkpoint() {
    let dir = match snapshot(DRAFT_SLUG) {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP dflash2_loader: {why}");
            return;
        }
    };

    let err =
        rmlx_models::speculative::dflash::DFlashDrafter::load(&dir, VERIFIER_HIDDEN, Device::Gpu)
            .err()
            .map_or_else(String::new, |e| e.to_string());

    assert!(
        err.contains("DFlashDrafter") && err.contains("23"),
        "the DFlash 1 loader must refuse this checkpoint, naming the count: {err}"
    );
    assert!(
        err.contains("candidate_selector.predecessor_codebook")
            && err.contains("layers.0.attention_conv.base_kernel"),
        "the refusal must name the families it cannot build: {err}"
    );
}

/// A drafter of a width the verifier does not share is refused on the real
/// config, before any of its 3.8 GB is read.
#[ignore]
#[test]
fn a_mismatched_verifier_width_is_refused_on_the_real_config() {
    let dir = match snapshot(DRAFT_SLUG) {
        Ok(p) => p,
        Err(why) => {
            println!("SKIP dflash2_loader: {why}");
            return;
        }
    };

    let err = DFlash2Drafter::load(&dir, 2048, Device::Gpu)
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("5120") && err.contains("2048"),
        "the refusal must name both widths: {err}"
    );
}
