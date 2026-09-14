//! What the round loop carries between rounds, and which rows it got it from.
//!
//! The loop no longer re-projects its whole conditioning history every round: it
//! projects each round's committed rows once and appends them to the projection
//! it is carrying. Both halves are invisible in an answer. `project_condition`
//! is row-wise, so the carried buffer is the projection of the whole history and
//! a re-projection agrees with it — exactly in exact arithmetic, and here within
//! `PROJECTION_TOL`, whose doc says why that is not zero and why an `f32`
//! fixture's figure does not carry to a checkpoint at its own dtype. And greedy
//! verification emits the verifier's own tokens whatever the drafter was
//! conditioned on, so a buffer built from the wrong rows moves the accept rate
//! before it moves anything a reader would notice. What is checked here is therefore the buffer itself: that it agrees
//! with the whole history's projection, and that the rows each round put into it
//! are the ones the round committed, by their position rather than by their
//! count.
//!
//! Rows that differ only in scale would pass the first check against almost any
//! implementation — a bias-free linear followed by an RMSNorm is scale-invariant
//! on a constant row — so the rows here differ in **direction**, and every
//! comparison carries a same-run control against rows the round did not commit.
//!
//! Runs on the CPU device against a synthetic drafter written to a temp dir; no
//! snapshot, no verifier, no Metal claim.

use std::path::Path;

use rmlx_mlx::{concatenate, Array, Device};

use super::DFlashDrafter;
use crate::speculative::{committed_rows, PROJECTION_TOL};

/// The synthetic drafter's width.
const HIDDEN: usize = 32;
/// Verifier layers it conditions on, so a conditioning row is `IDS * HIDDEN`.
const IDS: usize = 3;
/// Per-head width, and with it the head counts below.
const HEAD_DIM: usize = 8;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const INTERMEDIATE: usize = 16;

/// A deterministic spread of weights, distinct per tensor and per element, so no
/// two projections agree by symmetry.
fn weights(seed: f32, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.1 * (seed + 0.7 * i as f32).sin())
        .collect()
}

/// Rows that differ in direction, element `c` of absolute position `r` being
/// `sin(0.37 r + 0.11 c)`. Absolute position is what makes a row identifiable:
/// the wrong rows of the right shape do not match these.
#[allow(
    clippy::expect_used,
    reason = "test helper: an array that cannot be built is the assertion failing"
)]
fn rows_at(first: i32, rows: i32, width: i32) -> Array {
    let data: Vec<f32> = (first..first + rows)
        .flat_map(|r| (0..width).map(move |c| (0.37 * r as f32 + 0.11 * c as f32).sin()))
        .collect();
    Array::from_f32_slice(&data, &[1, rows, width]).expect("build rows")
}

/// Write a synthetic DFlash drafter — one decoder layer, F32 throughout — into
/// `dir`, and load it on the CPU device.
///
/// Only `fc` and `hidden_norm` carry the property under test; the rest exist
/// because the loader refuses a checkpoint missing them, and because it refuses
/// one carrying tensors it did not read.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a checkpoint that cannot be written or loaded is the assertion failing"
)]
fn fixture_drafter(dir: &Path) -> DFlashDrafter {
    let cfg = serde_json::json!({
        "architectures": ["DFlashDraftModel"],
        "block_size": 4,
        "dflash_config": {
            "mask_token_id": 7,
            "target_layer_ids": [0, 1, 2],
        },
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "model_type": "qwen3",
        "num_attention_heads": HEADS,
        "num_hidden_layers": 1,
        "num_key_value_heads": KV_HEADS,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1.0e7,
        "vocab_size": 16,
    });
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&cfg).expect("render config"),
    )
    .expect("write config");

    let q = HEADS * HEAD_DIM;
    let kv = KV_HEADS * HEAD_DIM;
    let shapes: Vec<(&str, Vec<usize>)> = vec![
        ("fc.weight", vec![HIDDEN, IDS * HIDDEN]),
        ("hidden_norm.weight", vec![HIDDEN]),
        ("norm.weight", vec![HIDDEN]),
        ("layers.0.input_layernorm.weight", vec![HIDDEN]),
        ("layers.0.post_attention_layernorm.weight", vec![HIDDEN]),
        ("layers.0.self_attn.q_proj.weight", vec![q, HIDDEN]),
        ("layers.0.self_attn.k_proj.weight", vec![kv, HIDDEN]),
        ("layers.0.self_attn.v_proj.weight", vec![kv, HIDDEN]),
        ("layers.0.self_attn.o_proj.weight", vec![HIDDEN, q]),
        ("layers.0.self_attn.q_norm.weight", vec![HEAD_DIM]),
        ("layers.0.self_attn.k_norm.weight", vec![HEAD_DIM]),
        ("layers.0.mlp.gate_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("layers.0.mlp.up_proj.weight", vec![INTERMEDIATE, HIDDEN]),
        ("layers.0.mlp.down_proj.weight", vec![HIDDEN, INTERMEDIATE]),
    ];
    let buffers: Vec<Vec<u8>> = shapes
        .iter()
        .enumerate()
        .map(|(i, (_, shape))| {
            let len = shape.iter().product::<usize>();
            weights(i as f32, len)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect()
        })
        .collect();
    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = shapes
        .iter()
        .zip(buffers.iter())
        .map(|((name, shape), buf)| {
            let tv =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), buf)
                    .expect("build tensor view");
            ((*name).to_owned(), tv)
        })
        .collect();
    std::fs::write(
        dir.join("model.safetensors"),
        safetensors::serialize(views, None).expect("serialize shard"),
    )
    .expect("write shard");

    DFlashDrafter::load(dir, HIDDEN, Device::Cpu).expect("the synthetic drafter loads")
}

#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn to_f32(a: &Array) -> Vec<f32> {
    let bytes = a.to_bytes().expect("read array bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes is an f32")))
        .collect()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let (x, y) = (to_f32(a), to_f32(b));
    assert_eq!(x.len(), y.len(), "compared arrays must have the same shape");
    x.iter()
        .zip(y.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max)
}

/// Across a full accept, a partial one and a zero accept, the buffer the loop
/// carries is the whole conditioning history projected, and each round put into
/// it exactly the rows it committed.
///
/// The control at every round is the projection of the same history shifted one
/// position along: the same shape and row count over rows the rounds did not
/// commit, so a tolerance that cannot separate the two is reported as such
/// rather than passing.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the rank is the one the drafter's projection returns"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a projection that cannot be taken is the assertion failing"
)]
fn the_carried_projection_is_the_history_and_the_rounds_commit_their_own_rows() {
    let dir = tempfile::tempdir().expect("temp dir");
    let drafter = fixture_drafter(dir.path());
    let width = (IDS * HIDDEN) as i32;

    // Round 0 conditions on the last prompt token alone, as the loop does.
    let mut history = rows_at(0, 1, width);
    let mut carried = drafter
        .project_condition(&history)
        .expect("project the first row");
    // The next position the verifier will score: the correction the last round
    // emitted, which is this round's carry token and its capture's row 0.
    let mut next = 1;

    // `(block, accept, committed)`: a full accept, a partial one, a zero accept,
    // and a round whose commit the request's remaining budget cut below the
    // `accept + 1` the caches kept — the case the row count is a caller's
    // argument for. The block varies per round because this drafter's schedule
    // varies it, so the capture the commit is sliced out of is a different
    // height each time.
    for (round, (block, accept, n_committed)) in
        [(5i32, 4usize, 5usize), (8, 1, 2), (3, 0, 1), (7, 4, 2)]
            .into_iter()
            .enumerate()
    {
        let v_hidden = rows_at(next, block, width);
        assert!(
            n_committed <= accept + 1,
            "round {round}: a round commits the carry token and the proposals it \
             kept, never more"
        );
        let committed =
            committed_rows(&v_hidden, n_committed, width, Device::Cpu).expect("commit the rows");

        assert_eq!(
            to_f32(&committed),
            to_f32(&rows_at(next, n_committed as i32, width)),
            "round {round}: the committed rows are not positions {next}..={} of the \
             capture",
            next + n_committed as i32 - 1
        );

        let projected_rows;
        (carried, projected_rows) = drafter
            .grow_conditioning(&carried, &committed, Device::Cpu)
            .expect("grow the conditioning");
        assert_eq!(
            projected_rows, n_committed as i32,
            "round {round}: the round projected {projected_rows} rows, not the \
             {n_committed} it committed"
        );

        history = concatenate(&[&history, &committed], 1, Device::Cpu).expect("extend the history");
        assert_eq!(
            carried.shape()[1],
            history.shape()[1],
            "round {round}: the carried buffer is not one row per committed position"
        );

        let want = drafter
            .project_condition(&history)
            .expect("project the history");
        let diff = max_abs_diff(&carried, &want);
        assert!(
            diff < PROJECTION_TOL,
            "round {round}: the carried buffer is not the whole history projected — \
             they differ by {diff:e}"
        );

        // The same history shifted one position along: the same shape and the
        // same row count, built from rows the rounds did not commit. It must not
        // sit inside the tolerance above, or that tolerance is telling us
        // nothing about which rows were taken.
        let wrong = drafter
            .project_condition(&rows_at(1, history.shape()[1], width))
            .expect("project the shifted history");
        let control = max_abs_diff(&carried, &wrong);
        assert!(
            control > 10.0 * PROJECTION_TOL,
            "round {round}: a history shifted by one position projects to within \
             {control:e}, so this case cannot tell one set of rows from another"
        );

        next += n_committed as i32;
    }

    // The first prompt row plus every round's commit, each position once.
    assert_eq!(
        next,
        1 + 5 + 2 + 1 + 2,
        "the rounds must have consumed each position exactly once"
    );
}

/// A capture holding fewer positions than the round commits is refused rather
/// than sliced past its own end.
#[test]
fn a_capture_shorter_than_the_commit_is_refused() {
    let width = 4;
    let v_hidden = rows_at(0, 3, width);
    let err = match committed_rows(&v_hidden, 6, width, Device::Cpu) {
        Ok(a) => panic!(
            "a capture of 3 rows committed 6 positions, got {:?}",
            a.shape()
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("holds 3 positions"),
        "the refusal does not say what it had: {err}"
    );
}
