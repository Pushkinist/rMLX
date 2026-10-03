//! The attention calls a run hands to MLX, and the oracle for the one
//! configuration a prefill must stay off.
//!
//! Shared by `tests/prefill_attention_configuration.rs`, which states the
//! configuration and the measurement behind it, and by the in-crate cells that
//! need a crate-private harness to reach a prefill path.

use std::sync::Arc;

use super::round_stream::{CapturedEvent, RoundStreamRecorder};
use rmlx_mlx::ATTENTION_CALL_TARGET;

/// One attention call as MLX received it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionCall {
    pub q_heads: i64,
    pub q_rows: i64,
    pub head_dim: i64,
    pub kv_heads: i64,
    pub k_rows: i64,
    pub v_head_dim: i64,
    pub dtype: String,
    pub mask: String,
    pub device: String,
}

impl AttentionCall {
    /// A missing field is a failure, not a call to skip: an event that lost a
    /// field would otherwise read as a prefill that issued no such call.
    pub fn from_event(event: &CapturedEvent) -> Self {
        let text = |name: &str| -> String {
            event
                .field(name)
                .unwrap_or_else(|| panic!("the attention-call event has no `{name}` field"))
                .trim_matches('"')
                .to_owned()
        };
        let int = |name: &str| -> i64 {
            text(name)
                .parse()
                .unwrap_or_else(|e| panic!("attention-call field `{name}` is not an integer: {e}"))
        };
        Self {
            q_heads: int("q_heads"),
            q_rows: int("q_rows"),
            head_dim: int("head_dim"),
            kv_heads: int("kv_heads"),
            k_rows: int("k_rows"),
            v_head_dim: int("v_head_dim"),
            dtype: text("dtype"),
            mask: text("mask"),
            device: text("device"),
        }
    }
}

/// Run `body` and return every attention call it handed to MLX on this thread.
pub fn recorded<T>(body: impl FnOnce() -> T) -> (T, Vec<AttentionCall>) {
    let recorder = RoundStreamRecorder::for_target(ATTENTION_CALL_TARGET);
    let out = tracing::subscriber::with_default(Arc::clone(&recorder), body);
    let accepted_another_target = recorder
        .questions()
        .iter()
        .any(|(target, _, answer)| *answer && target != ATTENTION_CALL_TARGET);
    assert!(
        !accepted_another_target,
        "the recorder enabled a callsite outside {ATTENTION_CALL_TARGET}: the run it \
         recorded is not the run that ships"
    );
    let calls = recorder
        .events()
        .iter()
        .map(AttentionCall::from_event)
        .collect();
    (out, calls)
}

/// Control, in the same process as the cells: the recorder is live. A call
/// outside the configuration, built and never evaluated, reaches it with its
/// own shape.
pub fn assert_the_recorder_is_live(device: rmlx_mlx::Device) {
    use rmlx_mlx::{zeros, Dtype};
    let ((), calls) = recorded(|| {
        let q = zeros(&[1, 4, 1025, 128], Dtype::Bf16, device).expect("q");
        let kv = zeros(&[1, 2, 3073, 128], Dtype::Bf16, device).expect("kv");
        let mask = zeros(&[1, 1, 1025, 3073], Dtype::Bf16, device).expect("mask");
        rmlx_mlx::scaled_dot_product_attention(&q, &kv, &kv, 1.0, "array", Some(&mask), device)
            .expect("control call");
    });
    let shapes: Vec<(i64, i64, i64)> = calls
        .iter()
        .map(|c| (c.q_rows, c.k_rows, c.head_dim))
        .collect();
    assert_eq!(
        shapes,
        vec![(1025, 3073, 128)],
        "control: the recorder must see the one call that was built"
    );
}

/// The head dim MLX 0.32.3 gives its head-dim-split attention kernel in a
/// masked prefill.
pub const SPLIT_KERNEL_HEAD_DIM: i64 = 256;
/// Below this many query rows MLX takes the unfused route at head dim 256.
pub const SPLIT_KERNEL_MIN_QUERY_ROWS: i64 = 1024;
/// The kernel's query and key block sizes: `align_Q` and `align_K` are "the row
/// count is a multiple of the block".
pub const QUERY_BLOCK: i64 = 64;
pub const KEY_BLOCK: i64 = 32;

/// Whether MLX 0.32.3 runs this call on the kernel specialization that returns
/// non-finite rows under device-memory validation.
///
/// Written from the MLX source and the measured cells, and shares no code with
/// the engine: an engine rule that states another configuration fails here.
/// The dtype is not a condition. MLX keeps `float32` off the kernel unless
/// `MLX_ENABLE_TF32` is set, so the oracle is stricter than the route for an
/// `f32` call.
pub fn is_faulty(call: &AttentionCall) -> bool {
    call.device == "Gpu"
        && call.mask == "array"
        && call.head_dim == SPLIT_KERNEL_HEAD_DIM
        && call.v_head_dim == SPLIT_KERNEL_HEAD_DIM
        && call.q_rows >= SPLIT_KERNEL_MIN_QUERY_ROWS
        && call.q_rows % QUERY_BLOCK != 0
        && call.k_rows % KEY_BLOCK != 0
}
