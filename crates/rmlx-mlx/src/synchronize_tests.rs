//! Tests for [`synchronize_gpu`]: what the live count holds after it, where
//! the mlx-c call is made, and the call it refuses.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use std::sync::mpsc;

use super::*;
use crate::within_limit::{within_limit, LIMIT};
use crate::{add, mlx_active_memory_bytes, mlx_clear_cache, Array, Dtype};

const ELEMENTS: usize = 1 << 20;

/// The live count with no temporary of an earlier evaluation in it, and with
/// no cached buffer for the next allocation to reuse.
fn settled_live() -> u64 {
    synchronize_gpu().unwrap();
    assert!(mlx_clear_cache());
    mlx_active_memory_bytes().unwrap()
}

/// After an evaluation with a large temporary, the settled live count is the
/// count before plus the output, exactly.
///
/// The temporary is the f32 sum, 4 MiB. The output is its bf16 cast, 2 MiB, a
/// different size, so MLX cannot give the output the buffer of the temporary.
/// MLX frees the temporary in the completion handler of the command buffer.
/// Both sizes are whole 16 KiB pages and both buffers are new, so the
/// allocator counts them at their own size.
#[test]
#[ignore = "GPU Metal context — run with `make gpu-test`"]
fn after_synchronize_gpu_the_live_count_holds_no_temporary() {
    let a = Array::from_f32_slice(&vec![1.0; ELEMENTS], &[ELEMENTS as i32]).unwrap();
    let cast_of_sum = || {
        let sum = add(&a, &a, Device::Gpu).unwrap();
        let out = sum.astype(Dtype::Bf16, Device::Gpu).unwrap();
        drop(sum);
        out.eval().unwrap();
        out
    };
    // The first evaluation compiles the kernels; keep that out of the reading.
    drop(cast_of_sum());

    let before = settled_live();
    let out = cast_of_sum();
    let after = settled_live();
    let output_bytes = (ELEMENTS * 2) as u64;
    assert_eq!(
        after - before,
        output_bytes,
        "the settled live count must hold the {output_bytes} B output and nothing else; \
         {} B more means the 4 MiB temporary is still counted",
        (after - before).saturating_sub(output_bytes)
    );
    drop(out);
}

/// A job on the MLX thread holds the evaluation lock. `synchronize_gpu` takes
/// that lock, so a call from such a job would wait forever. It must return an
/// error. The time limit ends the test binary if it waits.
#[test]
fn a_call_from_a_job_that_holds_the_evaluation_lock_is_refused() {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        done_tx
            .send(crate::with_eval_lock(|| {
                synchronize_gpu().map_err(|e| e.to_string())
            }))
            .ok();
    });
    let refused = within_limit(&done_rx, LIMIT, "synchronize_gpu inside a job")
        .unwrap_or_else(|| panic!("the thread of the job panicked"))
        .unwrap();
    let message = refused.expect_err("a call from a job on the MLX thread must be refused");
    assert!(
        message.contains("called from a job on the MLX thread"),
        "refused for another reason: {message}"
    );
}

/// The one `mlx_synchronize` call of the crate is made inside the
/// `with_eval_lock` job of `synchronize_gpu`: on the MLX thread, which owns
/// the encoder of the stream, and under the evaluation lock. The
/// `check-eval-lock` gate does not know the symbol, because it evaluates no
/// array. This reads the text, so an alias or a macro is outside it.
#[test]
fn the_synchronize_call_is_made_in_a_job_under_the_evaluation_lock() {
    let source = include_str!("mlx_thread.rs");
    assert_eq!(
        source.matches("mlx_synchronize").count() - source.matches("\"mlx_synchronize\"").count(),
        1,
        "mlx_thread.rs must name the mlx-c call once, outside the status context string"
    );
    let body = &source[source.find("pub fn synchronize_gpu").unwrap()..];
    let job = body
        .find("crate::with_eval_lock(move || {")
        .expect("the job");
    let call = body.find("sys::mlx_synchronize(").expect("the call");
    let job_end = job + body[job..].find("})?;").expect("the end of the job");
    assert!(
        job < call && call < job_end,
        "sys::mlx_synchronize is called outside the with_eval_lock job of synchronize_gpu"
    );
}
