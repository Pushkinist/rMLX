//! The row rule's check allocates nothing on a decode step.
//!
//! Every attention call passes the check of the row rule in
//! `scaled_dot_product_attention`. A decode step has one query row, and a
//! sliding-window layer sends it with an array mask on the GPU, so the check
//! reads the query row count there on every layer of every token.
//!
//! This binary holds one test, so nothing else allocates while it counts, and
//! its global allocator counts every Rust allocation of the process.
//! Allocations inside MLX and mlx-c do not reach it. The nodes are built and
//! never evaluated.

#![allow(unsafe_code, reason = "a global allocator implements an unsafe trait")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rmlx_mlx::{scaled_dot_product_attention, zeros, Device, Dtype};

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call goes to `System` unchanged; the count changes nothing
// else.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::SeqCst);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

const CALLS: usize = 1000;

/// Rust allocations of `CALLS` one-row attention calls with an array mask.
fn allocations_of_decode_calls(device: Device) -> Result<usize, Box<dyn std::error::Error>> {
    let q = zeros(&[1, 8, 1, 256], Dtype::Bf16, device)?;
    let kv = zeros(&[1, 1, 512, 256], Dtype::Bf16, device)?;
    let mask = zeros(&[1, 1, 1, 512], Dtype::Bf16, device)?;
    // The first calls start the MLX thread and its thread-local state.
    for _ in 0..10 {
        scaled_dot_product_attention(&q, &kv, &kv, 1.0, "array", Some(&mask), device)?;
    }
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    for _ in 0..CALLS {
        scaled_dot_product_attention(&q, &kv, &kv, 1.0, "array", Some(&mask), device)?;
    }
    Ok(ALLOCATIONS.load(Ordering::SeqCst) - before)
}

/// A whole one-row call with an array mask allocates nothing in Rust, on the
/// GPU, where the check reads the query row count, and on the CPU, where it
/// returns at the device compare.
#[test]
#[ignore = "builds attention nodes on the Metal GPU stream"]
fn the_row_rule_check_allocates_nothing_on_a_decode_step() -> Result<(), Box<dyn std::error::Error>>
{
    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let control = std::hint::black_box(Box::new(7u64));
    let seen = ALLOCATIONS.load(Ordering::SeqCst) - before;
    drop(control);
    assert_eq!(seen, 1, "the counter must see one allocation");

    for device in [Device::Cpu, Device::Gpu] {
        let during = allocations_of_decode_calls(device)?;
        assert_eq!(
            during, 0,
            "{CALLS} one-row attention calls with an array mask on {device:?} made {during} \
             allocations"
        );
    }
    Ok(())
}
