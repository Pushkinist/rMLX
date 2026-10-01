//! A hand-off to the MLX thread allocates nothing.
//!
//! The decode loop hands work to the MLX thread several times per layer per
//! token. This binary holds one test, so nothing else allocates while it
//! counts, and its global allocator counts every allocation of the process,
//! on every thread.

#![allow(unsafe_code, reason = "a global allocator implements an unsafe trait")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rmlx_mlx::scalar_f32;

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

const HAND_OFFS: usize = 1000;

/// `eval` of an array that is already available evaluates nothing, so the
/// count is the cost of the hand-off itself.
#[test]
fn a_hand_off_to_the_mlx_thread_allocates_nothing() -> Result<(), Box<dyn std::error::Error>> {
    let available = scalar_f32(1.0);
    // The first hand-offs start the MLX thread and its thread-local state.
    for _ in 0..10 {
        available.eval()?;
    }

    let before = ALLOCATIONS.load(Ordering::SeqCst);
    let control = std::hint::black_box(Box::new(7u64));
    let seen = ALLOCATIONS.load(Ordering::SeqCst) - before;
    drop(control);
    assert_eq!(seen, 1, "the counter must see one allocation");

    let before = ALLOCATIONS.load(Ordering::SeqCst);
    for _ in 0..HAND_OFFS {
        available.eval()?;
    }
    let during = ALLOCATIONS.load(Ordering::SeqCst) - before;
    assert_eq!(
        during, 0,
        "{HAND_OFFS} hand-offs to the MLX thread made {during} allocations"
    );
    Ok(())
}
