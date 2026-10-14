//! The stop of a request in the prefill of the speculative paths, under an
//! injected forward. No model, no GPU.
//!
//! This prefill fills the caches of the verifier and of a draft model. It
//! stores nothing in the prompt cache, so a stop here can leave no entry.

use super::*;

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::stop_scan::{error_events, STOP_OUTCOME};

/// Hand `stop` to the prefill of the speculative paths: one call that passes
/// `stop` to it, and does not wrap `forward`.
fn spec_prefill(
    tokens: &[u32],
    caches: &mut [KvCache],
    stop: &AtomicBool,
    forward: impl FnMut(&[u32], &mut [KvCache]) -> Result<()>,
) -> Result<()> {
    prefill_chunked_with(tokens, caches, 4, Device::Cpu, stop, forward)
}

/// What one scripted prefill did.
struct Prefilled {
    chunks: usize,
    /// Chunks that started with the stop set.
    chunks_after_stop: usize,
    /// The error the prefill returned, if it returned one.
    error: Option<Error>,
    /// Events the prefill logged at `ERROR`.
    logged_errors: usize,
    /// Caches that the prefill left in its prefill state.
    left_in_prefill: usize,
}

/// Prefill 16 tokens in 4 chunks. The request goes away while chunk `gone_in`
/// runs.
fn prefill(gone_in: Option<usize>) -> Prefilled {
    let mut caches: Vec<KvCache> = (0..3)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 32))
        .collect();
    let tokens: Vec<u32> = (0..16).collect();
    let stop = AtomicBool::new(false);
    let chunks = Cell::new(0usize);
    let chunks_after_stop = Cell::new(0usize);
    let forward = |_chunk: &[u32], caches: &mut [KvCache]| {
        assert!(
            caches.iter().all(KvCache::in_prefill),
            "the prefill did not enter its prefill state on every cache"
        );
        if stop.load(Ordering::Relaxed) {
            chunks_after_stop.set(chunks_after_stop.get() + 1);
        }
        if gone_in == Some(chunks.get()) {
            stop.store(true, Ordering::Relaxed);
        }
        chunks.set(chunks.get() + 1);
        Ok(())
    };
    let mut error = None;
    let logged_errors = error_events(|| {
        error = spec_prefill(&tokens, &mut caches, &stop, forward).err();
    });
    Prefilled {
        chunks: chunks.get(),
        chunks_after_stop: chunks_after_stop.get(),
        error,
        logged_errors,
        left_in_prefill: caches.iter().filter(|c| c.in_prefill()).count(),
    }
}

/// A stop that is never set changes nothing: every chunk runs.
#[test]
fn a_draft_prefill_that_is_not_stopped_runs_every_chunk() {
    let run = prefill(None);
    assert_eq!(run.chunks, 4);
    assert!(run.error.is_none(), "{:?}", run.error);
    assert_eq!(run.left_in_prefill, 0);
}

/// A stopped prefill starts no further chunk, and it still takes every cache
/// out of its prefill state: the exit sweep is a duty on every outcome. A stop
/// read with `?` at the head of the chunk loop, with nothing around the loop,
/// skips the sweep and fails here.
#[test]
fn a_stopped_draft_prefill_starts_no_further_chunk_and_leaves_no_cache_in_prefill() {
    let run = prefill(Some(1));
    assert_eq!(run.chunks, 2, "the stop did not end the prefill");
    assert_eq!(run.chunks_after_stop, 0);
    assert_eq!(run.left_in_prefill, 0, "caches left in their prefill state");
}

/// A stopped prefill returns the stop outcome, so the round loop above it
/// draws no seed and runs no round. It logs no error, and leaves no cache in
/// its prefill state.
#[test]
fn a_stopped_draft_prefill_returns_the_stop_outcome_and_logs_no_error() {
    let run = prefill(Some(1));
    assert!(
        run.error
            .as_ref()
            .is_some_and(|e| format!("{e:?}").starts_with(STOP_OUTCOME) && !e.is_migratable()),
        "{:?}",
        run.error
    );
    assert_eq!(run.logged_errors, 0);
    assert_eq!(run.left_in_prefill, 0, "caches left in their prefill state");
}
