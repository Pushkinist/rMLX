//! The rows a chunked verifier capture keeps.
//!
//! A chunked prefill produces one hidden capture per chunk and the caller reads
//! one array. What it reads back differs by drafter: an EAGLE-3 drafter
//! conditions its own KV prefill on every prompt position, a DFlash 2 drafter
//! attends over one sliding window and can never read a row older than it. The
//! second one is why this exists — each row is
//! `len(capture_layer_ids) * hidden_size` wide (51.2 KiB on the published
//! DFlash 2 pair), so holding the whole prompt's capture to the end of the
//! prefill costs gigabytes at a long prompt and hands back a few thousand rows.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, Array, Device};

#[cfg(test)]
#[path = "capture_tail_tests.rs"]
mod capture_tail_tests;

/// Accumulates a chunked capture, holding only the rows the caller will read.
///
/// `keep` is that many trailing rows; `None` keeps every row. Chunks that have
/// fallen out of the kept tail are released as the prefill walks forward, so
/// the peak is the tail plus one chunk rather than the whole prompt.
pub(super) struct CaptureTail {
    keep: Option<usize>,
    /// Row count and capture of each chunk still held, oldest first.
    chunks: Vec<(usize, Array)>,
    rows: usize,
}

impl CaptureTail {
    pub(super) fn new(keep: Option<usize>) -> Self {
        Self {
            keep,
            chunks: Vec::new(),
            rows: 0,
        }
    }

    /// Rows currently held, across every chunk not yet released.
    pub(super) fn retained_rows(&self) -> usize {
        self.rows
    }

    /// Add one chunk's capture, `[1, rows, width]`, oldest row first.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the chunk is not that rank and leading axis — the
    /// row count is read off axis 1 and a differently shaped array would be
    /// joined along the wrong one.
    pub(super) fn push(&mut self, chunk: Array) -> Result<()> {
        let shape = chunk.shape();
        let rows = match shape.as_slice() {
            [1, rows, _] if *rows >= 0 => *rows as usize,
            _ => {
                return Err(Error::Model(format!(
                    "CaptureTail: a capture chunk is {shape:?}, not [1, rows, width]"
                )))
            }
        };
        self.rows = self.rows.saturating_add(rows);
        self.chunks.push((rows, chunk));

        let Some(keep) = self.keep else {
            return Ok(());
        };
        while self.chunks.len() > 1 {
            let Some(&(oldest, _)) = self.chunks.first() else {
                break;
            };
            if self.rows - oldest < keep {
                break;
            }
            self.chunks.remove(0);
            self.rows -= oldest;
        }
        Ok(())
    }

    /// Join what is held into `[1, min(total_rows, keep), width]`.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when no chunk was pushed, or from the join and the
    /// slice below.
    #[allow(
        clippy::indexing_slicing,
        reason = "axis 1 and 2 are read after push has established every chunk's rank, and a join along axis 1 preserves it"
    )]
    pub(super) fn finish(self, device: Device) -> Result<Array> {
        let held: Vec<&Array> = self.chunks.iter().map(|(_, a)| a).collect();
        let joined = match held.as_slice() {
            [] => {
                return Err(Error::Model(
                    "CaptureTail: no capture chunk was pushed".into(),
                ))
            }
            [one] => one.try_clone()?,
            many => concatenate(many, 1, device)?,
        };
        let Some(keep) = self.keep else {
            return Ok(joined);
        };
        let keep = i32::try_from(keep).map_err(|_| {
            Error::Model(format!(
                "CaptureTail: {keep} kept rows is more than an array axis holds"
            ))
        })?;
        let shape = joined.shape();
        let rows = shape[1];
        if rows <= keep {
            return Ok(joined);
        }
        joined.slice(
            &[0, rows - keep, 0],
            &[1, rows, shape[2]],
            &[1, 1, 1],
            device,
        )
    }
}
