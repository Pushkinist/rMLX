//! The rows a chunked verifier capture keeps.
//!
//! A chunked prefill produces one hidden capture per chunk and the caller reads
//! one array. What it reads back differs by drafter: an EAGLE-3 drafter
//! conditions its own KV prefill on every prompt position, a DFlash 2 drafter
//! attends over one sliding window and can never read a row older than it. The
//! second one is why this exists — each row is
//! `len(capture_layer_ids) * hidden_size` wide (50 KiB on the published DFlash 2
//! pair), so holding the whole prompt's capture to the end of the prefill costs
//! gigabytes at a long prompt and hands back a few thousand rows.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{concatenate, Array, Device};

#[cfg(test)]
#[path = "capture_tail_tests.rs"]
mod capture_tail_tests;

/// Accumulates a chunked capture, holding only the rows the caller will read.
///
/// `keep` is that many trailing rows; `None` keeps every row. Chunks that have
/// fallen out of the kept tail are released as the prefill walks forward, and
/// the oldest one still held is cut to the part of it the tail reaches before
/// anything is joined, so the peak is the tail plus the chunk being filled.
pub(crate) struct CaptureTail {
    keep: Option<usize>,
    /// Row count and capture of each chunk still held, oldest first.
    chunks: Vec<(usize, Array)>,
    rows: usize,
}

impl CaptureTail {
    pub(crate) fn new(keep: Option<usize>) -> Self {
        Self {
            keep,
            chunks: Vec::new(),
            rows: 0,
        }
    }

    /// Rows currently held, across every chunk not yet released.
    pub(crate) fn retained_rows(&self) -> usize {
        self.rows
    }

    /// Add one chunk's capture, `[1, rows, width]`, oldest row first.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when the chunk is not that rank and leading axis — the
    /// row count is read off axis 1 and a differently shaped array would be
    /// joined along the wrong one.
    pub(crate) fn push(&mut self, chunk: Array) -> Result<()> {
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
    /// [`Error::Model`] when no chunk was pushed, or from the cut and the join
    /// below.
    #[allow(
        clippy::indexing_slicing,
        reason = "axis 1 and 2 are read after push has established every chunk's rank"
    )]
    pub(crate) fn finish(mut self, device: Device) -> Result<Array> {
        if self.chunks.is_empty() {
            return Err(Error::Model(
                "CaptureTail: no capture chunk was pushed".into(),
            ));
        }
        if let Some(keep) = self.keep {
            self.cut_oldest_to_tail(keep, device)?;
        }
        let held: Vec<&Array> = self.chunks.iter().map(|(_, a)| a).collect();
        match held.as_slice() {
            [one] => one.try_clone(),
            many => concatenate(many, 1, device),
        }
    }

    /// Replace the oldest chunk with the suffix of it the kept tail reaches.
    ///
    /// Cutting before the join rather than slicing after it is what keeps the
    /// transient at one copy of the tail: a join of everything held would
    /// materialise the overshoot too, and then throw it away.
    #[allow(
        clippy::indexing_slicing,
        reason = "the first chunk is read only after the empty case has returned, and its rank is what push established"
    )]
    fn cut_oldest_to_tail(&mut self, keep: usize, device: Device) -> Result<()> {
        if self.rows <= keep {
            return Ok(());
        }
        let drop_rows = self.rows - keep;
        let (oldest_rows, oldest) = &self.chunks[0];
        let (oldest_rows, kept_rows) = (*oldest_rows, oldest_rows.saturating_sub(drop_rows));
        let shape = oldest.shape();
        let (start, end, width) = (
            i32::try_from(drop_rows),
            i32::try_from(oldest_rows),
            shape[2],
        );
        let (Ok(start), Ok(end)) = (start, end) else {
            return Err(Error::Model(format!(
                "CaptureTail: a chunk of {oldest_rows} rows cut at {drop_rows} is more \
                 than an array axis holds"
            )));
        };
        let cut = oldest.slice(&[0, start, 0], &[1, end, width], &[1, 1, 1], device)?;
        self.chunks[0] = (kept_rows, cut);
        self.rows = keep;
        Ok(())
    }
}
