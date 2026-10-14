//! The token pick for a decoder that filters its logit rows on the host: one
//! seeded generator for each request, and this module's softmax and inverse
//! CDF. No second sampler.

use super::{sample_inverse_cdf, softmax_scaled, Pcg32, DEFAULT_SEED};
use rmlx_core::error::{Error, Result};

/// Picks tokens from host logit rows. One picker serves one request, so a
/// repeated request draws the same stream.
#[derive(Debug)]
pub struct RowPicker {
    rng: Pcg32,
}

impl Default for RowPicker {
    /// The picker of a request that names no seed.
    fn default() -> Self {
        Self::new(DEFAULT_SEED)
    }
}

impl RowPicker {
    /// A picker whose draws come from `Pcg32::new(seed)`.
    fn new(seed: u64) -> Self {
        Self {
            rng: Pcg32::new(seed),
        }
    }

    /// Pick one token id from `row`. A masked id has the logit `-inf` and is
    /// never picked: a row with no logit above `-inf` is an error, which
    /// names a logit that is not a number when the row has one. A
    /// temperature of 0 gives the first largest logit and draws nothing. A
    /// temperature above 0 draws one value from the generator and gives the id
    /// at that point of the cumulative distribution of
    /// `softmax(row / temperature)`; a row with a `NaN` or `+inf` logit is an
    /// error.
    pub fn pick(&mut self, row: &[f32], temperature: f32) -> Result<u32> {
        if !row.iter().any(|&logit| logit > f32::NEG_INFINITY) {
            let cause = if row.iter().any(|logit| logit.is_nan()) {
                "a logit is not a number and no logit is finite"
            } else {
                "each logit is masked"
            };
            return Err(Error::Other(format!(
                "row pick: {cause} in a row of {} logits; no token can be picked",
                row.len()
            )));
        }
        if temperature > 0.0 {
            let uniform = self.rng.next_f32();
            return pick_at(row, temperature, uniform).map(|id| id as u32);
        }
        let mut best = 0;
        let mut best_logit = f32::NEG_INFINITY;
        for (id, &logit) in row.iter().enumerate() {
            if logit > best_logit {
                best_logit = logit;
                best = id;
            }
        }
        Ok(best as u32)
    }
}

/// The index at point `uniform`, in `[0, 1)`, of the cumulative distribution
/// of `softmax(row / temperature)`.
fn pick_at(row: &[f32], temperature: f32, uniform: f32) -> Result<usize> {
    let probs = softmax_scaled(row, 1.0 / temperature)?;
    Ok(sample_inverse_cdf(&probs, uniform))
}

#[cfg(test)]
#[path = "row_pick_tests.rs"]
mod row_pick_tests;
