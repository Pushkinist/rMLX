//! The DFlash 2 candidate-path selector.
//!
//! Ported from the z-lab MLX reference `CandidateSelector.select`. The decoder
//! stack denoises every block position independently, so its per-position
//! argmaxes need not read as one sentence. The selector turns them into a chain:
//! it keeps the `selector_top_k` highest-scoring tokens at each position and
//! picks, left to right, the candidate that scores best *given the token already
//! chosen at the position before it*.
//!
//! For a predecessor `a`, a candidate `b` at block position `t` whose final
//! hidden state is `h_t`:
//!
//! ```text
//! S_t(a, b) = U_t(b) + <A(a) (*) H(h_t), B(b)>
//! ```
//!
//! `U_t(b)` is `b`'s logit — the unary term. `A` and `B` are the two
//! vocabulary codebooks at rank `selector_rank`, and `H` projects the hidden
//! state to that rank. `H(h_t)` enters as a Hadamard factor on the predecessor
//! embedding, which is the context gate: the same token pair scores differently
//! depending on what the block is about at that position.
//!
//! # Where the logits come from
//!
//! The unary term is the **verifier's** LM head applied to the drafter's final
//! hidden states, not a head of the drafter's own — the drafter has none. The
//! reference computes it inside `propose`; here the caller passes it in,
//! because the verifier lives one layer up. On the published pair that head is
//! 4-bit.
//!
//! # One host readback per call
//!
//! The chain is sequential — position `t` needs the token chosen at `t - 1` —
//! but the dependency is carried in a device array, not a host integer. Every
//! gather, product and argmax below is a lazy MLX op; nothing is read back
//! until the whole chain is stacked and evaluated once at the end. The reference
//! does the same. A port that read the winning id back at each position would
//! pay a full device synchronisation per block position.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{add, argmax, argpartition, multiply, stack_axis, sum_axis, take_along_axis, Array};

use super::DFlash2Drafter;

impl DFlash2Drafter {
    /// Trace one draft chain over a block's drafted positions.
    ///
    /// `hidden` is the drafter's final hidden states at the drafted positions,
    /// `[1, n, hidden_size]` — [`DFlash2Drafter::forward_hidden`]'s output with
    /// row 0 dropped, since row 0 is the seed token the block was anchored on
    /// and is not drafted. `logits` is the verifier's LM head over those same
    /// rows, `[1, n, vocab_size]`. `anchor_id` is that seed token: the
    /// predecessor the first position is scored against.
    ///
    /// Returns the `n` drafted token ids, in block order.
    ///
    /// # Errors
    ///
    /// [`Error::Model`] when either input's shape is not the one the config
    /// predicts, when the two disagree on how many positions are being drafted,
    /// when more positions are asked for than the block the drafter was trained
    /// at holds, or when the anchor is not a token of this vocabulary.
    #[allow(
        clippy::indexing_slicing,
        reason = "shape axes are established by construction: both inputs' ranks are validated at the entry point and every array below is reshaped or sliced from them"
    )]
    pub fn select_chain(&self, hidden: &Array, logits: &Array, anchor_id: u32) -> Result<Vec<u32>> {
        let device = self.device;
        let rank = self.cfg.selector_rank as i32;
        let vocab = self.cfg.vocab_size as i32;
        let k = self.cfg.selector_top_k as i32;
        let positions = self.check_inputs(hidden, logits, anchor_id)?;

        // The candidate set, and with it the reference's own tie-break.
        // `argpartition` leaves the top `k` in the trailing `k` slots in an
        // order MLX does not specify, and a bf16 head over a vocabulary this
        // size ties at the k-th place at nearly every block position — so which
        // tokens are considered there is decided by that unspecified rule.
        // Measured on the pinned MLX: the tie goes to the higher token id, and
        // `argsort` keeps the same set (differing only in the order within the
        // slice), so this is a coupling to MLX's tie-break rather than to the
        // choice between those two primitives. It reaches the accept rate and
        // nothing else, because greedy acceptance emits the verifier's own
        // argmax whatever was proposed. See
        // `a_tie_at_the_candidate_boundary_breaks_toward_the_higher_token_id`.
        let partitioned = argpartition(logits, -k, -1, device)?;
        let candidates = partitioned.slice(
            &[0, 0, vocab - k],
            &[1, positions, vocab],
            &[1, 1, 1],
            device,
        )?;
        let unary = take_along_axis(logits, &candidates, -1, device)?;

        // Every gather that does not depend on the chain is done once, outside
        // the walk: the successor side of the score reads only the candidates.
        let gated_hidden = self.selector.hidden_projection.forward(hidden, device)?;
        let successors = self
            .selector
            .successor_codebook
            .take(&candidates, 0, device)?;

        let mut predecessor = Array::from_i32_slice(&[anchor_id as i32], &[1])?;
        let mut chain: Vec<Array> = Vec::with_capacity(positions as usize);
        for t in 0..positions {
            let gate = gated_hidden
                .slice(&[0, t, 0], &[1, t + 1, rank], &[1, 1, 1], device)?
                .reshape(&[1, 1, rank], device)?;
            let a = self
                .selector
                .predecessor_codebook
                .take(&predecessor, 0, device)?
                .reshape(&[1, 1, rank], device)?;
            let b = successors
                .slice(&[0, t, 0, 0], &[1, t + 1, k, rank], &[1, 1, 1, 1], device)?
                .reshape(&[1, k, rank], device)?;
            let edges = sum_axis(
                &multiply(&multiply(&a, &gate, device)?, &b, device)?,
                -1,
                device,
            )?;
            let scores = add(
                &unary
                    .slice(&[0, t, 0], &[1, t + 1, k], &[1, 1, 1], device)?
                    .reshape(&[1, k], device)?,
                &edges,
                device,
            )?;
            let winner = argmax(&scores, -1, device)?.reshape(&[1, 1], device)?;
            let row = candidates
                .slice(&[0, t, 0], &[1, t + 1, k], &[1, 1, 1], device)?
                .reshape(&[1, k], device)?;
            predecessor = take_along_axis(&row, &winner, -1, device)?.reshape(&[1], device)?;
            chain.push(predecessor.try_clone()?);
        }

        let refs: Vec<&Array> = chain.iter().collect();
        let stacked = stack_axis(&refs, 1, device)?;
        stacked.eval()?;
        let bytes = stacked.to_bytes()?;
        let ids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if ids.len() != positions as usize {
            return Err(Error::Model(format!(
                "DFlash2Drafter: the selector traced {} ids for {positions} block positions",
                ids.len()
            )));
        }
        tracing::debug!(
            positions,
            top_k = k,
            anchor_id,
            "DFlash2Drafter: selector traced a draft chain"
        );
        Ok(ids)
    }

    /// Validate the selector's inputs and return the number of drafted
    /// positions they agree on.
    #[allow(
        clippy::indexing_slicing,
        reason = "each shape is indexed only after its rank has been compared against the expected one"
    )]
    fn check_inputs(&self, hidden: &Array, logits: &Array, anchor_id: u32) -> Result<i32> {
        let h_shape = hidden.shape();
        let l_shape = logits.shape();
        let hidden_size = self.cfg.hidden_size as i32;
        let vocab = self.cfg.vocab_size as i32;
        if h_shape.len() != 3 || h_shape[0] != 1 || h_shape[2] != hidden_size {
            return Err(Error::Model(format!(
                "DFlash2Drafter: selector hidden has shape {h_shape:?}, not the \
                 [1, positions, {hidden_size}] the config predicts"
            )));
        }
        if l_shape.len() != 3 || l_shape[0] != 1 || l_shape[2] != vocab {
            return Err(Error::Model(format!(
                "DFlash2Drafter: selector logits have shape {l_shape:?}, not the \
                 [1, positions, {vocab}] the config predicts"
            )));
        }
        let positions = h_shape[1];
        if positions != l_shape[1] {
            return Err(Error::Model(format!(
                "DFlash2Drafter: the selector was given {positions} hidden rows and \
                 {} rows of logits; the logits are the verifier's head over those \
                 same rows and a mismatch means the caller sliced one and not the \
                 other",
                l_shape[1]
            )));
        }
        let drafted = self.cfg.block_size as i32 - 1;
        if positions < 1 || positions > drafted {
            return Err(Error::Model(format!(
                "DFlash2Drafter: the selector was asked for {positions} drafted \
                 positions; this drafter's block is {} tokens, of which the first \
                 is the seed, so it drafts between 1 and {drafted}",
                self.cfg.block_size
            )));
        }
        if anchor_id >= self.cfg.vocab_size as u32 {
            return Err(Error::Model(format!(
                "DFlash2Drafter: the selector's anchor token {anchor_id} is outside \
                 this drafter's vocabulary of {}; the codebook gather would read \
                 a clamped row and score a chain against the wrong predecessor",
                self.cfg.vocab_size
            )));
        }
        Ok(positions)
    }
}

#[cfg(test)]
#[path = "selector_tests.rs"]
mod selector_tests;
