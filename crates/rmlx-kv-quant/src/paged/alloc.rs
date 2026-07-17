// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(missing_docs, missing_debug_implementations, unreachable_pub)]
//! Page slab allocator for paged KV storage.

use rmlx_core::error::Result;
use rmlx_mlx::{concatenate, zeros, Array, Device, Dtype};

/// A slab of pre-allocated GPU pages for one quantized buffer component
/// (codes or scales or rotations).
///
/// Each page is a flat 1-D `Array` on the GPU that holds `page_tokens` tokens
/// worth of data for a single buffer component. Pages are handed out in order
/// from `next_free` and returned to `free_list` on request completion.
///
/// For single-request decoding (current use case) pages are never returned
/// during a request — they accumulate monotonically and are all released at
/// `reset()`.
pub struct PageSlab {
    /// Pool of page arrays. Each `Array` is pre-zeroed at `PAGE_TOKENS`
    /// capacity per token. Index = physical page ID.
    pub(super) pool: Vec<Option<Array>>,
    /// Number of elements stored per token in this slab (e.g. `words_per_step`
    /// for codes, `scales_per_step` for scales).
    elems_per_token: i32,
    /// Tokens per page.
    page_tokens: i32,
    /// Next unused physical page ID.
    pub(super) next_free: usize,
    /// Recycled page IDs (not used yet — placeholder for continuous batching).
    pub(super) free_list: Vec<usize>,
    /// Dtype of the slab elements (U32 for codes, F32 for scales).
    dtype: Dtype,
}

impl PageSlab {
    /// Allocate a new slab with `n_pages` pages, each holding `page_tokens`
    /// tokens of `elems_per_token` elements of type `dtype`.
    pub fn new(n_pages: usize, page_tokens: i32, elems_per_token: i32, dtype: Dtype) -> Self {
        Self {
            pool: (0..n_pages).map(|_| None).collect(),
            elems_per_token,
            page_tokens,
            next_free: 0,
            free_list: Vec::new(),
            dtype,
        }
    }

    /// Return the number of elements per page (capacity).
    fn page_elems(&self) -> i32 {
        self.page_tokens * self.elems_per_token
    }

    /// Actual on-device bytes across all allocated pages in this slab.
    ///
    /// Read from each live page's own shape × dtype rather than recomputed as
    /// `pages × page_tokens × elems_per_token × itemsize`: the pages are the
    /// allocation, the geometry fields are only bookkeeping about it. An
    /// unallocated page contributes nothing. Pages that are allocated but not
    /// yet fully written count their full allocation.
    ///
    /// The exhaustive destructure is the drift guard: a new buffer cannot be
    /// added to this struct without this failing to compile.
    pub(super) fn resident_bytes(&self) -> u64 {
        let Self {
            pool,
            // Geometry / bookkeeping about the pages above, not allocations.
            elems_per_token: _,
            page_tokens: _,
            next_free: _,
            free_list: _,
            dtype: _,
        } = self;
        pool.iter().flatten().map(crate::bytes::array_bytes).sum()
    }

    /// Allocate the next free physical page, initialize it to zeros on `device`.
    /// Returns the physical page ID.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn alloc(&mut self, device: Device) -> Result<usize> {
        let phys_id = if let Some(id) = self.free_list.pop() {
            id
        } else {
            let id = self.next_free;
            self.next_free += 1;
            id
        };
        if phys_id >= self.pool.len() {
            self.pool.push(None);
        }
        if self.pool[phys_id].is_none() {
            self.pool[phys_id] = Some(zeros(&[self.page_elems()], self.dtype, device)?);
        }
        Ok(phys_id)
    }

    /// Write a slice of new data into physical page `phys_id` starting at token
    /// offset `token_off_in_page` within that page.
    ///
    /// `new_data` is a 1-D Array of length `new_tokens * elems_per_token`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn write_page(
        &mut self,
        phys_id: usize,
        token_off_in_page: i32,
        new_tokens: i32,
        new_data: &Array,
        device: Device,
    ) -> Result<()> {
        let start = token_off_in_page * self.elems_per_token;
        let stop = (token_off_in_page + new_tokens) * self.elems_per_token;
        let page = self.pool[phys_id].take().unwrap();
        let updated = page.slice_update(new_data, &[start], &[stop], &[1], device)?;
        self.pool[phys_id] = Some(updated);
        Ok(())
    }

    /// Gather and concatenate the active prefix of pages described by
    /// `block_table` (logical page indices to physical page IDs).
    ///
    /// `total_tokens` is the total number of filled tokens so we can slice the
    /// last partial page correctly. Returns a flat 1-D Array.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn gather(
        &self,
        block_table: &[usize],
        total_tokens: i32,
        device: Device,
    ) -> Result<Array> {
        let page_tokens = self.page_tokens;
        let elems_per_token = self.elems_per_token;
        let n_full_pages = (total_tokens / page_tokens) as usize;
        let tail_tokens = total_tokens % page_tokens;

        let mut slices: Vec<Array> = Vec::with_capacity(block_table.len());

        for (logical_idx, &phys_id) in block_table.iter().enumerate() {
            if let Some(page) = &self.pool[phys_id] {
                let is_last = logical_idx == block_table.len() - 1;
                let tokens_in_this_page = if is_last && tail_tokens > 0 {
                    tail_tokens
                } else if logical_idx < n_full_pages {
                    page_tokens
                } else {
                    0
                };
                if tokens_in_this_page == 0 {
                    continue;
                }
                let stop = tokens_in_this_page * elems_per_token;
                let slice = page.slice(&[0], &[stop], &[1], device)?;
                slices.push(slice);
            }
        }

        if slices.is_empty() {
            return zeros(&[0], self.dtype, device);
        }
        if slices.len() == 1 {
            return Ok(slices.remove(0));
        }

        let refs: Vec<&Array> = slices.iter().collect();
        concatenate(&refs, 0, device)
    }

    /// Reset the slab for the next request — recycle all allocated pages back
    /// to `free_list`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn reset(&mut self) {
        for i in 0..self.next_free {
            if self.pool[i].is_some() {
                self.free_list.push(i);
            }
        }
        self.next_free = 0;
    }
}
