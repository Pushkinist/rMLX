// promoted: types/fields/methods below were `pub(crate)` / `pub(super)`
// inside `rmlx-models::kv_cache` and are promoted to `pub` here so the SSD
// modules (block_io/hydrate/spill) can still reach them across the crate
// boundary. Doc/visibility warnings on the promoted surface are silenced; the
// API is otherwise unchanged.
#![allow(
    missing_docs,
    missing_debug_implementations,
    unreachable_pub,
    clippy::exhaustive_structs,
    clippy::exhaustive_enums
)]
//! Paged storage implementations: PagedKStorage, PagedVStorage, PagedPlanarVStorage.

#![allow(
    clippy::cognitive_complexity,
    clippy::needless_pass_by_value,
    clippy::unnecessary_wraps
)]

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use super::alloc::PageSlab;

// ── PagedKStorage ─────────────────────────────────────────────────────────────

/// Paged quantized K storage (q8_0). Replaces `QuantK` on the GPU path when
/// paged KV is enabled (`--paged-kv`).
///
/// Uses two `PageSlab`s — one for u32 codes, one for f32 scales — and a
/// `block_table` that maps logical page index to physical page ID.
pub struct PagedKStorage {
    pub codes: PageSlab,
    pub scales: PageSlab,
    pub block_table: Vec<usize>,
    /// Number of filled tokens.
    pub total_tokens: i32,
    pub page_tokens: i32,
    /// Tokens currently filled within the current (last) page.
    tokens_in_last_page: i32,
    /// Per-token element counts (derived from shape on first append).
    words_per_token: i32,
    scales_per_token: i32,
    /// Accumulated logical shape `[B, kv_h, S_total, D]`.
    pub shape: Vec<i32>,
    /// Max sequence length.
    #[allow(dead_code)]
    pub max_seq: i32,
}

impl PagedKStorage {
    pub fn new(max_seq: i32, page_tokens: i32, n_pages: usize) -> Self {
        Self {
            codes: PageSlab::new(n_pages, page_tokens, 1, Dtype::U32),
            scales: PageSlab::new(n_pages, page_tokens, 1, Dtype::F32),
            block_table: Vec::new(),
            total_tokens: 0,
            page_tokens,
            tokens_in_last_page: 0,
            words_per_token: 0,
            scales_per_token: 0,
            shape: Vec::new(),
            max_seq,
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn init_shape(&mut self, new_shape: &[i32], device: Device) -> Result<()> {
        use crate::q8::Q8_GROUP_SIZE;
        let b = new_shape[0];
        let kv_h = new_shape[1];
        let d = new_shape[3];
        let words_per_token = b * kv_h * d / 4;
        let scales_per_token = b * kv_h * d / Q8_GROUP_SIZE as i32;
        self.words_per_token = words_per_token;
        self.scales_per_token = scales_per_token;
        let n_pages = self.codes.pool.len().max(1);
        let page_tokens = self.page_tokens;
        self.codes = PageSlab::new(n_pages, page_tokens, words_per_token, Dtype::U32);
        self.scales = PageSlab::new(n_pages, page_tokens, scales_per_token, Dtype::F32);
        self.shape = vec![b, kv_h, 0, d];
        let _ = device;
        Ok(())
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn append(
        &mut self,
        new_shape: &[i32],
        new_codes: Array,
        new_scales: Array,
        device: Device,
    ) -> Result<()> {
        let new_tokens = new_shape[2];

        if self.shape.is_empty() {
            self.init_shape(new_shape, device)?;
        }
        self.shape[2] += new_tokens;

        let mut remaining = new_tokens;
        let mut src_token_off = 0i32;

        while remaining > 0 {
            if self.tokens_in_last_page == self.page_tokens || self.block_table.is_empty() {
                let phys_id = self.codes.alloc(device)?;
                let _ = self.scales.alloc(device)?;
                self.block_table.push(phys_id);
                self.tokens_in_last_page = 0;
            }

            let logical_page = self.block_table.len() - 1;
            let phys_id = self.block_table[logical_page];
            let space_in_page = self.page_tokens - self.tokens_in_last_page;
            let write_tokens = remaining.min(space_in_page);

            let wpt = self.words_per_token;
            let spt = self.scales_per_token;
            let codes_slice = if new_tokens == write_tokens && src_token_off == 0 {
                new_codes.try_clone()?
            } else {
                new_codes.slice(
                    &[src_token_off * wpt],
                    &[(src_token_off + write_tokens) * wpt],
                    &[1],
                    device,
                )?
            };
            let scales_slice = if new_tokens == write_tokens && src_token_off == 0 {
                new_scales.try_clone()?
            } else {
                new_scales.slice(
                    &[src_token_off * spt],
                    &[(src_token_off + write_tokens) * spt],
                    &[1],
                    device,
                )?
            };

            self.codes.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &codes_slice,
                device,
            )?;
            self.scales.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &scales_slice,
                device,
            )?;

            self.tokens_in_last_page += write_tokens;
            src_token_off += write_tokens;
            remaining -= write_tokens;
        }

        self.total_tokens += new_tokens;
        Ok(())
    }

    pub fn gather(&self, device: Device) -> Result<(Array, Array)> {
        let codes = self
            .codes
            .gather(&self.block_table, self.total_tokens, device)?;
        let scales = self
            .scales
            .gather(&self.block_table, self.total_tokens, device)?;
        Ok((codes, scales))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.tokens_in_last_page = 0;
        self.block_table.clear();
        if !self.shape.is_empty() {
            self.shape[2] = 0;
        }
        self.codes.reset();
        self.scales.reset();
    }

    /// Actual on-device bytes across all allocated pages (codes + scales slabs).
    ///
    /// The exhaustive destructure is the drift guard: a new slab cannot be
    /// added to this struct without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            codes,
            scales,
            // Block table / geometry, not page allocations.
            block_table: _,
            total_tokens: _,
            page_tokens: _,
            tokens_in_last_page: _,
            words_per_token: _,
            scales_per_token: _,
            shape: _,
            max_seq: _,
        } = self;
        codes.resident_bytes() + scales.resident_bytes()
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        if !self.shape.is_empty() {
            self.shape[2] = n;
        }
        let new_page_count = ((n + self.page_tokens - 1) / self.page_tokens) as usize;
        while self.block_table.len() > new_page_count {
            let phys_id = self.block_table.pop().unwrap();
            self.codes.free_list.push(phys_id);
            self.scales.free_list.push(phys_id);
        }
        self.total_tokens = n;
        self.tokens_in_last_page = if n > 0 && n % self.page_tokens == 0 {
            self.page_tokens
        } else {
            n % self.page_tokens
        };
    }
}

// ── PagedVStorage (TurboQuant V4) ─────────────────────────────────────────────

/// Paged TurboQuant V4 storage. Mirrors `PagedKStorage` for the V side.
pub struct PagedVStorage {
    pub codes: PageSlab,
    pub scales: PageSlab,
    pub block_table: Vec<usize>,
    pub total_tokens: i32,
    pub page_tokens: i32,
    tokens_in_last_page: i32,
    words_per_token: i32,
    scales_per_token: i32,
    pub shape: Vec<i32>,
    #[allow(dead_code)]
    pub max_seq: i32,
    #[allow(dead_code)]
    pub bits: u8,
}

impl PagedVStorage {
    pub fn new(max_seq: i32, page_tokens: i32, n_pages: usize, bits: u8) -> Self {
        Self {
            codes: PageSlab::new(n_pages, page_tokens, 1, Dtype::U32),
            scales: PageSlab::new(n_pages, page_tokens, 1, Dtype::F32),
            block_table: Vec::new(),
            total_tokens: 0,
            page_tokens,
            tokens_in_last_page: 0,
            words_per_token: 0,
            scales_per_token: 0,
            shape: Vec::new(),
            max_seq,
            bits,
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn init_shape(&mut self, new_shape: &[i32], device: Device) -> Result<()> {
        use crate::turboquant::GROUP_SIZE;
        let b = new_shape[0];
        let kv_h = new_shape[1];
        let d = new_shape[3];
        let words_per_token = b * kv_h * d * 4 / GROUP_SIZE as i32;
        let scales_per_token = b * kv_h * d / GROUP_SIZE as i32;
        self.words_per_token = words_per_token;
        self.scales_per_token = scales_per_token;
        let n_pages = self.codes.pool.len().max(1);
        let page_tokens = self.page_tokens;
        self.codes = PageSlab::new(n_pages, page_tokens, words_per_token, Dtype::U32);
        self.scales = PageSlab::new(n_pages, page_tokens, scales_per_token, Dtype::F32);
        self.shape = vec![b, kv_h, 0, d];
        let _ = device;
        Ok(())
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn append(
        &mut self,
        new_shape: &[i32],
        new_codes: Array,
        new_scales: Array,
        device: Device,
    ) -> Result<()> {
        let new_tokens = new_shape[2];

        if self.shape.is_empty() {
            self.init_shape(new_shape, device)?;
        }
        self.shape[2] += new_tokens;

        let mut remaining = new_tokens;
        let mut src_token_off = 0i32;

        while remaining > 0 {
            if self.tokens_in_last_page == self.page_tokens || self.block_table.is_empty() {
                let phys_id = self.codes.alloc(device)?;
                let _ = self.scales.alloc(device)?;
                self.block_table.push(phys_id);
                self.tokens_in_last_page = 0;
            }

            let logical_page = self.block_table.len() - 1;
            let phys_id = self.block_table[logical_page];
            let space_in_page = self.page_tokens - self.tokens_in_last_page;
            let write_tokens = remaining.min(space_in_page);

            let wpt = self.words_per_token;
            let spt = self.scales_per_token;
            let codes_slice = if new_tokens == write_tokens && src_token_off == 0 {
                new_codes.try_clone()?
            } else {
                new_codes.slice(
                    &[src_token_off * wpt],
                    &[(src_token_off + write_tokens) * wpt],
                    &[1],
                    device,
                )?
            };
            let scales_slice = if new_tokens == write_tokens && src_token_off == 0 {
                new_scales.try_clone()?
            } else {
                new_scales.slice(
                    &[src_token_off * spt],
                    &[(src_token_off + write_tokens) * spt],
                    &[1],
                    device,
                )?
            };

            self.codes.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &codes_slice,
                device,
            )?;
            self.scales.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &scales_slice,
                device,
            )?;

            self.tokens_in_last_page += write_tokens;
            src_token_off += write_tokens;
            remaining -= write_tokens;
        }

        self.total_tokens += new_tokens;
        Ok(())
    }

    pub fn gather(&self, device: Device) -> Result<(Array, Array)> {
        let codes = self
            .codes
            .gather(&self.block_table, self.total_tokens, device)?;
        let scales = self
            .scales
            .gather(&self.block_table, self.total_tokens, device)?;
        Ok((codes, scales))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.tokens_in_last_page = 0;
        self.block_table.clear();
        if !self.shape.is_empty() {
            self.shape[2] = 0;
        }
        self.codes.reset();
        self.scales.reset();
    }

    /// Actual on-device bytes across all allocated pages (codes + scales slabs).
    ///
    /// The exhaustive destructure is the drift guard: a new slab cannot be
    /// added to this struct without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            codes,
            scales,
            // Block table / geometry, not page allocations.
            block_table: _,
            total_tokens: _,
            page_tokens: _,
            tokens_in_last_page: _,
            words_per_token: _,
            scales_per_token: _,
            shape: _,
            max_seq: _,
            bits: _,
        } = self;
        codes.resident_bytes() + scales.resident_bytes()
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        if !self.shape.is_empty() {
            self.shape[2] = n;
        }
        let new_page_count = ((n + self.page_tokens - 1) / self.page_tokens) as usize;
        while self.block_table.len() > new_page_count {
            let phys_id = self.block_table.pop().unwrap();
            self.codes.free_list.push(phys_id);
            self.scales.free_list.push(phys_id);
        }
        self.total_tokens = n;
        self.tokens_in_last_page = if n > 0 && n % self.page_tokens == 0 {
            self.page_tokens
        } else {
            n % self.page_tokens
        };
    }
}

// ── PagedPlanarVStorage ───────────────────────────────────────────────────────

/// Paged PlanarQuant V4 storage. Three slabs: codes, scales, rotations.
pub struct PagedPlanarVStorage {
    pub codes: PageSlab,
    pub scales: PageSlab,
    pub rotations: PageSlab,
    pub block_table: Vec<usize>,
    pub total_tokens: i32,
    pub page_tokens: i32,
    tokens_in_last_page: i32,
    codes_words_per_token: i32,
    scales_per_token: i32,
    rotations_words_per_token: i32,
    pub shape: Vec<i32>,
    #[allow(dead_code)]
    pub max_seq: i32,
}

impl PagedPlanarVStorage {
    pub fn new(max_seq: i32, page_tokens: i32, n_pages: usize) -> Self {
        Self {
            codes: PageSlab::new(n_pages, page_tokens, 1, Dtype::U32),
            scales: PageSlab::new(n_pages, page_tokens, 1, Dtype::F32),
            rotations: PageSlab::new(n_pages, page_tokens, 1, Dtype::U32),
            block_table: Vec::new(),
            total_tokens: 0,
            page_tokens,
            tokens_in_last_page: 0,
            codes_words_per_token: 0,
            scales_per_token: 0,
            rotations_words_per_token: 0,
            shape: Vec::new(),
            max_seq,
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn init_shape(&mut self, new_shape: &[i32], device: Device) -> Result<()> {
        use crate::turboquant::GROUP_SIZE;
        let b = new_shape[0];
        let kv_h = new_shape[1];
        let d = new_shape[3];
        let total_per_token = b * kv_h * d;
        let codes_words_per_token = total_per_token * 4 / GROUP_SIZE as i32;
        let scales_per_token = total_per_token / 2;
        let rotations_words_per_token = total_per_token * 2 / GROUP_SIZE as i32;
        self.codes_words_per_token = codes_words_per_token;
        self.scales_per_token = scales_per_token;
        self.rotations_words_per_token = rotations_words_per_token;
        let n_pages = self.codes.pool.len().max(1);
        let page_tokens = self.page_tokens;
        self.codes = PageSlab::new(n_pages, page_tokens, codes_words_per_token, Dtype::U32);
        self.scales = PageSlab::new(n_pages, page_tokens, scales_per_token, Dtype::F32);
        self.rotations = PageSlab::new(n_pages, page_tokens, rotations_words_per_token, Dtype::U32);
        self.shape = vec![b, kv_h, 0, d];
        let _ = device;
        Ok(())
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn append(
        &mut self,
        new_shape: &[i32],
        new_codes: Array,
        new_scales: Array,
        new_rotations: Array,
        device: Device,
    ) -> Result<()> {
        let new_tokens = new_shape[2];

        if self.shape.is_empty() {
            self.init_shape(new_shape, device)?;
        }
        self.shape[2] += new_tokens;

        let mut remaining = new_tokens;
        let mut src_token_off = 0i32;

        while remaining > 0 {
            if self.tokens_in_last_page == self.page_tokens || self.block_table.is_empty() {
                let phys_id = self.codes.alloc(device)?;
                let _ = self.scales.alloc(device)?;
                let _ = self.rotations.alloc(device)?;
                self.block_table.push(phys_id);
                self.tokens_in_last_page = 0;
            }

            let logical_page = self.block_table.len() - 1;
            let phys_id = self.block_table[logical_page];
            let space_in_page = self.page_tokens - self.tokens_in_last_page;
            let write_tokens = remaining.min(space_in_page);

            let cw = self.codes_words_per_token;
            let sp = self.scales_per_token;
            let rw = self.rotations_words_per_token;

            let is_full = new_tokens == write_tokens && src_token_off == 0;
            let codes_slice = if is_full {
                new_codes.try_clone()?
            } else {
                new_codes.slice(
                    &[src_token_off * cw],
                    &[(src_token_off + write_tokens) * cw],
                    &[1],
                    device,
                )?
            };
            let scales_slice = if is_full {
                new_scales.try_clone()?
            } else {
                new_scales.slice(
                    &[src_token_off * sp],
                    &[(src_token_off + write_tokens) * sp],
                    &[1],
                    device,
                )?
            };
            let rotations_slice = if is_full {
                new_rotations.try_clone()?
            } else {
                new_rotations.slice(
                    &[src_token_off * rw],
                    &[(src_token_off + write_tokens) * rw],
                    &[1],
                    device,
                )?
            };

            self.codes.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &codes_slice,
                device,
            )?;
            self.scales.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &scales_slice,
                device,
            )?;
            self.rotations.write_page(
                phys_id,
                self.tokens_in_last_page,
                write_tokens,
                &rotations_slice,
                device,
            )?;

            self.tokens_in_last_page += write_tokens;
            src_token_off += write_tokens;
            remaining -= write_tokens;
        }

        self.total_tokens += new_tokens;
        Ok(())
    }

    pub fn gather(&self, device: Device) -> Result<(Array, Array, Array)> {
        let codes = self
            .codes
            .gather(&self.block_table, self.total_tokens, device)?;
        let scales = self
            .scales
            .gather(&self.block_table, self.total_tokens, device)?;
        let rotations = self
            .rotations
            .gather(&self.block_table, self.total_tokens, device)?;
        Ok((codes, scales, rotations))
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn reset(&mut self) {
        self.total_tokens = 0;
        self.tokens_in_last_page = 0;
        self.block_table.clear();
        if !self.shape.is_empty() {
            self.shape[2] = 0;
        }
        self.codes.reset();
        self.scales.reset();
        self.rotations.reset();
    }

    /// Actual on-device bytes across all allocated pages (codes + scales + rotations slabs).
    ///
    /// The exhaustive destructure is the drift guard: a new slab cannot be
    /// added to this struct without this failing to compile.
    pub fn resident_bytes(&self) -> u64 {
        let Self {
            codes,
            scales,
            rotations,
            // Block table / geometry, not page allocations.
            block_table: _,
            total_tokens: _,
            page_tokens: _,
            tokens_in_last_page: _,
            codes_words_per_token: _,
            scales_per_token: _,
            rotations_words_per_token: _,
            shape: _,
            max_seq: _,
        } = self;
        codes.resident_bytes() + scales.resident_bytes() + rotations.resident_bytes()
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    pub fn truncate_to(&mut self, n: i32) {
        if !self.shape.is_empty() {
            self.shape[2] = n;
        }
        let new_page_count = ((n + self.page_tokens - 1) / self.page_tokens) as usize;
        while self.block_table.len() > new_page_count {
            let phys_id = self.block_table.pop().unwrap();
            self.codes.free_list.push(phys_id);
            self.scales.free_list.push(phys_id);
            self.rotations.free_list.push(phys_id);
        }
        self.total_tokens = n;
        self.tokens_in_last_page = if n > 0 && n % self.page_tokens == 0 {
            self.page_tokens
        } else {
            n % self.page_tokens
        };
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "ops_tests.rs"]
mod tests;
