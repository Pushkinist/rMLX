use super::*;
use crate::paged::config::{
    paged_kv_page_tokens, resolve_paged_kv, resolve_paged_kv_page_tokens, DEFAULT_PAGE_TOKENS,
};

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn paged_kv_page_tokens_positive() {
    let val = paged_kv_page_tokens();
    assert!(val > 0, "page_tokens must be positive, got {val}");
}

// ── resolve_paged_kv ──────────────────────────────────────────────

#[test]
fn paged_resolver_default_absent() {
    assert!(!resolve_paged_kv(false));
}

#[test]
fn paged_resolver_cli_true() {
    assert!(resolve_paged_kv(true));
}

#[test]
fn paged_page_tokens_resolver_default() {
    assert_eq!(resolve_paged_kv_page_tokens(None), DEFAULT_PAGE_TOKENS);
}

#[test]
fn paged_page_tokens_resolver_cli() {
    assert_eq!(resolve_paged_kv_page_tokens(Some(64)), 64);
}

#[test]
fn paged_page_tokens_resolver_invalid_falls_through() {
    assert_eq!(resolve_paged_kv_page_tokens(Some(0)), DEFAULT_PAGE_TOKENS);
    assert_eq!(resolve_paged_kv_page_tokens(Some(-5)), DEFAULT_PAGE_TOKENS);
}

// Tests below require MLX runtime (zeros/slice_update/concatenate).
// They are marked #[ignore] so they don't crash in the unit-test harness
// which does not initialise the MLX C runtime. Run manually:
// cargo test -p rmlx-models --lib kv_cache::paged -- --ignored
#[test]
#[ignore = "requires mlx runtime"]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn page_slab_alloc_and_write_gather() {
    let device = Device::Cpu;
    let page_tokens = 4i32;
    let elems_per_token = 2i32;
    let n_pages = 4;
    let mut slab = PageSlab::new(n_pages, page_tokens, elems_per_token, Dtype::F32);

    let phys_id = slab.alloc(device).unwrap();
    assert_eq!(phys_id, 0);

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let bytes = f32_to_bytes(&data);
    let arr = Array::from_bytes(&bytes, &[4], Dtype::F32).unwrap();
    slab.write_page(phys_id, 0, 2, &arr, device).unwrap();

    let block_table = vec![0usize];
    let gathered = slab.gather(&block_table, 2, device).unwrap();
    // Force computation on CPU.
    let gbytes = gathered.to_bytes().unwrap();
    let gvals = bytes_to_f32(&gbytes);
    assert_eq!(gvals, data, "gather must return written data");
}

#[test]
#[ignore = "requires mlx runtime"]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn page_slab_multi_page_gather() {
    let device = Device::Cpu;
    let page_tokens = 2i32;
    let elems_per_token = 1i32;
    let n_pages = 4;
    let mut slab = PageSlab::new(n_pages, page_tokens, elems_per_token, Dtype::F32);

    let phys0 = slab.alloc(device).unwrap();
    let data0: Vec<f32> = vec![1.0, 2.0];
    let arr0 = Array::from_bytes(&f32_to_bytes(&data0), &[2], Dtype::F32).unwrap();
    slab.write_page(phys0, 0, 2, &arr0, device).unwrap();

    let phys1 = slab.alloc(device).unwrap();
    let data1: Vec<f32> = vec![3.0, 4.0];
    let arr1 = Array::from_bytes(&f32_to_bytes(&data1), &[2], Dtype::F32).unwrap();
    slab.write_page(phys1, 0, 2, &arr1, device).unwrap();

    let block_table = vec![phys0, phys1];
    let gathered = slab.gather(&block_table, 4, device).unwrap();
    let gvals = bytes_to_f32(&gathered.to_bytes().unwrap());
    assert_eq!(gvals, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[ignore = "requires mlx runtime"]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn page_slab_partial_last_page_gather() {
    let device = Device::Cpu;
    let page_tokens = 4i32;
    let elems_per_token = 1i32;
    let n_pages = 4;
    let mut slab = PageSlab::new(n_pages, page_tokens, elems_per_token, Dtype::F32);

    let phys0 = slab.alloc(device).unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let arr = Array::from_bytes(&f32_to_bytes(&data), &[3], Dtype::F32).unwrap();
    slab.write_page(phys0, 0, 3, &arr, device).unwrap();

    let block_table = vec![phys0];
    let gathered = slab.gather(&block_table, 3, device).unwrap();
    let gvals = bytes_to_f32(&gathered.to_bytes().unwrap());
    assert_eq!(
        gvals, data,
        "partial-page gather must only return filled tokens"
    );
}

#[test]
#[ignore = "requires mlx runtime"]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn page_slab_reset_recycles_pages() {
    let device = Device::Cpu;
    let mut slab = PageSlab::new(4, 4, 1, Dtype::F32);
    let phys0 = slab.alloc(device).unwrap();
    let phys1 = slab.alloc(device).unwrap();
    assert_ne!(phys0, phys1);

    slab.reset();
    assert!(!slab.free_list.is_empty() || slab.next_free == 0);
    let phys2 = slab.alloc(device).unwrap();
    assert!(phys2 < 4, "re-allocated page ID must be in pool range");
}
