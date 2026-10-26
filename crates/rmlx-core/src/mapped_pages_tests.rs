use super::*;

const PAGE: usize = 16_384;

#[test]
fn page_size_is_a_power_of_two() {
    let page = page_size().unwrap();
    assert!(page.is_power_of_two());
    assert!(page >= 4096);
}

#[test]
fn inner_page_range_rounds_start_up_and_end_down() {
    assert_eq!(
        inner_page_range(PAGE + 1, 4 * PAGE - 1, PAGE),
        Some(2 * PAGE..3 * PAGE)
    );
}

#[test]
fn inner_page_range_keeps_aligned_bounds() {
    assert_eq!(inner_page_range(PAGE, 3 * PAGE, PAGE), Some(PAGE..3 * PAGE));
}

#[test]
fn inner_page_range_is_none_below_one_whole_page() {
    assert_eq!(inner_page_range(PAGE + 1, 2 * PAGE + 100, PAGE), None);
    assert_eq!(inner_page_range(10, 100, PAGE), None);
    assert_eq!(inner_page_range(PAGE, 2 * PAGE - 1, PAGE), None);
    assert_eq!(inner_page_range(PAGE, PAGE, PAGE), None);
}

#[test]
fn inner_page_range_holds_exactly_one_page() {
    assert_eq!(
        inner_page_range(PAGE - 1, 2 * PAGE + 1, PAGE),
        Some(PAGE..2 * PAGE)
    );
}

#[test]
fn inner_page_range_is_none_for_a_zero_page() {
    assert_eq!(inner_page_range(0, 100, 0), None);
}

#[test]
fn inner_page_range_is_none_when_rounding_up_overflows() {
    assert_eq!(inner_page_range(usize::MAX - 1, usize::MAX, PAGE), None);
}

#[test]
fn file_pages_in_cache_has_one_entry_per_page_of_a_written_file() {
    let page = page_size().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pages.bin");
    std::fs::write(&path, vec![7u8; 3 * page + 1]).unwrap();
    File::open(&path).unwrap().sync_all().unwrap();
    let contents = std::fs::read(&path).unwrap();
    std::hint::black_box(contents.iter().fold(0u64, |sum, &b| sum + u64::from(b)));
    let cached = file_pages_in_cache(&path).unwrap();
    assert_eq!(cached.len(), 4);
    assert!(cached.iter().all(|&c| c));
}

#[test]
fn file_pages_in_cache_is_empty_for_an_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.bin");
    std::fs::write(&path, []).unwrap();
    assert_eq!(file_pages_in_cache(&path).unwrap(), Vec::<bool>::new());
}
