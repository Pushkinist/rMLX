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
