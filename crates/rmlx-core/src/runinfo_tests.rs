use super::*;

#[test]
fn run_id_shape() {
    let id = make_run_id();
    // YYYYMMDD-HHMMSS-... length >= 16
    assert!(id.len() >= 16, "got: {id}");
    assert_eq!(&id[8..9], "-");
}

#[test]
fn epoch_anchor() {
    assert_eq!(unix_to_ymdhms(0), (1970, 1, 1, 0, 0, 0));
}

#[test]
fn known_date() {
    // 2025-01-01 00:00:00 UTC = 1735689600 (well-known anchor).
    assert_eq!(unix_to_ymdhms(1_735_689_600), (2025, 1, 1, 0, 0, 0));
}
