use super::QuantRotorV3;

#[test]
fn rows_divides_by_the_width() {
    let store = QuantRotorV3::default();
    assert_eq!(store.rows(), 0);
}
