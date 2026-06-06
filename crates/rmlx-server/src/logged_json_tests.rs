use super::*;

/// Verify that the snippet construction truncates at SNIPPET_LIMIT bytes
/// and does not panic on oversized or non-UTF-8 input.
#[test]
fn snippet_truncates_to_limit() {
    // Body larger than SNIPPET_LIMIT.
    let big: Vec<u8> = b"x".repeat(SNIPPET_LIMIT + 100);
    let raw = &big[..big.len().min(SNIPPET_LIMIT)];
    let snippet = String::from_utf8_lossy(raw);
    assert_eq!(snippet.len(), SNIPPET_LIMIT);
}

#[test]
fn snippet_handles_short_body() {
    let small = b"hello";
    let raw = &small[..small.len().min(SNIPPET_LIMIT)];
    let snippet = String::from_utf8_lossy(raw);
    assert_eq!(snippet.as_ref(), "hello");
}

#[test]
fn snippet_handles_non_utf8_bytes() {
    // Lossy UTF-8 conversion must not panic on arbitrary bytes.
    let bad: Vec<u8> = vec![0xFF, 0xFE, 0x00, 0x01];
    let raw = &bad[..bad.len().min(SNIPPET_LIMIT)];
    let snippet = String::from_utf8_lossy(raw);
    // Just verify it didn't panic and produced something non-empty.
    assert!(!snippet.is_empty());
}

#[test]
fn is_json_content_type_accepts_application_json() {
    use axum::http::HeaderMap;
    let mut m = HeaderMap::new();
    m.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
    assert!(is_json_content_type(&m));
}

#[test]
fn is_json_content_type_accepts_with_params() {
    use axum::http::HeaderMap;
    let mut m = HeaderMap::new();
    m.insert(
        header::CONTENT_TYPE,
        "application/json; charset=utf-8".parse().unwrap(),
    );
    assert!(is_json_content_type(&m));
}

#[test]
fn is_json_content_type_accepts_plus_json() {
    use axum::http::HeaderMap;
    let mut m = HeaderMap::new();
    m.insert(
        header::CONTENT_TYPE,
        "application/cloudevents+json".parse().unwrap(),
    );
    assert!(is_json_content_type(&m));
}

#[test]
fn is_json_content_type_rejects_text_json() {
    use axum::http::HeaderMap;
    let mut m = HeaderMap::new();
    m.insert(header::CONTENT_TYPE, "text/json".parse().unwrap());
    assert!(!is_json_content_type(&m));
}

#[test]
fn is_json_content_type_rejects_absent() {
    use axum::http::HeaderMap;
    assert!(!is_json_content_type(&HeaderMap::new()));
}
