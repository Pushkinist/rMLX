use super::*;

#[test]
fn base64_matches_known_vectors() {
    assert_eq!(base64_encode(b""), "");
    assert_eq!(base64_encode(b"f"), "Zg==");
    assert_eq!(base64_encode(b"fo"), "Zm8=");
    assert_eq!(base64_encode(b"foo"), "Zm9v");
    assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
    assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
    assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
}

#[test]
fn f32_base64_roundtrips_little_endian() {
    let v = [1.0f32, -2.5, 0.0];
    let b = f32_vec_to_base64(&v);
    // Decode manually: standard base64 -> bytes -> f32 LE.
    // 12 bytes -> 16 base64 chars.
    assert_eq!(b.len(), 16);
    // First f32 is 1.0 = 0x3F800000 LE bytes 00 00 80 3F.
    // base64 of [00,00,80,3F,...] first 4 bytes -> "AACA" prefix start.
    assert!(b.starts_with("AACAP"));
}

fn texts(i: EmbeddingInput) -> Vec<String> {
    match i.normalize() {
        NormInput::Texts(v) => v,
        NormInput::Images(_) => panic!("expected text input"),
    }
}
fn images(i: EmbeddingInput) -> Vec<String> {
    match i.normalize() {
        NormInput::Texts(_) => panic!("expected image input"),
        NormInput::Images(v) => v,
    }
}

#[test]
fn input_single_and_many_normalize() {
    // Text wire contract: string | [string].
    let s: EmbeddingInput = serde_json::from_str(r#""hello""#).unwrap();
    assert_eq!(texts(s), vec!["hello".to_owned()]);
    let m: EmbeddingInput = serde_json::from_str(r#"["a","b"]"#).unwrap();
    assert_eq!(texts(m), vec!["a".to_owned(), "b".to_owned()]);
}

#[test]
fn input_image_object_and_list_normalize() {
    // oMLX image shapes: {"image": "..."} and [{"image": "..."}].
    let one: EmbeddingInput =
        serde_json::from_str(r#"{"image":"data:image/png;base64,AAAA"}"#).unwrap();
    assert_eq!(images(one), vec!["data:image/png;base64,AAAA".to_owned()]);
    let many: EmbeddingInput =
        serde_json::from_str(r#"[{"image":"/tmp/a.png"},{"image":"Zm9v"}]"#).unwrap();
    assert_eq!(
        images(many),
        vec!["/tmp/a.png".to_owned(), "Zm9v".to_owned()]
    );
}

#[test]
fn base64_decode_roundtrips_and_rejects() {
    use crate::image_io::base64_decode;
    for s in ["", "f", "fo", "foo", "foob", "fooba", "foobar"] {
        let enc = base64_encode(s.as_bytes());
        if s.is_empty() {
            // empty payload is an explicit error (no zero-length image).
            assert!(base64_decode(&enc).is_err());
        } else {
            assert_eq!(base64_decode(&enc).unwrap(), s.as_bytes(), "roundtrip {s}");
        }
    }
    // whitespace ignored, padding optional
    assert_eq!(base64_decode("Zm 9v\n").unwrap(), b"foo");
    assert_eq!(base64_decode("Zm9v").unwrap(), b"foo");
    // invalid alphabet char rejected
    assert!(base64_decode("Zm9v$").is_err());
}

#[test]
fn decode_image_source_data_uri_and_raw_base64() {
    use crate::image_io::{load_image, DEFAULT_HTTP_TIMEOUT};
    // "foo" base64 is "Zm9v".
    let via_uri = load_image("data:image/png;base64,Zm9v", DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(via_uri, b"foo");
    let via_raw = load_image("Zm9v", DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(via_raw, b"foo");
    // non-base64 data URI rejected
    assert!(load_image("data:image/png,hello", DEFAULT_HTTP_TIMEOUT).is_err());
    // pathish but missing file -> clear error
    assert!(load_image("/no/such/file.png", DEFAULT_HTTP_TIMEOUT).is_err());
}
