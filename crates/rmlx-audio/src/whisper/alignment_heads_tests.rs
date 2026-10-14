//! Unit tests for the read of the `alignment_heads` entry (no model).

use super::read_alignment_heads;
use crate::npz::WeightMap;
use crate::test_events::fields_recorded_by;
use crate::whisper::WhisperConfig;
use rmlx_mlx::{Array, Dtype};

const N_LAYERS: usize = 32;
const N_HEADS: usize = 20;

/// Small synthetic (layer, head) pairs, inside a decoder of 32 layers and 20 heads.
const PAIRS: [(usize, usize); 3] = [(0, 0), (3, 19), (31, 1)];

fn cfg() -> WhisperConfig {
    WhisperConfig::from_json(
        &serde_json::json!({
            "n_mels": 128,
            "n_audio_ctx": 1500,
            "n_audio_state": 1280,
            "n_audio_head": 20,
            "n_audio_layer": 32,
            "n_vocab": 51_866,
            "n_text_ctx": 448,
            "n_text_state": 1280,
            "n_text_head": N_HEADS,
            "n_text_layer": N_LAYERS,
        })
        .to_string(),
    )
    .unwrap()
}

fn weights_with(entry: Array) -> WeightMap {
    WeightMap::from([("alignment_heads".to_owned(), entry)])
}

fn pairs_i32(pairs: &[(i32, i32)]) -> WeightMap {
    let flat: Vec<i32> = pairs.iter().flat_map(|&(l, h)| [l, h]).collect();
    weights_with(Array::from_i32_slice(&flat, &[pairs.len() as i32, 2]).unwrap())
}

/// What a read gives, with the fields that its spans and events recorded.
fn read(weights: &WeightMap) -> (Option<Vec<(usize, usize)>>, Vec<(String, String)>) {
    let mut heads = None;
    let fields = fields_recorded_by(|| heads = read_alignment_heads(weights, &cfg()).unwrap());
    (heads, fields)
}

fn value_of<'a>(fields: &'a [(String, String)], name: &str) -> Vec<&'a str> {
    fields
        .iter()
        .filter(|(field, _)| field == name)
        .map(|(_, value)| value.as_str())
        .collect()
}

/// The read gave no heads and one warning, and the load event says so.
fn assert_one_warning_and_no_heads(weights: &WeightMap) -> Vec<(String, String)> {
    let (heads, fields) = read(weights);
    assert_eq!(heads, None, "{fields:?}");
    assert_eq!(
        value_of(&fields, "level")
            .iter()
            .filter(|l| **l == "WARN")
            .count(),
        1,
        "{fields:?}"
    );
    assert_eq!(value_of(&fields, "loaded"), ["false"], "{fields:?}");
    assert_eq!(value_of(&fields, "n_alignment_heads"), ["0"], "{fields:?}");
    fields
}

/// Integer pairs give the pairs, in file order, and warn of nothing.
#[test]
fn integer_pairs_give_the_pairs() {
    let weights = pairs_i32(&PAIRS.map(|(l, h)| (l as i32, h as i32)));
    let (heads, fields) = read(&weights);
    assert_eq!(heads, Some(PAIRS.to_vec()));
    assert!(!value_of(&fields, "level").contains(&"WARN"), "{fields:?}");
}

/// The load event says that the heads are loaded, and how many.
#[test]
fn the_load_event_gives_the_count_and_that_the_heads_are_loaded() {
    let weights = pairs_i32(&PAIRS.map(|(l, h)| (l as i32, h as i32)));
    let (_, fields) = read(&weights);
    assert_eq!(value_of(&fields, "n_alignment_heads"), ["3"], "{fields:?}");
    assert_eq!(value_of(&fields, "loaded"), ["true"], "{fields:?}");
}

/// The last layer and the last head are in range.
#[test]
fn the_last_layer_and_head_are_in_range() {
    let (heads, _) = read(&pairs_i32(&[(31, 19)]));
    assert_eq!(heads, Some(vec![(31, 19)]));
}

/// A snapshot with no entry has no heads: the accessor gives `None`, the load
/// event says that nothing is loaded, and there is no warning.
#[test]
fn no_entry_gives_no_heads_and_no_warning() {
    let (heads, fields) = read(&WeightMap::new());
    assert_eq!(heads, None);
    assert!(!value_of(&fields, "level").contains(&"WARN"), "{fields:?}");
    assert_eq!(value_of(&fields, "loaded"), ["false"], "{fields:?}");
    assert_eq!(value_of(&fields, "n_alignment_heads"), ["0"], "{fields:?}");
}

/// An entry that is not int32 or int64 pairs of shape (N, 2) gives no heads
/// and one warning with its dtype and its shape; the read does not fail.
#[test]
fn an_entry_in_a_wrong_form_gives_one_warning_with_its_dtype_and_shape() {
    let mask = Array::from_bytes(&[0; N_LAYERS * N_HEADS], &[32, 20], Dtype::U8).unwrap();
    let unsigned_pairs = Array::from_bytes(&[0; 16], &[2, 2], Dtype::U32).unwrap();
    let float_pairs = Array::from_f32_slice(&[7.0, 0.0], &[1, 2]).unwrap();
    let triples = Array::from_i32_slice(&[7, 0, 1, 10, 17, 1], &[2, 3]).unwrap();
    let vector = Array::from_i32_slice(&[7, 0], &[2]).unwrap();
    for (entry, dtype, shape) in [
        (mask, "U8", "[32, 20]"),
        (unsigned_pairs, "U32", "[2, 2]"),
        (float_pairs, "F32", "[1, 2]"),
        (triples, "I32", "[2, 3]"),
        (vector, "I32", "[2]"),
    ] {
        let fields = assert_one_warning_and_no_heads(&weights_with(entry));
        assert_eq!(value_of(&fields, "dtype"), [dtype], "{fields:?}");
        assert_eq!(value_of(&fields, "shape"), [shape], "{fields:?}");
    }
}

/// A pair outside the decoder or with a negative value gives no heads and one
/// warning with the pair; the read does not fail.
#[test]
fn a_pair_outside_the_decoder_gives_one_warning_with_the_pair() {
    for (pair, shown) in [
        ((32, 0), ["32", "0"]),
        ((7, 20), ["7", "20"]),
        ((-1, 0), ["-1", "0"]),
        ((0, -1), ["0", "-1"]),
    ] {
        let fields = assert_one_warning_and_no_heads(&pairs_i32(&[(1, 1), pair]));
        assert_eq!(
            value_of(&fields, "layer"),
            [shown[0]],
            "{pair:?} {fields:?}"
        );
        assert_eq!(value_of(&fields, "head"), [shown[1]], "{pair:?} {fields:?}");
    }
}
