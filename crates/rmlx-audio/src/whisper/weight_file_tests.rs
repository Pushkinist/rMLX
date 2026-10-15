//! Unit tests for the choice and the read of the weight file of a snapshot
//! (no model).

use super::load_weights;
use crate::npz::WeightMap;
use crate::whisper::alignment_heads::read_alignment_heads;
use crate::whisper::snapshot_fixture::{
    model_tensors, tiny_config_json, write_npz, write_safetensors, Tensor,
};
use crate::whisper::{WhisperConfig, WhisperModel};
use rmlx_mlx::Dtype;

/// Pairs inside the decoder of [`tiny_config_json`]: one layer of two heads.
const PAIRS: [(i64, i64); 2] = [(0, 0), (0, 1)];

/// Small tensors of each dtype that a Whisper snapshot holds, and the int64
/// `alignment_heads` entry.
fn small_tensors() -> Vec<Tensor> {
    vec![
        Tensor::f16("encoder.conv1.weight", &[2, 3, 4], 1),
        Tensor::f32("decoder.ln.bias", &[5], 2),
        Tensor::f16("decoder.positional_embedding", &[3, 2], 3),
        Tensor::int64_pairs(&[(0, 1), (3, 19), (31, 1)]),
    ]
}

fn load_error(dir: &std::path::Path) -> String {
    match load_weights(dir) {
        Ok(_) => panic!("the load must fail"),
        Err(e) => e.to_string(),
    }
}

fn config() -> WhisperConfig {
    WhisperConfig::from_json(&tiny_config_json(1, 1).to_string()).unwrap()
}

fn assert_same_weights(a: &WeightMap, b: &WeightMap) {
    let mut a_keys: Vec<&String> = a.keys().collect();
    let mut b_keys: Vec<&String> = b.keys().collect();
    a_keys.sort();
    b_keys.sort();
    assert_eq!(a_keys, b_keys);
    for (name, left) in a {
        let right = &b[name];
        assert_eq!(left.dtype(), right.dtype(), "{name}");
        assert_eq!(left.shape(), right.shape(), "{name}");
        assert_eq!(
            left.to_bytes().unwrap(),
            right.to_bytes().unwrap(),
            "{name}"
        );
    }
}

/// A directory with `weights.safetensors` only gives the weight map that the
/// same tensors give in a directory with `weights.npz` only: the same names,
/// shapes, dtypes and values.
#[test]
fn a_safetensors_file_gives_the_weights_of_the_same_npz_file() {
    let from_safetensors = tempfile::tempdir().unwrap();
    write_safetensors(from_safetensors.path(), &small_tensors());
    let from_npz = tempfile::tempdir().unwrap();
    write_npz(from_npz.path(), &small_tensors());

    let a = load_weights(from_safetensors.path()).unwrap();
    let b = load_weights(from_npz.path()).unwrap();

    assert_same_weights(&a, &b);
    assert_eq!(a["encoder.conv1.weight"].dtype(), Dtype::F16);
    assert_eq!(a["encoder.conv1.weight"].shape(), [2, 3, 4]);
    assert_eq!(a["decoder.ln.bias"].dtype(), Dtype::F32);
    assert_eq!(a["alignment_heads"].dtype(), Dtype::I32);
    assert_eq!(a["alignment_heads"].shape(), [3, 2]);
}

/// With both files in the directory, the values are those of
/// `weights.safetensors`.
#[test]
fn safetensors_is_read_when_both_files_exist() {
    let dir = tempfile::tempdir().unwrap();
    let mut other = small_tensors();
    other[1] = Tensor::f32("decoder.ln.bias", &[5], 50);
    write_npz(dir.path(), &other);
    write_safetensors(dir.path(), &small_tensors());

    let weights = load_weights(dir.path()).unwrap();

    let expected = small_tensors();
    assert_eq!(
        weights["decoder.ln.bias"].to_bytes().unwrap(),
        expected[1].bytes
    );
    assert_ne!(expected[1].bytes, other[1].bytes);
}

/// A directory with `weights.npz` only still loads.
#[test]
fn an_npz_file_is_read_when_there_is_no_safetensors_file() {
    let dir = tempfile::tempdir().unwrap();
    write_npz(dir.path(), &small_tensors());
    let weights = load_weights(dir.path()).unwrap();
    assert_eq!(weights.len(), 4);
}

/// The server sends a load error to the client: with no weight file it names
/// both file names and never the directory.
#[test]
fn no_weight_file_is_an_error_naming_both_files_and_no_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let err = load_error(dir.path());
    assert!(err.contains("weights.safetensors"), "{err}");
    assert!(err.contains("weights.npz"), "{err}");
    assert!(!err.contains(&path), "{err}");
}

/// A `weights.safetensors` that cannot be read is an error that names the
/// file and not the directory, also when a good `weights.npz` is beside it.
#[test]
fn an_unreadable_safetensors_file_is_an_error_and_is_not_replaced_by_the_npz_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    write_npz(dir.path(), &small_tensors());
    std::fs::write(
        dir.path().join("weights.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();
    let err = load_error(dir.path());
    assert!(err.contains("weights.safetensors"), "{err}");
    assert!(!err.contains(&path), "{err}");
}

/// A dtype that the reader does not support is an error with its own text:
/// it names the file, the tensor and the dtype, and holds no MLX text.
#[test]
fn a_tensor_of_an_unsupported_dtype_is_an_error_naming_the_tensor_and_the_dtype() {
    for (dtype, width) in [("F64", 8), ("I16", 2), ("U16", 2), ("U64", 8)] {
        let dir = tempfile::tempdir().unwrap();
        let tensors = [Tensor::new("encoder.odd", dtype, &[1], vec![0; width])];
        write_safetensors(dir.path(), &tensors);
        assert_eq!(
            load_error(dir.path()),
            format!(
                "weight file error: weights.safetensors: tensor encoder.odd: dtype {dtype} is not supported"
            )
        );
    }
}

/// A boolean or int8 tensor reads as uint8, as in `weights.npz`.
#[test]
fn a_one_byte_tensor_reads_as_uint8_as_in_the_npz_file() {
    let tensors = || {
        vec![
            Tensor::new("mask", "BOOL", &[3], vec![1, 0, 1]),
            Tensor::new("small", "I8", &[2], vec![1, 2]),
        ]
    };
    let from_safetensors = tempfile::tempdir().unwrap();
    write_safetensors(from_safetensors.path(), &tensors());
    let from_npz = tempfile::tempdir().unwrap();
    write_npz(from_npz.path(), &tensors());

    let a = load_weights(from_safetensors.path()).unwrap();
    assert_eq!(a["mask"].dtype(), Dtype::U8);
    assert_same_weights(&a, &load_weights(from_npz.path()).unwrap());
}

/// A `weights.safetensors` that is a broken symlink is an error that names the
/// file, and a good `weights.npz` beside it is not read in its place.
#[test]
fn a_broken_safetensors_symlink_is_an_error_and_is_not_replaced_by_the_npz_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    write_npz(dir.path(), &small_tensors());
    std::os::unix::fs::symlink(
        dir.path().join("missing"),
        dir.path().join("weights.safetensors"),
    )
    .unwrap();
    let err = load_error(dir.path());
    assert!(err.contains("weights.safetensors"), "{err}");
    assert!(!err.contains(&path), "{err}");
}

/// An int64 value that int32 cannot hold is an error that names the tensor,
/// as for `weights.npz`.
#[test]
fn an_int64_value_beyond_int32_is_an_error_naming_the_tensor() {
    let dir = tempfile::tempdir().unwrap();
    write_safetensors(dir.path(), &[Tensor::int64_pairs(&[(0, 1 << 40)])]);
    let err = load_error(dir.path());
    assert!(err.contains("alignment_heads"), "{err}");
    assert!(err.contains(&(1_i64 << 40).to_string()), "{err}");
}

type Heads = Option<Vec<(usize, usize)>>;

/// The heads that the read gives for `tensors` written as safetensors and as npz.
fn heads_of(tensors: &[Tensor]) -> (Heads, Heads) {
    let read = |write: fn(&std::path::Path, &[Tensor])| {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), tensors);
        read_alignment_heads(&load_weights(dir.path()).unwrap(), &config()).unwrap()
    };
    (read(write_safetensors), read(write_npz))
}

/// `alignment_heads` as int64 in `weights.safetensors` gives the same pairs as
/// in `weights.npz`.
#[test]
fn int64_alignment_heads_in_safetensors_give_the_pairs_of_the_npz_file() {
    let (from_safetensors, from_npz) = heads_of(&[Tensor::int64_pairs(&PAIRS)]);
    assert_eq!(from_safetensors, Some(vec![(0, 0), (0, 1)]));
    assert_eq!(from_safetensors, from_npz);
}

/// A shape other than `(N, 2)` and a pair outside the decoder give no heads,
/// for both file kinds.
#[test]
fn bad_alignment_heads_in_safetensors_give_no_heads_as_in_the_npz_file() {
    let three_wide = {
        let bytes = (0..6_i64).flat_map(i64::to_le_bytes).collect();
        Tensor::new("alignment_heads", "I64", &[2, 3], bytes)
    };
    for entry in [
        three_wide,
        Tensor::int64_pairs(&[(1, 0)]),
        Tensor::int64_pairs(&[(0, 2)]),
        Tensor::int64_pairs(&[(-1, 0)]),
        Tensor::new("alignment_heads", "BOOL", &[2, 2], vec![1, 0, 0, 1]),
    ] {
        let (from_safetensors, from_npz) = heads_of(&[entry]);
        assert_eq!(from_safetensors, None);
        assert_eq!(from_npz, None);
    }
}

fn model_dir(write: fn(&std::path::Path, &[Tensor])) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        tiny_config_json(1, 1).to_string(),
    )
    .unwrap();
    write(dir.path(), &model_tensors(Tensor::int64_pairs(&PAIRS)));
    dir
}

/// A snapshot directory with `config.json` and `weights.safetensors` loads
/// as a model, and its alignment heads are those of the `.npz` snapshot.
#[test]
fn a_safetensors_snapshot_loads_as_a_model_with_its_alignment_heads() {
    let heads = |dir: &tempfile::TempDir| match WhisperModel::load(dir.path()) {
        Ok(model) => model.alignment_heads().map(<[_]>::to_vec),
        Err(e) => panic!("the snapshot must load: {e}"),
    };
    let from_safetensors = heads(&model_dir(write_safetensors));
    let from_npz = heads(&model_dir(write_npz));
    assert_eq!(from_safetensors, Some(vec![(0, 0), (0, 1)]));
    assert_eq!(from_safetensors, from_npz);
}
