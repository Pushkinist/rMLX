use super::*;

// ---------------------------------------------------------------------------
// Mask-mode selection ("additive" is NOT a valid mlx-c mode).
// ---------------------------------------------------------------------------

/// The mlx-c Metal SDPA kernel accepts only these mask_mode strings. A drafter
/// layer must never emit anything else (the old "additive" string aborted every
/// decode step).
const KERNEL_ACCEPTED_MODES: [&str; 3] = ["causal", "array", ""];

#[test]
fn windowed_layer_uses_array_mode() {
    // A present window-bias mask routes through "array" (NOT "additive").
    let bias = make_bias();
    let (mode, arr) = swa_sdpa_mode(Some(&bias));
    assert_eq!(mode, "array");
    assert!(arr.is_some());
    assert!(KERNEL_ACCEPTED_MODES.contains(&mode));
}

#[test]
fn global_or_covered_layer_uses_empty_mode() {
    // Full-attention layers (and sliding layers whose window covers all KV)
    // pass no mask: empty mode, no array.
    let (mode, arr) = swa_sdpa_mode(None);
    assert_eq!(mode, "");
    assert!(arr.is_none());
    assert!(KERNEL_ACCEPTED_MODES.contains(&mode));
}

#[test]
fn no_additive_mode_ever_emitted() {
    // Regression guard: neither branch may select "additive".
    let bias = make_bias();
    for m in [swa_sdpa_mode(Some(&bias)).0, swa_sdpa_mode(None).0] {
        assert_ne!(m, "additive", "mlx-c rejects mask_mode 'additive'");
        assert!(KERNEL_ACCEPTED_MODES.contains(&m));
    }
}

// ---------------------------------------------------------------------------
// Window-bias values — pure mirror of build_swa_mask's banded loop.
// ---------------------------------------------------------------------------

/// Pure mirror of the additive bidirectional-window bias built in
/// `build_swa_mask`. Locks the value convention: 0.0 in-window, -1e30 out of
/// window (NOT f32::NEG_INFINITY — a fully-masked softmax row over -inf is NaN,
/// matching the verifier `crate::layers::build_swa_*` builders).
#[allow(
    clippy::indexing_slicing,
    reason = "bias sized query_len*kv_len; (qi,ki) loop bounds keep the index in range"
)]
fn window_bias(q_off: i32, query_len: i32, kv_len: i32, window: i32) -> Vec<f32> {
    let mut bias = vec![0.0f32; (query_len * kv_len) as usize];
    for qi in 0..query_len {
        let q = q_off + qi;
        for ki in 0..kv_len {
            let dist = q - ki;
            if !(dist > -window && dist < window) {
                bias[(qi * kv_len + ki) as usize] = -1e30;
            }
        }
    }
    bias
}

#[allow(
    clippy::expect_used,
    reason = "test fixture: tiny constant-shape array, failure is a test bug"
)]
fn make_bias() -> Array {
    let data = window_bias(0, 1, 4, 2);
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[1, 1, 1, 4], Dtype::F32).expect("bias array")
}

// Exact bit-pattern compare avoids the float_cmp lint while still asserting
// the precise constants the kernel sees (these cells are assigned literals).
fn allowed(v: f32) -> bool {
    v.to_bits() == 0.0f32.to_bits()
}
fn masked(v: f32) -> bool {
    v.to_bits() == (-1e30f32).to_bits()
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "fixed-shape bias [1,1,1,6]; indices are in-bounds by construction"
)]
fn bias_masks_outside_window_with_finite_penalty() {
    // window=2, query at offset 5, kv_len 6: keys within (q-window, q+window)
    // exclusive are allowed (0.0), the rest get -1e30.
    let bias = window_bias(5, 1, 6, 2);
    // q=5: dist = 5-ki. allowed when -2 < dist < 2 => ki in {4,5} (dist 1,0).
    assert!(allowed(bias[4]));
    assert!(allowed(bias[5]));
    // ki=3 -> dist 2 (not < window) => masked. ki=0 far past => masked.
    assert!(masked(bias[3]));
    assert!(masked(bias[0]));
    // No -inf / NaN anywhere (NaN-avoidance invariant vs f32::NEG_INFINITY).
    assert!(bias.iter().all(|v| v.is_finite()));
}

#[test]
fn bias_all_allowed_when_window_covers_kv() {
    // window large enough to cover every key => all-zero bias.
    let bias = window_bias(0, 1, 4, 64);
    assert!(bias.iter().all(|v| allowed(*v)));
}

// ---------------------------------------------------------------------------
// Plain tied-head LM head — full-vocab argmax over h @ embed_tokens.T.
// Exercises the plain-head path (centroids None) for 26B/31B assistants.
// ---------------------------------------------------------------------------

#[allow(
    clippy::expect_used,
    reason = "test fixture: tiny constant-shape arrays, any failure is a test bug"
)]
fn mk_f32(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("f32 array")
}

/// Build a minimal plain-head drafter: only `embed_tokens`, `draft_hidden`,
/// `device`, and `centroids = None` are load-bearing for `plain_argmax`. The
/// remaining fields are dummies never touched by the head.
fn plain_head_drafter(embed_weight: Array, draft_hidden: usize) -> Gemma4AssistantDrafter {
    let dummy_lin = || Linear::Plain {
        weight: mk_f32(&[0.0], &[1, 1]),
    };
    let dummy_norm = RmsNorm {
        weight: None,
        eps: 1e-6,
    };
    Gemma4AssistantDrafter {
        embed_tokens: Embedding::Plain {
            weight: embed_weight,
        },
        pre_projection: dummy_lin(),
        post_projection: dummy_lin(),
        norm: dummy_norm,
        layers: Vec::new(),
        centroids: None,
        token_ordering: None,
        draft_hidden,
        backbone_hidden: 1,
        n_heads: 1,
        sliding_window: 1,
        rope_sliding_theta: 10000.0,
        num_centroids: 1,
        centroid_top_k: 1,
        vocab_per_centroid: 1,
        device: Device::Cpu,
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: .expect on tiny fixtures, failure is a test bug"
)]
fn plain_argmax_picks_max_dot_product_token() {
    // embed_tokens: vocab=4, draft_hidden=3. Token 2's row aligns best with h.
    let draft_hidden = 3usize;
    let embed = mk_f32(
        &[
            1.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, // token 1
            5.0, 5.0, 5.0, // token 2 — largest dot with all-positive h
            0.0, 0.0, 1.0, // token 3
        ],
        &[4, draft_hidden as i32],
    );
    let drafter = plain_head_drafter(embed, draft_hidden);

    let h = mk_f32(&[1.0, 1.0, 1.0], &[1, 1, draft_hidden as i32]);
    let tok = drafter.plain_argmax(&h).expect("plain_argmax");
    assert_eq!(tok, 2, "max-dot-product token is 2");
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test: .expect on tiny fixtures, failure is a test bug"
)]
fn plain_argmax_respects_hidden_direction() {
    // Different h selects a different token (no centroid shortlist — full vocab).
    let draft_hidden = 3usize;
    let embed = mk_f32(
        &[
            1.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, // token 1
            0.0, 0.0, 1.0, // token 2
        ],
        &[3, draft_hidden as i32],
    );
    let drafter = plain_head_drafter(embed, draft_hidden);

    // h points along the token-1 basis vector → argmax = 1.
    let h = mk_f32(&[0.1, 9.0, 0.2], &[1, 1, draft_hidden as i32]);
    let tok = drafter.plain_argmax(&h).expect("plain_argmax");
    assert_eq!(tok, 1, "h aligned with token-1 basis");
}

// ---------------------------------------------------------------------------
// Round block
// ---------------------------------------------------------------------------

/// The request decides the block, up to what one verify forward can score, and
/// a block with no room for a draft token is refused rather than raised.
#[test]
fn the_round_block_is_bounded_above_and_refused_below_two() {
    for requested in [2, 6, 64, MAX_BLOCK_SIZE] {
        assert_eq!(block_from_request(requested).ok(), Some(requested));
    }
    for requested in [MAX_BLOCK_SIZE + 1, usize::MAX] {
        assert_eq!(block_from_request(requested).ok(), Some(MAX_BLOCK_SIZE));
    }
    for requested in [0, 1] {
        assert!(block_from_request(requested).is_err(), "{requested}");
    }
}

// ---------------------------------------------------------------------------
// Pairing with the verifier: config-only drafters, no weights.
// ---------------------------------------------------------------------------

/// The verifier the config-only drafters below are paired with: Gemma4 E2B.
const PAIRED_VERIFIER: VerifierTrunk = VerifierTrunk {
    hidden_size: 1536,
    num_hidden_layers: 35,
    vocab_size: 262_144,
};

/// A drafter folder holding nothing but a `config.json` shaped like a shipped
/// assistant's: the backbone width at the top level, the drafter's own width and
/// the vocabulary in `text_config`. `backbone_hidden_size` is left out when `None`.
#[allow(
    clippy::expect_used,
    reason = "test-only: the tempdir is this process's own, so a failure to write into it names a broken environment"
)]
fn config_only_drafter(
    backbone_hidden_size: Option<usize>,
    vocab_size: usize,
) -> tempfile::TempDir {
    let mut config = serde_json::json!({
        "architectures": ["Gemma4AssistantForCausalLM"],
        "num_centroids": 2048,
        "text_config": { "hidden_size": 256, "num_hidden_layers": 4, "vocab_size": vocab_size },
    });
    if let Some(width) = backbone_hidden_size {
        config
            .as_object_mut()
            .expect("config is an object")
            .insert("backbone_hidden_size".to_owned(), serde_json::json!(width));
    }
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config.json");
    dir
}

/// The error the loader stops with on a folder that has no weights.
fn load_error(dir: &tempfile::TempDir) -> Error {
    match Gemma4AssistantDrafter::load(dir.path(), PAIRED_VERIFIER, Device::Cpu) {
        Ok(_) => Error::Other("a config-only drafter loaded".to_owned()),
        Err(e) => e,
    }
}

/// A drafter that restates a backbone width or a vocabulary other than the
/// verifier's is refused before a weight is read, as a pairing refusal naming
/// the key and both values.
#[test]
fn a_drafter_built_for_another_base_is_refused_naming_both_values() {
    for (key, backbone, vocab, foreign, held) in [
        ("backbone_hidden_size", Some(2560), 262_144, 2560, 1536),
        ("vocab_size", Some(1536), 256_000, 256_000, 262_144),
    ] {
        let err = load_error(&config_only_drafter(backbone, vocab));
        let Error::SpeculativePairing { reason } = &err else {
            panic!("{key}={foreign}: expected a pairing refusal, got: {err}");
        };
        assert!(
            reason.contains(key)
                && reason.contains(&foreign.to_string())
                && reason.contains(&held.to_string()),
            "the refusal must name {key}, the drafter's {foreign} and the verifier's {held}: \
             {reason}"
        );
    }
}

/// A drafter that agrees with the verifier goes on to its weights, and so does
/// one that does not declare its backbone.
#[test]
fn a_drafter_that_agrees_or_declares_no_backbone_goes_on_to_its_weights() {
    for backbone in [Some(1536), None] {
        let err = load_error(&config_only_drafter(backbone, 262_144));
        let msg = err.to_string();
        assert!(
            !matches!(err, Error::SpeculativePairing { .. }) && msg.contains("shard index"),
            "backbone {backbone:?}: expected the load to reach the weights, got: {msg}"
        );
    }
}

/// Serialise a zero-filled bf16 safetensors file. Offsets follow the header's
/// key order, which the format requires to be contiguous.
fn safetensors_bytes(tensors: &[(&str, Vec<usize>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut sorted: Vec<&(&str, Vec<usize>)> = tensors.iter().collect();
    sorted.sort_by_key(|(name, _)| *name);
    let mut offset = 0usize;
    for (name, shape) in sorted {
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let header = serde_json::Value::Object(header).to_string();
    let mut out = Vec::with_capacity(8 + header.len() + offset);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(out.len() + offset, 0);
    out
}

/// A drafter folder whose config declares no backbone and whose weights hold the
/// embedding and the two projections at the given shapes, and nothing else.
#[allow(
    clippy::expect_used,
    reason = "test-only: the tempdir is this process's own, so a failure to write into it names a broken environment"
)]
fn drafter_with_projections(pre: Vec<usize>, post: Vec<usize>) -> tempfile::TempDir {
    let dir = config_only_drafter(None, 262_144);
    let tensors = [
        ("model.embed_tokens.weight", vec![8, 256]),
        ("pre_projection.weight", pre),
        ("post_projection.weight", post),
    ];
    std::fs::write(
        dir.path().join("model.safetensors"),
        safetensors_bytes(&tensors),
    )
    .expect("write model.safetensors");
    dir
}

/// A projection sized for another backbone is refused whether or not the config
/// declares the backbone: the weights carry the width. The case is an E4B-sized
/// assistant (backbone 2560) over the E2B-sized verifier (1536).
#[test]
fn a_projection_sized_for_another_backbone_is_refused_naming_both_sizes() {
    for (name, pre, post, found, needed) in [
        (
            "pre_projection.weight",
            vec![256, 5120],
            vec![1536, 256],
            5120,
            3072,
        ),
        (
            "post_projection.weight",
            vec![256, 3072],
            vec![2560, 256],
            2560,
            1536,
        ),
    ] {
        let err = load_error(&drafter_with_projections(pre, post));
        let Error::SpeculativePairing { reason } = &err else {
            panic!("{name}: expected a pairing refusal, got: {err}");
        };
        assert!(
            reason.contains(name)
                && reason.contains(&found.to_string())
                && reason.contains(&needed.to_string()),
            "the refusal must name {name}, the found {found} and the needed {needed}: {reason}"
        );
    }
}

/// Projections sized for the verifier go on to the next tensor.
#[test]
fn projections_sized_for_the_verifier_go_on_to_the_next_tensor() {
    let err = load_error(&drafter_with_projections(vec![256, 3072], vec![1536, 256]));
    let msg = err.to_string();
    assert!(
        !matches!(err, Error::SpeculativePairing { .. }) && msg.contains("model.norm.weight"),
        "expected the load to go on to model.norm.weight, got: {msg}"
    );
}
