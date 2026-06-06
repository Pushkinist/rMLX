use super::*;
use serde_json::json;

fn primary_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

// ── extract_token_str ─────────────────────────────────────────────────────

#[test]
fn extract_from_string() {
    let v = json!("<bos>");
    assert_eq!(extract_token_str(&v), Some("<bos>".to_owned()));
}

#[test]
fn extract_from_null() {
    let v = json!(null);
    assert_eq!(extract_token_str(&v), None);
}

#[test]
fn extract_from_object_with_content() {
    let v = json!({"content": "<bos>", "lstrip": false, "normalized": false});
    assert_eq!(extract_token_str(&v), Some("<bos>".to_owned()));
}

#[test]
fn extract_from_object_without_content() {
    let v = json!({"lstrip": false});
    assert_eq!(extract_token_str(&v), None);
}

// ── load_tokenizer_config ─────────────────────────────────────────────────

#[test]
fn load_config_from_synthetic_json() {
    let json = r#"{"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"}"#;
    let cfg: TokenizerConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(cfg.bos_token.as_deref(), Some("<bos>"));
    assert_eq!(cfg.eos_token.as_deref(), Some("<eos>"));
}

#[test]
fn load_config_null_tokens() {
    let json = r#"{"bos_token": null, "eos_token": null}"#;
    let cfg: TokenizerConfig = serde_json::from_str(json).expect("parse");
    assert!(cfg.bos_token.is_none());
    assert!(cfg.eos_token.is_none());
}

#[test]
fn load_config_object_tokens() {
    let json = r#"{"bos_token": {"content": "<bos>", "lstrip": false}, "eos_token": {"content": "<eos>", "rstrip": false}}"#;
    let cfg: TokenizerConfig = serde_json::from_str(json).expect("parse");
    assert_eq!(cfg.bos_token.as_deref(), Some("<bos>"));
    assert_eq!(cfg.eos_token.as_deref(), Some("<eos>"));
}

// ── Integration: real tokenizer.json ─────────────────────────────────────

#[test]
fn real_tokenizer_encodes_hello_world() {
    let Some(snap_buf) = primary_snap_dir() else {
        tracing::warn!(
            "RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping real_tokenizer_encodes_hello_world"
        );
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping real_tokenizer_encodes_hello_world"
        );
        return;
    }

    let tk = load_tokenizer(snap).expect("load tokenizer");
    let ids = encode(&tk, "hello world").expect("encode");
    assert!(
        ids.len() > 1,
        "expected >1 tokens for 'hello world', got {ids:?}"
    );
}

// ── PARO model tokenizer regression ──────────────────────────────────────
//
// Reference (HF AutoTokenizer with add_special_tokens=False):
// input: '<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n'
// ids: [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198, 248068, 198]
//
// Key: <|im_start|>=248045, <|im_end|>=248046, <think>=248068 must be
// single tokens, not split into sub-word pieces.
#[test]
fn paro_tokenizer_encodes_chat_template_to_reference_ids() {
    let Some(snap_buf) =
        std::env::var_os("RMLX_TEST_MODEL_QWEN36_PARO").map(std::path::PathBuf::from)
    else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36_PARO not set — skipping paro_tokenizer_encodes_chat_template_to_reference_ids");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(path = %snap.display(), "PARO snapshot absent — skipping");
        return;
    }
    let tk = load_tokenizer(snap).expect("load PARO tokenizer");
    let rendered =
        "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let ids = encode(&tk, rendered).expect("encode");
    let expected: Vec<u32> = vec![
        248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198, 248068, 198,
    ];
    assert_eq!(
        ids, expected,
        "PARO tokenizer IDs diverge from HF reference.\n  got (len={}):      {:?}\n  expected (len={}): {:?}",
        ids.len(), ids, expected.len(), expected
    );
}

// ── Regression: tokenizer must encode the HF reference string to exact IDs ──
//
// Reference (HF tokenizer, `add_special_tokens=False`):
// input: '<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n'
// token ids: [2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862,
// 236778, 236881, 106, 107, 105, 4368, 107] (length 16)
//
// Skip gracefully when the primary snapshot is absent.
#[test]
fn gemma4_tokenizer_encodes_rendered_string_to_reference_ids() {
    let Some(snap_buf) = primary_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping gemma4_tokenizer_encodes_rendered_string_to_reference_ids");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping gemma4_tokenizer_encodes_rendered_string_to_reference_ids"
        );
        return;
    }
    let tk = load_tokenizer(snap).expect("load tokenizer");
    let rendered = "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n";
    let ids = encode(&tk, rendered).expect("encode");
    let expected: Vec<u32> = vec![
        2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368,
        107,
    ];
    assert_eq!(
        ids, expected,
        "tokenizer IDs diverge from HF reference.\n  got (len={}):      {:?}\n  expected (len={}): {:?}",
        ids.len(), ids, expected.len(), expected
    );
}
