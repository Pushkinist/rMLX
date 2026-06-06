use super::*;

const SAMPLE: &str = r#"
[[model]]
namespace = "mlx-community"
name = "gemma-4-e2b-it-mxfp8"
arch = "Gemma4 small"
weight_quant_display = "mxfp8 g32"
order = 1
unsupported = [
  { backend = "ollama", reason = "no mxfp8 support" },
]

[[model]]
namespace = "z-lab"
name = "Qwen3.6-27B-PARO"
arch = "Qwen3.5MoE"
weight_quant_display = "paroquant int4"
order = 2
aliases = [
  { namespace = "hf", name = "z-lab/Qwen3.6-27B-PARO" },
]
"#;

#[test]
fn parses_two_models() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    assert_eq!(s.models.len(), 2);
    assert_eq!(s.models[0].order, 1);
    assert_eq!(s.models[1].order, 2);
}

#[test]
fn primary_namespace_matches() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    assert!(s.matches("mlx-community", "gemma-4-e2b-it-mxfp8").is_some());
    assert!(s.matches("z-lab", "Qwen3.6-27B-PARO").is_some());
}

#[test]
fn alias_matches() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    let hit = s.matches("hf", "z-lab/Qwen3.6-27B-PARO").unwrap();
    assert_eq!(hit.namespace, "z-lab");
    assert_eq!(hit.name, "Qwen3.6-27B-PARO");
}

#[test]
fn miss_when_not_in_scope() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    assert!(s.matches("mlx-community", "Laguna-XS.2-mxfp8").is_none());
}

#[test]
fn unsupported_backend_lookup() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    let m = s.matches("mlx-community", "gemma-4-e2b-it-mxfp8").unwrap();
    assert!(m.is_backend_unsupported("ollama"));
    assert!(!m.is_backend_unsupported("rmlx"));
}

#[test]
fn duplicate_order_rejected() {
    let bad = r#"
[[model]]
namespace = "a"
name = "x"
arch = "X"
weight_quant_display = "q"
order = 1

[[model]]
namespace = "b"
name = "y"
arch = "Y"
weight_quant_display = "q"
order = 1
"#;
    let err = ScopeFile::parse(bad).unwrap_err();
    assert!(err.contains("duplicate order"));
}

#[test]
fn display_id_uses_double_underscore() {
    let s = ScopeFile::parse(SAMPLE).unwrap();
    assert_eq!(
        s.models[0].display_id(),
        "mlx-community__gemma-4-e2b-it-mxfp8"
    );
}
