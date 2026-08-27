// ---------------------------------------------------------------------------
// Known-architecture registry
// ---------------------------------------------------------------------------

/// Every `architectures[0]` value that rMLX can currently handle.
///
/// Kept at module level (rather than inside `load_model`) so that the server
/// startup path can validate a model's architecture before paying the full
/// I/O cost of loading weights. `is_arch_supported` is the public entry-point
/// for that early check.
pub const KNOWN_ARCHS: &[&str] = &[
    "Gemma4ForConditionalGeneration",
    // 12B unified variant — text decoder is identical to Gemma4ForConditionalGeneration.
    // The multimodal-embedder tensors (embed_vision, embed_audio, vision_embedder.*)
    // drive the encoder-free vision and audio front-ends wired in the gemma4::vision
    // and gemma4::audio unified paths (is_unified_arch, VisionBundle::Gemma4Unified,
    // build_unified_inputs_embeds). Vision + audio input are fully supported.
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "LagunaForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "BitNetForCausalLM",
    "MapleForCausalLM",
    // jina-embeddings-v4 is an encoder, NOT a causal LM. It is accepted
    // by the registry/loader gate so the server can route it to the
    // `/v1/embeddings` embedding path, but it has no `Architecture` enum
    // variant and no `Generator` impl — the match arm below rejects any
    // attempt to load it via the generative `load_model` path.
    "JinaEmbeddingsV4Model",
];

/// Returns `true` if `arch` is a known architecture that rMLX can load.
///
/// Used at serve-startup to fail fast before paying I/O cost; the same
/// check also lives inside `load_model` as defense-in-depth.
#[inline]
pub fn is_arch_supported(arch: &str) -> bool {
    KNOWN_ARCHS.contains(&arch)
}

/// Declared arch strings that rMLX deliberately reports under a different,
/// canonical class — an alias, not a mismatch.
///
/// `Architecture::arch_class()` reports the resolved class, so for these pairs
/// the declared and resolved names differ on every load of a perfectly
/// well-formed snapshot. Their consumers all carry an explicit arm for the
/// alias, so nothing is keyed on a name it does not handle. Callers that
/// report a declared-vs-resolved divergence must stay quiet for these, or the
/// signal is noise on a supported model and gets ignored where it matters.
///
/// This is NOT the Qwen3.5 dense/MoE pair: those two names describe genuinely
/// different models, and a snapshot declaring one while building the other is
/// the mismatch worth reporting.
#[inline]
pub(crate) fn is_declared_arch_alias(declared: &str, resolved: &str) -> bool {
    matches!(
        (declared, resolved),
        (
            "Gemma4UnifiedForConditionalGeneration",
            "Gemma4ForConditionalGeneration"
        )
    )
}
