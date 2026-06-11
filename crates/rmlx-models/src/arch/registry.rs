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
    // 12B unified variant — text decoder is identical to Gemma4ForConditionalGeneration;
    // extra multimodal-embedder tensors (embed_vision, embed_audio, vision_embedder.*)
    // are not read by the text loader and are inert. Text only; vision/audio out of scope.
    "Gemma4UnifiedForConditionalGeneration",
    "Gemma3ForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "LagunaForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
    "BitNetForCausalLM",
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
