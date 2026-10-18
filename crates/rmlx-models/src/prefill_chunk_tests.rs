use super::*;

#[test]
fn defaults_match_recommendations() {
    // Lock the per-arch defaults so tuning changes are intentional.
    // Test is racy w.r.t. env vars; we only assert when no override
    // is set so CI runs (which set RMLX_PREFILL_CHUNK=64 for VG.2)
    // don't false-fail.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() {
        return;
    }
    assert_eq!(arch_default("qwen3"), Some(1024));
    assert_eq!(arch_default("qwen3_5_moe"), Some(2048));
    assert_eq!(arch_default("qwen3_vl_moe"), Some(512));
    assert_eq!(arch_default("gemma3"), Some(256));
    assert_eq!(arch_default("gemma4"), Some(1024));
    assert_eq!(arch_default("qwen2"), Some(256));
    assert_eq!(arch_default("laguna"), Some(256));
    assert_eq!(arch_default("bitnet"), Some(64));
    assert_eq!(arch_default("unknown_arch"), None);
}

#[test]
fn module_key_for_class_maps_supported_classes() {
    // Each supported config `architectures[0]` class maps to the same
    // module-style key its own generate path passes to `prefill_chunk_for`.
    assert_eq!(
        module_key_for_class("Gemma4ForConditionalGeneration"),
        "gemma4"
    );
    assert_eq!(
        module_key_for_class("Gemma4UnifiedForConditionalGeneration"),
        "gemma4"
    );
    assert_eq!(
        module_key_for_class("Gemma3ForConditionalGeneration"),
        "gemma3"
    );
    assert_eq!(module_key_for_class("Qwen2ForCausalLM"), "qwen2");
    assert_eq!(module_key_for_class("Qwen3ForCausalLM"), "qwen3");
    assert_eq!(module_key_for_class("LagunaForCausalLM"), "laguna");
    assert_eq!(
        module_key_for_class("Qwen3_5MoeForConditionalGeneration"),
        "qwen3_5_moe"
    );
    assert_eq!(
        module_key_for_class("Qwen3_5ForConditionalGeneration"),
        "qwen3_5_moe"
    );
    assert_eq!(module_key_for_class("BitNetForCausalLM"), "bitnet");
    // Qwen3-VL-MoE chunks its image prefill (native tiling → thousands of soft
    // tokens would trip the Metal watchdog in one forward).
    assert_eq!(
        module_key_for_class("Qwen3VLMoeForConditionalGeneration"),
        "qwen3_vl_moe"
    );

    // Unknown classes → "" → FALLBACK chunk, never the oversized gemma4 default.
    assert_eq!(module_key_for_class("JinaEmbeddingsV4Model"), "");
    assert_eq!(module_key_for_class("TotallyUnknownArch"), "");
}

#[test]
fn module_key_resolves_to_arch_default_chunk() {
    // The class→key→chunk chain lands on the arch's known default. Racy w.r.t.
    // env vars; only assert when no global override is set.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() {
        return;
    }
    // qwen3_5_moe resolves to its own 2048 default (GDN kernel handles any T),
    // distinct from gemma4's 1024.
    let key = module_key_for_class("Qwen3_5MoeForConditionalGeneration");
    assert_eq!(prefill_chunk_for(key), 2048);
    assert_eq!(
        prefill_chunk_for(module_key_for_class("Gemma4ForConditionalGeneration")),
        1024
    );
    // Unknown class resolves through "" to FALLBACK, not gemma4's default.
    assert_eq!(
        prefill_chunk_for(module_key_for_class("TotallyUnknownArch")),
        FALLBACK
    );
}

#[test]
fn unknown_arch_falls_back() {
    // Env-var path may override; only assert pure default behaviour.
    if env::var("RMLX_PREFILL_CHUNK").is_ok() || env::var("RMLX_PREFILL_CHUNK_BOGUS").is_ok() {
        return;
    }
    assert_eq!(prefill_chunk_for("bogus"), FALLBACK);
}

/// Every rule in the resolution order reports its own name, and reads the
/// variable that belongs to it.
///
/// The chunk a run prefilled at is logged with the rule that produced it, and
/// the two are read together: a label naming the arch default while an
/// override supplied the number would describe a measurement of somebody's
/// environment as a measurement of the shipped configuration. The env reader
/// is injected, so the two variables can both be set at once — the only way to
/// separate them, since swapping two `Option<usize>` arguments compiles.
#[test]
fn each_resolution_rule_reports_its_own_source() {
    // Both variables set, per-arch differing from global, so precedence is
    // decided rather than defaulted.
    let both = |name: &str| match name {
        "RMLX_PREFILL_CHUNK_QWEN3" => Some(512),
        "RMLX_PREFILL_CHUNK" => Some(256),
        _ => None,
    };
    let global_only = |name: &str| (name == "RMLX_PREFILL_CHUNK").then_some(256);
    let none = |_: &str| None;

    assert_eq!(resolve_with(1024, "qwen3", both), (1024, "adaptive"));
    assert_eq!(resolve_with(0, "qwen3", both), (512, "env_arch"));
    assert_eq!(resolve_with(0, "qwen3", global_only), (256, "env_global"));
    assert_eq!(resolve_with(0, "qwen3", none), (1024, "arch_default"));
    assert_eq!(
        resolve_with(0, "no_such_arch", none),
        (FALLBACK, "fallback")
    );

    // The per-arch name is built from the arch, so another arch does not see
    // qwen3's variable.
    assert_eq!(resolve_with(0, "gemma4", both), (256, "env_global"));
}

#[test]
fn pieces_without_a_cut_are_the_chunks() {
    assert_eq!(prefill_pieces(0, 5, 2, None), vec![0..2, 2..4, 4..5]);
    assert_eq!(prefill_pieces(0, 4, 2, None), vec![0..2, 2..4]);
    assert_eq!(prefill_pieces(3, 8, 4, None), vec![3..7, 7..8]);
    assert_eq!(prefill_pieces(5, 5, 4, None), Vec::<Range<usize>>::new());
    assert_eq!(prefill_pieces(0, 3, 0, None), vec![0..1, 1..2, 2..3]);
}

#[test]
fn a_cut_inside_a_chunk_splits_it() {
    assert_eq!(
        prefill_pieces(0, 3893, 2048, Some(3840)),
        vec![0..2048, 2048..3840, 3840..3893]
    );
    assert_eq!(
        prefill_pieces(0, 700, 1024, Some(512)),
        vec![0..512, 512..700]
    );
    assert_eq!(prefill_pieces(3, 8, 4, Some(5)), vec![3..5, 5..7, 7..8]);
}

#[test]
fn a_cut_on_a_boundary_or_outside_adds_nothing() {
    let plain = prefill_pieces(0, 5, 2, None);
    assert_eq!(prefill_pieces(0, 5, 2, Some(2)), plain, "on a chunk edge");
    assert_eq!(prefill_pieces(0, 5, 2, Some(4)), plain, "on the last edge");
    assert_eq!(prefill_pieces(0, 5, 2, Some(5)), plain, "at the end");
    assert_eq!(prefill_pieces(0, 5, 2, Some(0)), plain, "at the start");
    assert_eq!(prefill_pieces(0, 5, 2, Some(9)), plain, "past the end");
    assert_eq!(
        prefill_pieces(3, 8, 4, Some(3)),
        prefill_pieces(3, 8, 4, None)
    );
    assert_eq!(
        prefill_pieces(3, 8, 4, Some(1)),
        prefill_pieces(3, 8, 4, None),
        "behind the start"
    );
}

/// A speculative prefill forwards the prompt less its last token. When that
/// head ends on the cut, the cut adds no boundary, and the last token is a
/// piece of its own, as in a prefill of the whole prompt.
#[test]
fn a_cut_at_the_end_of_the_speculative_head_matches_the_whole_prompt() {
    assert_eq!(
        prefill_pieces(0, 3840, 2048, Some(3840)),
        prefill_pieces(0, 3840, 2048, None)
    );
    assert_eq!(
        prefill_pieces(0, 3841, 2048, Some(3840)),
        vec![0..2048, 2048..3840, 3840..3841]
    );
}

/// The pieces cover `start..end` once, in order, whatever the cut, and none is
/// empty.
#[test]
fn the_pieces_partition_the_range() {
    for end in [1, 2, 700, 1024, 1025, 4096, 4097] {
        for chunk in [1, 64, 1024, 2048] {
            for cut in [None, Some(0), Some(1), Some(256), Some(512), Some(3840)] {
                let pieces = prefill_pieces(0, end, chunk, cut);
                let mut covered = 0;
                for piece in &pieces {
                    assert!(piece.start < piece.end, "empty piece in {pieces:?}");
                    assert_eq!(piece.start, covered, "gap or overlap in {pieces:?}");
                    assert!(piece.len() <= chunk, "{pieces:?} over the chunk {chunk}");
                    covered = piece.end;
                }
                assert_eq!(covered, end, "end={end} chunk={chunk} cut={cut:?}");
            }
        }
    }
}
