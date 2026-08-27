use super::ssd_attachment_blocked;

#[test]
fn maple_ssd_attachment_is_blocked_until_swa_rings_can_be_hydrated() {
    assert!(ssd_attachment_blocked("MapleForCausalLM"));

    for arch in [
        "Gemma4ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3ForCausalLM",
    ] {
        assert!(
            !ssd_attachment_blocked(arch),
            "established SSD-capable architecture {arch} must remain enabled"
        );
    }
}
