use super::is_maple_ssd_attachment_target;
use crate::prompt_cache::{PromptCacheEntry, SpillSink, SsdHydrate};
use rmlx_kv_ssd::{SsdHydrator, SsdSpiller};

#[test]
fn maple_is_eligible_for_ssd_attachment_dispatch() {
    assert!(is_maple_ssd_attachment_target("MapleForCausalLM"));
    assert!(!is_maple_ssd_attachment_target("UnknownForCausalLM"));
}

#[test]
fn maple_entry_satisfies_spill_and_hydrate_attachment_contracts() {
    fn assert_contracts<E: PromptCacheEntry>()
    where
        SsdSpiller: SpillSink<E>,
        SsdHydrator: SsdHydrate<E>,
    {
    }

    assert_contracts::<crate::maple::prompt_cache::MapleEntry>();
}
