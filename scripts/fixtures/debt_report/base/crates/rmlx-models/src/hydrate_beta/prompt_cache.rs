struct BetaEntry {
    prompt_token_ids: Vec<u32>,
    block_hashes: Vec<u64>,
    kv_caches: Vec<KvCache>,
    is_ssd_hydrated: bool,
}

impl SsdHydrate<BetaEntry> for SsdHydrator {
    fn hydrate(&self, prompt_ids: &[u32], seed: u64) -> Option<BetaEntry> {
        let (block, block_hashes) = self.lookup_seeded(prompt_ids, seed, false)?;
        let HydratedBlock {
            prompt_ids,
            kv_caches,
            lin_caches: _,
        } = block;
        Some(BetaEntry {
            prompt_token_ids: prompt_ids,
            block_hashes,
            kv_caches,
            is_ssd_hydrated: true,
        })
    }
}
