pub struct ProbeStep {
    pub token: u32,
}

pub fn dflash2_generate(
    prompt_ids: &[u32],
    n_tokens: usize,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n_tokens {
        let next = prompt_ids.last().copied().unwrap_or(0) + 1;
        if step_fn(&ProbeStep { token: next }).is_none() {
            break;
        }
        out.push(next);
    }
    out
}
