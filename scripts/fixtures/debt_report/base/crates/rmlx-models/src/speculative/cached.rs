struct ProbeStep {
    token: u32,
}

fn spec_generate_greedy_cached(
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

fn spec_generate_stochastic_cached(
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
